from __future__ import annotations

import argparse
import io
import sqlite3
from datetime import datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nailong_classifier.bot import (
    build_catalog_post,
    build_correction_confirmation_card,
    build_daily_stats_card,
    build_prediction_card,
    format_catalog_caption,
    format_chain_caption,
    format_daily_stats,
    format_existing_observation,
    format_help_message,
    has_direct_image,
    is_catalog_command,
    is_daily_stats_command,
    is_help_command,
    is_help_request,
    media_response_mode,
    parse_asset_ids,
    parse_correction_card_action,
    parse_correction_command,
    parse_daily_report_time,
    parse_douyin_video_urls,
    should_reply_media_error,
)
from nailong_classifier.model import Prediction
from nailong_classifier.storage import LOCAL_TIMEZONE, BotStore, DailyStats


def make_image_bytes(color: str, *, image_format: str = "PNG") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (48, 48), color).save(output, format=image_format)
    return output.getvalue()


def walk_card(value):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from walk_card(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from walk_card(nested)


def card_text(card: dict) -> str:
    return "\n".join(
        node["content"]
        for node in walk_card(card)
        if isinstance(node.get("content"), str)
    )


def record(
    store: BotStore,
    image_bytes: bytes,
    *,
    message_id: str,
    label: str,
    chat_id: str = "chat-a",
    chat_type: str = "group",
):
    return store.record_observation(
        image_bytes=image_bytes,
        source_message_id=message_id,
        chat_id=chat_id,
        chat_type=chat_type,
        sender_id="user-a",
        predicted_label=label,
        confidence=0.91,
        accepted=True,
    )


def test_hash_deduplicates_assets_and_message_events(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    image_bytes = make_image_bytes("yellow")

    first = record(store, image_bytes, message_id="message-1", label="nailong")
    second = record(store, image_bytes, message_id="message-2", label="nailong")
    duplicate_event = record(store, image_bytes, message_id="message-1", label="nailong")

    assert first.is_new_asset is True
    assert second.is_new_asset is False
    assert second.asset.id == first.asset.id
    assert second.asset.seen_count == 2
    assert duplicate_event.is_new_observation is False
    assert duplicate_event.asset.seen_count == 2
    assert len(list((tmp_path / "assets").iterdir())) == 1


def test_store_migrates_existing_catalog_document_sync_table(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(tmp_path / "bot.sqlite3") as connection:
        connection.execute(
            """
            CREATE TABLE catalog_document_sync (
                document_id TEXT NOT NULL,
                asset_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                image_block_id TEXT,
                file_token TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at TEXT NOT NULL,
                synced_at TEXT,
                PRIMARY KEY (document_id, asset_id)
            )
            """
        )

    BotStore(tmp_path)

    with sqlite3.connect(tmp_path / "bot.sqlite3") as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(catalog_document_sync)"
            ).fetchall()
        }
    assert "parent_block_id" in columns
    assert "duplicate_of" in columns


def test_video_link_pushes_use_a_persistent_rolling_per_chat_limit(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    start = datetime(2026, 10, 23, 10, 0, tzinfo=LOCAL_TIMEZONE)

    first = store.reserve_video_link_push(
        "chat-a", window_seconds=3600, max_pushes=2, now=start
    )
    second = store.reserve_video_link_push(
        "chat-a", window_seconds=3600, max_pushes=2, now=start + timedelta(minutes=30)
    )
    blocked = store.reserve_video_link_push(
        "chat-a", window_seconds=3600, max_pushes=2, now=start + timedelta(minutes=40)
    )
    other_chat = store.reserve_video_link_push(
        "chat-b", window_seconds=3600, max_pushes=2, now=start + timedelta(minutes=40)
    )

    assert first is not None
    assert second is not None
    assert blocked is None
    assert other_chat is not None
    store.release_video_link_push(second)
    assert (
        store.reserve_video_link_push(
            "chat-a", window_seconds=3600, max_pushes=2, now=start + timedelta(minutes=40)
        )
        is not None
    )
    assert (
        store.reserve_video_link_push(
            "chat-a", window_seconds=3600, max_pushes=2, now=start + timedelta(hours=1)
        )
        is not None
    )


def test_parse_douyin_video_urls_accepts_share_domains_and_rejects_other_links() -> None:
    urls = parse_douyin_video_urls(
        "https://v.douyin.com/example-one/, https://www.douyin.com/video/123"
    )

    assert urls == [
        "https://v.douyin.com/example-one/",
        "https://www.douyin.com/video/123",
    ]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_douyin_video_urls("https://example.com/video")


def test_chain_rotation_prefers_collected_assets_without_repeating(
    tmp_path: Path,
) -> None:
    store = BotStore(tmp_path)
    seed_assets = [
        store.register_seed(make_image_bytes(color), "nailong")
        for color in ("gold", "orange")
    ]
    collected_assets = [
        record(
            store,
            make_image_bytes(color),
            message_id=f"collected-{index}",
            label="nailong",
        ).asset
        for index, color in enumerate(("yellow", "pink"))
    ]

    chosen_assets = []
    for index in range(4):
        asset = store.next_chain_asset(
            chat_id="chat-a",
            label="nailong",
            exclude_sha256=(
                collected_assets[0].sha256 if index == 0 else None
            ),
        )
        assert asset is not None
        chosen_assets.append(asset)
        store.record_chain_reply(
            chat_id="chat-a",
            label="nailong",
            asset_id=asset.id,
            response_message_id=f"response-{index}",
        )

    assert chosen_assets[0].id == collected_assets[1].id
    assert chosen_assets[1].id == collected_assets[0].id
    assert {asset.id for asset in chosen_assets[2:]} == {
        asset.id for asset in seed_assets
    }
    assert len({asset.id for asset in chosen_assets}) == 4

    next_cycle = store.next_chain_asset(chat_id="chat-a", label="nailong")
    other_chat = store.next_chain_asset(chat_id="chat-b", label="nailong")
    assert next_cycle is not None
    assert next_cycle.id in {asset.id for asset in chosen_assets}
    assert other_chat is not None
    assert other_chat.origin == "collected"


def test_correction_accepts_source_or_response_message_id(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    observation = record(
        store,
        make_image_bytes("pink"),
        message_id="source-message",
        label="naiwa",
    )
    store.attach_response(observation.id, "response-message")

    correction = store.correct_label("response-message", "nailong")

    assert correction is not None
    assert correction.old_label == "naiwa"
    assert correction.new_label == "nailong"
    assert store.catalog_stats().total == 1
    assert store.random_asset("nailong").id == observation.asset.id
    corrected_observation = record(
        store,
        make_image_bytes("pink"),
        message_id="source-message",
        label="naiwa",
    )
    prediction = Prediction(
        label="naiwa",
        confidence=0.6,
        accepted=False,
        scores={"nailong": 0.35, "naiwa": 0.6, "other": 0.05},
    )
    assert format_existing_observation(corrected_observation, prediction) == (
        f"这张图已经确认是奶龙，图鉴编号 #{observation.asset.id:04d}。"
    )
    assert store.correct_label("missing-message", "other") is None


def test_daily_stats_are_scoped_to_the_current_chat(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    nailong = record(
        store,
        make_image_bytes("yellow"),
        message_id="nailong-a",
        label="nailong",
    )
    record(
        store,
        make_image_bytes("pink"),
        message_id="naiwa-a",
        label="naiwa",
    )
    record(
        store,
        make_image_bytes("gray"),
        message_id="other-a",
        label="other",
    )
    record(
        store,
        make_image_bytes("green"),
        message_id="nailong-b",
        label="nailong",
        chat_id="chat-b",
    )

    stats = store.daily_stats("chat-a")

    assert stats.total == 3
    assert stats.nailong == 1
    assert stats.naiwa == 1
    assert stats.other == 1
    assert str(stats.milk_percent) == "66.7"
    assert stats.new_nailong == 1
    assert "今日含奶量：66.7%" in format_daily_stats(stats)
    assert f"#{nailong.asset.id:04d}" in format_chain_caption(nailong)


def test_catalog_uses_collected_milk_assets_and_builds_multi_image_post(
    tmp_path: Path,
) -> None:
    store = BotStore(tmp_path)
    store.register_seed(make_image_bytes("gold"), "nailong")
    record(
        store,
        make_image_bytes("yellow"),
        message_id="collected-nailong",
        label="nailong",
    )
    record(
        store,
        make_image_bytes("pink"),
        message_id="collected-naiwa",
        label="naiwa",
    )
    store.record_observation(
        image_bytes=make_image_bytes("green"),
        source_message_id="pending-milk",
        chat_id="chat-a",
        chat_type="group",
        sender_id="user-a",
        predicted_label="naiwa",
        confidence=0.6,
        accepted=False,
    )
    store.record_observation(
        image_bytes=make_image_bytes("gray"),
        source_message_id="confirmed-other",
        chat_id="chat-a",
        chat_type="group",
        sender_id="user-a",
        predicted_label="naiwa",
        confidence=0.6,
        accepted=False,
    )
    store.correct_label("confirmed-other", "other")

    stats = store.catalog_stats()
    assets = store.catalog_assets()
    image_keys = [f"img_{index}" for index in range(len(assets))]
    post = build_catalog_post(assets, image_keys, stats)

    assert stats.total == 2
    assert stats.nailong == 1
    assert stats.naiwa == 1
    assert stats.today_new == 2
    assert len(assets) == 2
    assert all(asset.origin == "collected" for asset in assets)
    assert sorted(asset.label for asset in assets) == [
        "nailong",
        "naiwa",
    ]
    assert post["zh_cn"]["title"] == "奶龙图鉴"
    post_images = [
        node["image_key"]
        for row in post["zh_cn"]["content"]
        for node in row
        if node["tag"] == "img"
    ]
    assert post_images == image_keys
    assert format_catalog_caption(stats, len(assets)) == (
        "共收录 2 张：奶龙 1 · 奶蛙 1。"
        "今日新增 2 张，当前展示最近 2 张。"
    )


def test_catalog_document_sync_tracks_baseline_and_resumes_stages(
    tmp_path: Path,
) -> None:
    store = BotStore(tmp_path)
    baseline = record(
        store,
        make_image_bytes("yellow"),
        message_id="baseline",
        label="nailong",
    ).asset
    pending = record(
        store,
        make_image_bytes("pink"),
        message_id="pending",
        label="naiwa",
    ).asset
    other = record(
        store,
        make_image_bytes("gray"),
        message_id="other",
        label="other",
    ).asset
    document_id = "doc-a"

    assert store.mark_catalog_document_baseline(
        document_id=document_id,
        asset_ids=[baseline.id, baseline.id],
    ) == 1
    assert store.mark_catalog_document_baseline(
        document_id=document_id,
        asset_ids=[baseline.id],
    ) == 0

    records = store.pending_catalog_document_sync(document_id=document_id)
    assert [record.asset.id for record in records] == [pending.id]
    assert records[0].status == "pending"

    store.mark_catalog_document_target_assigned(
        document_id=document_id,
        asset_id=pending.id,
        parent_block_id="column-a",
    )
    store.mark_catalog_document_block_created(
        document_id=document_id,
        asset_id=pending.id,
        image_block_id="block-a",
    )
    records = store.pending_catalog_document_sync(document_id=document_id)
    assert records[0].status == "block_created"
    assert records[0].parent_block_id == "column-a"
    assert records[0].image_block_id == "block-a"

    store.mark_catalog_document_media_uploaded(
        document_id=document_id,
        asset_id=pending.id,
        image_block_id="block-a",
        file_token="file-a",
    )
    store.mark_catalog_document_sync_failed(
        document_id=document_id,
        asset_id=pending.id,
        error="temporary error",
    )
    records = store.pending_catalog_document_sync(document_id=document_id)
    assert records[0].status == "failed"
    assert records[0].image_block_id == "block-a"
    assert records[0].file_token == "file-a"
    assert records[0].attempt_count == 1

    store.mark_catalog_document_synced(
        document_id=document_id,
        asset_id=pending.id,
        image_block_id="block-a",
        file_token="file-a",
    )
    assert store.pending_catalog_document_sync(document_id=document_id) == []
    assert store.catalog_document_sync_counts(document_id=document_id) == {
        "baseline": 1,
        "synced": 1,
    }

    store.correct_label("other", "naiwa")
    records = store.pending_catalog_document_sync(document_id=document_id)
    assert [record.asset.id for record in records] == [other.id]

    # 内容和文档里已有图相同的素材标记为 duplicate 后，不再进入待同步列表。
    store.mark_catalog_document_duplicate(
        document_id=document_id,
        asset_id=other.id,
        duplicate_of="file-a",
    )
    assert store.pending_catalog_document_sync(document_id=document_id) == []
    assert store.catalog_document_sync_counts(document_id=document_id)["duplicate"] == 1


def test_catalog_document_image_hash_cache(tmp_path: Path) -> None:
    store = BotStore(tmp_path)

    store.record_catalog_document_image_hash(
        document_id="doc-a",
        file_token="file-a",
        content_hash="ab",
    )
    store.record_catalog_document_image_hash(
        document_id="doc-a",
        file_token="file-broken",
        content_hash=None,
    )
    store.record_catalog_document_image_hash(
        document_id="doc-b",
        file_token="file-b",
        content_hash="cd",
    )

    assert store.catalog_document_image_hashes(document_id="doc-a") == {
        "file-a": "ab",
        "file-broken": None,
    }


def test_commands_are_exact_and_support_correction_aliases() -> None:
    assert parse_correction_command("纠正为奶龙") == "nailong"
    assert parse_correction_command("改成 奶蛙") == "naiwa"
    assert parse_correction_command("这是其他") == "other"
    assert parse_correction_command("帮我纠正为奶龙") is None
    assert is_daily_stats_command(" 今日含奶量 ") is True
    assert is_catalog_command("查看奶龙图鉴") is True
    assert is_help_command("HELP") is True
    assert is_help_command("帮助") is True
    assert is_help_request("help", mentioned_bot=True) is True
    assert is_help_request("help", mentioned_bot=False) is False
    assert "每天 23:55 也会自动播报" in format_help_message(
        time(hour=23, minute=55)
    )
    assert has_direct_image(
        SimpleNamespace(resources=[SimpleNamespace(type="sticker")])
    )
    assert not has_direct_image(SimpleNamespace(resources=[]))
    assert should_reply_media_error(mentioned_bot=False, chat_type="group") is False
    assert should_reply_media_error(mentioned_bot=True, chat_type="group") is True
    assert should_reply_media_error(mentioned_bot=False, chat_type="p2p") is True
    assert parse_daily_report_time("23:55") == time(hour=23, minute=55)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_daily_report_time("24:00")
    assert parse_asset_ids("") == []
    assert parse_asset_ids("3, 1,3") == [3, 1]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_asset_ids("3,bad")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_asset_ids("0")
    assert (
        media_response_mode(
            mentioned_bot=True,
            chat_type="group",
            label="nailong",
        )
        == "prediction"
    )
    assert (
        media_response_mode(
            mentioned_bot=False,
            chat_type="group",
            label="nailong",
        )
        == "chain"
    )
    assert (
        media_response_mode(
            mentioned_bot=False,
            chat_type="p2p",
            label="naiwa",
        )
        == "chain"
    )
    assert (
        media_response_mode(
            mentioned_bot=False,
            chat_type="group",
            label="other",
        )
        == "silent"
    )
    assert (
        media_response_mode(
            mentioned_bot=False,
            chat_type="p2p",
            label="other",
        )
        == "prediction"
    )


def test_daily_stats_card_uses_card2_metrics_and_native_progress_chart() -> None:
    stats = DailyStats(total=10, nailong=4, naiwa=3, other=3, new_nailong=2)

    card = build_daily_stats_card(stats)

    assert card["schema"] == "2.0"
    assert "header" not in card
    assert card["config"]["summary"]["content"] == "今日含奶量 70.0%，共识别 10 张"
    root = card["body"]["elements"][0]
    assert root["tag"] == "interactive_container"
    assert root["background_style"] == "wb_handdrawn_vibes_canvas"
    chart = next(node for node in walk_card(card) if node.get("tag") == "chart")
    assert chart["chart_spec"]["type"] == "bar"
    assert [row["value"] for row in chart["chart_spec"]["data"]["values"]] == [
        70.0,
        30.0,
    ]
    text = card_text(card)
    assert "今日含奶量" in text
    assert "奶龙" in text and ">4<" in text
    assert "奶蛙" in text and ">3<" in text
    assert "其他" in text


def test_prediction_card_shows_confidence_concentration_and_rarity(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    observation = record(
        store,
        make_image_bytes("yellow"),
        message_id="card-nailong",
        label="nailong",
    )
    prediction = Prediction(
        label="nailong",
        confidence=0.61,
        accepted=True,
        scores={"nailong": 0.61, "naiwa": 0.30, "other": 0.09},
    )

    card = build_prediction_card(
        observation,
        prediction,
        store.asset_rarity(observation.asset.id),
        image_key="img_preview",
    )

    assert card["schema"] == "2.0"
    assert "header" not in card
    assert card["config"]["summary"]["content"] == (
        "识别结果：奶龙，置信度 61.0%，奶含量 91%"
    )
    text = card_text(card)
    assert "识别结果：奶龙" in text
    assert "这是一张奶龙图" in text
    assert "稀有度：⭐⭐⭐⭐⭐ 传说" in text
    assert "群里出现过 1 次" in text
    images = [node for node in walk_card(card) if node.get("tag") == "img"]
    assert [image["img_key"] for image in images] == ["img_preview"]
    correction_controls = [
        node
        for node in walk_card(card)
        if node.get("tag") == "interactive_container"
        and node.get("behaviors")
        and node["behaviors"][0].get("type") == "callback"
    ]
    assert len(correction_controls) == 3
    assert {
        control["behaviors"][0]["value"]["label"] for control in correction_controls
    } == {"nailong", "naiwa", "other"}
    chart = next(node for node in walk_card(card) if node.get("tag") == "chart")
    assert [row["value"] for row in chart["chart_spec"]["data"]["values"]] == [
        91.0,
        9.0,
    ]


def test_uncertain_prediction_card_uses_warning_header_without_rarity(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    observation = record(
        store,
        make_image_bytes("pink"),
        message_id="card-uncertain",
        label="naiwa",
    )
    prediction = Prediction(
        label="naiwa",
        confidence=0.52,
        accepted=False,
        scores={"nailong": 0.31, "naiwa": 0.52, "other": 0.17},
    )

    card = build_prediction_card(observation, prediction, None)

    assert "识别结果暂不确定" in card_text(card)
    assert all(node.get("tag") != "hr" for node in walk_card(card))


def test_corrected_prediction_card_uses_confirmed_label_color(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    image_bytes = make_image_bytes("yellow")
    record(
        store,
        image_bytes,
        message_id="card-before-correction",
        label="naiwa",
    )
    store.correct_label("card-before-correction", "nailong")
    observation = record(
        store,
        image_bytes,
        message_id="card-after-correction",
        label="naiwa",
    )
    prediction = Prediction(
        label="naiwa",
        confidence=0.52,
        accepted=False,
        scores={"nailong": 0.31, "naiwa": 0.52, "other": 0.17},
    )

    card = build_prediction_card(observation, prediction, None)

    assert "已确认是奶龙" in card_text(card)
    assert not any(
        node.get("behaviors")
        and node["behaviors"][0].get("type") == "callback"
        for node in walk_card(card)
    )


def test_correction_action_parser_rejects_invalid_payloads() -> None:
    assert parse_correction_card_action(None) is None
    assert parse_correction_card_action({"action": "other"}) is None
    assert parse_correction_card_action(
        {"action": "correct_label", "observation_id": True, "label": "nailong"}
    ) is None
    assert parse_correction_card_action(
        {"action": "correct_label", "observation_id": 1, "label": "bad"}
    ) is None
    assert parse_correction_card_action(
        {
            "action": "correct_label",
            "observation_id": 3,
            "label": "naiwa",
            "image_key": "img_card",
        }
    ) == (3, "naiwa", "img_card")


def test_card_correction_is_bound_to_sent_message_and_builds_confirmation(
    tmp_path: Path,
) -> None:
    store = BotStore(tmp_path)
    observation = record(
        store,
        make_image_bytes("yellow"),
        message_id="source-card",
        label="naiwa",
    )
    store.attach_response(observation.id, "response-card")

    assert store.correct_card_label("wrong-card", observation.id, "nailong") is None
    correction = store.correct_card_label("response-card", observation.id, "nailong")
    assert correction is not None
    updated = store.observation_by_id(observation.id)
    assert updated is not None
    assert updated.asset.label == "nailong"
    assert updated.asset.corrected is True

    card = build_correction_confirmation_card(
        updated,
        store.asset_rarity(updated.asset.id),
        image_key="img_confirmed",
    )
    text = card_text(card)
    assert "已确认：奶龙" in text
    assert "人工校正已生效" in text
    assert "img_confirmed" in str(card)


def test_uncorrected_duplicate_keeps_the_model_uncertainty_message(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    image_bytes = make_image_bytes("green")
    first = store.record_observation(
        image_bytes=image_bytes,
        source_message_id="uncertain-message",
        chat_id="chat-a",
        chat_type="group",
        sender_id="user-a",
        predicted_label="naiwa",
        confidence=0.6,
        accepted=False,
    )
    duplicate = store.record_observation(
        image_bytes=image_bytes,
        source_message_id="uncertain-message",
        chat_id="chat-a",
        chat_type="group",
        sender_id="user-a",
        predicted_label="naiwa",
        confidence=0.6,
        accepted=False,
    )
    prediction = Prediction(
        label="naiwa",
        confidence=0.6,
        accepted=False,
        scores={"nailong": 0.35, "naiwa": 0.6, "other": 0.05},
    )

    assert first.is_new_observation is True
    assert duplicate.is_new_observation is False
    assert format_existing_observation(duplicate, prediction) == (
        "我不太确定。最高候选是「奶蛙」，置信度 60.0%。"
    )


def test_daily_reports_target_active_groups_once_per_day(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    record(
        store,
        make_image_bytes("yellow"),
        message_id="group-message",
        label="nailong",
        chat_id="group-chat",
    )
    record(
        store,
        make_image_bytes("pink"),
        message_id="p2p-message",
        label="naiwa",
        chat_id="p2p-chat",
        chat_type="p2p",
    )
    report_day = datetime.now(LOCAL_TIMEZONE).date().isoformat()

    targets = store.pending_daily_report_targets(report_day)

    assert [target.chat_id for target in targets] == ["group-chat"]
    store.mark_daily_report_sent(
        chat_id="group-chat",
        report_day=report_day,
        response_message_id="report-message",
    )
    assert store.pending_daily_report_targets(report_day) == []
