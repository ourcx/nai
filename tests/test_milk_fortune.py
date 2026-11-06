from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta
from decimal import Decimal

from test_hdr_bot import run_mock_bot

from nailong_classifier.application.bot import (
    format_fortune_collection,
    format_milk_fortune,
)
from nailong_classifier.domain.milk_fortune import (
    FORTUNE_CHARACTERS,
    FORTUNE_FACES,
    FORTUNE_POOLS,
    FORTUNE_TEXT_FIELDS,
    NAIWA_SETS,
    THEME_FACES,
    fortune_text_combination_count,
    make_milk_fortune,
)
from nailong_classifier.domain.milk_fortune_memes import MEME_TITLES, THEME_SETS
from nailong_classifier.persistence.storage import LOCAL_TIMEZONE, BotStore


def test_daily_snapshot_survives_concurrent_draws_and_new_content(tmp_path, monkeypatch):
    store = BotStore(tmp_path)
    options = dict(chat_id="chat-a", sender_id="user-a", day="2026-11-04")
    with ThreadPoolExecutor(max_workers=8) as workers:
        fortunes = list(workers.map(lambda _: store.draw_milk_fortune(**options), range(16)))
    assert all(fortune == fortunes[0] for fortune in fortunes)
    monkeypatch.setattr(
        "nailong_classifier.persistence.storage.make_milk_fortune",
        lambda **_: replace(fortunes[0], message="新文案", score=100),
    )
    restored = BotStore(tmp_path)
    assert restored.draw_milk_fortune(**options) == fortunes[0]
    collection = restored.milk_fortune_collection(
        chat_id="chat-a", sender_id="user-a", today="2026-11-04"
    )
    assert collection.total == 1
    assert collection.streak == 1
    with store._connect() as connection:
        row = dict(connection.execute("SELECT * FROM milk_fortunes").fetchone())
    assert "chat-a" not in str(row) and "user-a" not in str(row)
    assert row["payload_version"] == 3


def test_collection_isolation_pages_streaks_and_saved_details(tmp_path):
    store = BotStore(tmp_path)
    start = date(2026, 10, 26)
    for offset in range(9):
        day = (start + timedelta(days=offset)).isoformat()
        store.draw_milk_fortune(chat_id="chat-a", sender_id="user-a", day=day)
    options = dict(chat_id="chat-a", sender_id="user-a")
    yesterday = store.milk_fortune_collection(**options, today="2026-11-04", page=2)
    assert yesterday.total == 9 and yesterday.pages == 2
    assert len(yesterday.entries) == 3
    assert yesterday.streak == yesterday.longest_streak == 9
    assert "一周不掉奶" in format_fortune_collection(yesterday)
    missed = store.milk_fortune_collection(**options, today="2026-11-05")
    assert missed.streak == 0 and missed.longest_streak == 9
    store.draw_milk_fortune(**options, day="2026-11-05")
    assert store.milk_fortune_collection(**options, today="2026-11-05").streak == 1
    assert store.milk_fortune_on(**options, day="2026-11-04") is None
    assert store.milk_fortune_on(**options, day="2026-10-26") is not None
    for isolated in (
        dict(chat_id="chat-b", sender_id="user-a"),
        dict(chat_id="chat-a", sender_id="user-b"),
    ):
        assert store.milk_fortune_collection(**isolated, today="2026-11-05").total == 0


def test_complete_sets_and_legacy_scores(tmp_path, monkeypatch):
    store = BotStore(tmp_path)
    original = make_milk_fortune(chat_id="chat", sender_id="user", day="2026-11-04")
    digest = hashlib.sha256(b"2026-11-04\0chat\0user").digest()
    assert original.score == int(Decimal(int.from_bytes(digest[0:4], "big")) % Decimal(101))
    assert original.banter == int(Decimal(int.from_bytes(digest[4:8], "big")) % Decimal(101))
    assert all(word in format_milk_fortune(original, day="2026-11-04")
               for word in (original.special_luck_label, "幸运物", "今日小事件",
                            original.face, "仅供娱乐"))
    for offset, (set_name, face) in enumerate(FORTUNE_FACES):
        monkeypatch.setattr(
            "nailong_classifier.persistence.storage.make_milk_fortune",
            lambda set_name=set_name, face=face, **_: replace(
                original, set_name=set_name, face=face
            ),
        )
        store.draw_milk_fortune(
            chat_id="chat", sender_id="user",
            day=(date(2026, 10, 1) + timedelta(days=offset)).isoformat(),
        )
    collection = store.milk_fortune_collection(
        chat_id="chat", sender_id="user", today="2026-10-16"
    )
    assert len(collection.faces) == 128
    assert len(collection.completed_sets) == 20


def test_collection_commands_handle_empty_and_invalid_arguments(tmp_path, monkeypatch):
    for command, expected in (
        ("奶签收藏册", "还没有收藏"), ("奶签历史 0", "页码"),
        ("奶签历史 999999999", "页码"), ("奶签详情 昨天", "日期格式"),
        ("奶签详情 2026-02-30", "日期格式"), ("奶签详情 2026-11-04", "没有收藏"),
    ):
        sent, _ = run_mock_bot(
            tmp_path, monkeypatch, text=command, mentioned_bot=True,
            random_push_probability=0,
        )
        assert len(sent) == 1
        assert expected in str(sent[0][0])


def test_naiwa_fortunes_have_their_own_content_and_both_characters_can_appear():
    fortunes = [
        make_milk_fortune(chat_id="chat", sender_id=str(index), day="2026-11-04")
        for index in range(24)
    ]
    assert {fortune.character for fortune in fortunes} == set(FORTUNE_CHARACTERS)
    assert len({fortune.daily_event for fortune in fortunes}) > 8
    assert len({fortune.mood for fortune in fortunes}) > 8
    for index, fortune in enumerate(fortunes):
        if fortune.character == "奶蛙":
            assert fortune.set_name in NAIWA_SETS
            assert fortune.special_luck == fortune.pond_luck
            assert "荷塘运" in format_milk_fortune(fortune, day="2026-11-04")
            assert "鸡腿运" not in format_milk_fortune(fortune, day="2026-11-04")
        else:
            assert fortune.set_name in THEME_SETS[fortune.character]
        assert fortune == make_milk_fortune(
            chat_id="chat", sender_id=str(index), day="2026-11-04"
        )


def test_old_saved_fortunes_do_not_get_replaced_by_character_update(tmp_path):
    store = BotStore(tmp_path)
    fortune = make_milk_fortune(chat_id="chat", sender_id="user", day="2026-11-04")
    payload = asdict(fortune)
    for field in ("character", "pond_luck", "mood", "theme_luck"):
        payload.pop(field)
    payload.update(title="天选奶龙", set_name="鸡腿食堂", face="鸡腿自由")
    with store._connect() as connection:
        connection.execute(
            "INSERT INTO milk_fortunes (chat_hash, sender_hash, fortune_day, payload) "
            "VALUES (?, ?, ?, ?)",
            (store.identity_hash("chat"), store.identity_hash("user"),
             "2026-11-04", json.dumps(payload)),
        )
    restored = store.draw_milk_fortune(chat_id="chat", sender_id="user", day="2026-11-04")
    assert restored.character == "奶龙" and restored.mood == ""
    assert restored.title == "天选奶龙" and restored.face == "鸡腿自由"
    assert "鸡腿运" in format_milk_fortune(restored, day="2026-11-04")


def test_daily_fortune_calls_saved_milk_name_without_renaming_user(tmp_path, monkeypatch):
    store = BotStore(tmp_path)
    store.assign_milk_name("user-a", original_name="王小明", milk_name="荷塘奶蛙队长")
    today = datetime.now(LOCAL_TIMEZONE).date().isoformat()
    fortune = store.draw_milk_fortune(
        chat_id="chat-a", sender_id="user-a", day=today,
    )
    for command in ("今日奶运", "奶签收藏册", "奶签历史"):
        sent, _ = run_mock_bot(
            tmp_path, monkeypatch, text=command, mentioned_bot=True, random_push_probability=0,
        )
        assert "荷塘奶蛙队长" in str(sent[0][0])
        assert "王小明" not in str(sent[0][0])
        assert store.milk_name_for_user("user-a").milk_name == "荷塘奶蛙队长"
    store.assign_milk_name(
        "user-a", original_name="王小明", milk_name="旧名字",
        name_options=lambda _: ("鸡腿奶龙大王",),
    )
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text=f"奶签详情 {today}",
        mentioned_bot=True, random_push_probability=0,
    )
    assert "鸡腿奶龙大王" in str(sent[0][0])
    assert store.milk_fortune_on(
        chat_id="chat-a", sender_id="user-a", day=today,
    ) == fortune


def test_first_daily_fortune_assigns_one_name_and_reuses_it(tmp_path, monkeypatch):
    store = BotStore(tmp_path)
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text="今日奶运", mentioned_bot=True, random_push_probability=0,
    )
    name = store.milk_name_for_user("user-a").milk_name
    assert name in str(sent[0][0])
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text="今日奶运", mentioned_bot=True, random_push_probability=0,
    )
    assert store.milk_name_for_user("user-a").milk_name == name
    assert name in str(sent[0][0])


def test_hundred_level_catalog_keeps_old_sets_and_has_large_visible_combination_space():
    assert len(FORTUNE_FACES) == len({face for _, face in FORTUNE_FACES}) == 128
    assert all(len(faces) == 32 for faces in THEME_FACES.values())
    assert THEME_SETS["奶龙"]["鸡腿食堂"] == (
        "鸡腿自由", "奶茶续杯", "最后一口", "开饭大吉"
    )
    assert THEME_SETS["奶蛙"]["奶蛙荷塘"] == (
        "荷叶躺平", "池塘团宠", "水面冒泡", "莲蓬大丰收"
    )
    for field in FORTUNE_TEXT_FIELDS:
        choices = {text for pools in FORTUNE_POOLS.values() for text in pools[field]}
        assert len(choices) >= 100
    assert len({title for grades in MEME_TITLES.values()
                for grade in grades for title in grade}) == 192
    assert fortune_text_combination_count() >= 35_000_000_000_000


def test_new_meme_themes_use_their_own_luck_and_call_the_milk_name():
    fortunes = [
        make_milk_fortune(chat_id="chat", sender_id=str(index), day="2026-11-04")
        for index in range(24)
    ]
    expected_labels = {"大肥鱼DeepSeek": "深潜运", "豆包": "嘴甜运"}
    for character, label in expected_labels.items():
        fortune = next(fortune for fortune in fortunes if fortune.character == character)
        assert fortune.special_luck_label == label
        assert fortune.special_luck == fortune.theme_luck
        text = format_milk_fortune(fortune, day="2026-11-04", milk_name="荷塘奶蛙队长")
        assert "荷塘奶蛙队长，收好你的奶签。" in text
        assert character in text and label in text
        assert fortune.message in text and fortune.daily_event in text
