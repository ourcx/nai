from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from nailong_classifier.bot import (
    format_concentration,
    format_guess_already_settled,
    format_guess_correct,
    format_guess_no_assets,
    format_guess_no_open_round,
    format_guess_prompt,
    format_guess_ranking,
    format_guess_reveal,
    format_guess_wrong,
    format_mention_reply,
    format_rarity,
    is_guess_ranking_command,
    is_guess_reveal_command,
    is_guess_start_command,
    milk_concentration,
    parse_guess_answer,
)
from nailong_classifier.model import Prediction
from nailong_classifier.rarity import classify_rarity
from nailong_classifier.storage import BotStore


def make_image_bytes(color: str, *, image_format: str = "PNG") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (48, 48), color).save(output, format=image_format)
    return output.getvalue()


def make_gradient_bytes() -> bytes:
    """一张左右明暗渐变的图，感知指纹和纯色图差异极大，用来当“独一无二”的素材。"""
    image = Image.new("L", (48, 48))
    pixels = image.load()
    for x in range(48):
        for y in range(48):
            pixels[x, y] = 255 - int(x / 48 * 255)
    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def record(
    store: BotStore,
    image_bytes: bytes,
    *,
    message_id: str,
    label: str,
    chat_id: str = "chat-a",
):
    return store.record_observation(
        image_bytes=image_bytes,
        source_message_id=message_id,
        chat_id=chat_id,
        chat_type="group",
        sender_id="user-a",
        predicted_label=label,
        confidence=0.91,
        accepted=True,
    )


def make_prediction(nailong: float, naiwa: float, *, label: str, accepted: bool) -> Prediction:
    other = max(0.0, 1.0 - nailong - naiwa)
    scores = {"nailong": nailong, "naiwa": naiwa, "other": other}
    confidence = scores[label]
    return Prediction(label=label, confidence=confidence, accepted=accepted, scores=scores)


def test_classify_rarity_thresholds() -> None:
    # 稀有度按发送次数：出现越少越稀有。
    assert classify_rarity(1).name == "传说"
    assert classify_rarity(1).stars == 5
    assert classify_rarity(2).name == "史诗"
    assert classify_rarity(3).name == "史诗"
    assert classify_rarity(4).name == "稀有"
    assert classify_rarity(7).name == "稀有"
    assert classify_rarity(8).name == "常见"
    assert classify_rarity(16).name == "烂大街"
    assert classify_rarity(99).stars == 1
    assert classify_rarity(1).stars_text == "⭐⭐⭐⭐⭐"


def test_milk_concentration_sums_milk_scores() -> None:
    prediction = make_prediction(0.5, 0.4, label="nailong", accepted=True)
    assert milk_concentration(prediction) == 90
    dry = make_prediction(0.05, 0.03, label="other", accepted=False)
    assert milk_concentration(dry) == 8


def test_format_concentration_tiers() -> None:
    high = make_prediction(0.6, 0.35, label="nailong", accepted=True)
    assert "纯度爆表" in format_concentration(high)
    assert "本图奶含量 95%" in format_concentration(high)
    mid = make_prediction(0.5, 0.2, label="nailong", accepted=True)
    assert "奶味浓郁" in format_concentration(mid)
    low = make_prediction(0.3, 0.2, label="nailong", accepted=False)
    assert "有点奶" in format_concentration(low)
    dry = make_prediction(0.05, 0.05, label="other", accepted=False)
    assert "查无此奶" in format_concentration(dry)


def test_format_mention_reply_appends_concentration_and_rarity(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    observation = record(
        store,
        make_gradient_bytes(),
        message_id="only-nailong",
        label="nailong",
    )
    prediction = make_prediction(0.6, 0.31, label="nailong", accepted=True)
    rarity = store.asset_rarity(observation.asset.id)

    reply = format_mention_reply(observation, prediction, rarity)

    assert "识别结果：奶龙。置信度 60.0%。" in reply
    assert "奶龙浓度检测仪" in reply
    assert "本图奶含量 91%" in reply
    assert "稀有度：" in reply
    # 传入 None 稀有度（如“其他”类）时不带稀有度行。
    other_only = format_mention_reply(observation, prediction, None)
    assert "稀有度：" not in other_only


def test_asset_rarity_scales_with_send_count(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    # 同一张图发一次：只出现 1 次，属于传说。
    rare = record(
        store,
        make_image_bytes("yellow"),
        message_id="rare-1",
        label="nailong",
    ).asset
    assert store.asset_rarity(rare.id).seen_count == 1
    assert store.asset_rarity(rare.id).name == "传说"

    # 另一张图被不同消息反复发送，seen_count 累加后变成烂大街。
    for index in range(16):
        popular = record(
            store,
            make_image_bytes("pink"),
            message_id=f"popular-{index}",
            label="naiwa",
        ).asset
    assert store.asset_rarity(popular.id).seen_count == 16
    assert store.asset_rarity(popular.id).name == "烂大街"

    # 查不到的 asset 返回 None。
    assert store.asset_rarity(999999) is None


def test_guess_round_lifecycle_and_scoring(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    # 只放奶龙图，答案恒为 nailong，便于断言。
    record(store, make_image_bytes("yellow"), message_id="g-1", label="nailong")
    record(store, make_gradient_bytes(), message_id="g-2", label="nailong")

    round_ = store.start_guess_round(chat_id="chat-a")
    assert round_ is not None
    assert round_.answer == "nailong"
    assert store.open_guess_round(chat_id="chat-a").id == round_.id

    wrong = store.settle_guess_round(
        chat_id="chat-a",
        guess="naiwa",
        sender_id="user-wrong",
        display_name="猜错的人",
    )
    assert wrong.correct is False
    assert wrong.first_correct is False
    assert wrong.already_settled is False

    first = store.settle_guess_round(
        chat_id="chat-a",
        guess="nailong",
        sender_id="user-win",
        display_name="冠军",
    )
    assert first.first_correct is True

    late = store.settle_guess_round(
        chat_id="chat-a",
        guess="nailong",
        sender_id="user-late",
        display_name="慢了一步",
    )
    assert late.first_correct is False
    assert late.already_settled is True

    assert store.open_guess_round(chat_id="chat-a") is None
    ranking = store.guess_ranking(chat_id="chat-a")
    assert [(row.display_name, row.score) for row in ranking] == [("冠军", 1)]


def test_guess_round_scoped_per_chat_and_none_when_empty(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    assert store.start_guess_round(chat_id="chat-a") is None
    assert store.settle_guess_round(
        chat_id="chat-a",
        guess="nailong",
        sender_id="user-a",
        display_name="无局",
    ) is None

    record(store, make_image_bytes("yellow"), message_id="g-1", label="nailong")
    store.start_guess_round(chat_id="chat-a")
    # 其它群没有进行中的局。
    assert store.open_guess_round(chat_id="chat-b") is None


def test_reveal_closes_round_without_scoring(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    record(store, make_image_bytes("yellow"), message_id="g-1", label="nailong")
    round_ = store.start_guess_round(chat_id="chat-a")

    assert store.close_guess_round(round_id=round_.id) is True
    assert store.close_guess_round(round_id=round_.id) is False
    assert store.open_guess_round(chat_id="chat-a") is None
    # 公布后再答对不加分。
    after = store.settle_guess_round(
        chat_id="chat-a",
        guess="nailong",
        sender_id="user-a",
        display_name="马后炮",
    )
    assert after.first_correct is False
    assert after.already_settled is True
    assert store.guess_ranking(chat_id="chat-a") == []


def test_guess_commands_and_answer_parsing() -> None:
    assert is_guess_start_command("猜图") is True
    assert is_guess_start_command("盲盒猜图") is True
    assert is_guess_reveal_command("公布答案") is True
    assert is_guess_ranking_command("猜图榜") is True
    assert parse_guess_answer("奶龙") == "nailong"
    assert parse_guess_answer("猜奶蛙") == "naiwa"
    assert parse_guess_answer("是奶龙") == "nailong"
    assert parse_guess_answer("答案是奶蛙") == "naiwa"
    assert parse_guess_answer("其他") is None
    assert parse_guess_answer("奶龙好可爱") is None


def test_guess_message_formatters() -> None:
    assert "盲盒猜图开始" in format_guess_prompt()
    result = store_guess_result("nailong", first_correct=True)
    assert "冠军 抢答正确" in format_guess_correct(result, "冠军")
    assert "答案是奶龙" in format_guess_correct(result, "冠军")
    assert "答错了" in format_guess_wrong()
    assert "被抢答" in format_guess_already_settled()
    assert "没有可用的奶图" in format_guess_no_assets()
    assert "没有进行中的猜图" in format_guess_no_open_round()


def test_format_guess_reveal_and_ranking(tmp_path: Path) -> None:
    from nailong_classifier.storage import GuessRanking

    store = BotStore(tmp_path)
    record(store, make_image_bytes("yellow"), message_id="g-1", label="nailong")
    round_ = store.start_guess_round(chat_id="chat-a")
    reveal = format_guess_reveal(round_)
    assert "这是奶龙" in reveal
    assert f"#{round_.asset.id:04d}" in reveal

    assert "还没有人上榜" in format_guess_ranking([])
    ranking = [
        GuessRanking(sender_hash="h1", display_name="甲", wins=3, score=3),
        GuessRanking(sender_hash="h2", display_name="乙", wins=1, score=1),
    ]
    text = format_guess_ranking(ranking)
    assert "🥇 甲：3 分" in text
    assert "🥈 乙：1 分" in text


def test_format_rarity_line() -> None:
    line = format_rarity(classify_rarity(1))
    assert "传说" in line
    assert "⭐⭐⭐⭐⭐" in line
    assert "出现过 1 次" in line


def store_guess_result(answer: str, *, first_correct: bool):
    from nailong_classifier.storage import GuessResult

    return GuessResult(
        round_id=1,
        answer=answer,
        correct=True,
        first_correct=first_correct,
        already_settled=False,
        winner_hash="h1",
    )


def test_make_puzzle_image_crops_blurs_and_is_deterministic() -> None:
    import random

    from nailong_classifier.puzzle import (
        PUZZLE_OUTPUT_SIZE,
        make_puzzle_image,
        try_make_puzzle_image,
    )

    source = make_image_bytes("yellow")
    puzzle = make_puzzle_image(source, rng=random.Random(1))
    image = Image.open(io.BytesIO(puzzle))
    # 题图固定边长、能正常解析。
    assert image.size == (PUZZLE_OUTPUT_SIZE, PUZZLE_OUTPUT_SIZE)
    # 同样的随机种子得到一致结果，方便复现。
    again = make_puzzle_image(source, rng=random.Random(1))
    assert puzzle == again
    # 坏图回退 None，由调用方发原图。
    assert try_make_puzzle_image(b"not-an-image") is None


def test_make_puzzle_image_rejects_bad_params() -> None:
    import pytest

    from nailong_classifier.puzzle import make_puzzle_image

    source = make_image_bytes("yellow")
    with pytest.raises(ValueError):
        make_puzzle_image(source, fraction=0)
    with pytest.raises(ValueError):
        make_puzzle_image(source, blur_radius=-1)

