from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from nailong_classifier.bot import (
    OTHER_MILKY_REACTIONS,
    OTHER_REACTIONS,
    build_prediction_card,
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
    format_prediction,
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


@pytest.mark.parametrize(
    ("nailong", "naiwa", "accepted", "suspected"),
    [
        (0.30, 0.09, False, False),
        (0.30, 0.10, False, True),
        (0.30, 0.30, False, True),
        (0.30, 0.10, True, True),
        (0.04, 0.05, True, False),
    ],
)
def test_other_replies_suspect_missed_milk_without_losing_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nailong: float,
    naiwa: float,
    accepted: bool,
    suspected: bool,
) -> None:
    # “其他”最高也可能奶龙+奶蛙合计更高，默认置信度门槛会拒绝这类图。
    prediction = make_prediction(nailong, naiwa, label="other", accepted=accepted)
    store = BotStore(tmp_path)
    observation = record(
        store, make_gradient_bytes(), message_id="suspected-other", label="other"
    )
    monkeypatch.setattr("nailong_classifier.bot.secrets.choice", lambda choices: choices[0])

    text = format_prediction(prediction)
    card = build_prediction_card(observation, prediction, None)
    card_content = str(card)
    concentration = format_concentration(prediction)

    assert f"置信度 {prediction.confidence:.1%}" in text
    assert observation.asset.label == "other"
    if suspected:
        assert text.startswith(OTHER_MILKY_REACTIONS[0])
        assert card["config"]["summary"]["content"].startswith(OTHER_MILKY_REACTIONS[0])
        assert "可能漏识别了奶龙或奶蛙" in card_content
        assert "可能是没认出来的奶龙或奶蛙" in concentration
        assert "查无此奶" not in concentration
        assert "不像奶龙或奶蛙" not in card_content
    elif accepted:
        assert text.startswith(OTHER_REACTIONS[0])
        assert "不像奶龙或奶蛙" in card_content
    else:
        assert text.startswith("我不太确定。")
        assert "可能漏识别" not in card_content
    if not accepted:
        assert "我不太确定。最高候选是「其他」" in text
        assert "识别结果暂不确定" in card_content


def test_corrected_other_does_not_get_suspected_milk_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BotStore(tmp_path)
    original = record(store, make_gradient_bytes(), message_id="confirmed-other", label="other")
    assert store.correct_label("confirmed-other", "other") is not None
    observation = store.observation_by_id(original.id)
    assert observation is not None
    prediction = make_prediction(0.3, 0.3, label="other", accepted=False)

    def unexpected_reaction(choices):
        raise AssertionError("人工确认的图片不应使用随机怀疑文案")

    monkeypatch.setattr("nailong_classifier.bot.secrets.choice", unexpected_reaction)

    reply = format_mention_reply(observation, prediction, None)
    assert reply.startswith("这张图已经确认是其他")
    assert "可能是没认出来" not in reply
    assert "分类以人工纠正为准" in reply
    assert "已确认是其他" in str(build_prediction_card(observation, prediction, None))


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


def test_guess_round_balances_labels_despite_skewed_inventory(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    # 故意造成库存失衡：其他远多于奶龙/奶蛙。均衡抽样应让三类都稳定出现，而非被“其他”占满。
    record(store, make_image_bytes("yellow"), message_id="n-1", label="nailong")
    record(store, make_image_bytes("pink"), message_id="w-1", label="naiwa")
    for index in range(20):
        store.record_observation(
            image_bytes=make_image_bytes(f"#{index:02x}{index:02x}{index:02x}"),
            source_message_id=f"o-{index}",
            chat_id="chat-a",
            chat_type="group",
            sender_id="user-a",
            predicted_label="other",
            confidence=0.9,
            accepted=True,
        )

    seen: set[str] = set()
    for _ in range(60):
        round_ = store.start_guess_round(chat_id="chat-a")
        assert round_ is not None
        seen.add(round_.answer)
        store.close_guess_round(round_id=round_.id)

    # 60 次抽样里三类都应出现过（均衡抽样下每类概率约 1/3，漏掉某类几乎不可能）。
    assert seen == {"nailong", "naiwa", "other"}


def test_guess_commands_and_answer_parsing() -> None:
    assert is_guess_start_command("猜图") is True
    assert is_guess_start_command("盲盒猜图") is True
    assert is_guess_reveal_command("公布答案") is True
    assert is_guess_ranking_command("猜图榜") is True
    assert parse_guess_answer("奶龙") == "nailong"
    assert parse_guess_answer("猜奶蛙") == "naiwa"
    assert parse_guess_answer("是奶龙") == "nailong"
    assert parse_guess_answer("答案是奶蛙") == "naiwa"
    assert parse_guess_answer("其他") == "other"
    assert parse_guess_answer("猜其他") == "other"
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


def test_deactivate_chat_excludes_from_daily_report(tmp_path: Path) -> None:
    from datetime import datetime

    from nailong_classifier.storage import LOCAL_TIMEZONE

    store = BotStore(tmp_path)
    record(store, make_image_bytes("yellow"), message_id="m-1", label="nailong", chat_id="grp")
    day = datetime.now(LOCAL_TIMEZONE).date().isoformat()

    # 失效前该群在待播报列表里。
    assert [t.chat_id for t in store.pending_daily_report_targets(day)] == ["grp"]

    store.deactivate_chat("grp")
    # 失效后不再出现。
    assert store.pending_daily_report_targets(day) == []


def test_random_milk_asset_only_milk_and_balanced(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    # 没有任何奶图时返回 None。
    assert store.random_milk_asset() is None

    store.register_seed(make_image_bytes("gold"), "nailong")
    record(store, make_image_bytes("pink"), message_id="w-1", label="naiwa")
    # 加一张“其他”，不应被主动推送选中。
    store.record_observation(
        image_bytes=make_image_bytes("#101010"),
        source_message_id="o-1",
        chat_id="chat-a",
        chat_type="group",
        sender_id="user-a",
        predicted_label="other",
        confidence=0.9,
        accepted=True,
    )

    labels = {store.random_milk_asset().label for _ in range(40)}
    assert labels <= {"nailong", "naiwa"}
    assert "other" not in labels


def test_discoverer_recorded_on_first_capture_only(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    first = store.record_observation(
        image_bytes=make_image_bytes("yellow"),
        source_message_id="m-1",
        chat_id="chat-a",
        chat_type="group",
        sender_id="user-finder",
        predicted_label="nailong",
        confidence=0.9,
        accepted=True,
        discoverer_name="首发哥",
    )
    assert first.asset.discoverer_name == "首发哥"

    # 同一张图第二次由别人发：首发人保持不变（成就归第一个发现者）。
    again = store.record_observation(
        image_bytes=make_image_bytes("yellow"),
        source_message_id="m-2",
        chat_id="chat-a",
        chat_type="group",
        sender_id="user-latecomer",
        predicted_label="nailong",
        confidence=0.9,
        accepted=True,
        discoverer_name="后来者",
    )
    assert again.asset.id == first.asset.id
    assert again.asset.discoverer_name == "首发哥"


def test_legacy_asset_without_discoverer_shows_no_line(tmp_path: Path) -> None:
    from nailong_classifier.bot import build_catalog_post

    store = BotStore(tmp_path)
    # 不传 discoverer_name，模拟老图：图鉴条目不含「首发」。
    record(store, make_image_bytes("yellow"), message_id="m-1", label="nailong")
    record(store, make_image_bytes("pink"), message_id="m-2", label="naiwa")
    store.record_observation(
        image_bytes=make_image_bytes("gold"),
        source_message_id="m-3",
        chat_id="chat-a",
        chat_type="group",
        sender_id="user-a",
        predicted_label="nailong",
        confidence=0.9,
        accepted=True,
        discoverer_name="金牌发现者",
    )

    assets = store.catalog_assets()
    stats = store.catalog_stats()
    post = build_catalog_post(assets, [f"img_{i}" for i in range(len(assets))], stats)
    captions = [
        node["text"]
        for row in post["zh_cn"]["content"]
        for node in row
        if node["tag"] == "text"
    ]
    credit_lines = [c for c in captions if "首发" in c]
    # 只有带署名的那张图出现「首发」行，另两张老图不出现。
    assert len(credit_lines) == 1
    assert "首发：金牌发现者" in credit_lines[0]


def test_hunter_rankings_counts_catalog_discoveries(tmp_path: Path) -> None:
    store = BotStore(tmp_path)
    # A 首发两张入典奶图，B 首发一张，无署名的不计入。
    for index, name in enumerate(["猎人A", "猎人A", "猎人B"]):
        store.record_observation(
            image_bytes=make_image_bytes(f"#{index:02x}00{index:02x}"),
            source_message_id=f"m-{index}",
            chat_id="chat-a",
            chat_type="group",
            sender_id=f"user-{name}",
            predicted_label="nailong",
            confidence=0.9,
            accepted=True,
            discoverer_name=name,
        )
    record(store, make_image_bytes("teal"), message_id="legacy", label="naiwa")

    rankings = store.hunter_rankings()
    assert [(r.display_name, r.discovered) for r in rankings] == [
        ("猎人A", 2),
        ("猎人B", 1),
    ]


def test_hunter_ranking_command_and_formatter() -> None:
    from nailong_classifier.bot import format_hunter_ranking, is_hunter_ranking_command
    from nailong_classifier.storage import HunterRanking

    assert is_hunter_ranking_command("奶龙猎人榜") is True
    assert is_hunter_ranking_command("猎人榜") is True
    assert is_hunter_ranking_command("猜图榜") is False
    assert "还没有人上榜" in format_hunter_ranking([])
    text = format_hunter_ranking(
        [
            HunterRanking(display_name="甲", discovered=5),
            HunterRanking(display_name="乙", discovered=2),
        ]
    )
    assert "🥇 甲：5 张" in text
    assert "🥈 乙：2 张" in text
