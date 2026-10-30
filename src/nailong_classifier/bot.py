from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import time as time_module
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .adapters.base import BotAdapter, CardAction, ErrorCode, ImageTarget, MembersAdded
from .adapters.config import create_adapter, load_adapter_configs
from .doubao import DoubaoDetector
from .douyin import (
    SEARCH_KEYWORDS,
    DouyinSearchError,
    DouyinVideo,
    choose_douyin_video,
    download_douyin_cover,
    search_douyin_videos,
)
from .hdr import (
    DEFAULT_HDR_PEAK_NITS,
    DEFAULT_HDR_PROBABILITY,
    random_milk_image_content,
    validate_hdr_settings,
)
from .image_fingerprint import find_duplicate, try_content_hash, try_file_content_hash
from .milk import combined_milk_concentration
from .milk_name_words import (
    MILK_NAME_CHARACTERS,
    MILK_NAME_FOREIGN,
    MILK_NAME_PREFIXES,
    MILK_NAME_TITLES,
)
from .model import NailongPredictor, Prediction
from .puzzle import try_make_puzzle_image
from .rarity import Rarity
from .services import BackendServices, SerializedPredictor
from .storage import (
    LEGACY_NAME_FALLBACKS,
    LOCAL_TIMEZONE,
    AssetRecord,
    BotStore,
    CatalogStats,
    DailyStats,
    GuessRanking,
    GuessResult,
    GuessRound,
    HunterRanking,
    MilkFamilyMember,
    ObservationRecord,
)

if TYPE_CHECKING:
    from .catalog_document import CatalogDocumentClient

LOGGER = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
DISPLAY_LABELS = {
    "nailong": "奶龙",
    "naiwa": "奶蛙",
    "other": "其他",
}
# “其他”图按现有奶含量分档接梗；偏奶时怀疑漏识别，文字和卡片共用文案。
OTHER_MILKY_THRESHOLD = 40
OTHER_MILKY_REACTIONS = (
    "这不是奶娃吧？！",
    "别吓我，这不会真是奶龙吧？！",
    "坏了，该不会是奶娃混进来了吧？！",
    "等会儿，我怎么越看越像奶龙？！",
    "救命，这奶味不对劲啊？！",
    "嘶……你最好真不是奶龙！",
    "完了，不会又让奶娃蒙混过关了吧？！",
    "别吧？！奶龙已经藏成这样了？！",
    "我有点慌，这张真不是奶娃？！",
    "不敢细看了，怎么有股奶龙味？！",
    "你别过来啊！该不会是奶娃吧？！",
    "先别笑，我怕这真是没认出来的奶龙！",
)
OTHER_REACTIONS = (
    "等等，奶龙呢？奶蛙呢？！",
    "好家伙，奶龙奶蛙都没来？！",
    "不是，怎么还混进来一张别的？！",
    "嚯？！这位走错片场了吧！",
    "嗯？！说好的奶龙奶蛙呢？",
    "我奶呢？！那么大一口奶呢？！",
    "啊这？！查无此奶！",
    "等会儿？！奶龙奶蛙一个都没对上！",
    "你管这叫奶图？！我得再看一眼！",
    "救命？！奶龙奶蛙集体缺席！",
)
CORRECTION_PATTERN = re.compile(r"^(?:纠正(?:为|成)?|改成|这是)\s*(奶龙|奶蛙|其他)$")
COMMAND_LABELS = {
    "奶龙": "nailong",
    "奶蛙": "naiwa",
    "其他": "other",
}
# 猜图抢答：在奶龙/奶蛙/其他三类之间抢答。
GUESS_ANSWER_LABELS = {
    "奶龙": "nailong",
    "奶蛙": "naiwa",
    "其他": "other",
}
DEFAULT_DAILY_REPORT_TIME = time(hour=23, minute=55)
DEFAULT_CATALOG_SYNC_RETRY_SECONDS = 60
# 每日播报的轮询间隔：每分钟看一眼是否过了播报点，错过或重启都能靠下一次轮询补上。
DAILY_REPORT_POLL_SECONDS = 60
# 主动推送抽象奶图：群里有人正常聊天时，以此概率冷不丁丢一张图（只发图不说话）。
DEFAULT_RANDOM_PUSH_PROBABILITY = 0.03
# 两次主动推送之间的最小间隔（秒），同一个群内生效，避免连着刷屏。默认 2 小时。
DEFAULT_RANDOM_PUSH_COOLDOWN_SECONDS = 7200
# 普通群消息触发抖音奶龙/奶娃视频链接的概率；未配置 Cookie 搜索时功能关闭。
DEFAULT_DOUYIN_VIDEO_PUSH_PROBABILITY = 0.4
# 同一群在滚动一小时内最多发送两条链接。
DEFAULT_DOUYIN_VIDEO_PUSH_WINDOW_SECONDS = 3600
DEFAULT_DOUYIN_VIDEO_PUSH_MAX_PER_WINDOW = 2
DEFAULT_DOUYIN_SEARCH_CACHE_SECONDS = 900


@dataclass(frozen=True)
class MilkFortune:
    score: int
    banter: int
    title: str
    keyword: str
    good: str
    avoid: str
    message: str


def _decode_message_content(raw: dict[str, Any]) -> Any:
    body = raw.get("body")
    content = body.get("content") if isinstance(body, dict) else raw.get("content")
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}
    return content or {}


def _find_image_key(value: Any) -> str | None:
    if isinstance(value, dict):
        image_key = value.get("image_key")
        if isinstance(image_key, str) and image_key:
            return image_key
        for nested in value.values():
            found = _find_image_key(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_image_key(nested)
            if found:
                return found
    return None


def _find_file_key(value: Any) -> str | None:
    if isinstance(value, dict):
        file_key = value.get("file_key")
        if isinstance(file_key, str) and file_key:
            return file_key
        for nested in value.values():
            found = _find_file_key(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_file_key(nested)
            if found:
                return found
    return None


def _raw_parent_id(message) -> str | None:
    raw_value = getattr(message, "raw", None)
    raw = raw_value if isinstance(raw_value, dict) else {}
    nested = raw.get("message")
    if isinstance(nested, dict) and nested.get("parent_id"):
        return nested["parent_id"]
    parent_id = raw.get("parent_id")
    return parent_id if isinstance(parent_id, str) and parent_id else None


async def resolve_image_target(channel: BotAdapter, message) -> ImageTarget:
    return await channel.resolve_image_target(message)


async def download_target(channel: BotAdapter, target: ImageTarget) -> bytes | None:
    return await channel.download_target(target)


def format_prediction(prediction: Prediction) -> str:
    display_label = DISPLAY_LABELS.get(prediction.label, prediction.label)
    confidence = f"{prediction.confidence:.1%}"
    if not prediction.accepted:
        uncertainty = f"我不太确定。最高候选是「{display_label}」，置信度 {confidence}。"
        if is_milky_other(prediction):
            return f"{other_reaction(prediction)} {uncertainty}"
        return uncertainty
    if prediction.label == "other":
        return f"{other_reaction(prediction)} 识别结果：其他。置信度 {confidence}。"
    return f"识别结果：{display_label}。置信度 {confidence}。"


def milk_concentration(prediction: Prediction) -> int:
    """用模型奶概率和可见黄色占比合成奶含量，所有文案与卡片使用同一读数。"""
    return combined_milk_concentration(prediction.scores, prediction.yellow_ratio)


def is_milky_other(prediction: Prediction) -> bool:
    """沿用“有点奶”的分界，以模型奶概率和黄色占比的加权读数判断疑似漏识别。

    “其他”也可能因置信度不足而未被接受，所以不以 accepted 限制这类怀疑语气。
    """
    return prediction.label == "other" and milk_concentration(prediction) >= OTHER_MILKY_THRESHOLD


def other_reaction(prediction: Prediction) -> str:
    reactions = OTHER_MILKY_REACTIONS if is_milky_other(prediction) else OTHER_REACTIONS
    return secrets.choice(reactions)


def filter_douyin_videos_by_cover(
    videos: list[DouyinVideo],
    predictor: NailongPredictor,
) -> list[DouyinVideo]:
    """只保留封面被本地模型高置信判为奶龙或奶蛙的搜索结果。"""
    verified: list[DouyinVideo] = []
    for video in videos:
        if not video.cover_url:
            LOGGER.info("Skipping Douyin video without cover URL url=%s", video.url)
            continue
        try:
            prediction = predictor.predict_bytes(download_douyin_cover(video.cover_url))
        except (DouyinSearchError, OSError, ValueError):
            LOGGER.info("Skipping Douyin video with unavailable cover url=%s", video.url)
            continue
        if prediction.accepted and prediction.label in {"nailong", "naiwa"}:
            verified.append(video)
        else:
            LOGGER.info(
                "Skipping non-milk Douyin cover url=%s label=%s confidence=%.3f accepted=%s",
                video.url,
                prediction.label,
                prediction.confidence,
                prediction.accepted,
            )
    return verified


def format_concentration(prediction: Prediction, *, corrected: bool = False) -> str:
    """奶龙浓度检测仪：把识别置信度包装成一句分级吐槽，仅用于 @机器人 的娱乐回复。"""
    percent = milk_concentration(prediction)
    display_label = DISPLAY_LABELS.get(prediction.label, prediction.label)
    if is_milky_other(prediction):
        verdict = (
            "奶含量是娱乐读数，分类以人工纠正为准。"
            if corrected
            else "奶味有点可疑，可能是没认出来的奶龙或奶蛙。"
        )
    elif percent >= 90:
        verdict = f"纯度爆表！纯正{display_label}血统，建议立刻收藏。"
    elif percent >= 65:
        verdict = f"奶味浓郁，{display_label}认证通过。"
    elif percent >= 40:
        verdict = "有点奶，但不够纯，再加把劲。"
    else:
        verdict = "查无此奶，这含量约等于清水。"
    return f"🥛 奶龙浓度检测仪\n本图奶含量 {percent}%\n{verdict}"


def format_rarity(rarity: Rarity) -> str:
    """把稀有度评级拼成一行展示，附带这张图在群里出现过多少次。"""
    return (
        f"稀有度：{rarity.stars_text} {rarity.name}"
        f"（{rarity.note}，群里出现过 {rarity.seen_count} 次）"
    )


def format_mention_reply(
    observation: ObservationRecord,
    prediction: Prediction,
    rarity: Rarity | None,
) -> str:
    """@机器人识别一张图时的完整回复：识别结论 + 奶龙浓度，奶图再附稀有度。

    识别结论尊重人工纠正和模型置信度，浓度和稀有度作为附加娱乐信息。
    """
    lines = [
        format_existing_observation(observation, prediction),
        format_concentration(prediction, corrected=observation.asset.corrected),
    ]
    if rarity is not None:
        lines.append(format_rarity(rarity))
    return "\n".join(lines)


def parse_correction_command(text: str) -> str | None:
    match = CORRECTION_PATTERN.fullmatch(text.strip())
    return COMMAND_LABELS[match.group(1)] if match else None


def is_daily_stats_command(text: str) -> bool:
    return text.strip() in {"今日含奶量", "含奶量"}


def is_catalog_command(text: str) -> bool:
    return text.strip() in {"奶龙图鉴", "查看奶龙图鉴", "打开奶龙图鉴"}


def is_hdr_milk_command(text: str) -> bool:
    return text.strip().lower() in {"hdr奶图", "hdr奶龙", "来张hdr奶图"}


def is_help_command(text: str) -> bool:
    return text.strip().lower() in {"help", "帮助", "使用帮助", "奶龙帮助"}


def is_help_request(text: str, *, mentioned_bot: bool) -> bool:
    return mentioned_bot and is_help_command(text)


def mentioned_user_ids(
    message,
    *,
    body_text: str | None = None,
) -> list[tuple[str, str | None]]:
    """取同条消息里被 @ 的普通用户，排除机器人和无身份的 mention。

    飞书 SDK 会把 @ 解析成带 open_id 的 Mention，不依赖正文里的显示名或占位符。
    个别事件会漏标机器人 mention 的 is_bot；此时 body_text 已去掉当前机器人的
    @，只保留仍出现在正文中的目标用户，避免把机器人自己当成查询对象。
    """
    users: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    visible_text = body_text if getattr(message, "mentioned_bot", False) else None
    for mention in getattr(message, "mentions", ()):
        open_id = getattr(mention, "open_id", None)
        name = getattr(mention, "name", None)
        if (
            not isinstance(open_id, str)
            or not open_id
            or getattr(mention, "is_bot", False)
            or (
                visible_text is not None
                and isinstance(name, str)
                and name
                and name not in visible_text
            )
            or open_id in seen
        ):
            continue
        seen.add(open_id)
        users.append((open_id, name if isinstance(name, str) and name else None))
    return users


def format_milk_family_check(
    display_name: str,
    member: MilkFamilyMember | None,
) -> str:
    if member is None:
        return f"{display_name} 还没被我认到是奶家人，先在群里冒个泡吧。"
    return f"{display_name} 是奶家人，接一张奶图。"


def milk_nickname_variants(display_name: str) -> tuple[str, ...]:
    """奶字可以替换任意位置，也可以加在名字前后；已有奶字保留主体。"""
    name = "".join(display_name.split())
    if not name:
        return ()
    if "奶" in name:
        return (name,)
    if len(name) == 1:
        return (f"奶{name}",)
    replaced = (
        "".join("奶" if index == chosen else char for index, char in enumerate(name))
        for chosen in range(len(name))
    )
    return tuple(dict.fromkeys((*replaced, f"奶{name}", f"{name}奶")))


def make_milk_name(display_name: str) -> str | None:
    """随机选一种奶字位置，入群和主动求赐名使用同一规则。"""
    variants = milk_nickname_variants(display_name)
    return secrets.choice(variants) if variants else None


def is_milk_name_command(text: str) -> bool:
    return "".join(text.split()) == "求赐名"


def is_milk_name_grant_for_other_command(text: str) -> bool:
    """只认“给某人求赐名”，避免普通 @ 群友时意外替对方换名。"""
    compact = "".join(text.split())
    return "给" in compact and compact.endswith("求赐名")


def milk_name_options(original_name: str) -> Iterator[str]:
    """每类候选都含奶龙或奶蛙；存储层在写锁内查重，撞名再取下一候选。

    称号按抖音文案改写后整条抽取，避免无关甜品词、称号随机拼出不顺口的名字。
    欢迎和求赐名共用词库；已有名字是否更新仍由存储层的 replace_existing 决定。
    """
    base_name = make_milk_name(original_name)
    if base_name is None:
        raise ValueError("Original name must not be empty")
    for _ in range(96):
        style = secrets.choice(("nickname", "foreign", "foreign_milk", "title", "title_name"))
        if style == "nickname":
            yield f"{secrets.choice(MILK_NAME_PREFIXES)} {base_name}"
        elif style == "foreign":
            yield (
                f"{secrets.choice(MILK_NAME_FOREIGN)}·"
                f"{secrets.choice(MILK_NAME_CHARACTERS)}·{base_name}"
            )
        elif style == "foreign_milk":
            yield f"{secrets.choice(MILK_NAME_FOREIGN)}·{secrets.choice(MILK_NAME_TITLES)}"
        else:
            title = secrets.choice(MILK_NAME_TITLES)
            if style == "title":
                yield title
            else:
                yield f"{title}·{base_name}"


def format_milk_name_grant(
    original_name: str, milk_name: str, *, original_name_known: bool = True,
) -> str:
    mapping = (
        f"{original_name} → {milk_name}，已经记住啦。"
        if original_name_known else "奶名已经记住啦。"
    )
    return (
        f"你的奶名是「{milk_name}」🥛\n"
        f"{mapping}\n"
        "想换一个，再求赐名就行。"
    )


def format_milk_name_for_other(display_name: str, milk_name: str, *, existing: bool) -> str:
    if existing:
        return f"{display_name} 是奶家人，是你【{milk_name}】。"
    return f"{display_name} 已入奶籍，奶名是【{milk_name}】。"


def build_milk_name_image_request(open_id: str, milk_name: str) -> dict[str, Any]:
    """飞书富文本必须使用真实 open_id，不能使用业务层命名空间后的用户 ID。"""
    return {
        "zh_cn": {
            "title": "",
            "content": [[
                {"tag": "at", "user_id": open_id, "user_name": "神笔奶良"},
                {"tag": "text", "text": f" 请生成「{milk_name}」这个奶名的图片。"},
            ]],
        },
    }


def format_milk_name_welcome(entries: list[tuple[str, str]]) -> str:
    """合并同一批入群事件，避免转群或批量拉人时刷屏。"""
    if len(entries) == 1:
        display_name, milk_name = entries[0]
        if display_name == milk_name:
            return f"新奶友「{display_name}」进群，本来就带奶，奶名不用改。"
        return f"新奶友「{display_name}」进群，奶名已定：{milk_name}。"
    shown = "、".join(f"{display_name}→{milk_name}" for display_name, milk_name in entries[:5])
    suffix = f"等 {len(entries)} 位" if len(entries) > 5 else ""
    return f"新奶友进群，今日奶名：{shown}{suffix}。"


def is_guess_start_command(text: str) -> bool:
    return text.strip() in {"猜图", "盲盒猜图", "来一局", "猜一猜"}


def is_guess_reveal_command(text: str) -> bool:
    return text.strip() in {"公布答案", "揭晓答案", "公布"}


def is_guess_ranking_command(text: str) -> bool:
    return text.strip() in {"猜图榜", "猜图排行榜", "猜图积分榜"}


def is_hunter_ranking_command(text: str) -> bool:
    return text.strip() in {"奶龙猎人榜", "猎人榜", "奶图猎人榜"}


def is_milk_fortune_command(text: str) -> bool:
    return text.strip() in {"今日奶运", "奶运", "奶龙签", "抽奶签"}


MILK_FORTUNE_GOOD = (
    "主动接住群友的梗",
    "发一张压箱底奶图",
    "夸群友一句真心话",
    "猜图时相信第一直觉",
    "忙里偷闲摸鱼五分钟",
    "给今天定个小目标",
)
MILK_FORTUNE_AVOID = (
    "把奶蛙认成奶龙",
    "连续刷屏十张表情",
    "空腹参加长会议",
    "把群聊聊成工单群",
    "为了最后一句争输赢",
    "潜水到大家以为你掉线",
)
MILK_FORTUNE_KEYWORDS = (
    "接梗",
    "松弛",
    "好奇",
    "开怀",
    "摸鱼",
    "行动",
    "分享",
    "好运",
)
MILK_FORTUNE_MESSAGES = (
    "今天的快乐藏在第一句主动开口里。",
    "别急着证明自己，先把群里的气氛接住。",
    "适合做一点不费劲、但能让人开心的小事。",
    "你的好运来得很轻，可能就是一个刚好的表情包。",
    "先完成最小的一步，剩下的奶气会自己跟上。",
    "今天很适合把严肃的话说得稍微可爱一点。",
)


def make_milk_fortune(
    *,
    chat_id: str,
    sender_id: str,
    day: str,
) -> MilkFortune:
    """同一群、同一用户、同一天得到相同奶运，不落库也不会重复刷随机值。"""
    digest = hashlib.sha256(f"{day}\0{chat_id}\0{sender_id}".encode()).digest()
    score = int.from_bytes(digest[0:4], "big") % 101
    banter = int.from_bytes(digest[4:8], "big") % 101
    if score >= 95:
        title = "天选奶龙"
    elif score >= 80:
        title = "群聊锦鲤"
    elif score >= 60:
        title = "元气奶龙"
    elif score >= 40:
        title = "稳稳奶友"
    elif score >= 20:
        title = "慢热奶龙"
    else:
        title = "潜水奶苗"
    return MilkFortune(
        score=score,
        banter=banter,
        title=title,
        keyword=MILK_FORTUNE_KEYWORDS[digest[8] % len(MILK_FORTUNE_KEYWORDS)],
        good=MILK_FORTUNE_GOOD[digest[9] % len(MILK_FORTUNE_GOOD)],
        avoid=MILK_FORTUNE_AVOID[digest[10] % len(MILK_FORTUNE_AVOID)],
        message=MILK_FORTUNE_MESSAGES[digest[11] % len(MILK_FORTUNE_MESSAGES)],
    )


def format_milk_fortune(fortune: MilkFortune, *, day: str) -> str:
    return (
        f"今日奶运：{fortune.title}\n"
        f"奶气值 {fortune.score} · 接梗力 {fortune.banter}\n"
        f"宜：{fortune.good}\n"
        f"忌：{fortune.avoid}\n"
        f"今日关键词：{fortune.keyword}\n"
        f"{fortune.message}\n"
        f"{day}，同一群同一天签固定。"
    )


GUESS_ANSWER_PATTERN = re.compile(r"^(?:猜|是|答案是?)?\s*(奶龙|奶蛙|其他)$")


def parse_guess_answer(text: str) -> str | None:
    """把「奶龙」「猜奶蛙」「是奶龙」这类抢答解析成内部标签，其它文本返回 None。"""
    match = GUESS_ANSWER_PATTERN.fullmatch(text.strip())
    return GUESS_ANSWER_LABELS[match.group(1)] if match else None


def format_guess_prompt(*, mentioned_answers: bool = False) -> str:
    answer_instruction = "@我后发" if mentioned_answers else "直接发"
    return (
        "🎲 盲盒猜图开始！\n"
        "这是打了码的局部，猜猜它是奶龙、奶蛙还是其他？"
        f"{answer_instruction}「奶龙」「奶蛙」或「其他」抢答，首答对者得 1 分。"
    )


def format_guess_correct(result: GuessResult, display_name: str) -> str:
    answer = DISPLAY_LABELS.get(result.answer, result.answer)
    return f"🎉 {display_name} 抢答正确！答案是{answer}，积 1 分。发「猜图」再来一局。"


def format_guess_wrong() -> str:
    return "答错了，再想想～（答案保密，让别人也猜猜）"


def format_guess_already_settled() -> str:
    return "这一局已经被抢答啦，发「猜图」开新的一局。"


def format_guess_no_assets() -> str:
    return "图鉴里还没有可用的奶图，先让大家多发几张再来玩吧。"


def format_guess_reveal(round_: GuessRound) -> str:
    answer = DISPLAY_LABELS.get(round_.answer, round_.answer)
    return f"本局答案揭晓：这是{answer}（图鉴 #{round_.asset.id:04d}）。发「猜图」开新的一局。"


def format_guess_no_open_round() -> str:
    return "现在没有进行中的猜图，发「猜图」开一局吧。"


def format_guess_ranking(rankings: list[GuessRanking]) -> str:
    if not rankings:
        return "还没有人上榜，发「猜图」开一局，抢答得分。"
    medals = ["🥇", "🥈", "🥉"]
    lines = ["🏆 猜图积分榜"]
    for index, ranking in enumerate(rankings):
        prefix = medals[index] if index < len(medals) else f"{index + 1}."
        lines.append(f"{prefix} {ranking.display_name}：{ranking.score} 分")
    return "\n".join(lines)


def format_hunter_ranking(rankings: list[HunterRanking]) -> str:
    if not rankings:
        return "还没有人上榜。发奶龙或奶蛙图被收进图鉴，就能成为首发猎人。"
    medals = ["🥇", "🥈", "🥉"]
    lines = ["🏹 奶龙猎人榜（首发入典数）"]
    for index, ranking in enumerate(rankings):
        prefix = medals[index] if index < len(medals) else f"{index + 1}."
        lines.append(f"{prefix} {ranking.display_name}：{ranking.discovered} 张")
    return "\n".join(lines)


def parse_daily_report_time(value: str) -> time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as error:
        raise argparse.ArgumentTypeError("daily report time must use HH:MM") from error


def parse_asset_ids(value: str) -> list[int]:
    if not value.strip():
        return []
    try:
        asset_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "catalog baseline asset IDs must be comma-separated integers"
        ) from error
    if any(asset_id <= 0 for asset_id in asset_ids):
        raise argparse.ArgumentTypeError(
            "catalog baseline asset IDs must be positive integers"
        )
    return list(dict.fromkeys(asset_ids))


def parse_douyin_search_keywords(value: str) -> tuple[str, ...]:
    """解析逗号或换行分隔的话题；# 前缀仅作展示习惯，不传入抖音搜索接口。"""
    keywords = tuple(
        dict.fromkeys(
            item.strip().lstrip("#").strip()
            for item in re.split(r"[\n,]+", value)
            if item.strip().lstrip("#").strip()
        )
    )
    if not keywords:
        raise argparse.ArgumentTypeError("at least one Douyin search keyword is required")
    if len(keywords) > 10:
        raise argparse.ArgumentTypeError("at most 10 Douyin search keywords are allowed")
    return keywords


DOUYIN_TOPIC_PATTERN = re.compile(r"(?<!\S)#([^\s#，。！？、,.!?]{1,30})")


def parse_douyin_topic_request(text: str) -> tuple[str, ...]:
    """提取 @机器人消息中的 #话题；最多三项，避免一次请求过多搜索词。"""
    return tuple(dict.fromkeys(match.group(1) for match in DOUYIN_TOPIC_PATTERN.finditer(text)))[:3]


def has_direct_image(message) -> bool:
    return any(resource.type in {"image", "sticker"} for resource in message.resources)


def should_reply_media_error(*, mentioned_bot: bool, chat_type: str) -> bool:
    # 群聊里没 @ 机器人的图片属于自动识别，出错时只记日志，避免在群里刷报错。
    return mentioned_bot or chat_type == "p2p"


def media_response_mode(
    *,
    mentioned_bot: bool,
    chat_type: str,
    label: str,
) -> Literal["prediction", "chain", "silent"]:
    if mentioned_bot:
        return "prediction"
    if label in {"nailong", "naiwa"}:
        return "chain"
    if chat_type == "p2p":
        return "prediction"
    return "silent"


def format_daily_stats(stats: DailyStats) -> str:
    return (
        f"今日含奶量：{stats.milk_percent}%\n"
        f"奶龙 {stats.nailong} · 奶蛙 {stats.naiwa} · 其他 {stats.other}\n"
        f"今日共识别 {stats.total} 张图片/表情，新收录奶龙 {stats.new_nailong} 张。"
    )


CARD_THEME_COLORS = {
    "wb_handdrawn_vibes_canvas": {
        "light_mode": "rgba(255,250,238,1)",
        "dark_mode": "rgba(25,25,24,1)",
    },
    "wb_handdrawn_vibes_body": {
        "light_mode": "rgba(255,250,238,1)",
        "dark_mode": "rgba(25,25,24,1)",
    },
    "wb_handdrawn_vibes_panel": {
        "light_mode": "rgba(255,255,246,1)",
        "dark_mode": "rgba(38,38,36,1)",
    },
    "wb_handdrawn_vibes_ink": {
        "light_mode": "rgba(15,15,14,1)",
        "dark_mode": "rgba(242,240,234,1)",
    },
    "wb_handdrawn_vibes_muted": {
        "light_mode": "rgba(95,94,90,1)",
        "dark_mode": "rgba(184,181,173,1)",
    },
    "wb_handdrawn_vibes_secondary": {
        "light_mode": "rgba(221,214,200,1)",
        "dark_mode": "rgba(70,69,65,1)",
    },
    "wb_handdrawn_vibes_accent": {
        "light_mode": "rgba(173,222,255,1)",
        "dark_mode": "rgba(45,91,111,1)",
    },
    "wb_handdrawn_vibes_accent_2": {
        "light_mode": "rgba(254,197,84,1)",
        "dark_mode": "rgba(112,82,24,1)",
    },
    "wb_handdrawn_vibes_button": {
        "light_mode": "rgba(15,15,14,1)",
        "dark_mode": "rgba(231,228,218,1)",
    },
    "wb_handdrawn_vibes_button_text": {
        "light_mode": "rgba(255,250,238,1)",
        "dark_mode": "rgba(25,25,24,1)",
    },
    "wb_handdrawn_vibes_button_secondary": {
        "light_mode": "rgba(255,255,246,1)",
        "dark_mode": "rgba(38,38,36,1)",
    },
    "wb_handdrawn_vibes_button_secondary_text": {
        "light_mode": "rgba(15,15,14,1)",
        "dark_mode": "rgba(242,240,234,1)",
    },
    "wb_handdrawn_vibes_icon_surface": {
        "light_mode": "rgba(255,250,238,1)",
        "dark_mode": "rgba(38,38,36,1)",
    },
    "wb_handdrawn_vibes_icon_border": {
        "light_mode": "rgba(31,35,41,0.10)",
        "dark_mode": "rgba(242,240,234,0.14)",
    },
    "wb_handdrawn_vibes_on_canvas": {
        "light_mode": "rgba(15,15,14,1)",
        "dark_mode": "rgba(242,240,234,1)",
    },
    "wb_handdrawn_vibes_on_body": {
        "light_mode": "rgba(15,15,14,1)",
        "dark_mode": "rgba(242,240,234,1)",
    },
    "wb_handdrawn_vibes_on_panel": {
        "light_mode": "rgba(15,15,14,1)",
        "dark_mode": "rgba(242,240,234,1)",
    },
    "wb_handdrawn_vibes_on_accent": {
        "light_mode": "rgba(15,15,14,1)",
        "dark_mode": "rgba(242,240,234,1)",
    },
    "wb_handdrawn_vibes_on_accent_2": {
        "light_mode": "rgba(15,15,14,1)",
        "dark_mode": "rgba(242,240,234,1)",
    },
}


def _theme_token(name: str) -> str:
    return f"wb_handdrawn_vibes_{name}"


def _theme_text(
    text: str,
    *,
    color: str = "on_canvas",
    size: str = "normal",
    bold: bool = False,
    align: str = "left",
    margin: str | None = None,
) -> dict[str, Any]:
    content = f"<font color='{_theme_token(color)}'>{text}</font>"
    if bold:
        content = f"**{content}**"
    node: dict[str, Any] = {
        "tag": "markdown",
        "content": content,
        "text_align": align,
        "text_size": size,
    }
    if margin:
        node["margin"] = margin
    return node


def _theme_pill(text: str) -> dict[str, Any]:
    return {
        "tag": "column_set",
        "flex_mode": "none",
        "columns": [
            {
                "tag": "column",
                "width": "auto",
                "elements": [
                    {
                        "tag": "interactive_container",
                        "behaviors": [],
                        "width": "auto",
                        "height": "auto",
                        "corner_radius": "999px",
                        "has_border": True,
                        "border_color": _theme_token("icon_border"),
                        "background_style": _theme_token("panel"),
                        "padding": "3px 8px 3px 8px",
                        "horizontal_align": "center",
                        "disabled": False,
                        "elements": [
                            _theme_text(
                                text,
                                color="muted",
                                size="notation",
                                align="center",
                            )
                        ],
                    }
                ],
            }
        ],
    }


def _theme_highlights(items: list[tuple[str, str, str, str]]) -> dict[str, Any]:
    return {
        "tag": "column_set",
        "flex_mode": "bisect" if len(items) == 2 else "trisect",
        "horizontal_spacing": "8px",
        "columns": [
            {
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "background_style": _theme_token(background),
                "padding": "14px 12px 14px 12px",
                "horizontal_align": "left",
                "vertical_spacing": "0px",
                "elements": [
                    _theme_text(label, color=foreground, size="notation"),
                    _theme_text(
                        value,
                        color=foreground,
                        size="heading-2",
                        bold=True,
                        margin="4px 0px 0px 0px",
                    ),
                ],
            }
            for label, value, background, foreground in items
        ],
    }


def _theme_image(image_key: str, *, alt: str) -> dict[str, Any]:
    return {
        "tag": "img",
        "img_key": image_key,
        "alt": {"tag": "plain_text", "content": alt, "text_align": "left"},
        "preview": True,
        "transparent": False,
        "scale_type": "crop_center",
        "corner_radius": "16px",
    }


def _theme_panel(lines: list[tuple[str, str, bool]]) -> dict[str, Any]:
    return {
        "tag": "interactive_container",
        "behaviors": [],
        "background_style": _theme_token("panel"),
        "has_border": True,
        "border_color": _theme_token("icon_border"),
        "corner_radius": "14px",
        "padding": "12px 14px 12px 14px",
        "vertical_spacing": "0px",
        "disabled": False,
        "elements": [
            _theme_text(
                text,
                color=color,
                size="normal" if bold else "notation",
                bold=bold,
                margin="4px 0px 0px 0px" if index else None,
            )
            for index, (text, color, bold) in enumerate(lines)
        ],
    }


def _handdrawn_card(
    *,
    summary: str,
    pill: str,
    title: str,
    subtitle: str,
    elements: list[dict[str, Any]],
) -> dict[str, Any]:
    shell_elements = [
        _theme_pill(pill),
        _theme_text(title, size="heading", bold=True, margin="4px 0px 0px 0px"),
        _theme_text(subtitle, color="muted", size="notation", margin="2px 0px 8px 0px"),
        *elements,
    ]
    return {
        "schema": "2.0",
        "config": {
            "compact_width": False,
            "enable_forward": True,
            "summary": {"content": summary},
            "style": {"color": CARD_THEME_COLORS},
        },
        "body": {
            "padding": "0px 0px 0px 0px",
            "elements": [
                {
                    "tag": "interactive_container",
                    "behaviors": [],
                    "background_style": _theme_token("canvas"),
                    "padding": "18px 18px 18px 18px",
                    "corner_radius": "0px",
                    "has_border": False,
                    "direction": "vertical",
                    "vertical_spacing": "10px",
                    "disabled": False,
                    "elements": shell_elements,
                }
            ],
        },
    }


def _milk_progress_chart(percent: int | float, *, accent: str) -> dict[str, Any]:
    """用 Card 2.0 原生图表画一条奶含量进度条。"""
    value = max(0.0, min(100.0, float(percent)))
    return {
        "tag": "chart",
        "height": "48px",
        "color_theme": "primary",
        "preview": False,
        "chart_spec": {
            "type": "bar",
            "background": "rgba(0,0,0,0)",
            "direction": "horizontal",
            "data": {
                "values": [
                    {"metric": "奶含量", "segment": "奶含量", "value": value},
                    {"metric": "奶含量", "segment": "剩余", "value": 100.0 - value},
                ]
            },
            "xField": "value",
            "yField": "metric",
            "seriesField": "segment",
            "stack": True,
            "color": [accent, "rgba(143,149,158,0.18)"],
            "bar": {"style": {"cornerRadius": 12}},
            "label": {"visible": False},
            "axes": [
                {"orient": "bottom", "visible": False, "min": 0, "max": 100},
                {"orient": "left", "visible": False},
            ],
            "legends": {"visible": False},
            "tooltip": {"visible": False},
            "padding": {"top": 2, "right": 0, "bottom": 2, "left": 0},
        },
    }


def build_daily_stats_card(
    stats: DailyStats,
    *,
    title: str = "今日含奶量",
    image_key: str | None = None,
) -> dict[str, Any]:
    percent = f"{stats.milk_percent}"
    summary = f"{title} {percent}%，共识别 {stats.total} 张"
    elements: list[dict[str, Any]] = [
        _theme_highlights(
            [
                ("今日含奶量", f"{percent}%", "accent_2", "on_accent_2"),
                ("今日奶图", f"{stats.milk} 张", "accent", "on_accent"),
            ]
        )
    ]
    if image_key:
        elements.append(_theme_image(image_key, alt="奶龙图鉴主视觉"))
    elements.extend(
        [
            _milk_progress_chart(float(stats.milk_percent), accent="#FEC554"),
            _theme_highlights(
                [
                    ("奶龙", str(stats.nailong), "accent_2", "on_accent_2"),
                    ("奶蛙", str(stats.naiwa), "accent", "on_accent"),
                    ("其他", str(stats.other), "panel", "on_panel"),
                ]
            ),
            _theme_panel(
                [
                    (f"今日新收录奶龙 {stats.new_nailong} 张", "muted", False),
                ]
            ),
        ]
    )
    return _handdrawn_card(
        summary=summary,
        pill="MILK DAILY",
        title=title,
        subtitle=f"今日共识别 {stats.total} 张图片或表情",
        elements=elements,
    )


def build_milk_fortune_card(
    fortune: MilkFortune,
    *,
    day: str,
) -> dict[str, Any]:
    advice = {
        "tag": "column_set",
        "flex_mode": "bisect",
        "horizontal_spacing": "8px",
        "columns": [
            {
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "background_style": _theme_token(background),
                "padding": "12px 10px 12px 10px",
                "vertical_spacing": "0px",
                "elements": [
                    _theme_text(label, color=foreground, size="notation"),
                    _theme_text(
                        value,
                        color=foreground,
                        size="normal",
                        bold=True,
                        margin="4px 0px 0px 0px",
                    ),
                ],
            }
            for label, value, background, foreground in (
                ("宜", fortune.good, "accent", "on_accent"),
                ("忌", fortune.avoid, "panel", "on_panel"),
            )
        ],
    }
    return _handdrawn_card(
        summary=f"今日奶运 {fortune.score} 分，称号是{fortune.title}",
        pill="每日一签",
        title=f"今日奶运：{fortune.title}",
        subtitle=f"{day} · 同一群同一天签固定",
        elements=[
            _theme_highlights(
                [
                    ("奶气值", str(fortune.score), "accent_2", "on_accent_2"),
                    ("接梗力", str(fortune.banter), "accent", "on_accent"),
                ]
            ),
            advice,
            _theme_panel(
                [
                    (f"今日关键词：{fortune.keyword}", "on_panel", True),
                    (fortune.message, "muted", False),
                ]
            ),
        ],
    )


def _correction_buttons(
    observation_id: int,
    *,
    image_key: str | None,
) -> dict[str, Any]:
    buttons: list[tuple[str, str, str, str]] = [
        ("奶龙", "nailong", "accent_2", "on_accent_2"),
        ("奶蛙", "naiwa", "accent", "on_accent"),
        ("其他", "other", "button", "button_text"),
    ]
    columns: list[dict[str, Any]] = []
    for text, label, background, foreground in buttons:
        value: dict[str, Any] = {
            "action": "correct_label",
            "observation_id": observation_id,
            "label": label,
        }
        if image_key:
            value["image_key"] = image_key
        columns.append(
            {
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "elements": [
                    {
                        "tag": "interactive_container",
                        "behaviors": [{"type": "callback", "value": value}],
                        "background_style": _theme_token(background),
                        "has_border": True,
                        "border_color": _theme_token("icon_border"),
                        "corner_radius": "14px",
                        "padding": "10px 8px 10px 8px",
                        "horizontal_align": "center",
                        "disabled": False,
                        "confirm": {
                            "title": {"tag": "plain_text", "content": f"确认是{text}？"},
                            "text": {
                                "tag": "plain_text",
                                "content": "确认后会更新图鉴与后续统计。",
                            },
                        },
                        "elements": [
                            _theme_text(
                                text,
                                color=foreground,
                                size="normal",
                                bold=True,
                                align="center",
                            )
                        ],
                    }
                ],
            }
        )
    return {
        "tag": "column_set",
        "flex_mode": "trisect",
        "horizontal_spacing": "8px",
        "margin": "6px 0px 0px 0px",
        "columns": columns,
    }


def build_prediction_card(
    observation: ObservationRecord,
    prediction: Prediction,
    rarity: Rarity | None,
    *,
    image_key: str | None = None,
) -> dict[str, Any]:
    label = DISPLAY_LABELS.get(prediction.label, prediction.label)
    confidence = f"{prediction.confidence:.1%}"
    concentration = milk_concentration(prediction)

    if observation.asset.corrected:
        title = f"已确认是{DISPLAY_LABELS[observation.asset.label]}"
        conclusion = f"图鉴编号 #{observation.asset.id:04d}"
    elif not prediction.accepted:
        if is_milky_other(prediction):
            title = other_reaction(prediction)
            conclusion = f"识别结果暂不确定，最高候选是「{label}」，可能漏识别了奶龙或奶蛙"
        else:
            title = "识别结果暂不确定"
            conclusion = f"最高候选是「{label}」"
    elif prediction.label == "other":
        title = other_reaction(prediction)
        conclusion = (
            "识别结果：其他，可能漏识别了奶龙或奶蛙"
            if is_milky_other(prediction)
            else "识别结果：其他，这张图不像奶龙或奶蛙"
        )
    else:
        title = f"识别结果：{label}"
        conclusion = f"这是一张{label}图"

    summary = f"{title}，置信度 {confidence}，奶含量 {concentration}%"
    elements: list[dict[str, Any]] = [
        _theme_highlights(
            [
                ("模型置信度", confidence, "accent", "on_accent"),
                ("奶含量", f"{concentration}%", "accent_2", "on_accent_2"),
            ]
        ),
    ]
    if image_key:
        elements.append(_theme_image(image_key, alt="本次识别的图片"))
    elements.extend(
        [
            _theme_panel([(conclusion, "on_panel", True)]),
            _milk_progress_chart(concentration, accent="#FEC554"),
        ]
    )
    if rarity is not None:
        elements.append(
            _theme_panel(
                [
                    (f"稀有度：{rarity.stars_text} {rarity.name}", "on_panel", True),
                    (f"{rarity.note} · 群里出现过 {rarity.seen_count} 次", "muted", False),
                ]
            )
        )
    if not observation.asset.corrected:
        elements.extend(
            [
                _theme_text("识别不对？点一下直接纠正", color="muted", size="notation"),
                _correction_buttons(observation.id, image_key=image_key),
            ]
        )
    return _handdrawn_card(
        summary=summary,
        pill="AI 识别",
        title=title,
        subtitle=f"置信度 {confidence} · 奶含量 {concentration}%",
        elements=elements,
    )


def build_correction_confirmation_card(
    observation: ObservationRecord,
    rarity: Rarity | None,
    *,
    image_key: str | None = None,
) -> dict[str, Any]:
    label = DISPLAY_LABELS[observation.asset.label]
    elements: list[dict[str, Any]] = [
        _theme_highlights(
            [
                ("确认分类", label, "accent_2", "on_accent_2"),
                ("图鉴编号", f"#{observation.asset.id:04d}", "accent", "on_accent"),
            ]
        )
    ]
    if image_key:
        elements.append(_theme_image(image_key, alt=f"已确认是{label}的图片"))
    if rarity is not None:
        elements.append(
            _theme_panel(
                [
                    (f"稀有度：{rarity.stars_text} {rarity.name}", "on_panel", True),
                    (f"{rarity.note} · 群里出现过 {rarity.seen_count} 次", "muted", False),
                ]
            )
        )
    elements.append(
        _theme_panel(
            [
                ("人工校正已生效", "on_panel", True),
                ("后续识别、图鉴和统计会使用新分类。", "muted", False),
            ]
        )
    )
    return _handdrawn_card(
        summary=f"图鉴 #{observation.asset.id:04d} 已确认是{label}",
        pill="已确认",
        title=f"已确认：{label}",
        subtitle=f"图鉴 #{observation.asset.id:04d} · 分类已更新",
        elements=elements,
    )


def parse_correction_card_action(
    value: Any,
) -> tuple[int, str, str | None] | None:
    if not isinstance(value, dict) or value.get("action") != "correct_label":
        return None
    observation_id = value.get("observation_id")
    label = value.get("label")
    if isinstance(observation_id, bool) or not isinstance(observation_id, int):
        return None
    if observation_id <= 0 or label not in COMMAND_LABELS.values():
        return None
    image_key = value.get("image_key")
    if not isinstance(image_key, str) or not image_key.startswith("img_"):
        image_key = None
    return observation_id, label, image_key


def format_help_message(
    report_time: time, *, platform: str = "feishu", allow_proactive: bool = False
) -> str:
    if platform == "qq":
        report_note = (
            f"已开启主动播报，每天 {report_time.strftime('%H:%M')} 发送，需群主允许主动发言。"
            if allow_proactive else "当前未开启定时主动播报，可手动查询。"
        )
        return (
            "我是奶龙，QQ群里先 @我再发送下面的内容，私聊直接发：\n"
            "1. 图片：识别奶龙/奶蛙，查看奶含量和稀有度；QQ表情请保存成图片发送。\n"
            "2. 直接发图片并 @我：识图后自动接同类奶图；"
            "回复图片并 @我：同样识图并接龙，附奶含量和稀有度。\n"
            "3. 回复原图或接龙图，再 @我发“纠正为奶龙/奶蛙/其他”：修正分类。\n"
            "4. 奶龙图鉴：查看最近收录的奶图。\n"
            f"5. 今日含奶量：查看本群已收到的图片统计。{report_note}\n"
            "6. 猜图：开盲盒；@我 奶龙/奶蛙/其他 抢答；"
            "公布答案：揭晓；猜图榜：看积分。\n"
            "7. 奶龙猎人榜：看首发收图排行；今日奶运：抽当天奶签。\n"
            "8. 接奶（或奶家人）：认领你为奶家人，领一张随机奶图，不用 @群友或输入昵称。\n"
            "9. 求赐名：随机起一个不撞名的奶名并记住，再求一次就换名。\n"
            "10. #奶龙 / #奶蛙 / #其他话题：找相关抖音视频，需搜索可用且链接通过QQ审核。\n"
            "群主允许接收全量消息后，普通发图也能自动接龙；"
            "普通成员入群暂不能自动欢迎。奶运和识图使用文字回复。"
        )
    return (
        "我是奶龙，可以这样玩：\n"
        "1. 直接发送图片或表情：自动接龙，不用 @。\n"
        "2. 回复图片并 @我：查看识别结果、奶龙浓度和稀有度。\n"
        "3. 回复原图或接龙图，发送“纠正为奶龙/奶蛙/其他”：修正分类。\n"
        "4. 发送“奶龙图鉴”：查看最近收录的奶龙图。\n"
        "5. 发送“今日含奶量”：立即查看；"
        f"每天 {report_time.strftime('%H:%M')} 也会自动播报。\n"
        "6. 发送“猜图”：开一局盲盒猜图，抢答“奶龙/奶蛙/其他”得分；"
        "“公布答案”揭晓本局，“猜图榜”看积分排行。\n"
        "7. 发送“奶龙猎人榜”：看谁首发收进图鉴的奶图最多。\n"
        "8. 发送“今日奶运”：抽当天固定奶签，看看奶气值、宜忌和称号。\n"
        "9. @我发送“求赐名”：随机起一个不撞名的奶名并记住，再求一次就换名。\n"
        "10. 发送“HDR奶图”：立即来一张高亮奶图；自动接龙也会随机转成 HDR。"
    )


def format_catalog_caption(stats: CatalogStats, shown: int) -> str:
    return (
        f"共收录 {stats.total} 张：奶龙 {stats.nailong} · 奶蛙 {stats.naiwa}。"
        f"今日新增 {stats.today_new} 张，"
        f"当前展示最近 {shown} 张。"
    )


def build_catalog_post(
    assets: list[AssetRecord],
    image_keys: list[str],
    stats: CatalogStats,
) -> dict[str, Any]:
    if len(assets) != len(image_keys):
        raise ValueError("Catalog assets and image keys must have the same length")

    rows: list[list[dict[str, str]]] = [
        [{"tag": "text", "text": format_catalog_caption(stats, len(assets))}]
    ]
    for asset, image_key in zip(assets, image_keys, strict=True):
        display_label = DISPLAY_LABELS[asset.label]
        rows.append([{"tag": "img", "image_key": image_key}])
        caption = f"#{asset.id:04d} · {display_label} · 出现 {asset.seen_count} 次"
        # 有首发人署名的新图才加这一句；老库存量图没有署名，保持原样不加。
        if asset.discoverer_name:
            caption += f" · 首发：{asset.discoverer_name}"
        rows.append([{"tag": "text", "text": caption}])
    return {"zh_cn": {"title": "奶龙图鉴", "content": rows}}


def format_chain_caption(observation: ObservationRecord) -> str:
    if observation.asset.label == "nailong":
        if observation.is_new_asset and observation.asset.origin == "collected":
            return f"奶龙接上了，新收录图鉴 #{observation.asset.id:04d}。"
        return "奶龙接上了。"
    return "奶蛙接上了。"


def format_existing_observation(
    observation: ObservationRecord,
    prediction: Prediction,
) -> str:
    if observation.asset.corrected:
        display_label = DISPLAY_LABELS[observation.asset.label]
        return f"这张图已经确认是{display_label}，图鉴编号 #{observation.asset.id:04d}。"
    return format_prediction(prediction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image classifier bot adapters")
    parser.add_argument(
        "--bots-config",
        type=Path,
        default=Path(os.environ["BOT_ADAPTERS_CONFIG"])
        if os.getenv("BOT_ADAPTERS_CONFIG")
        else None,
        help="TOML file containing [[bots]] adapter instances; defaults to legacy Feishu.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("MODEL_PATH", "artifacts/model.pt")),
    )
    parser.add_argument("--device", default=os.getenv("MODEL_DEVICE", "cpu"))
    parser.add_argument(
        "--milk-model",
        type=Path,
        default=Path(os.environ["MILK_MODEL_PATH"]) if os.getenv("MILK_MODEL_PATH") else None,
        help="Verified milk-priority release manifest; shares classification and Doubao filtering.",
    )
    parser.add_argument(
        "--doubao-model",
        type=Path,
        default=Path(os.getenv("DOUBAO_MODEL_PATH"))
        if os.getenv("DOUBAO_MODEL_PATH")
        else None,
        help="Optional standalone Doubao presence detector checkpoint.",
    )
    parser.add_argument(
        "--doubao-threshold",
        type=float,
        default=(
            float(os.environ["DOUBAO_MODEL_THRESHOLD"])
            if os.getenv("DOUBAO_MODEL_THRESHOLD")
            else None
        ),
        help="Override the Doubao detector's checkpoint threshold.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.getenv("BOT_STATE_DIR", "runtime")),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ["MODEL_THRESHOLD"]) if os.getenv("MODEL_THRESHOLD") else None,
    )
    parser.add_argument(
        "--daily-report-time",
        type=parse_daily_report_time,
        default=parse_daily_report_time(
            os.getenv(
                "DAILY_REPORT_TIME",
                DEFAULT_DAILY_REPORT_TIME.strftime("%H:%M"),
            )
        ),
    )
    parser.add_argument(
        "--catalog-document-id",
        default=os.getenv("CATALOG_DOCUMENT_ID") or None,
    )
    parser.add_argument(
        "--catalog-section-block-id",
        default=os.getenv("CATALOG_SECTION_BLOCK_ID") or None,
    )
    parser.add_argument(
        "--catalog-baseline-asset-ids",
        type=parse_asset_ids,
        default=parse_asset_ids(os.getenv("CATALOG_BASELINE_ASSET_IDS", "")),
    )
    parser.add_argument(
        "--catalog-sync-retry-seconds",
        type=int,
        default=int(
            os.getenv(
                "CATALOG_SYNC_RETRY_SECONDS",
                str(DEFAULT_CATALOG_SYNC_RETRY_SECONDS),
            )
        ),
    )
    parser.add_argument(
        "--random-push-probability",
        type=float,
        default=float(
            os.getenv(
                "RANDOM_PUSH_PROBABILITY",
                str(DEFAULT_RANDOM_PUSH_PROBABILITY),
            )
        ),
    )
    parser.add_argument(
        "--random-push-cooldown-seconds",
        type=int,
        default=int(
            os.getenv(
                "RANDOM_PUSH_COOLDOWN_SECONDS",
                str(DEFAULT_RANDOM_PUSH_COOLDOWN_SECONDS),
            )
        ),
    )
    parser.add_argument(
        "--douyin-cookie-file",
        type=Path,
        default=Path(os.getenv("DOUYIN_COOKIE_FILE"))
        if os.getenv("DOUYIN_COOKIE_FILE")
        else None,
        help="Path to a private browser-exported Douyin cookie JSON file.",
    )
    parser.add_argument(
        "--douyin-search-keywords",
        type=parse_douyin_search_keywords,
        default=parse_douyin_search_keywords(
            os.getenv("DOUYIN_SEARCH_KEYWORDS", ",".join(SEARCH_KEYWORDS))
        ),
        help="Comma-separated Douyin topics. A leading # is optional.",
    )
    parser.add_argument(
        "--douyin-search-cache-seconds",
        type=int,
        default=int(
            os.getenv(
                "DOUYIN_SEARCH_CACHE_SECONDS",
                str(DEFAULT_DOUYIN_SEARCH_CACHE_SECONDS),
            )
        ),
    )
    parser.add_argument(
        "--douyin-video-push-probability",
        type=float,
        default=float(
            os.getenv(
                "DOUYIN_VIDEO_PUSH_PROBABILITY",
                str(DEFAULT_DOUYIN_VIDEO_PUSH_PROBABILITY),
            )
        ),
    )
    parser.add_argument(
        "--douyin-video-push-window-seconds",
        type=int,
        default=int(
            os.getenv(
                "DOUYIN_VIDEO_PUSH_WINDOW_SECONDS",
                str(DEFAULT_DOUYIN_VIDEO_PUSH_WINDOW_SECONDS),
            )
        ),
    )
    parser.add_argument(
        "--douyin-video-push-max-per-window",
        type=int,
        default=int(
            os.getenv(
                "DOUYIN_VIDEO_PUSH_MAX_PER_WINDOW",
                str(DEFAULT_DOUYIN_VIDEO_PUSH_MAX_PER_WINDOW),
            )
        ),
    )
    parser.add_argument(
        "--random-push-hdr-probability",
        type=float,
        default=float(os.getenv("RANDOM_PUSH_HDR_PROBABILITY", str(DEFAULT_HDR_PROBABILITY))),
    )
    parser.add_argument(
        "--milk-image-generator-open-id",
        default=os.getenv("MILK_IMAGE_GENERATOR_OPEN_ID") or None,
        help="Optional Feishu open_id of the image bot to mention after a public milk-name lookup.",
    )
    parser.add_argument(
        "--milk-image-generator-name",
        default=os.getenv("MILK_IMAGE_GENERATOR_NAME", "神笔奶良"),
        help="Feishu group-member name used to discover the image bot when no open_id is set.",
    )
    parser.add_argument(
        "--hdr-peak-nits",
        type=float,
        default=float(os.getenv("HDR_PEAK_NITS", str(DEFAULT_HDR_PEAK_NITS))),
    )
    args = parser.parse_args()
    try:
        validate_hdr_settings(args.random_push_hdr_probability, args.hdr_peak_nits)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 <= args.douyin_video_push_probability <= 1:
        parser.error("douyin video push probability must be between 0 and 1")
    if args.douyin_video_push_window_seconds <= 0:
        parser.error("douyin video push window seconds must be positive")
    if args.douyin_video_push_max_per_window <= 0:
        parser.error("douyin video push max per window must be positive")
    if args.douyin_search_cache_seconds <= 0:
        parser.error("douyin search cache seconds must be positive")
    if args.doubao_threshold is not None and not 0 < args.doubao_threshold < 1:
        parser.error("doubao threshold must be between 0 and 1")
    return args


async def run_bot(args: argparse.Namespace) -> None:
    configs = load_adapter_configs(getattr(args, "bots_config", None))
    adapters: list[BotAdapter] = []
    tasks: list[asyncio.Task] = []
    try:
        # 先校验所有接入凭证，再加载一份共用模型，避免每个机器人重复占内存。
        # SDK 导入时捕获模块级事件循环；即使只有一个飞书实例，在运行中的
        # 主循环里才导入 SDK 也会捕获错误的 loop。每个飞书 SDK 固定放进子进程，
        # 由 worker 在 asyncio.run 前导入，模型和业务仍只在主进程加载一次。
        isolate_feishu = True
        for config in configs:
            adapters.append(create_adapter(config, isolate_feishu=isolate_feishu))
        if getattr(args, "milk_model", None):
            from .milk_runtime import MilkPriorityPredictor

            milk_predictor = MilkPriorityPredictor(
                args.milk_model, device=args.device, threshold=args.threshold,
                guard_threshold=args.doubao_threshold,
            )
            predictor = SerializedPredictor(milk_predictor)
            doubao_detector = SerializedPredictor(milk_predictor.doubao_detector)
        else:
            predictor = SerializedPredictor(NailongPredictor(
                args.model, device=args.device, threshold=args.threshold
            ))
            doubao_detector = (
                SerializedPredictor(DoubaoDetector(
                    args.doubao_model, device=args.device, threshold=args.doubao_threshold
                ))
                if args.doubao_model
                else None
            )
        services = BackendServices(predictor, BotStore(args.state_dir), doubao_detector)
        for config, adapter in zip(configs, adapters, strict=True):
            runtime_args = argparse.Namespace(**vars(args))
            if getattr(args, "bots_config", None):
                # 多实例配置中的文档同步由一个明确的飞书实例负责；
                # 不能把旧全局文档配置无意间复制给所有 QQ/飞书实例。
                runtime_args.catalog_document_id = config.catalog_document_id
                runtime_args.catalog_section_block_id = config.catalog_section_block_id
            tasks.append(asyncio.create_task(
                run_adapter(runtime_args, adapter, services),
                name=f"bot:{config.instance_id}",
            ))
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        # 一个实例异常退出时也必须收尾其余实例；重抛失败任务会跳过 close。
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for adapter in adapters:
            try:
                await adapter.close()
            except Exception:
                LOGGER.exception("Adapter shutdown failed instance=%s", adapter.scope.instance_id)


async def run_adapter(
    args: argparse.Namespace, channel: BotAdapter, services: BackendServices
) -> None:
    predictor = services.predictor
    doubao_detector = services.doubao_detector
    store = services.store
    chain_lock = asyncio.Lock()
    guess_lock = asyncio.Lock()
    contact_resolver = channel
    # 每个群上次主动推送的单调时钟时间，用于冷却；进程内存着即可，重启清零无妨。
    last_random_push: dict[str, float] = {}
    # 第三方生图机器人不属于奶家人，也不写入数据库；仅在当前进程记住其群内身份。
    milk_image_generator_ids: dict[str, str] = {}
    douyin_search_lock = asyncio.Lock()
    douyin_video_cache: dict[tuple[str, ...], tuple[float, list[DouyinVideo]]] = {}
    catalog_sync_event = asyncio.Event()
    catalog_sync_lock = asyncio.Lock()
    catalog_document: CatalogDocumentClient | None = None
    if bool(args.catalog_document_id) != bool(args.catalog_section_block_id):
        raise RuntimeError(
            "CATALOG_DOCUMENT_ID and CATALOG_SECTION_BLOCK_ID must be set together"
        )
    if args.catalog_document_id and args.catalog_section_block_id:
        catalog_document = channel.catalog_document(
            args.catalog_document_id, args.catalog_section_block_id
        )
        baseline_count = await asyncio.to_thread(
            store.mark_catalog_document_baseline,
            document_id=args.catalog_document_id,
            asset_ids=args.catalog_baseline_asset_ids,
        )
        LOGGER.info(
            "Catalog document sync enabled document_id=%s section_block_id=%s "
            "baseline_inserted=%d",
            args.catalog_document_id,
            args.catalog_section_block_id,
            baseline_count,
        )

    async def reply(message, text: str):
        result = await channel.send(
            message.chat_id,
            {"text": text},
            {"reply_to": message.message_id},
        )
        if not result.success:
            LOGGER.error("Failed to reply: %s", result.error)
        return result

    async def reply_card(message, card: dict[str, Any], fallback_text: str):
        result = await channel.send(
            message.chat_id,
            {"card": card, "fallback_text": fallback_text},
            {"reply_to": message.message_id},
        )
        if result.success:
            return result
        LOGGER.error("Failed to reply with card: %s", result.error)
        if result.error and result.error.code in {ErrorCode.TARGET_REVOKED, ErrorCode.SEND_TIMEOUT}:
            return result
        LOGGER.info("Falling back to plain-text reply after card send failure")
        return await reply(message, fallback_text)

    async def reply_image(message, source: bytes | Path, caption: str):
        image_source = source if isinstance(source, bytes) else str(source)
        result = await channel.send(
            message.chat_id,
            {"image": {"source": image_source}, "caption": caption},
            {"reply_to": message.message_id},
        )
        if not result.success:
            LOGGER.error("Failed to reply with image: %s", result.error)
        return result

    async def request_milk_name_image(message, milk_name: str) -> None:
        image_bot_open_id = (
            args.milk_image_generator_open_id
            or milk_image_generator_ids.get(message.chat_id)
        )
        if channel.scope.platform != "feishu":
            return
        if not image_bot_open_id:
            target_name = args.milk_image_generator_name.strip()
            if not target_name:
                return
            try:
                members = await channel.get_chat_members(message.chat_id)
            except Exception:
                LOGGER.warning(
                    "Milk name image bot lookup failed instance=%s",
                    channel.scope.instance_id,
                )
                return
            target = next(
                (
                    member
                    for member in members
                    if member.name and member.name.strip() == target_name
                ),
                None,
            )
            if target is None:
                LOGGER.info(
                    "Milk name image bot not in chat instance=%s target_name=%s",
                    channel.scope.instance_id,
                    target_name,
                )
                return
            image_bot_open_id = channel.scope.raw("user", target.id)
        result = await channel.send(
            message.chat_id,
            {"post": build_milk_name_image_request(image_bot_open_id, milk_name)},
            {"reply_to": message.message_id},
        )
        if not result.success:
            LOGGER.warning(
                "Milk name image request failed instance=%s error=%s",
                channel.scope.instance_id,
                result.error,
            )

    def remember_milk_image_generator(message) -> None:
        if channel.scope.platform != "feishu":
            return
        target_name = args.milk_image_generator_name.strip()
        if not target_name:
            return
        candidates = [(message.sender_id, message.sender_name)]
        candidates.extend(
            (mention.user_id, mention.name) for mention in message.mentions
        )
        for user_id, display_name in candidates:
            if (
                user_id
                and display_name
                and display_name.strip() == target_name
                and channel.scope.owns("user", user_id)
            ):
                milk_image_generator_ids[message.chat_id] = channel.scope.raw(
                    "user", user_id
                )
                LOGGER.info(
                    "Milk name image bot remembered instance=%s",
                    channel.scope.instance_id,
                )
                return

    async def enroll_existing_group_members() -> None:
        """启动时把已知活跃群的成员补登记为奶家人。

        单个群缺少群成员权限或机器人已被移出时只记日志，其余群继续处理。
        """
        chat_ids = await asyncio.to_thread(store.active_group_chat_ids)
        enrolled = 0
        failed = 0
        for chat_id in chat_ids:
            if not channel.scope.owns("chat", chat_id):
                continue
            try:
                members = await channel.get_chat_members(chat_id)
            except Exception:
                failed += 1
                LOGGER.warning(
                    "Milk family enrollment failed chat_hash=%s; "
                    "检查 im:chat.members:read 权限或机器人群状态",
                    store.identity_hash(chat_id),
                    exc_info=True,
                )
                continue
            enrolled += await asyncio.to_thread(
                store.record_milk_family_members,
                [(member.id, member.name) for member in members],
            )
        LOGGER.info(
            "Milk family enrollment completed groups=%d members=%d failed_groups=%d",
            len(chat_ids),
            enrolled,
            failed,
        )

    async def send_milk_image(
        message,
        asset: AssetRecord,
        *,
        hdr_probability: float,
        reply_to: bool,
        caption: str | None = None,
    ):
        """接龙、主动推送和点名闪亮奶图共用发送入口。"""
        if doubao_detector:
            doubao_prediction = await asyncio.to_thread(
                doubao_detector.predict_path,
                asset.path,
            )
            if doubao_prediction.detected:
                LOGGER.warning(
                    "Blocked bot asset with Doubao asset_id=%d confidence=%.3f",
                    asset.id,
                    doubao_prediction.confidence,
                )
                return None
        content = await asyncio.to_thread(
            random_milk_image_content,
            asset.path,
            probability=hdr_probability,
            peak_nits=args.hdr_peak_nits,
        )
        is_flash_gif = isinstance(content["image"]["source"], bytes)
        if caption and not is_flash_gif:
            content["caption"] = caption
        options = {"reply_to": message.message_id} if reply_to else None
        result = await channel.send(message.chat_id, content, options)
        # GIF 上传失败时只回退一次原图；群已撤销或发送结果不确定时不重复发。
        if (
            not result.success
            and is_flash_gif
            and result.error
            and result.error.code == ErrorCode.UPLOAD_FAILED
        ):
            LOGGER.warning(
                "Flash GIF upload failed, falling back to original asset_id=%d",
                asset.id,
            )
            result = await channel.send(
                message.chat_id, {"image": {"source": str(asset.path)}}, options
            )
            is_flash_gif = False
        if result.success:
            LOGGER.info(
                "Milk image sent chat_hash=%s asset_id=%d flash_gif=%s message_id=%s",
                store.identity_hash(message.chat_id),
                asset.id,
                is_flash_gif,
                result.message_id,
            )
        elif result.error and result.error.code == ErrorCode.TARGET_REVOKED:
            await asyncio.to_thread(store.deactivate_chat, message.chat_id)
        else:
            LOGGER.error(
                "Milk image send failed asset_id=%d flash_gif=%s error=%s",
                asset.id,
                is_flash_gif,
                result.error,
            )
        return result

    async def cached_douyin_videos(keywords: tuple[str, ...]) -> list[DouyinVideo]:
        if not args.douyin_cookie_file:
            return []
        async with douyin_search_lock:
            now = time_module.monotonic()
            fetched_at, videos = douyin_video_cache.get(keywords, (0.0, []))
            if now - fetched_at >= args.douyin_search_cache_seconds:
                try:
                    videos = await asyncio.to_thread(
                        search_douyin_videos,
                        args.douyin_cookie_file,
                        keywords=keywords,
                    )
                    searched_count = len(videos)
                    videos = await asyncio.to_thread(
                        filter_douyin_videos_by_cover,
                        videos,
                        predictor,
                    )
                    douyin_video_cache[keywords] = (now, videos)
                    LOGGER.info(
                        "Douyin search cache refreshed topics=%s searched_count=%d "
                        "verified_count=%d",
                        ",".join(keywords),
                        searched_count,
                        len(videos),
                    )
                except DouyinSearchError:
                    LOGGER.warning("Douyin search failed; skipping video link push", exc_info=True)
                    return []
            return videos

    async def reply_catalog(
        message,
        assets: list[AssetRecord],
        stats: CatalogStats,
    ) -> None:
        gallery = []
        for asset in assets:
            caption = (
                f"#{asset.id:04d} · {DISPLAY_LABELS[asset.label]} · 出现 {asset.seen_count} 次"
            )
            if asset.discoverer_name:
                caption += f" · 首发：{asset.discoverer_name}"
            gallery.append({"source": asset.path, "caption": caption})
        result = await channel.send(
            message.chat_id,
            {"gallery": gallery, "caption": format_catalog_caption(stats, len(assets))},
            {"reply_to": message.message_id},
        )
        if not result.success:
            LOGGER.error("Failed to reply with catalog: %s", result.error)

    async def handle_command(message) -> bool:
        if message.raw_content_type not in {"text", "post"}:
            return False
        text = message.body_text or message.content_text
        normalized_text = text.strip().replace(" ", "")
        mentioned_users = mentioned_user_ids(message, body_text=text)

        if (
            message.mentioned_bot
            and mentioned_users
            and is_milk_name_grant_for_other_command(text)
        ):
            # 只有明确说“给某人求赐名”才替群友起名；普通求赐名仍只作用于本人。
            user_id, mention_name = mentioned_users[0]
            existing = await asyncio.to_thread(store.milk_name_for_user, user_id)
            already_named = existing is not None
            member = await asyncio.to_thread(store.milk_family_member, user_id)
            display_name = mention_name or (
                member.display_name if member and member.display_name else None
            )
            if not display_name:
                display_name = await contact_resolver.resolve(user_id, fallback="这位奶友")
            display_name = display_name.strip() or "这位奶友"
            if existing is None:
                await asyncio.to_thread(
                    store.record_milk_family_member, user_id, display_name=display_name
                )
            # 明确替人求赐名就是重抽入口：存储层在写锁内排除旧名并保证全局不撞名。
            existing = await asyncio.to_thread(
                store.assign_milk_name,
                user_id,
                original_name=display_name,
                milk_name=make_milk_name(display_name),
                name_options=milk_name_options,
            )
            await reply(
                message,
                format_milk_name_for_other(
                    display_name, existing.milk_name, existing=already_named
                ),
            )
            await request_milk_name_image(message, existing.milk_name)
            return True

        if (
            is_milk_name_command(text)
            and (message.mentioned_bot or message.chat_type != "group")
        ):
            # 指令只给发送者本人起名，优先于群友点名和引用图片逻辑。
            # 已有真实原昵称不重复查询；系统兜底留下的空缺允许后续补全。
            if not message.sender_id:
                await reply(message, "暂时拿不到你的身份，请重新发送“求赐名”试试。")
                return True
            assigned = await asyncio.to_thread(store.milk_name_for_user, message.sender_id)
            name_known = assigned is not None and assigned.original_name_known
            if name_known:
                display_name = assigned.original_name
            else:
                member = await asyncio.to_thread(store.milk_family_member, message.sender_id)
                display_name = message.sender_name or (
                    member.display_name
                    if member and member.display_name not in LEGACY_NAME_FALLBACKS else None
                )
                display_name = display_name.strip() if display_name else ""
                if not display_name:
                    try:
                        display_name = await contact_resolver.resolve(
                            message.sender_id, fallback=""
                        )
                    except Exception:
                        LOGGER.warning(
                            "Milk name lookup failed instance=%s", channel.scope.instance_id
                        )
                    display_name = display_name.strip() if isinstance(display_name, str) else ""
                name_known = bool(display_name)
                if name_known:
                    await asyncio.to_thread(
                        store.record_milk_family_member, message.sender_id,
                        display_name=display_name,
                    )
                else:
                    display_name = "神秘奶友"
            assigned = await asyncio.to_thread(
                store.assign_milk_name,
                message.sender_id,
                original_name=display_name,
                milk_name=make_milk_name(display_name),
                name_options=milk_name_options,
                original_name_known=name_known,
            )
            await reply(
                message, format_milk_name_grant(
                    assigned.original_name, assigned.milk_name,
                    original_name_known=assigned.original_name_known,
                )
            )
            return True

        if message.mentioned_bot and normalized_text in {"安静一点", "少说话", "闭嘴"}:
            await asyncio.to_thread(store.set_proactive_enabled, message.chat_id, False)
            await reply(message, "好，我安静一点。以后只在被 @、识图或收到命令时说话。")
            return True

        if message.mentioned_bot and normalized_text in {"开始说话", "活跃一点", "别安静"}:
            await asyncio.to_thread(store.set_proactive_enabled, message.chat_id, True)
            await reply(message, "收到，恢复正常参与。")
            return True

        # QQ 用发送者的可靠身份直接领图，不依赖群成员 @ 或昵称解析。
        # on_message 已登记发送者；图片仍优先走识图，飞书保留原来的群友点名。
        if (
            channel.scope.platform == "qq"
            and message.mentioned_bot
            and text.strip() in {"接奶", "奶家人"}
            and not has_direct_image(message)
        ):
            member = await asyncio.to_thread(store.milk_family_member, message.sender_id)
            display_name = message.sender_name or (
                member.display_name if member and member.display_name else None
            ) or "奶友"
            asset = await asyncio.to_thread(store.random_milk_asset)
            if asset:
                await send_milk_image(
                    message,
                    asset,
                    hdr_probability=0,
                    reply_to=True,
                    caption=f"{display_name}，奶家人接奶！🥛",
                )
            else:
                await reply(
                    message,
                    f"{display_name}，你已经是奶家人啦！"
                    "图鉴里还没有奶图，先发张奶龙或奶蛙图片给我收录吧。",
                )
            return True
        # “回复图片后 @我”优先走识图。否则图片回复会被误判成奶家人点名。
        if (
            message.mentioned_bot
            and mentioned_users
            and not has_direct_image(message)
            and not (message.reply_to_message_id or _raw_parent_id(message))
        ):
            user_id, mention_name = mentioned_users[0]
            member = await asyncio.to_thread(store.milk_family_member, user_id)
            display_name = mention_name or (
                member.display_name if member and member.display_name else None
            )
            if not display_name:
                display_name = await contact_resolver.resolve(
                    user_id,
                    fallback="这位奶友",
                )
            if member is None:
                await reply(message, format_milk_family_check(display_name, None))
                return True
            assigned = await asyncio.to_thread(store.milk_name_for_user, user_id)
            if assigned is None:
                assigned = await asyncio.to_thread(
                    store.assign_milk_name,
                    user_id,
                    original_name=display_name,
                    milk_name=make_milk_name(display_name),
                    name_options=milk_name_options,
                )
            await reply(
                message,
                format_milk_name_for_other(
                    display_name, assigned.milk_name, existing=True
                ),
            )
            await request_milk_name_image(message, assigned.milk_name)
            return True

        requested_topics = parse_douyin_topic_request(text)
        if message.mentioned_bot and requested_topics:
            videos = await cached_douyin_videos(requested_topics)
            if not videos:
                await reply(message, "这个话题暂时没有找到可用视频，换个 #话题 试试。")
                return True
            video = choose_douyin_video(videos, keywords=requested_topics)
            await reply(message, f"来点#{video.keyword}：{video.url}")
            return True

        if is_help_request(text, mentioned_bot=message.mentioned_bot):
            await reply(message, format_help_message(
                args.daily_report_time,
                platform=channel.scope.platform,
                allow_proactive=getattr(channel, "allow_proactive", False),
            ))
            return True

        if is_hdr_milk_command(text):
            asset = await asyncio.to_thread(store.random_milk_asset)
            if asset:
                await send_milk_image(message, asset, hdr_probability=1.0, reply_to=True)
            else:
                await reply(message, "还没有可用奶图，先发几张奶龙或奶蛙吧。")
            return True

        if is_milk_fortune_command(text):
            day = datetime.now(LOCAL_TIMEZONE).date().isoformat()
            fortune = make_milk_fortune(
                chat_id=message.chat_id,
                sender_id=message.sender_id,
                day=day,
            )
            await reply_card(
                message,
                build_milk_fortune_card(fortune, day=day),
                format_milk_fortune(fortune, day=day),
            )
            return True

        corrected_label = parse_correction_command(text)
        if corrected_label:
            reference_id = message.reply_to_message_id or _raw_parent_id(message)
            if not reference_id:
                await reply(message, "请回复原图片或我的接龙图片，再发送“纠正为奶龙/奶蛙/其他”。")
                return True
            correction = await asyncio.to_thread(
                store.correct_label,
                reference_id,
                corrected_label,
            )
            if not correction:
                await reply(message, "没有找到这张图的识别记录，请重新发送图片后再纠正。")
                return True
            if correction.new_label in {"nailong", "naiwa"}:
                services.notify_catalog_changed()
            display_label = DISPLAY_LABELS[correction.new_label]
            if correction.old_label == correction.new_label:
                text = f"已确认：图鉴 #{correction.asset_id:04d} 是{display_label}。"
            else:
                old_label = DISPLAY_LABELS[correction.old_label]
                text = (
                    f"已纠正：图鉴 #{correction.asset_id:04d} "
                    f"从{old_label}改为{display_label}。"
                )
            await reply(message, text)
            return True

        if is_daily_stats_command(text):
            stats = await asyncio.to_thread(store.daily_stats, message.chat_id)
            await reply_card(
                message,
                build_daily_stats_card(stats),
                format_daily_stats(stats),
            )
            return True

        if is_catalog_command(text):
            assets = await asyncio.to_thread(store.catalog_assets)
            if not assets:
                await reply(message, "图鉴还没有收集到群聊表情，先发几张奶图吧。")
                return True
            stats = await asyncio.to_thread(store.catalog_stats)
            await reply_catalog(message, assets, stats)
            return True

        if is_guess_start_command(text):
            await start_guess(message)
            return True

        if is_guess_reveal_command(text):
            async with guess_lock:
                round_ = await asyncio.to_thread(
                    store.open_guess_round,
                    chat_id=message.chat_id,
                )
                if round_:
                    await asyncio.to_thread(
                        store.close_guess_round,
                        round_id=round_.id,
                    )
            if not round_:
                await reply(message, format_guess_no_open_round())
                return True
            await reply(message, format_guess_reveal(round_))
            return True

        if is_guess_ranking_command(text):
            rankings = await asyncio.to_thread(
                store.guess_ranking,
                chat_id=message.chat_id,
            )
            await reply(message, format_guess_ranking(rankings))
            return True

        if is_hunter_ranking_command(text):
            hunters = await asyncio.to_thread(store.hunter_rankings)
            await reply(message, format_hunter_ranking(hunters))
            return True

        guess_label = parse_guess_answer(text)
        if guess_label and (
            not message.mentioned_bot or getattr(channel, "guess_answers_with_mention", False)
        ):
            await settle_guess(message, guess_label)
            return True

        return False

    async def start_guess(message) -> None:
        async with guess_lock:
            round_ = await asyncio.to_thread(
                store.start_guess_round,
                chat_id=message.chat_id,
            )
        if not round_:
            await reply(message, format_guess_no_assets())
            return
        # 出题只露原图的一块局部，放大后发出来；裁剪失败（图损坏等）才回退发完整图。
        puzzle = await asyncio.to_thread(
            try_make_puzzle_image,
            round_.asset.path.read_bytes(),
        )
        prompt = format_guess_prompt(
            mentioned_answers=getattr(channel, "guess_answers_with_mention", False)
        )
        if puzzle is not None:
            await reply_image(message, puzzle, prompt)
        else:
            await reply_image(message, round_.asset.path, prompt)

    async def settle_guess(message, guess_label: str) -> None:
        # 群消息常拿不到 sender_name，为空时按 open_id 反查通讯录真名，兜底“神秘奶友”。
        display_name = message.sender_name
        if not display_name:
            display_name = await contact_resolver.resolve(message.sender_id)
        async with guess_lock:
            round_ = await asyncio.to_thread(
                store.open_guess_round,
                chat_id=message.chat_id,
            )
            result = await asyncio.to_thread(
                store.settle_guess_round,
                chat_id=message.chat_id,
                guess=guess_label,
                sender_id=message.sender_id,
                display_name=display_name,
            )
        if result is None:
            # 没有进行中的局，静默忽略，避免把群聊里正常的“奶龙/奶蛙”闲聊都怼回去。
            return
        if result.first_correct:
            # 答对后亮出完整原图，让大家看清这局到底是哪张。
            caption = format_guess_correct(result, display_name)
            if round_:
                await reply_image(message, round_.asset.path, caption)
            else:
                await reply(message, caption)
        elif result.already_settled:
            await reply(message, format_guess_already_settled())
        elif not result.correct:
            await reply(message, format_guess_wrong())

    async def continue_chain(
        message,
        observation: ObservationRecord,
        prediction: Prediction,
        *,
        include_prediction: bool = False,
    ) -> None:
        if not message.mentioned_bot and not await asyncio.to_thread(
            store.proactive_enabled, message.chat_id
        ):
            return
        async with chain_lock:
            reply_asset = await asyncio.to_thread(
                store.next_chain_asset,
                chat_id=message.chat_id,
                label=observation.asset.label,
                exclude_sha256=observation.asset.sha256,
            )
            if not reply_asset:
                LOGGER.warning("No reply asset available for label=%s", observation.asset.label)
                if message.mentioned_bot:
                    if include_prediction:
                        await reply_mention_result(message, observation, prediction)
                    else:
                        await reply(message, format_prediction(prediction))
                return

            caption = format_chain_caption(observation)
            if include_prediction:
                rarity = await asyncio.to_thread(store.asset_rarity, observation.asset.id)
                caption += "\n" + format_mention_reply(observation, prediction, rarity)
            result = await send_milk_image(
                message,
                reply_asset,
                hdr_probability=0 if include_prediction else args.random_push_hdr_probability,
                reply_to=True,
                caption=caption,
            )
            if result and result.success and result.message_id:
                await asyncio.to_thread(
                    store.attach_response,
                    observation.id,
                    result.message_id,
                )
                await asyncio.to_thread(
                    store.record_chain_reply,
                    chat_id=message.chat_id,
                    label=observation.asset.label,
                    asset_id=reply_asset.id,
                    response_message_id=result.message_id,
                )

    async def reply_mention_result(
        message,
        observation: ObservationRecord,
        prediction: Prediction,
    ):
        # 稀有度只对真正入典的奶龙/奶蛙图有意义，“其他”不算稀有度。
        rarity = None
        if observation.asset.label in {"nailong", "naiwa"}:
            rarity = await asyncio.to_thread(store.asset_rarity, observation.asset.id)
        result = await reply_card(
            message,
            build_prediction_card(
                observation,
                prediction,
                rarity,
            ),
            format_mention_reply(observation, prediction, rarity),
        )
        if result.success and result.message_id:
            await asyncio.to_thread(
                store.attach_response,
                observation.id,
                result.message_id,
            )
        return result

    async def send_pending_daily_reports(report_day: str) -> bool:
        targets = await asyncio.to_thread(
            store.pending_daily_report_targets,
            report_day,
        )
        if targets:
            LOGGER.info(
                "Daily report pending report_day=%s targets=%d",
                report_day,
                len(targets),
            )
        all_sent = True
        for target in targets:
            if not channel.scope.owns("chat", target.chat_id):
                continue
            try:
                stats = await asyncio.to_thread(
                    store.daily_stats,
                    target.chat_id,
                    day=target.report_day,
                )
                result = await channel.send(
                    target.chat_id,
                    {
                        "card": build_daily_stats_card(
                            stats,
                            title="每日含奶量播报",
                        ),
                        "fallback_text": f"每日含奶量播报\n{format_daily_stats(stats)}",
                    },
                )
                if (
                    not result.success
                    and not (
                        result.error
                        and result.error.code in {
                            ErrorCode.TARGET_REVOKED, ErrorCode.SEND_TIMEOUT
                        }
                    )
                ):
                    LOGGER.warning(
                        "Daily report card failed; falling back to plain text "
                        "chat_hash=%s error=%s",
                        store.identity_hash(target.chat_id),
                        result.error,
                    )
                    result = await channel.send(
                        target.chat_id,
                        {"text": f"每日含奶量播报\n{format_daily_stats(stats)}"},
                    )
                if not result.success:
                    all_sent = False
                    if (
                        result.error
                        and result.error.code == ErrorCode.TARGET_REVOKED
                    ):
                        # 机器人已被踢出该群，标记失效，不再反复尝试刷错误日志。
                        await asyncio.to_thread(store.deactivate_chat, target.chat_id)
                        LOGGER.warning(
                            "Deactivated revoked chat chat_hash=%s",
                            store.identity_hash(target.chat_id),
                        )
                        continue
                    LOGGER.error(
                        "Failed to send daily report chat_hash=%s error=%s",
                        store.identity_hash(target.chat_id),
                        result.error,
                    )
                    continue
                await asyncio.to_thread(
                    store.mark_daily_report_sent,
                    chat_id=target.chat_id,
                    report_day=target.report_day,
                    response_message_id=result.message_id,
                )
                LOGGER.info(
                    "Sent daily report chat_hash=%s report_day=%s",
                    store.identity_hash(target.chat_id),
                    target.report_day,
                )
            except Exception:
                all_sent = False
                LOGGER.exception(
                    "Daily report failed chat_hash=%s report_day=%s",
                    store.identity_hash(target.chat_id),
                    target.report_day,
                )
        return all_sent

    async def load_catalog_document_hashes() -> dict[str, str]:
        """返回图鉴章节里现有图片的「素材 token → 内容指纹」；没算过的素材下载一次后缓存。"""
        tokens = await catalog_document.list_section_image_tokens()
        cached = await asyncio.to_thread(
            store.catalog_document_image_hashes,
            document_id=args.catalog_document_id,
        )
        for token in tokens:
            if token in cached:
                continue
            image_bytes = await catalog_document.download_media(token)
            cached[token] = await asyncio.to_thread(try_content_hash, image_bytes)
            await asyncio.to_thread(
                store.record_catalog_document_image_hash,
                document_id=args.catalog_document_id,
                file_token=token,
                content_hash=cached[token],
            )
        return {
            token: cached[token]
            for token in tokens
            if cached.get(token) is not None
        }

    async def sync_catalog_document_once() -> tuple[int, int]:
        if not catalog_document or not args.catalog_document_id:
            return 0, 0
        synced_count = 0
        failed_count = 0
        async with catalog_sync_lock:
            records = await asyncio.to_thread(
                store.pending_catalog_document_sync,
                document_id=args.catalog_document_id,
                limit=100,
            )
            if not records:
                return 0, 0
            # 同一张图换个尺寸或编码字节 hash 就变了，追加前按内容指纹和文档里已有的图比一遍。
            # 文档指纹拿不全时宁可这轮不追加新图，等下一轮重试，避免把重复图放进文档。
            known_hashes: dict[str, str] | None
            try:
                known_hashes = await load_catalog_document_hashes()
            except Exception:
                known_hashes = None
                failed_count += 1
                LOGGER.exception("Failed to load catalog document image hashes")

            for record in records:
                image_block_id = record.image_block_id
                file_token = record.file_token
                asset_hash: str | None = None
                try:
                    if not image_block_id:
                        if known_hashes is None:
                            continue
                        asset_hash = await asyncio.to_thread(
                            try_file_content_hash,
                            record.asset.path,
                        )
                        duplicate_of = (
                            find_duplicate(asset_hash, known_hashes)
                            if asset_hash
                            else None
                        )
                        if duplicate_of:
                            await asyncio.to_thread(
                                store.mark_catalog_document_duplicate,
                                document_id=args.catalog_document_id,
                                asset_id=record.asset.id,
                                duplicate_of=duplicate_of,
                            )
                            LOGGER.info(
                                "Catalog document image skipped as duplicate "
                                "asset_id=%d duplicate_of=%s",
                                record.asset.id,
                                duplicate_of,
                            )
                            continue
                        if asset_hash:
                            # 先占位，同一批里后面内容相同的图即使这张还没传完也会被拦下。
                            known_hashes[f"asset:{record.asset.id}"] = asset_hash

                        parent_block_id = record.parent_block_id
                        if not parent_block_id:
                            parent_block_id = (
                                await catalog_document.resolve_target_column()
                            )
                            await asyncio.to_thread(
                                store.mark_catalog_document_target_assigned,
                                document_id=args.catalog_document_id,
                                asset_id=record.asset.id,
                                parent_block_id=parent_block_id,
                            )
                        image_block_id = await catalog_document.create_image_block(
                            parent_block_id=parent_block_id,
                            client_token=catalog_document.client_token(
                                record.asset.sha256,
                                "create-image-block",
                            ),
                        )
                        await asyncio.to_thread(
                            store.mark_catalog_document_block_created,
                            document_id=args.catalog_document_id,
                            asset_id=record.asset.id,
                            image_block_id=image_block_id,
                        )

                    if not file_token:
                        file_token = await catalog_document.upload_image(
                            record.asset.path,
                            image_block_id=image_block_id,
                        )
                        await asyncio.to_thread(
                            store.mark_catalog_document_media_uploaded,
                            document_id=args.catalog_document_id,
                            asset_id=record.asset.id,
                            image_block_id=image_block_id,
                            file_token=file_token,
                        )

                    await catalog_document.replace_image(
                        image_block_id=image_block_id,
                        file_token=file_token,
                        client_token=catalog_document.client_token(
                            record.asset.sha256,
                            "replace-image",
                        ),
                    )
                    await asyncio.to_thread(
                        store.mark_catalog_document_synced,
                        document_id=args.catalog_document_id,
                        asset_id=record.asset.id,
                        image_block_id=image_block_id,
                        file_token=file_token,
                    )
                    if asset_hash:
                        # 刚上传的素材本地就有原图，直接记指纹，下一轮不用再下载。
                        await asyncio.to_thread(
                            store.record_catalog_document_image_hash,
                            document_id=args.catalog_document_id,
                            file_token=file_token,
                            content_hash=asset_hash,
                        )
                    synced_count += 1
                    LOGGER.info(
                        "Catalog document image synced asset_id=%d block_id=%s",
                        record.asset.id,
                        image_block_id,
                    )
                except Exception as error:
                    failed_count += 1
                    await asyncio.to_thread(
                        store.mark_catalog_document_sync_failed,
                        document_id=args.catalog_document_id,
                        asset_id=record.asset.id,
                        error=str(error),
                    )
                    LOGGER.exception(
                        "Catalog document sync failed asset_id=%d attempt=%d",
                        record.asset.id,
                        record.attempt_count + 1,
                    )
        return synced_count, failed_count

    async def catalog_document_sync_loop() -> None:
        if not catalog_document:
            return
        catalog_sync_event.set()

        while True:
            await catalog_sync_event.wait()
            catalog_sync_event.clear()
            synced_count, failed_count = await sync_catalog_document_once()
            if synced_count:
                catalog_sync_event.set()
            if failed_count:
                await asyncio.sleep(args.catalog_sync_retry_seconds)
                catalog_sync_event.set()

    async def daily_report_loop() -> None:
        # 不用 connection_snapshot().ready 作门禁：阻塞式 connect() 下该标志恒为 False，
        # 旧写法会在这里死循环导致播报从不执行。启动先等几秒让 WS 建连，之后靠每分钟轮询，
        # 没连上时 send 会失败并记日志，下一轮自动重试。
        await asyncio.sleep(10)

        # 每分钟轮询：昨天的播报点一定已过，每轮都补发昨天没发出去的；今天过了播报点再发今天的。
        # 发没发过由数据库按 (群, 日期) 去重，重复轮询不会重发；
        # 过零点、重启、发送失败都能在下一轮补上。
        while True:
            now = datetime.now(LOCAL_TIMEZONE)
            report_at = datetime.combine(
                now.date(),
                args.daily_report_time,
                tzinfo=LOCAL_TIMEZONE,
            )
            report_days = [(now.date() - timedelta(days=1)).isoformat()]
            if now >= report_at:
                report_days.append(now.date().isoformat())
            for report_day in report_days:
                try:
                    await send_pending_daily_reports(report_day)
                except Exception:
                    LOGGER.exception("Daily report polling failed report_day=%s", report_day)
            await asyncio.sleep(DAILY_REPORT_POLL_SECONDS)

    async def maybe_douyin_video_push(message) -> bool:
        """按配置概率发送抖音奶龙/奶娃链接，并按群限制滚动窗口内的发送次数。"""
        if message.chat_type not in {"group", "topic"}:
            return False
        if not await asyncio.to_thread(store.proactive_enabled, message.chat_id):
            return False
        if not args.douyin_cookie_file:
            return False
        if args.douyin_video_push_probability <= 0:
            return False
        if secrets.SystemRandom().random() >= args.douyin_video_push_probability:
            return False
        douyin_videos = await cached_douyin_videos(args.douyin_search_keywords)
        if not douyin_videos:
            return False
        video_url = choose_douyin_video(
            douyin_videos,
            keywords=args.douyin_search_keywords,
        ).url
        reservation_id = await asyncio.to_thread(
            store.reserve_video_link_push,
            message.chat_id,
            window_seconds=args.douyin_video_push_window_seconds,
            max_pushes=args.douyin_video_push_max_per_window,
        )
        if reservation_id is None:
            return False
        result = await channel.send(message.chat_id, {"text": f"来点奶：{video_url}"})
        if result.success:
            LOGGER.info(
                "Douyin video link sent chat_hash=%s reservation_id=%d",
                store.identity_hash(message.chat_id),
                reservation_id,
            )
            return True
        await asyncio.to_thread(store.release_video_link_push, reservation_id)
        if result.error and result.error.code == ErrorCode.TARGET_REVOKED:
            await asyncio.to_thread(store.deactivate_chat, message.chat_id)
            LOGGER.warning(
                "Deactivated revoked chat chat_hash=%s",
                store.identity_hash(message.chat_id),
            )
        else:
            LOGGER.error("Douyin video link push failed error=%s", result.error)
        return False

    async def maybe_random_push(message) -> None:
        """群聊有人说话时，按概率+冷却冷不丁丢一张抽象奶图，只发图不说话。

        仅在群聊生效；被踢出的群和冷却期内跳过；发送遇到 TARGET_REVOKED 标记该群失效。
        """
        if message.chat_type not in {"group", "topic"}:
            return
        if not await asyncio.to_thread(store.proactive_enabled, message.chat_id):
            return
        if args.random_push_probability <= 0:
            return
        now = time_module.monotonic()
        last = last_random_push.get(message.chat_id)
        if last is not None and now - last < args.random_push_cooldown_seconds:
            return
        if secrets.SystemRandom().random() >= args.random_push_probability:
            return
        asset = await asyncio.to_thread(store.random_milk_asset)
        if not asset:
            return
        # 先占住冷却，避免并发消息同时触发多张。
        last_random_push[message.chat_id] = now
        result = await send_milk_image(
            message, asset, hdr_probability=args.random_push_hdr_probability, reply_to=False
        )
        if result and result.success:
            LOGGER.info(
                "Random push sent chat_hash=%s asset_id=%d",
                store.identity_hash(message.chat_id),
                asset.id,
            )
            return
        if result and result.error and result.error.code == ErrorCode.TARGET_REVOKED:
            await asyncio.to_thread(store.deactivate_chat, message.chat_id)
            LOGGER.warning(
                "Deactivated revoked chat chat_hash=%s",
                store.identity_hash(message.chat_id),
            )
        elif result:
            LOGGER.error("Random push failed error=%s", result.error)

    async def on_message(message) -> None:
        remember_milk_image_generator(message)
        if message.sender_is_bot:
            return
        LOGGER.info(
            "Received message platform=%s instance=%s chat_type=%s content_type=%s "
            "resource_types=%s quoted=%s mentioned=%s",
            channel.scope.platform,
            channel.scope.instance_id,
            message.chat_type,
            message.raw_content_type,
            [resource.type for resource in message.resources],
            bool(message.reply_to_message_id),
            message.mentioned_bot,
        )
        try:
            # 每条群消息都记录会话，纯文字群也能在下次启动时补登记所有成员。
            await asyncio.to_thread(store.record_chat, message.chat_id, message.chat_type)
            # 用户发过一条可被机器人收到的消息，就登记为奶家人。原始 open_id
            # 不落库，查询时使用同一哈希规则匹配。
            await asyncio.to_thread(
                store.record_milk_family_member,
                message.sender_id,
                display_name=message.sender_name,
            )
            # 用平台解析出的 @ 显示名补记录，不从正文昵称猜身份，不将机器人登记为群友。
            named_mentions = [
                (mention.user_id, mention.name)
                for mention in message.mentions
                if mention.user_id and mention.name and not mention.is_bot
                and channel.scope.owns("user", mention.user_id)
            ]
            if named_mentions:
                await asyncio.to_thread(store.remember_milk_family_names, named_mentions)
            if await handle_command(message):
                return
            if not has_direct_image(message) and not message.mentioned_bot:
                # 普通聊天最多触发一种主动内容，避免一条消息连发视频和奶图。
                if await maybe_douyin_video_push(message):
                    return
                # 群里有人正常聊天（没发图、没 @），按概率冷不丁丢一张抽象奶图。
                await maybe_random_push(message)
                return

            target = await resolve_image_target(channel, message)
            if target.reason == "quote_unavailable":
                if message.mentioned_bot:
                    await reply(
                        message,
                        "暂时无法读取被回复的历史消息，请直接发送图片后再试。",
                    )
                return
            if not target.file_key:
                if message.mentioned_bot:
                    if (
                        channel.scope.platform == "qq"
                        and message.chat_type == "group"
                        and not message.reply_to_message_id
                    ):
                        await reply(
                            message,
                            "想接奶：@我发“接奶”，直接领一张奶图。\n"
                            "想识图：直接发送图片并 @我，或回复图片后 @我。",
                        )
                    else:
                        await reply(message, "请发送图片，或回复一条图片消息后 @我。")
                return

            reply_error = should_reply_media_error(
                mentioned_bot=message.mentioned_bot,
                chat_type=message.chat_type,
            )
            image_bytes = await download_target(channel, target)
            if not image_bytes:
                LOGGER.warning(
                    "Failed to download resource kind=%s message_id=%s",
                    target.resource_type,
                    target.message_id,
                )
                if not reply_error:
                    return
                if target.resource_type == "sticker":
                    await reply(
                        message,
                        "这张表情暂时拿不到图像数据。请把它保存或转发成普通图片后再试。",
                    )
                else:
                    await reply(message, "图片下载失败，请重新发送。")
                return
            if len(image_bytes) > MAX_IMAGE_BYTES:
                LOGGER.warning(
                    "Skipping oversized resource bytes=%d message_id=%s",
                    len(image_bytes),
                    target.message_id,
                )
                if reply_error:
                    await reply(message, "图片超过 20 MB，暂时无法识别。")
                return
            if doubao_detector:
                doubao_prediction = await asyncio.to_thread(
                    doubao_detector.predict_bytes,
                    image_bytes,
                )
                if doubao_prediction.detected:
                    LOGGER.warning(
                        "Rejected Doubao image chat_hash=%s message_id=%s confidence=%.3f",
                        store.identity_hash(message.chat_id),
                        target.message_id or message.message_id,
                        doubao_prediction.confidence,
                    )
                    return
            prediction = await asyncio.to_thread(predictor.predict_bytes, image_bytes)
            # 首发署名：发图者昵称为空时按 open_id 反查，拿不到就留空（图鉴不显示该行）。
            discoverer_name = message.sender_name
            if not discoverer_name:
                discoverer_name = await contact_resolver.resolve(
                    message.sender_id,
                    fallback="",
                )
            observation = await asyncio.to_thread(
                store.record_observation,
                image_bytes=image_bytes,
                source_message_id=target.message_id or message.message_id,
                chat_id=message.chat_id,
                chat_type=message.chat_type,
                sender_id=message.sender_id,
                predicted_label=prediction.label,
                confidence=prediction.confidence,
                accepted=prediction.accepted,
                discoverer_name=discoverer_name or None,
            )
            services.notify_catalog_changed()
            qq_chain = (
                channel.scope.platform == "qq"
                and message.chat_type == "group"
                and message.mentioned_bot
                and observation.asset.label in {"nailong", "naiwa"}
            )
            if not observation.is_new_observation:
                LOGGER.info(
                    "Skipping duplicate message source_message_id=%s",
                    target.message_id or message.message_id,
                )
                if qq_chain:
                    # 同一原图被再次引用时，统计去重但仍接龙；WebSocket
                    # 重投的同一事件已经在 adapter 挡住，不会重复发送。
                    await continue_chain(message, observation, prediction, include_prediction=True)
                elif message.mentioned_bot:
                    await reply_mention_result(message, observation, prediction)
                return

            response_mode = media_response_mode(
                mentioned_bot=message.mentioned_bot,
                chat_type=message.chat_type,
                label=observation.asset.label,
            )
            if response_mode == "prediction":
                if qq_chain:
                    # QQ 图片可能封装在引用节点中；顶层图片和引用图片都接龙，
                    # 同一回复保留识图读数。私聊与飞书继续沿用原响应方式。
                    await continue_chain(message, observation, prediction, include_prediction=True)
                elif message.mentioned_bot:
                    await reply_mention_result(message, observation, prediction)
                else:
                    await reply(message, format_prediction(prediction))
            elif response_mode == "chain":
                await continue_chain(message, observation, prediction)
        except Exception:
            LOGGER.exception("Message handling failed")
            if not has_direct_image(message) or should_reply_media_error(
                mentioned_bot=message.mentioned_bot,
                chat_type=message.chat_type,
            ):
                await reply(message, "消息处理失败，请稍后重试。")

    async def on_error(error) -> None:
        LOGGER.error("Adapter error instance=%s: %s", channel.scope.instance_id, error)

    async def on_user_added(event: MembersAdded) -> None:
        """为入群新人登记奶家人并起一次奶名。"""
        chat_id = event.chat_id
        entries: list[tuple[str, str]] = []
        claimed_ids: list[str] = []
        sent = False
        LOGGER.info(
            "Milk name welcome event received instance=%s members=%d",
            channel.scope.instance_id,
            len(event.members),
        )
        await asyncio.to_thread(store.record_chat, chat_id, "group")
        if not await asyncio.to_thread(store.proactive_enabled, chat_id):
            LOGGER.info(
                "Milk name welcome skipped instance=%s reason=proactive_disabled",
                channel.scope.instance_id,
            )
            return
        try:
            for user in event.members:
                user_id = user.id
                if not isinstance(user_id, str) or not user_id:
                    continue
                claimed = await asyncio.to_thread(
                    store.claim_milk_name_welcome, chat_id, user_id
                )
                if not claimed:
                    continue
                claimed_ids.append(user_id)
                display_name = user.name.strip() if isinstance(user.name, str) else ""
                if not display_name:
                    # 飞书事件可能没有 name；缺通讯录权限时也欢迎新人，不静默跳过。
                    try:
                        display_name = await channel.resolve(user_id, fallback="")
                    except Exception:
                        LOGGER.warning(
                            "Milk name welcome name lookup failed instance=%s",
                            channel.scope.instance_id,
                        )
                    display_name = display_name.strip() if isinstance(display_name, str) else ""
                name_known = bool(display_name)
                display_name = display_name or "新朋友"
                assigned = await asyncio.to_thread(
                    store.assign_milk_name,
                    user_id,
                    original_name=display_name,
                    milk_name=make_milk_name(display_name),
                    original_name_known=name_known,
                    name_options=milk_name_options,
                    replace_existing=False,
                )
                entries.append((display_name, assigned.milk_name))

            if not entries:
                LOGGER.info(
                    "Milk name welcome skipped instance=%s reason=no_new_members",
                    channel.scope.instance_id,
                )
                return
            result = await channel.send(chat_id, {"text": format_milk_name_welcome(entries)})
            sent = result.success
            if not sent:
                LOGGER.error(
                    "Milk name welcome failed instance=%s members=%d error_code=%s",
                    channel.scope.instance_id,
                    len(entries),
                    getattr(result.error, "code", None),
                )
                return
            LOGGER.info(
                "Milk name welcome sent instance=%s members=%d",
                channel.scope.instance_id,
                len(entries),
            )
        except Exception:
            LOGGER.error(
                "Milk name welcome handling failed instance=%s members=%d",
                channel.scope.instance_id,
                len(entries),
            )
        finally:
            # 只释放当前处理者占用的新人，不触碰以前成功欢迎过的人。
            if not sent:
                for user_id in claimed_ids:
                    await asyncio.to_thread(store.release_milk_name_welcome, chat_id, user_id)

    async def on_card_action(event: CardAction) -> None:
        parsed = parse_correction_card_action(event.value)
        if parsed is None:
            LOGGER.info("Ignored unknown card action tag=%s", event.tag)
            return
        observation_id, label, image_key = parsed
        correction = await asyncio.to_thread(
            store.correct_card_label,
            event.message_id,
            observation_id,
            label,
        )
        if correction is None:
            LOGGER.warning(
                "Rejected stale card correction message_id=%s observation_id=%s",
                event.message_id,
                observation_id,
            )
            return
        if correction.new_label in {"nailong", "naiwa"}:
            services.notify_catalog_changed()
        observation = await asyncio.to_thread(store.observation_by_id, observation_id)
        if observation is None:
            LOGGER.error("Corrected observation disappeared id=%s", observation_id)
            return
        rarity = None
        if observation.asset.label in {"nailong", "naiwa"}:
            rarity = await asyncio.to_thread(store.asset_rarity, observation.asset.id)
        updated = build_correction_confirmation_card(
            observation,
            rarity,
            image_key=image_key,
        )
        result = await channel.update_card(event.message_id, updated)
        if result.success:
            LOGGER.info(
                "Updated correction card message_id=%s asset_id=%s label=%s",
                event.message_id,
                observation.asset.id,
                observation.asset.label,
            )
        else:
            LOGGER.error("Failed to update correction card: %s", result.error)

    async def on_reject(event) -> None:
        LOGGER.info("Message rejected reason=%s", event.reason)

    channel.on("message", on_message)
    channel.on("card_action", on_card_action)
    channel.on("error", on_error)
    channel.on("reject", on_reject)
    channel.on("members_added", on_user_added)
    await enroll_existing_group_members()
    LOGGER.info(
        "Starting adapter platform=%s instance=%s model=%s device=%s threshold=%.3f "
        "doubao_detector=%s "
        "doubao_threshold=%s state_dir=%s "
        "daily_report_time=%s catalog_document_sync=%s hdr_probability=%.2f hdr_peak_nits=%.0f "
        "random_push_probability=%.2f random_push_cooldown_seconds=%d "
        "douyin_cookie_search=%s douyin_video_push_probability=%.2f "
        "douyin_video_push_window_seconds=%d douyin_video_push_max_per_window=%d",
        channel.scope.platform,
        channel.scope.instance_id,
        getattr(args, "milk_model", None) or args.model,
        predictor.device,
        predictor.threshold,
        bool(doubao_detector),
        doubao_detector.threshold if doubao_detector else None,
        args.state_dir,
        args.daily_report_time.strftime("%H:%M"),
        bool(catalog_document),
        args.random_push_hdr_probability,
        args.hdr_peak_nits,
        args.random_push_probability,
        args.random_push_cooldown_seconds,
        bool(args.douyin_cookie_file),
        args.douyin_video_push_probability,
        args.douyin_video_push_window_seconds,
        args.douyin_video_push_max_per_window,
    )
    report_task = asyncio.create_task(daily_report_loop())
    catalog_sync_task = (
        asyncio.create_task(catalog_document_sync_loop())
        if catalog_document
        else None
    )
    if catalog_document:
        services.catalog_sync_events.append(catalog_sync_event)
    try:
        await channel.connect()
    finally:
        if catalog_document:
            services.catalog_sync_events.remove(catalog_sync_event)
        tasks = [report_task]
        if catalog_sync_task:
            tasks.append(catalog_sync_task)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("Lark").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run_bot(parse_args()))


if __name__ == "__main__":
    main()
