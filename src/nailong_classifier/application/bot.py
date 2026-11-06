from __future__ import annotations

import argparse
import asyncio
from difflib import SequenceMatcher
import io
import json
import logging
import os
import re
import secrets
import time as time_module
from collections import Counter
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image

from ..adapters.base import BotAdapter, CardAction, ErrorCode, ImageTarget, MembersAdded
from ..adapters.config import create_adapter, load_adapter_configs
from ..classification.doubao import DoubaoDetector
from ..classification.milk import combined_milk_concentration
from ..classification.model import NailongPredictor, Prediction
from ..domain.milk_fortune import (
    FORTUNE_FACES,
    FORTUNE_SETS,
    FortuneCollection,
    MilkFortune,
)
from ..domain.milk_fortune import (
    make_milk_fortune as make_milk_fortune,
)
from ..domain.milk_name_long_words import (
    MILK_NAME_CHARACTERS,
    MILK_NAME_FOREIGN,
    MILK_NAME_PREFIXES,
    MILK_NAME_TITLES,
)
from ..domain.milk_name_words import (
    MILK_NAME_ANIMALS,
    MILK_NAME_LONG_PERCENT,
    MILK_NAME_MAX_LENGTH,
    MILK_NAME_SUFFIXES,
    MILK_NAME_VARIANTS,
)
from ..domain.rarity import Rarity
from ..imaging.hdr import (
    DEFAULT_HDR_PEAK_NITS,
    DEFAULT_HDR_PROBABILITY,
    random_milk_image_content,
    validate_hdr_settings,
)
from ..imaging.image_fingerprint import find_duplicate, try_content_hash, try_file_content_hash
from ..imaging.milk_text import MILK_TEXT_USAGE, parse_milk_text_command, render_milk_text
from ..imaging.puzzle import try_make_puzzle_image
from ..integrations.douyin import (
    SEARCH_KEYWORDS,
    DouyinSearchError,
    DouyinVideo,
    choose_douyin_video,
    download_douyin_cover,
    search_douyin_videos,
)
from ..integrations.speech import MAX_SPEECH_CHARS, NailongSpeechClient, SpeechError
from ..integrations.social_media import SocialMediaError, find_social_link, iter_social_images
from ..persistence.storage import (
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
from .chain_selector import ChainSelector
from .context import MentionRef, MessageContext, ResourceRef
from .dispatch import (
    AsyncTaskDegraded,
    DeliveryGateway,
    ResponseBundle,
    ResponseCoordinator,
    SpeechDispatcher,
)
from .inference_dispatch import InferenceDispatcher, InferenceTaskDegraded
from .features import (
    AdminFeature,
    CallbackFeature,
    HelpFeature,
    InitializationFeature,
    SocialImportFeature,
    SpeechFeature,
    SpeechRequest,
    Workload,
)
from .response import AsyncTask, MessageAction, ResponsePlan
from .router import FeatureRegistry, FeatureRouter
from .services import BackendServices, SerializedPredictor
from .work_scheduler import (
    LaneConfig,
    OverloadPolicy,
    TaskContext,
    WorkOutcome,
    WorkScheduler,
)

if TYPE_CHECKING:
    from ..integrations.catalog_document import CatalogDocumentClient

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
    "这是奶龙吗？！",
    "奶蛋救我",
    "你们奶家人"
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


def is_random_milk_command(text: str) -> bool:
    return text.strip() in {"随机奶图", "来张奶图", "来张随机奶图"}


def is_hdr_milk_command(text: str) -> bool:
    return text.strip().lower() in {"hdr奶图", "hdr奶龙", "来张hdr奶图"}


def parse_speech_command(text: str) -> str | None:
    match = re.fullmatch(r"奶龙说(?:[\s：:]+(.*))?", text.strip(), flags=re.DOTALL)
    return (match.group(1) or "").strip() if match else None


def parse_naiwa_speech_command(text: str) -> str | None:
    stripped = text.strip()
    if stripped == "奶蛙大笑":
        return "哈哈哈哈哈哈哈哈！"
    match = re.fullmatch(r"奶蛙说(?:[\s：:]+(.*))?", stripped, flags=re.DOTALL)
    return (match.group(1) or "").strip() if match else None


def is_help_command(text: str) -> bool:
    return text.strip().lower() in {"help", "/help", "帮助", "/帮助", "使用帮助", "奶龙帮助"}


def is_help_request(text: str, *, mentioned_bot: bool, chat_type: str = "group") -> bool:
    return (mentioned_bot or chat_type == "p2p") and is_help_command(text)


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


def parse_custom_milk_name_command(text: str) -> str | None:
    """识别自选名字命令；保留名字内空格，单独发送命令时返回空串提示用法。"""
    text = text.strip()
    if text == "取奶名":
        return ""
    match = re.fullmatch(r"取奶名(?:\s+|[：:]\s*)(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else None


def is_milk_name_grant_for_other_command(text: str) -> bool:
    """只认“给某人求赐名”，避免普通 @ 群友时意外替对方换名。"""
    compact = "".join(text.split())
    return "给" in compact and compact.endswith("求赐名")


def milk_name_options(original_name: str) -> Iterator[str]:
    """90% 抽短奶名，10% 抽旧版定语、洋名和长梗称号。

    每次赐名先决定风格，撞名重试不切换，避免短词撞名后长名概率被抬高。
    短名的四种叫法等概率；长名保留旧版五种叫法的随机组合。
    存储层继续排除当前和历史奶名；是否换已有名由 replace_existing 决定。
    """
    if not milk_nickname_variants(original_name):
        raise ValueError("Original name must not be empty")
    long_form = secrets.randbelow(100) < MILK_NAME_LONG_PERCENT
    styles = (
        ("prefix", "foreign", "foreign_milk", "title", "title_name")
        if long_form else ("animal", "suffix", "nickname", "variant")
    )
    seen: set[str] = set()
    for _ in range(96):
        style = secrets.choice(styles)
        if long_form:
            base_name = make_milk_name(original_name)
            if style == "prefix":
                candidate = f"{secrets.choice(MILK_NAME_PREFIXES)} {base_name}"
            elif style == "foreign":
                candidate = (
                    f"{secrets.choice(MILK_NAME_FOREIGN)}·"
                    f"{secrets.choice(MILK_NAME_CHARACTERS)}·{base_name}"
                )
            elif style == "foreign_milk":
                candidate = f"{secrets.choice(MILK_NAME_FOREIGN)}·{secrets.choice(MILK_NAME_TITLES)}"
            elif style == "title":
                candidate = secrets.choice(MILK_NAME_TITLES)
            else:
                candidate = f"{secrets.choice(MILK_NAME_TITLES)}·{base_name}"
        elif style == "animal":
            candidate = secrets.choice(MILK_NAME_ANIMALS)
        elif style == "suffix":
            candidate = secrets.choice(MILK_NAME_SUFFIXES)
        elif style == "nickname":
            candidate = make_milk_name(original_name)
            # 保留首次原昵称的映射，但不把一大串昵称直接当作新奶名。
            if candidate is None or not 2 <= len(candidate) <= MILK_NAME_MAX_LENGTH:
                candidate = secrets.choice(MILK_NAME_VARIANTS)
        else:
            candidate = secrets.choice(MILK_NAME_VARIANTS)
        # 单次分配不反复查询同一个候选；数据库仍负责跨用户、跨请求的去重。
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


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


def format_milk_name_speech(milk_name: str) -> str:
    return secrets.choice((
        f"奶龙赐名！从今天起，你的奶名就是，{milk_name}！",
        f"鸡腿敲章，奶籍生效！以后就叫你，{milk_name}！",
        f"奶龙认真想了一下，决定赐你奶名，{milk_name}！",
        f"欢迎加入奶家人！你的新名字是，{milk_name}！",
    ))


def format_milk_name_welcome_speech(
    entries: list[tuple[str, str]], voice: str = "nailong",
) -> str:
    prefixes = {
        "nailong": (
            "欢迎新奶友！奶龙赐名：",
            "新朋友来啦！鸡腿敲章，今日奶名是：",
            "奶龙点名！从今天起，你们就叫：",
            "奶龙带着鸡腿来迎接新朋友啦！你们的奶名是：",
            "欢迎来到奶家人！奶龙宣布，你们的新名字是：",
            "让奶龙看看是谁来啦！今天赐名：",
            "奶龙抱紧鸡腿跑来迎新啦！今日奶名是：",
            "鸡腿还没吃完，新朋友先安排上！你们就叫：",
            "奶龙从圆滚滚的肚皮后面探出头来，宣布：",
            "奶龙听说有新人，端着饭碗就赶来了！赐名：",
            "今天不减肥，先给新奶友取名字！你们叫：",
            "奶龙把小七喊来一起见证，新奶友名叫：",
            "奶龙摇着小尾巴欢迎大家！今日奶名：",
            "一口鸡腿一口奶名，奶龙宣布你们叫：",
            "奶龙肚子咕噜一声，灵感来了！赐名：",
            "奶龙从表情包里滚出来迎新啦！你们叫：",
            "奶龙端起小饭碗，郑重宣布新奶友名叫：",
            "奶龙的鸡腿雷达发现新朋友！今日赐名：",
            "谁把新奶友带来啦？奶龙决定叫你们：",
            "奶龙拍了拍圆肚皮，新名字已经想好啦：",
            "奶龙胖嘟嘟地挤到前排，宣布你们叫：",
            "奶龙决定少吃一口，把时间留给赐名！你们叫：",
            "奶龙守住鸡腿，也守住新奶友！今日奶名是：",
        ),
        "naiwa": (
            "呱！奶蛙来欢迎新朋友啦！你们的奶名是：",
            "新奶友到啦！奶蛙蹲在荷叶上宣布：",
            "奶蛙点名！从今天起，你们就叫：",
            "欢迎加入奶家人！奶蛙为你们赐名：",
            "荷叶晃一晃，新朋友登场！今日奶名是：",
            "奶蛙已经笑着等半天啦！你们的新名字是：",
            "奶蛙捧腹大笑着来迎新啦！你们就叫：",
            "奶蛙一个蛙跳落在群里，宣布今日奶名：",
            "呱声一响，新奶友登场！奶蛙赐名：",
            "奶蛙蹲稳荷叶，认真给大家取名：",
            "奶蛙笑得肚子疼，也没忘记欢迎新人！你们叫：",
            "奶蛙开启战斗模式，只为抢先赐名：",
            "上掌下拳准备完毕，奶蛙宣布新名字：",
            "奶蛙似笑非笑地看了一眼，决定叫你们：",
            "荷叶今天有点挤，因为新奶友叫：",
            "奶蛙从池塘边飞奔而来，呱一声赐名：",
            "奶蛙把笑声收一收，郑重宣布你们叫：",
            "一声呱，两声呱，新奶友的名字是：",
            "奶蛙抱着肚子滚了一圈，灵感来了！赐名：",
            "奶蛙占好最佳荷叶位置，欢迎新朋友：",
            "奶蛙呱呱点名，今天来的新奶友叫：",
            "奶蛙蹲了半天，终于等到你们啦！赐名：",
            "奶蛙笑声已经就位，新奶友名字也定了：",
        ),
    }
    prefix = secrets.choice(prefixes[voice])
    invitation = secrets.choice((
        "！欢迎加入奶家人，鸡腿和荷叶都给你留位置啦！",
        "！快来抽奶签，看看今天是鸡腿运还是荷叶运！",
        "！有空来猜图，看看你能不能分清奶龙和奶蛙！",
        "！先领一张随机奶图，正式开启含奶生活吧！",
        "！从今天起一起吃鸡腿、蹲荷叶、捧腹大笑吧！",
    ))
    spoken: list[str] = []
    for _, milk_name in entries:
        candidate = "、".join((*spoken, milk_name))
        remaining = len(entries) - len(spoken) - 1
        suffix = f"，还有{remaining}位奶友" if remaining else ""
        if len(prefix + candidate + suffix + invitation) > MAX_SPEECH_CHARS:
            break
        spoken.append(milk_name)
    remaining = len(entries) - len(spoken)
    suffix = f"，还有{remaining}位奶友" if remaining else ""
    return f"{prefix}{'、'.join(spoken)}{suffix}{invitation}"


def is_guess_start_command(text: str) -> bool:
    return text.strip() in {"猜图", "盲盒猜图", "来一局", "猜一猜"}


def is_guess_reveal_command(text: str) -> bool:
    return text.strip() in {"公布答案", "揭晓答案", "公布"}


def is_guess_ranking_command(text: str) -> bool:
    return text.strip() in {"猜图榜", "猜图排行榜", "猜图积分榜"}


def is_hunter_ranking_command(text: str) -> bool:
    return text.strip() in {"奶龙猎人榜", "猎人榜", "奶图猎人榜"}


def is_milk_fortune_command(text: str) -> bool:
    # 两种角色的别名都抽同一份日签，不用换命令刷角色或刷分。
    return text.strip() in {"今日奶运", "奶运", "奶龙签", "奶蛙签", "抽奶签"}


def format_milk_fortune(
    fortune: MilkFortune, *, day: str, milk_name: str | None = None,
) -> str:
    greeting = f"{milk_name}，收好你的奶签。\n" if milk_name else ""
    mood = f"今日状态：{fortune.mood}\n" if fortune.mood else ""
    return (
        f"今日奶运：{fortune.title}\n"
        f"{greeting}{day} · {fortune.character}签 · {fortune.set_name}「{fortune.face}」\n\n"
        f"{fortune.headline}\n"
        f"奶气值 {fortune.score} · 接梗力 {fortune.banter} · "
        f"{fortune.special_luck_label} {fortune.special_luck}\n\n"
        f"{mood}"
        f"宜：{fortune.good}\n"
        f"忌：{fortune.avoid}\n"
        f"今日关键词：{fortune.keyword}\n"
        f"幸运物：{fortune.lucky_item}\n\n"
        f"今日小事件：{fortune.daily_event}\n"
        f"{fortune.message}\n\n"
        "已存入本群收藏册。同一天签固定，运势仅供娱乐。\n"
        "发送「奶签收藏册」看收集进度，「奶签历史」回看。"
    )


def format_fortune_collection(
    collection: FortuneCollection, *, history: bool = False, milk_name: str | None = None,
) -> str:
    title = "奶签历史" if history else "奶签收藏册"
    greeting = f"\n{milk_name}，这是你的奶签收藏。" if milk_name else ""
    if not collection.total:
        return f"{title}{greeting}\n还没有收藏的奶签。发送「今日奶运」，抽到的签会自动保存。"
    lines = [
        f"{title}{greeting}",
        f"累计抽签 {collection.total} 天 · 签面 {len(collection.faces)}/{len(FORTUNE_FACES)}",
        f"当前连签 {collection.streak} 天 · 最长连签 {collection.longest_streak} 天", "",
    ]
    if not history:
        for name, faces in FORTUNE_SETS.items():
            collected = [face for face in faces if face in collection.faces]
            missing = [face for face in faces if face not in collection.faces]
            lines.append(f"{name} {len(collected)}/{len(faces)}")
            lines.append("已收：" + ("、".join(collected) or "暂无"))
            if missing:
                lines.append("待收：" + "、".join(missing))
            lines.append("")
        badges = []
        if collection.longest_streak >= 3:
            badges.append("三日奶友")
        if collection.longest_streak >= 7:
            badges.append("一周不掉奶")
        if collection.longest_streak >= 30:
            badges.append("满月奶友")
        badges.extend(f"{name}集齐" for name in collection.completed_sets)
        lines.extend(["收藏徽章：" + ("、".join(badges) or "继续抽签，集签面或连签可解锁。"), ""])
    lines.append(f"最近奶签 · 第 {collection.page}/{collection.pages} 页")
    lines.extend(
        f"{entry.day}  {entry.fortune.face} · {entry.fortune.title} · {entry.fortune.score} 分"
        for entry in collection.entries
    )
    lines.extend(["", "发送「奶签历史 2」翻页；「奶签详情 2026-11-04」回看完整签文。"])
    return "\n".join(lines)


def parse_fortune_collection_command(text: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(奶签收藏册|收藏册|奶签历史|奶签详情)(?:\s+(.*))?", text.strip())
    return (match[1], (match[2] or "").strip()) if match else None


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


def gif_recognition_frame(image_bytes: bytes) -> bytes:
    """GIF 用中间帧识别，避免封面或透明过渡帧导致漏识别；保存时仍保留原图。"""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if getattr(image, "n_frames", 1) <= 1:
                return image_bytes
            image.seek(image.n_frames // 2)
            frame = image.convert("RGBA")
            background = Image.new("RGBA", frame.size, "white")
            background.alpha_composite(frame)
            output = io.BytesIO()
            background.convert("RGB").save(output, format="JPEG", quality=95)
            return output.getvalue()
    except (OSError, ValueError):
        return image_bytes


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
    historical: bool = False,
    milk_name: str | None = None,
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
        pill=f"每日一签 · {fortune.character}",
        title=f"{'奶签回看' if historical else '今日奶运'}：{fortune.title}",
        subtitle=f"{day} · {fortune.set_name} · {fortune.face}",
        elements=[
            *([_theme_panel([(f"{milk_name}，收好你的奶签。", "on_panel", True)])]
              if milk_name else []),
            _theme_highlights(
                [
                    ("奶气值", str(fortune.score), "accent_2", "on_accent_2"),
                    ("接梗力", str(fortune.banter), "accent", "on_accent"),
                    (fortune.special_luck_label, str(fortune.special_luck), "panel", "on_panel"),
                ]
            ),
            advice,
            _theme_panel(
                [
                    (f"今日关键词：{fortune.keyword}", "on_panel", True),
                    (fortune.headline, "on_panel", True),
                    *([(f"今日状态：{fortune.mood}", "on_panel", False)] if fortune.mood else []),
                    (f"幸运物：{fortune.lucky_item}", "on_panel", False),
                    (f"今日小事件：{fortune.daily_event}", "on_panel", False),
                    (fortune.message, "muted", False),
                    ("已存入本群收藏册 · 发送「奶签收藏册」回看 · 仅供娱乐", "muted", False),
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


HELP_SECTION_ALIASES = {
    "奶名": ("奶名", "取名", "赐名", "奶家人"),
    "奶签": ("奶签", "抽签", "奶运", "运势", "收藏"),
    "识图": ("识图", "图片", "接奶", "图鉴", "怪图", "纠正"),
    "游戏": ("游戏", "猜图", "排行榜", "排行", "积分"),
    "创作": ("创作", "奶字", "文字图", "生成器", "表情包"),
    "语音": ("语音", "说话", "奶龙说", "奶蛙说"),
    "设置": ("设置", "统计", "日报", "安静", "初始化"),
}

COMMAND_INTENTS = (
    ("取奶名 奶狗", "自选奶名", "奶名", ("取名", "改名", "自定义奶名", "换个名字")),
    ("求赐名", "随机换一个奶名", "奶名", ("赐名", "随机名字", "随机奶名", "重新取名")),
    ("今日奶运", "抽今天的奶签", "奶签", ("抽签", "今日运势", "今天运气", "奶运")),
    ("奶签收藏册", "查看奶签收藏", "奶签", ("看奶签", "收藏奶签", "签册", "奶签图鉴")),
    ("奶签历史", "查看历史奶签", "奶签", ("历史签", "以前的奶签", "奶签记录")),
    ("随机奶图", "随机领取一张奶图", "识图", ("随机图片", "来张图", "发奶图", "给张奶龙")),
    ("奶龙图鉴", "查看最近收录的奶图", "识图", ("图库", "看图鉴", "奶图列表")),
    ("标怪图", "把回复的图片加入怪图池", "识图", ("怪图", "加入怪图", "收藏怪图")),
    ("纠正为奶龙", "修正回复图片的分类", "识图", ("识别错了", "改分类", "纠正识别")),
    ("猜图", "开始一局盲盒猜图", "游戏", ("猜图片", "猜一猜", "开一局", "盲盒")),
    ("公布答案", "揭晓当前猜图答案", "游戏", ("答案", "揭晓", "看答案")),
    ("猜图榜", "查看猜图积分排行", "游戏", ("积分榜", "猜图排行", "排行榜")),
    ("奶峰生成器 文字", "制作奶系文字图", "创作", ("文字图", "做表情包", "奶字", "生成图片")),
    ("奶龙说 你好", "用奶龙声音读出台词", "语音", ("奶龙语音", "读一下", "说句话")),
    ("奶蛙说 你好", "用奶蛙声音读出台词", "语音", ("奶蛙语音", "奶蛙说话")),
    ("今日含奶量", "查看本群今日统计", "设置", ("统计", "日报", "群数据", "含奶量")),
    ("安静一点", "关闭主动推送", "设置", ("闭嘴", "少说话", "别主动发")),
    ("开始说话", "恢复主动推送", "设置", ("恢复说话", "活跃一点", "继续推送")),
    ("初始化", "登记群成员并扫描历史表情", "设置", ("初始化群聊", "扫描群聊", "登记成员")),
)


def _normalize_command_intent(text: str) -> str:
    normalized = re.sub(r"[\s，。,.！!？?：:/_#@\-]+", "", text.casefold())
    for filler in ("帮我", "给我", "我想", "请", "一下", "怎么", "如何", "命令", "指令"):
        normalized = normalized.replace(filler, "")
    return normalized


def parse_help_section(text: str) -> str | None:
    topic = re.sub(
        r"^(?:/?help|/?帮助|使用帮助|奶龙帮助)[\s：:]*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    normalized = _normalize_command_intent(topic)
    if not normalized:
        return None
    for section, aliases in HELP_SECTION_ALIASES.items():
        if any(
            normalized == _normalize_command_intent(alias)
            or _normalize_command_intent(alias) in normalized
            for alias in aliases
        ):
            return section
    return None


def suggest_command(text: str) -> tuple[str, str, str] | None:
    normalized = _normalize_command_intent(text)
    if len(normalized) < 2 or len(normalized) > 40 or "http" in text.casefold():
        return None
    best: tuple[float, str, str, str] | None = None
    for command, description, section, aliases in COMMAND_INTENTS:
        for candidate in (command, *aliases):
            target = _normalize_command_intent(candidate)
            score = (
                1.0
                if normalized == target
                else 0.9
                if min(len(normalized), len(target)) >= 2
                and (normalized in target or target in normalized)
                else SequenceMatcher(None, normalized, target).ratio()
            )
            if best is None or score > best[0]:
                best = (score, command, description, section)
    if best is None or best[0] < 0.58:
        return None
    return best[1], best[2], best[3]


def format_help_message(
    report_time: time,
    *,
    platform: str = "feishu",
    allow_proactive: bool = False,
    speech_enabled: bool = False,
    section: str | None = None,
) -> str:
    """默认展示分类目录，指定分类时只展示对应命令。"""
    qq = platform == "qq"
    intro = (
        "奶龙玩法\nQQ群里先 @我，私聊直接发。"
        if qq else "奶龙玩法\n群聊先 @我，私聊直接发。"
    )
    image_note = (
        "发图或回复图片并 @我：识图、接奶，显示奶含量和稀有度。"
        if qq else "群里发图、表情会自动接奶；@我识图会显示奶含量和稀有度。"
    )
    family_note = (
        "接奶 / 奶家人：领一张随机奶图。"
        if qq else "@我和群友：查看奶名；给 @群友 求赐名：替对方随机换名。"
    )
    report_note = (
        (
            f"每天 {report_time.strftime('%H:%M')} 主动播报。"
            if allow_proactive else "当前只支持手动查询。"
        )
        if qq else f"每天 {report_time.strftime('%H:%M')} 自动播报。"
    )
    generator_note = "QQ 返回静态 PNG。" if qq else "飞书支持动态 GIF。"
    video_note = "需通过 QQ 审核。" if qq else "搜索可用时返回链接。"
    speech_note = (
        "奶龙说 你好 / 奶蛙说 你好：用对应声音读出台词，最多 200 字。\n"
        "奶蛙大笑：直接播放奶蛙笑声。"
        if speech_enabled else "当前未启用语音功能。"
    )
    sections = {
        "奶名": (
            "取奶名 奶狗：自选奶名，每 7 天成功一次。\n"
            "求赐名：随机换奶名，不限次数。\n"
            f"{family_note}"
        ),
        "奶签": (
            "今日奶运：抽今日签，当天固定、零点更新。\n"
            f"奶签收藏册：查看 {len(FORTUNE_FACES)} 张签面和徽章。\n"
            "奶签历史 2：翻页；奶签详情 2026-11-04：回看指定日期。"
        ),
        "识图": (
            f"{image_note}\n"
            "随机奶图：随机领一张奶龙或奶蛙。\n"
            "标怪图 / 取消怪图：回复图片后调整怪图池。\n"
            "纠正为奶龙 / 奶蛙 / 其他：回复图片后修正分类。\n"
            "奶龙图鉴：查看最近收录；HDR奶图：发送静态高亮奶图。\n"
            f"#奶龙 / #奶蛙 / #话题：查找抖音视频，{video_note}"
        ),
        "游戏": (
            f"猜图：开始盲盒，{'@我后发' if qq else '直接发'}奶龙 / 奶蛙 / 其他抢答。\n"
            "公布答案：揭晓；猜图榜：积分排行；奶龙猎人榜：首发收图排行。"
        ),
        "创作": (
            "奶峰生成器 鸡腿拿来！：制作奶龙文字图。\n"
            "奶蛙字 笑得肚子疼：制作奶蛙文字图。\n"
            f"最多 60 字、四行；发送「奶峰生成器」查看参数。{generator_note}"
        ),
        "语音": speech_note,
        "设置": (
            f"今日含奶量：查看本群统计。{report_note}\n"
            "安静一点 / 开始说话：关闭或恢复主动推送。\n"
            "初始化：登记群成员并扫描历史表情包。"
        ),
    }
    if section in sections:
        return f"{intro}\n\n{section}\n{sections[section]}\n\n发送「help」返回分类目录。"
    categories = " / ".join(HELP_SECTION_ALIASES)
    return (
        f"{intro}\n\n"
        f"选择分类\n{categories}\n\n"
        "发送「help 奶名」「help 识图」这类命令查看详情。"
    )


def build_help_card(text: str) -> dict[str, Any]:
    """帮助使用飞书原生分组卡片；每组保留常用命令，详细参数走独立命令。"""
    elements = []
    for index, paragraph in enumerate(text.split("\n\n")):
        lines = paragraph.splitlines()
        if index and lines:
            lines[0] = f"**{lines[0]}**"
        content = "\n".join(lines)
        content = re.sub(r"^([^：\n]{1,32})：", r"**\1**：", content, flags=re.MULTILINE)
        elements.append({"tag": "markdown", "content": content})
    return {
        "schema": "2.0",
        "config": {"enable_forward": True, "summary": {"content": "奶龙玩法"}},
        "header": {
            "title": {"tag": "plain_text", "content": "奶龙玩法"},
            "template": "yellow",
        },
        "body": {"vertical_spacing": "12px", "elements": elements},
    }


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


def parse_chain_weights(value: str) -> tuple[int, ...]:
    try:
        weights = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("chain weights must be integers: similar,weird,random") from exc
    if len(weights) != 3 or any(weight < 0 for weight in weights) or not any(weights):
        raise argparse.ArgumentTypeError("chain weights require three nonnegative values with a positive total")
    return weights


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
        "--chain-weights", type=parse_chain_weights,
        default=os.getenv("BOT_CHAIN_WEIGHTS", "20,40,30"),
        help="Relative weights for similar,weird,random image chaining.",
    )
    parser.add_argument(
        "--feishu-emoji-ids-file",
        type=Path,
        default=os.getenv("NAILONG_FEISHU_EMOJI_IDS_FILE") or None,
        help="JSON mapping of emoji names to Feishu enterprise emoji IDs.",
    )
    parser.add_argument(
        "--chain-top-k", type=int, default=int(os.getenv("BOT_CHAIN_TOP_K", "5")),
    )
    parser.add_argument(
        "--chain-min-similarity", type=float,
        default=float(os.getenv("BOT_CHAIN_MIN_SIMILARITY", "0.55")),
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
        "--xiaohongshu-cookie-file", type=Path,
        default=Path(os.environ["XIAOHONGSHU_COOKIE_FILE"])
        if os.getenv("XIAOHONGSHU_COOKIE_FILE") else None,
        help="Private browser-exported Xiaohongshu cookie JSON for shared note parsing.",
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
    if args.chain_top_k < 1 or not 0 <= args.chain_min_similarity <= 1:
        parser.error("chain top-k must be positive; similarity must be between 0 and 1")
    return args


async def run_bot(args: argparse.Namespace) -> None:
    configs = load_adapter_configs(getattr(args, "bots_config", None))
    adapters: list[BotAdapter] = []
    tasks: list[asyncio.Task] = []
    index_task: asyncio.Task | None = None
    speech_dispatcher: SpeechDispatcher | None = None
    inference_dispatcher: InferenceDispatcher | None = None
    work_scheduler: WorkScheduler | None = None
    try:
        # 先校验所有接入凭证，再加载一份共用模型，避免每个机器人重复占内存。
        # SDK 导入时捕获模块级事件循环；即使只有一个飞书实例，在运行中的
        # 主循环里才导入 SDK 也会捕获错误的 loop。每个飞书 SDK 固定放进子进程，
        # 由 worker 在 asyncio.run 前导入，模型和业务仍只在主进程加载一次。
        isolate_feishu = True
        for config in configs:
            adapters.append(create_adapter(config, isolate_feishu=isolate_feishu))
        if getattr(args, "milk_model", None):
            from ..classification.milk_runtime import MilkPriorityPredictor

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
        store = BotStore(args.state_dir)
        inference_dispatcher = InferenceDispatcher(predictor, doubao_detector)
        inference_dispatcher.start()
        work_scheduler = WorkScheduler({
            "background": LaneConfig(
                concurrency=2,
                queue_capacity=8,
                timeout=120,
                overload_policy=OverloadPolicy.DEGRADE_TEXT,
                circuit_failure_threshold=3,
                circuit_reset_timeout=60,
            ),
            "social": LaneConfig(
                concurrency=1,
                queue_capacity=2,
                timeout=180,
                overload_policy=OverloadPolicy.DEGRADE_TEXT,
                circuit_failure_threshold=3,
                circuit_reset_timeout=60,
            ),
            "initialization": LaneConfig(
                concurrency=1,
                queue_capacity=1,
                timeout=300,
                overload_policy=OverloadPolicy.DEGRADE_TEXT,
                circuit_failure_threshold=3,
                circuit_reset_timeout=60,
            ),
            "proactive": LaneConfig(
                concurrency=1,
                queue_capacity=1,
                timeout=60,
                overload_policy=OverloadPolicy.DROP,
                circuit_failure_threshold=3,
                circuit_reset_timeout=60,
            ),
        })
        await work_scheduler.start()
        chain_selector = ChainSelector(
            store, weights=getattr(args, "chain_weights", (20, 40, 30)),
            top_k=getattr(args, "chain_top_k", 5),
            min_similarity=getattr(args, "chain_min_similarity", 0.55),
        )
        speech_client = NailongSpeechClient.from_environment()
        speech_dispatcher = SpeechDispatcher(speech_client) if speech_client else None
        if speech_dispatcher:
            speech_dispatcher.start()
        services = BackendServices(
            predictor, store, doubao_detector,
            inference_dispatcher=inference_dispatcher,
            work_scheduler=work_scheduler,
            speech_client=speech_client,
            speech_dispatcher=speech_dispatcher,
            chain_selector=chain_selector,
        )
        # 独立索引任务不参加接入实例的 gather；图库没准备好也能正常启动和随机接龙。
        index_task = asyncio.create_task(chain_selector.build_index(), name="chain-index")
        for config, adapter in zip(configs, adapters, strict=True):
            runtime_args = argparse.Namespace(**vars(args))
            if getattr(args, "bots_config", None):
                # 多实例配置中的文档同步由一个明确的飞书实例负责；
                # 不能把旧全局文档配置无意间复制给所有 QQ/飞书实例。
                runtime_args.catalog_document_id = config.catalog_document_id
                runtime_args.catalog_section_block_id = config.catalog_section_block_id
                # 企业表情 ID 只在对应租户可用，多实例时按机器人单独配置。
                runtime_args.feishu_emoji_ids_file = config.settings.get("emoji_ids_file")
            tasks.append(asyncio.create_task(
                run_adapter(runtime_args, adapter, services),
                name=f"bot:{config.instance_id}",
            ))
        await asyncio.gather(*tasks)
    finally:
        if index_task:
            index_task.cancel()
            await asyncio.gather(index_task, return_exceptions=True)
        for task in tasks:
            task.cancel()
        # 一个实例异常退出时也必须收尾其余实例；重抛失败任务会跳过 close。
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if work_scheduler:
            await work_scheduler.close()
        if speech_dispatcher:
            await speech_dispatcher.close()
        if inference_dispatcher:
            await inference_dispatcher.close()
        for adapter in adapters:
            try:
                await adapter.close()
            except Exception:
                LOGGER.exception("Adapter shutdown failed instance=%s", adapter.scope.instance_id)


def load_feishu_emoji_ids(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    try:
        entries = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(entries, dict) or not entries or any(
            not name.strip() or not isinstance(emoji_id, str)
            or not re.fullmatch(r"[0-9]+", emoji_id)
            for name, emoji_id in entries.items()
        ):
            raise ValueError("Expected a JSON object of names and numeric ID strings")
        return {name.strip(): emoji_id for name, emoji_id in entries.items()}
    except (OSError, ValueError):
        LOGGER.warning("Could not load Feishu enterprise emoji mapping")
        return {}


async def run_adapter(
    args: argparse.Namespace, channel: BotAdapter, services: BackendServices
) -> None:
    predictor = services.predictor
    doubao_detector = services.doubao_detector
    inference_dispatcher = services.inference_dispatcher
    if inference_dispatcher is None:
        raise RuntimeError("Inference dispatcher is not configured")
    store = services.store
    delivery = DeliveryGateway(channel)
    speech_dispatcher = services.speech_dispatcher
    responses = ResponseCoordinator(delivery, speech_dispatcher)
    chain_lock = asyncio.Lock()
    guess_lock = asyncio.Lock()
    contact_resolver = channel
    feishu_emoji_ids = (
        load_feishu_emoji_ids(getattr(args, "feishu_emoji_ids_file", None))
        if channel.scope.platform == "feishu" else {}
    )
    feishu_emoji_lookup = {name.casefold(): emoji_id for name, emoji_id in feishu_emoji_ids.items()}
    # 每个群上次主动推送的单调时钟时间，用于冷却；进程内存着即可，重启清零无妨。
    last_random_push: dict[str, float] = {}
    # 第三方生图机器人不属于奶家人，也不写入数据库；仅在当前进程记住其群内身份。
    milk_image_generator_ids: dict[str, str] = {}
    douyin_search_lock = asyncio.Lock()
    social_import_lock = asyncio.Lock()
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

    async def reply(message, text: str, *, in_thread: bool = False):
        options = {"reply_to": message.message_id}
        if in_thread:
            options["reply_in_thread"] = True
        result = await delivery.send(
            message.chat_id,
            {"text": text},
            options,
        )
        if not result.success:
            LOGGER.error("Failed to reply: %s", result.error)
        return result

    async def reply_card(
        message, card: dict[str, Any], fallback_text: str, *, in_thread: bool = False,
    ):
        options = {"reply_to": message.message_id}
        if in_thread:
            options["reply_in_thread"] = True
        result = await delivery.send(
            message.chat_id,
            {"card": card, "fallback_text": fallback_text},
            options,
        )
        if result.success:
            return result
        LOGGER.error("Failed to reply with card: %s", result.error)
        # QQ 在适配层完成确定拒绝时的样式降级，不再重复发送或猜测未知错误。
        if channel.scope.platform == "qq":
            return result
        if result.error and result.error.code in {ErrorCode.TARGET_REVOKED, ErrorCode.SEND_TIMEOUT}:
            return result
        LOGGER.info("Falling back to plain-text reply after card send failure")
        return await reply(message, fallback_text, in_thread=in_thread)

    async def fortune_milk_name(message) -> str:
        """直接喊已登记的奶名；首次抽签分配一次，普通抽签不触发换名或额外发图。"""
        assigned = await asyncio.to_thread(store.milk_name_for_user, message.sender_id)
        if assigned is not None:
            return assigned.milk_name
        member = await asyncio.to_thread(store.milk_family_member, message.sender_id)
        display_name = (message.sender_name or (
            member.display_name if member else None
        ) or "").strip()
        name_known = bool(display_name and display_name not in LEGACY_NAME_FALLBACKS)
        original_name = display_name if name_known else "神秘奶友"
        assigned = await asyncio.to_thread(
            store.assign_milk_name, message.sender_id,
            original_name=original_name, milk_name=make_milk_name(original_name),
            name_options=milk_name_options, original_name_known=name_known,
            replace_existing=False,
        )
        return assigned.milk_name

    async def reply_image(message, source: bytes | Path, caption: str):
        image_source = source if isinstance(source, bytes) else str(source)
        result = await delivery.send(
            message.chat_id,
            {"image": {"source": image_source}, "caption": caption},
            {"reply_to": message.message_id},
        )
        if not result.success:
            LOGGER.error("Failed to reply with image: %s", result.error)
        return result

    async def send_milk_name_speech(
        chat_id: str, text: str, *, reply_to: str | None = None,
        voice: str = "nailong",
    ) -> None:
        """赐名语音是文字结果的互动补充，失败时不影响赐名和新人欢迎。"""
        if channel.scope.platform != "feishu" or speech_dispatcher is None:
            return
        try:
            audio = await speech_dispatcher.generate(
                message_id=reply_to or f"proactive:{secrets.token_hex(8)}",
                chat_id=chat_id,
                user_id="",
                voice=voice,
                text=text,
            )
            options = {"reply_to": reply_to} if reply_to else None
            result = await delivery.send(
                chat_id,
                {"audio": {
                    "source": audio.opus,
                    "duration_ms": audio.duration_ms,
                }},
                options,
            )
            if not result.success:
                LOGGER.warning(
                    "Milk name speech send failed instance=%s",
                    channel.scope.instance_id,
                )
        except AsyncTaskDegraded:
            LOGGER.info(
                "Milk name speech degraded instance=%s",
                channel.scope.instance_id,
            )
        except Exception:
            LOGGER.exception(
                "Milk name speech generation failed instance=%s",
                channel.scope.instance_id,
            )

    async def reply_with_speech(
        message,
        text: str,
        speech_text: str,
        *,
        voice: str = "nailong",
    ):
        if channel.scope.platform != "feishu" or speech_dispatcher is None:
            return await reply(message, text)
        text_result, audio_result = await responses.send_text_with_optional_speech(
            ResponseBundle(
                message_id=message.message_id,
                chat_id=message.chat_id,
                user_id=message.sender_id,
                text_content={"text": text},
                speech_text=speech_text,
                voice=voice,
                audio_source="opus",
                options={"reply_to": message.message_id},
            )
        )
        if audio_result is not None and not audio_result.success:
            LOGGER.warning(
                "Paired speech send failed instance=%s",
                channel.scope.instance_id,
            )
        return text_result

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
        result = await delivery.send(
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
        """接龙、主动推送、随机奶图和手动 HDR 奶图共用发送入口。"""
        if doubao_detector:
            try:
                doubao_prediction = await inference_dispatcher.detect_doubao_path(
                    asset.path,
                    message_id=message.message_id,
                    chat_id=message.chat_id,
                    user_id=message.sender_id,
                )
            except InferenceTaskDegraded:
                return None
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
        is_derived = isinstance(content["image"]["source"], bytes)
        if is_derived and channel.scope.platform == "feishu":
            content["image"]["presentation"] = "card"
        if caption and not is_derived:
            content["caption"] = caption
        options = {"reply_to": message.message_id} if reply_to else None
        result = await delivery.send(message.chat_id, content, options)
        # 衍生图上传失败时只回退一次原图；群已撤销或发送结果不确定时不重复发。
        if (
            not result.success
            and is_derived
            and result.error
            and result.error.code == ErrorCode.UPLOAD_FAILED
        ):
LOGGER.warning(
                "Derived image upload failed, falling back to original asset_id=%d",
                asset.id,
            )
            result = await delivery.send(
                message.chat_id, {"image": {"source": str(asset.path)}}, options
            )
            is_derived = False
        if result.success:
            if result.message_id:
                await asyncio.to_thread(
                    store.record_chain_reply,
                    chat_id=message.chat_id,
                    label=asset.label,
                    asset_id=asset.id,
                    response_message_id=result.message_id,
                )
            LOGGER.info(
                "Milk image sent chat_hash=%s asset_id=%d flash_gif=%s hdr_jpeg=%s message_id=%s",
                store.identity_hash(message.chat_id),
                asset.id,
                is_derived and effect == "flash",
                is_derived and effect == "hdr",
                result.message_id,
            )
        elif result.error and result.error.code == ErrorCode.TARGET_REVOKED:
            await asyncio.to_thread(store.deactivate_chat, message.chat_id)
        else:
            LOGGER.error(
                "Milk image send failed asset_id=%d effect=%s error=%s",
                asset.id,
                effect if is_derived else "original",
                result.error,
            )
        return result

    async def filter_douyin_covers(videos: list[DouyinVideo]) -> list[DouyinVideo]:
        verified = []
        for index, video in enumerate(videos):
            if not video.cover_url:
                continue
            try:
                cover = await asyncio.to_thread(download_douyin_cover, video.cover_url)
                prediction = await inference_dispatcher.predict_bytes(
                    cover,
                    message_id=f"douyin-cover:{index}:{secrets.token_hex(4)}",
                    chat_id="",
                    user_id="",
                )
            except (DouyinSearchError, InferenceTaskDegraded, OSError, ValueError):
                continue
            if prediction.accepted and prediction.label in {"nailong", "naiwa"}:
                verified.append(video)
        return verified

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
                    videos = await filter_douyin_covers(videos)
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

    async def import_social_link(message, url: str, *, announce: bool = True) -> None:
        """用户明确分享链接才解析；逐图识别、收录，文档仍走原有去重与重试流程。"""
        async with social_import_lock:
            if announce:
                await reply(message, "正在读取图片或抽取视频画面并识别奶蛙、奶龙，稍等一下。")
            images = iter_social_images(
                url, douyin_cookie_file=args.douyin_cookie_file,
                xiaohongshu_cookie_file=getattr(args, "xiaohongshu_cookie_file", None),
                share_text=message.body_text or message.content_text,
            )
            counts: Counter = Counter()
            known_hashes: dict[str, str] = {}
            error_text = ""
            try:
                while (image := await asyncio.to_thread(next, images, None)) is not None:
                    counts.update(["processed"])
                    if not image.content:
                        counts.update(["failed"])
                        continue
                    fingerprint = await asyncio.to_thread(try_content_hash, image.content)
                    if fingerprint and find_duplicate(fingerprint, known_hashes):
                        counts.update(["duplicate"])
                        continue
                    if fingerprint:
                        known_hashes[image.source_key] = fingerprint
                    try:
                        if doubao_detector:
                            guard = await inference_dispatcher.detect_doubao_bytes(
                                image.content,
                                message_id=f"{message.message_id}:{image.source_key}:guard",
                                chat_id=message.chat_id,
                                user_id=message.sender_id,
                            )
                            if guard.detected:
                                counts.update(["filtered"])
                                continue
                        prediction = await inference_dispatcher.predict_bytes(
                            image.content,
                            message_id=f"{message.message_id}:{image.source_key}",
                            chat_id=message.chat_id,
                            user_id=message.sender_id,
                        )
                        if not prediction.accepted or prediction.label not in {"nailong", "naiwa"}:
                            counts.update(["filtered"])
                            continue
                        observation = await asyncio.to_thread(
                            store.record_observation,
                            image_bytes=image.content,
                            source_message_id=f"{message.message_id}:social:{image.source_key}",
                            chat_id=message.chat_id, chat_type=message.chat_type,
                            sender_id=message.sender_id, discoverer_name=message.sender_name,
                            predicted_label=prediction.label, confidence=prediction.confidence,
                            accepted=True, promote_seed=True,
                        )
                        if observation.asset.label not in {"nailong", "naiwa"}:
                            counts.update(["filtered"])
                            continue
                        counts.update([observation.asset.label])
                        if observation.is_new_asset:
                            counts.update(["new"])
                        services.notify_catalog_changed()
                    except Exception:
                        LOGGER.exception("Social image recognition failed kind=%s", image.kind)
                        counts.update(["failed"])
            except SocialMediaError as exc:
                error_text = str(exc)
            finally:
                await asyncio.to_thread(images.close)
            if error_text and not counts["processed"]:
                await reply(message, error_text)
                return
            lines = [
                f"链接识别：处理 {counts['processed']} 张图片或视频帧，"
                f"奶蛙 {counts['naiwa']} 张，奶龙 {counts['nailong']} 张，"
                f"新收录 {counts['new']} 张。",
            ]
            if counts["duplicate"]:
                lines.append(f"跳过 {counts['duplicate']} 张重复或近重复图。")
            if counts["filtered"]:
                lines.append(f"{counts['filtered']} 张未通过奶图识别或豆包过滤，不收录。")
            if counts["failed"]:
                lines.append(f"{counts['failed']} 张下载或识别失败，其他图片已继续处理。")
            if counts["naiwa"] or counts["nailong"]:
                lines.append(
                    "已进入图鉴文档同步队列，重复图片不会再插入。"
                    if services.catalog_sync_events else "已存入图库；当前未配置图鉴文档同步。"
                )
                if args.catalog_document_id:
                    lines.append(f"https://bytedance.feishu.cn/docx/{args.catalog_document_id}")
            if error_text:
                lines.append(error_text)
            await reply(message, "\n".join(lines))

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
        result = await delivery.send(
            message.chat_id,
            {"gallery": gallery, "caption": format_catalog_caption(stats, len(assets))},
            {"reply_to": message.message_id},
        )
        if not result.success:
            LOGGER.error("Failed to reply with catalog: %s", result.error)

    async def handle_command_capability(message, *, capability: str) -> bool:
        if message.raw_content_type not in {"text", "post"}:
            return False
        text = message.body_text or message.content_text
        mentioned_users = mentioned_user_ids(message, body_text=text)
        custom_milk_name = parse_custom_milk_name_command(text)

        if (
            capability == "enterprise_emoji"
            and feishu_emoji_ids
            and (message.mentioned_bot or message.chat_type == "p2p")
        ):
            emoji_command = re.fullmatch(r"(?:奶龙|奶蛙)表情[ \t:：]*(.*)", text.strip())
            if emoji_command:
                name = emoji_command[1].strip().removeprefix("奶龙").strip()
                emoji_id = feishu_emoji_lookup.get(name.casefold())
                if emoji_id is None:
                    await reply(message, "可以点播这些小表情：" + "、".join(feishu_emoji_ids))
                    return True
                # 飞书富文本需要企业表情的原始 ID 字符串，避免长整数精度丢失。
                emotion = {"tag": "emotion", "emoji_type": emoji_id}
                result = await delivery.send(
                    message.chat_id,
                    {"post": {"zh_cn": {"content": [[emotion]]}}},
                    {"reply_to": message.message_id},
                )
                if not result.success:
                    LOGGER.warning("Feishu enterprise emoji reply failed: %s", result.error)
                return True

        # 明确要随机图时优先响应命令，不因引用图片或 @群友转去识图、点名。
        # 复用图库两类等概率抽图和已有发送护栏，只返回普通图片。
        if capability == "random_milk_image" and is_random_milk_command(text):
            asset = await asyncio.to_thread(store.random_milk_asset)
            if asset is None:
                await reply(message, "图库里还没有奶图，先发张奶龙或奶蛙图片给我收录吧。")
            else:
                result = await send_milk_image(
                    message, asset, hdr_probability=0, reply_to=True,
                )
                if result is None:
                    await reply(message, "这张奶图暂时不能发送，再试一次吧。")
                elif not result.success and (
                    result.error is None
                    or result.error.code not in {ErrorCode.TARGET_REVOKED, ErrorCode.SEND_TIMEOUT}
                ):
                    await reply(message, "奶图没发出去，稍后再试试。")
            return True

        if capability == "milk_text":
            try:
                milk_text = parse_milk_text_command(text)
            except ValueError as error:
                if message.mentioned_bot or message.chat_type == "p2p":
                    await reply(message, str(error))
                    return True
                return False
            if (
                milk_text is not None
                and (message.mentioned_bot or message.chat_type == "p2p")
            ):
                if not milk_text.text:
                    await reply(message, MILK_TEXT_USAGE)
                else:
                    try:
                        if channel.scope.platform == "qq":
                            # QQ 适配器会移除透明通道，先合成奶油底，避免透明区域变成黑底。
                            # QQ 当前富媒体发送只保留 PNG 首帧；仍保留随机倾斜和错位的字形。
                            milk_text = replace(milk_text, background="cream", motion="static")
                        image = await asyncio.to_thread(render_milk_text, milk_text)
                        result = await reply_image(message, image, "")
                        if not result.success:
                            await reply(message, "奶字图片没发出去，稍后再试试。")
                    except ValueError as error:
                        await reply(message, str(error))
                    except Exception:
                        LOGGER.exception("Milk text render failed")
                        await reply(message, "奶字制作失败，稍后再试试。")
                return True

        if (
            capability == "milk_name_for_other"
            and message.mentioned_bot
            and mentioned_users
            and custom_milk_name is None
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
            await reply_with_speech(
                message,
                format_milk_name_for_other(
                    display_name, existing.milk_name, existing=already_named
                ),
                format_milk_name_speech(existing.milk_name),
            )
            await request_milk_name_image(message, existing.milk_name)
            return True

        if (
            capability == "milk_name"
            and (is_milk_name_command(text) or custom_milk_name is not None)
            and (message.mentioned_bot or message.chat_type != "group")
        ):
            # 指令只给发送者本人起名，优先于群友点名和引用图片逻辑。
            # 已有真实原昵称不重复查询；系统兜底留下的空缺允许后续补全。
            if custom_milk_name == "":
                await reply(message, "发送「取奶名 奶狗」，把奶狗换成你想要的名字就行。")
                return True
            if not message.sender_id:
                command = "取奶名 你想要的名字" if custom_milk_name is not None else "求赐名"
                await reply(message, f"暂时拿不到你的身份，请重新发送「{command}」试试。")
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
            try:
                assigned = await asyncio.to_thread(
                    store.assign_milk_name,
                    message.sender_id,
                    original_name=display_name,
                    milk_name=(
                        custom_milk_name
                        if custom_milk_name is not None else make_milk_name(display_name)
                    ),
                    name_options=None if custom_milk_name is not None else milk_name_options,
                    original_name_known=name_known,
                    exact_name=custom_milk_name is not None,
                    weekly_limit=custom_milk_name is not None,
                )
            except ValueError as error:
                if custom_milk_name is None:
                    raise
                await reply(message, str(error))
                return True
            if custom_milk_name is not None:
                await reply(
                    message,
                    f"你的奶名是「{assigned.milk_name}」，已经记住啦。\n"
                    "以后抽奶签就喊这个名字。「取奶名」每 7 天成功一次，"
                    "「求赐名」可无限次随机换名。",
                )
            else:
                await reply_with_speech(
                    message,
                    format_milk_name_grant(
                        assigned.original_name, assigned.milk_name,
                        original_name_known=assigned.original_name_known,
                    ),
                    format_milk_name_speech(assigned.milk_name),
                )
            return True

        collection_command = (
            parse_fortune_collection_command(text)
            if capability == "milk_fortune" else None
        )
        if capability == "milk_fortune" and (
            is_milk_fortune_command(text) or collection_command
        ):
            if not message.sender_id:
                await reply(message, "暂时拿不到你的身份，请重新发送奶签命令。")
                return True
            day = datetime.now(LOCAL_TIMEZONE).date().isoformat()
            if is_milk_fortune_command(text):
                fortune = await asyncio.to_thread(
                    store.draw_milk_fortune,
                    chat_id=message.chat_id, sender_id=message.sender_id, day=day,
                )
                milk_name = await fortune_milk_name(message)
                await reply_card(
                    message, build_milk_fortune_card(fortune, day=day, milk_name=milk_name),
                    format_milk_fortune(fortune, day=day, milk_name=milk_name),
                )
                return True
            command, argument = collection_command
            if command == "奶签详情":
                try:
                    requested_day = datetime.strptime(argument, "%Y-%m-%d").date().isoformat()
                except ValueError:
                    await reply(message, "请发送「奶签详情 2026-11-04」，日期格式为 年-月-日。")
                    return True
                fortune = await asyncio.to_thread(
                    store.milk_fortune_on,
                    chat_id=message.chat_id, sender_id=message.sender_id, day=requested_day,
                )
                if fortune is None:
                    await reply(message, "那天没有收藏奶签。发送「奶签历史」查看已收藏的日期。")
                else:
                    milk_name = await fortune_milk_name(message)
                    await reply_card(
                        message,
                        build_milk_fortune_card(
                            fortune, day=requested_day, historical=True, milk_name=milk_name,
                        ),
                        format_milk_fortune(
                            fortune, day=requested_day, milk_name=milk_name,
                        ).replace(
                            "今日奶运：", "奶签回看：", 1
                        ),
                    )
                return True
            if argument and (not re.fullmatch(r"[0-9]{1,5}", argument)
                             or not 1 <= int(argument) <= 10000):
                await reply(message, "页码请填 1 到 10000，例如「奶签历史 2」。")
                return True
            collection = await asyncio.to_thread(
                store.milk_fortune_collection,
                chat_id=message.chat_id, sender_id=message.sender_id, today=day,
                page=int(argument or "1"),
            )
            if collection.page > collection.pages:
                await reply(message, f"目前共 {collection.pages} 页，请发送「奶签历史 1」回看。")
                return True
            milk_name = await fortune_milk_name(message)
            summary = format_fortune_collection(
                collection, history=command == "奶签历史", milk_name=milk_name,
            )
            card = _handdrawn_card(
                summary=summary.splitlines()[0], pill="奶签收藏",
                title=summary.splitlines()[0],
                subtitle="只展示你在当前聊天中收藏的奶签",
                # 每套签面共用一个面板；百级收藏按段落展示，避免每一行都占一张容器。
                elements=[
                    _theme_panel([(line, "on_panel", False) for line in paragraph.splitlines()])
                    for paragraph in "\n".join(summary.splitlines()[1:]).split("\n\n")
                    if paragraph
                ],
            )
            await reply_card(message, card, summary)
            return True

        # QQ 用发送者的可靠身份直接领图，不依赖群成员 @ 或昵称解析。
        # on_message 已登记发送者；图片仍优先走识图，飞书保留原来的群友点名。
        if (
            capability == "qq_milk_family"
            and channel.scope.platform == "qq"
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
            capability == "milk_family_lookup"
            and message.mentioned_bot
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

        requested_topics = (
            parse_douyin_topic_request(text)
            if capability == "douyin_topic" else ()
        )
        if capability == "douyin_topic" and message.mentioned_bot and requested_topics:
            videos = await cached_douyin_videos(requested_topics)
            if not videos:
                await reply(message, "这个话题暂时没有找到可用视频，换个 #话题 试试。")
                return True
            video = choose_douyin_video(videos, keywords=requested_topics)
            await reply(message, f"来点#{video.keyword}：{video.url}")
            return True

        if capability == "hdr_milk_image" and is_hdr_milk_command(text):
            asset = await asyncio.to_thread(store.random_milk_asset)
            if asset:
                await send_milk_image(message, asset, hdr_probability=1.0, reply_to=True)
            else:
                await reply(message, "还没有可用奶图，先发几张奶龙或奶蛙吧。")
            return True

        if capability == "weird_image" and text.strip() in {"标怪图", "取消怪图"}:
            reference_id = message.reply_to_message_id or _raw_parent_id(message)
            if not reference_id:
                await reply(message, "请回复一张奶图，再发“标怪图”或“取消怪图”。")
                return True
            weird = text.strip() == "标怪图"
            asset = await asyncio.to_thread(store.mark_weird_reference, reference_id, weird)
            if asset is None:
                await reply(message, "没找到这张奶图，请先发给我识别，再回复图片标记。")
            else:
                action = "已加入怪图池" if weird else "已移出怪图池"
                await reply(message, f"图鉴 #{asset.id:04d} {action}。")
            return True

        corrected_label = (
            parse_correction_command(text)
            if capability == "image_correction" else None
        )
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

        if capability == "daily_stats" and is_daily_stats_command(text):
            stats = await asyncio.to_thread(store.daily_stats, message.chat_id)
            await reply_card(
                message,
                build_daily_stats_card(stats),
                format_daily_stats(stats),
            )
            return True

        if capability == "catalog" and is_catalog_command(text):
            assets = await asyncio.to_thread(store.catalog_assets)
            if not assets:
                await reply(message, "图鉴还没有收集到群聊表情，先发几张奶图吧。")
                return True
            stats = await asyncio.to_thread(store.catalog_stats)
            await reply_catalog(message, assets, stats)
            return True

        if capability == "guess_game" and is_guess_start_command(text):
            await start_guess(message)
            return True

        if capability == "guess_game" and is_guess_reveal_command(text):
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

        if capability == "guess_game" and is_guess_ranking_command(text):
            rankings = await asyncio.to_thread(
                store.guess_ranking,
                chat_id=message.chat_id,
            )
            await reply(message, format_guess_ranking(rankings))
            return True

        if capability == "hunter_ranking" and is_hunter_ranking_command(text):
            hunters = await asyncio.to_thread(store.hunter_rankings)
            await reply(message, format_hunter_ranking(hunters))
            return True

        guess_label = (
            parse_guess_answer(text)
            if capability == "guess_game" else None
        )
        if guess_label and (
            not message.mentioned_bot or getattr(channel, "guess_answers_with_mention", False)
        ):
            await settle_guess(message, guess_label)
            return True

        return False

    async def reply_usage(message, *, unknown_command: bool = False) -> None:
        text = format_help_message(
            args.daily_report_time,
            platform=channel.scope.platform,
            allow_proactive=getattr(channel, "allow_proactive", False),
            speech_enabled=services.speech_client is not None,
        )
        if unknown_command:
            raw_text = message.body_text or message.content_text
            suggestion = suggest_command(raw_text)
            if suggestion:
                command, description, section = suggestion
                text = (
                    f"你是不是想用「{command}」？\n"
                    f"{description}。\n\n"
                    f"发送「help {section}」查看这一类指令。"
                )
            else:
                text = "没认出这个指令。\n\n" + text
        if channel.scope.platform == "feishu":
            # 只有长帮助放进话题，识图、奶签等其他回复保持普通消息。
            await reply_card(
                message, build_help_card(text), text,
                in_thread=message.chat_type in {"group", "topic"},
            )
        else:
            await reply(message, text)

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
            if services.chain_selector:
                reply_asset = await asyncio.to_thread(
                    services.chain_selector.select, observation.asset, message.chat_id,
                )
            else:
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
                # 接龙频率较高，固定发普通图片，避免高亮或闪烁影响群友。
                hdr_probability=0,
                reply_to=True,
                caption=caption,
            )
            if result and result.success and result.message_id:
                await asyncio.to_thread(
                    store.attach_response,
                    observation.id,
                    result.message_id,
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
                result = await delivery.send(
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
                    result = await delivery.send(
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
        result = await delivery.send(message.chat_id, {"text": f"来点奶：{video_url}"})
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
        scheduler = services.work_scheduler
        if (
            (scheduler and scheduler.busy("social"))
            or (speech_dispatcher and speech_dispatcher.busy)
        ):
            return
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

    def build_help_plan(context: MessageContext) -> ResponsePlan:
        section = parse_help_section(context.text)
        text = format_help_message(
            args.daily_report_time,
            platform=context.platform,
            allow_proactive=getattr(channel, "allow_proactive", False),
            speech_enabled=services.speech_client is not None,
            section=section,
        )
        if feishu_emoji_ids and section == "识图":
            text += "\n奶龙表情 点赞 · 点播企业表情；发「奶龙表情」查看列表。"
        options = {"reply_to": context.message_id}
        if context.platform == "feishu" and context.chat_type in {"group", "topic"}:
            options["reply_in_thread"] = True
        content = (
            {"card": build_help_card(text), "fallback_text": text}
            if context.platform == "feishu"
            else {"text": text}
        )
        return ResponsePlan(
            messages=(MessageAction(content=content, options=options),),
            handled=True,
        )

    def build_social_plan(context: MessageContext, url: str) -> ResponsePlan:
        message = context.metadata["legacy_message"]

        async def run_import() -> ResponsePlan | None:
            await import_social_link(message, url, announce=False)
            return None

        return ResponsePlan(
            messages=(
                MessageAction(
                    content={
                        "text": "正在读取图片或抽取视频画面并识别奶蛙、奶龙，稍等一下。"
                    },
                    options={"reply_to": context.message_id},
                ),
            ),
            tasks=(
                AsyncTask(
                    name="social_import",
                    lane="social",
                    run=run_import,
                    fallback_messages=(
                        MessageAction(
                            content={
                                "text": "当前链接解析任务较多，文字功能正常，请稍后再试。"
                            },
                            options={"reply_to": context.message_id},
                        ),
                    ),
                ),
            ),
            handled=True,
        )

    async def build_speech_plan(
        context: MessageContext,
        request: SpeechRequest,
    ) -> ResponsePlan:
        if speech_dispatcher is None:
            return ResponsePlan(
                messages=(MessageAction(
                    content={"text": "语音功能暂未启用。"},
                    options={"reply_to": context.message_id},
                ),),
                handled=True,
            )
        if not request.text or len(request.text) > MAX_SPEECH_CHARS:
            command = "奶蛙说 你好" if request.voice == "naiwa" else "奶龙说 你好"
            return ResponsePlan(
                messages=(MessageAction(
                    content={
                        "text": f"发送“{command}”，台词最多 {MAX_SPEECH_CHARS} 字。"
                    },
                    options={"reply_to": context.message_id},
                ),),
                handled=True,
            )

        async def run_speech() -> ResponsePlan | None:
            try:
                with services.speech_limiter.admit(
                    chat_id=context.chat_id,
                    user_id=context.sender_id,
                ):
                    text_result, audio_result = (
                        await responses.send_text_with_optional_speech(
                            ResponseBundle(
                                message_id=context.message_id,
                                chat_id=context.chat_id,
                                user_id=context.sender_id,
                                text_content={"text": request.text},
                                speech_text=request.text,
                                voice=request.voice,
                                audio_source=(
                                    "wav" if context.platform == "qq" else "opus"
                                ),
                                options={"reply_to": context.message_id},
                            )
                        )
                    )
                if text_result.success and (
                    audio_result is None or audio_result.success
                ):
                    return None
                error_text = "语音发送失败，请稍后重试。"
            except SpeechError as error:
                error_text = str(error)
            except Exception:
                LOGGER.exception("Speech feature failed")
                error_text = "语音生成失败，请稍后重试。"
            return ResponsePlan(
                messages=(MessageAction(
                    content={"text": error_text},
                    options={"reply_to": context.message_id},
                ),),
                handled=True,
            )

        return ResponsePlan(
            tasks=(AsyncTask(
                name=f"speech:{request.voice}",
                lane="background",
                run=run_speech,
                fallback_messages=(MessageAction(
                    content={"text": request.text},
                    options={"reply_to": context.message_id},
                ),),
            ),),
            handled=True,
        )

    async def build_admin_plan(
        context: MessageContext,
        action: str,
    ) -> ResponsePlan:
        quiet = action == "quiet"
        await asyncio.to_thread(
            store.set_proactive_enabled,
            context.chat_id,
            not quiet,
        )
        text = (
            "好，我安静一点。以后只在被 @、识图或收到命令时说话。"
            if quiet else "收到，恢复正常参与。"
        )
        return ResponsePlan(
            messages=(MessageAction(
                content={"text": text},
                options={"reply_to": context.message_id},
            ),),
            handled=True,
        )

    def command_capability_matches(
        context: MessageContext,
        capability: str,
    ) -> bool:
        if context.content_type not in {"text", "post"}:
            return False
        message = context.metadata["legacy_message"]
        text = context.text
        addressed = context.mentioned_bot or context.chat_type == "p2p"
        mentioned_users = mentioned_user_ids(message, body_text=text)

        if capability == "enterprise_emoji":
            return bool(
                feishu_emoji_ids
                and addressed
                and re.fullmatch(r"(?:奶龙|奶蛙)表情[ \t:：]*(.*)", text.strip())
            )
        if capability == "random_milk_image":
            return is_random_milk_command(text)
        if capability == "milk_text":
            if not addressed:
                return False
            try:
                return parse_milk_text_command(text) is not None
            except ValueError:
                return True
        if capability == "milk_name_for_other":
            return bool(
                context.mentioned_bot
                and mentioned_users
                and parse_custom_milk_name_command(text) is None
                and is_milk_name_grant_for_other_command(text)
            )
        if capability == "milk_name":
            return bool(
                (
                    is_milk_name_command(text)
                    or parse_custom_milk_name_command(text) is not None
                )
                and (context.mentioned_bot or context.chat_type != "group")
            )
        if capability == "milk_fortune":
            return bool(
                is_milk_fortune_command(text)
                or parse_fortune_collection_command(text)
            )
        if capability == "qq_milk_family":
            return bool(
                context.platform == "qq"
                and context.mentioned_bot
                and text.strip() in {"接奶", "奶家人"}
                and not has_direct_image(message)
            )
        if capability == "milk_family_lookup":
            return bool(
                context.mentioned_bot
                and mentioned_users
                and not has_direct_image(message)
                and not (context.reply_to_message_id or _raw_parent_id(message))
            )
        if capability == "douyin_topic":
            return bool(context.mentioned_bot and parse_douyin_topic_request(text))
        if capability == "hdr_milk_image":
            return is_hdr_milk_command(text)
        if capability == "weird_image":
            return text.strip() in {"标怪图", "取消怪图"}
        if capability == "image_correction":
            return parse_correction_command(text) is not None
        if capability == "daily_stats":
            return is_daily_stats_command(text)
        if capability == "catalog":
            return is_catalog_command(text)
        if capability == "guess_game":
            guess_label = parse_guess_answer(text)
            return bool(
                is_guess_start_command(text)
                or is_guess_reveal_command(text)
                or is_guess_ranking_command(text)
                or (
                    guess_label
                    and (
                        not context.mentioned_bot
                        or getattr(channel, "guess_answers_with_mention", False)
                    )
                )
            )
        if capability == "hunter_ranking":
            return is_hunter_ranking_command(text)
        return False

    async def build_command_capability_plan(
        context: MessageContext,
        capability: str,
    ) -> ResponsePlan:
        handled = await handle_command_capability(
            context.metadata["legacy_message"],
            capability=capability,
        )
        return ResponsePlan(handled=handled)

    async def build_event_plan(
        context: MessageContext,
        callback,
    ) -> ResponsePlan:
        await callback(context.metadata["legacy_event"])
        return ResponsePlan(handled=True)

    def build_message_context(message) -> MessageContext:
        return MessageContext(
            platform=channel.scope.platform,
            message_id=message.message_id,
            chat_id=message.chat_id,
            sender_id=message.sender_id,
            sender_name=message.sender_name,
            text=message.body_text or message.content_text,
            chat_type=message.chat_type,
            content_type=message.raw_content_type,
            mentioned_bot=message.mentioned_bot,
            reply_to_message_id=message.reply_to_message_id,
            resources=tuple(
                ResourceRef(resource.type, resource.file_key)
                for resource in message.resources
            ),
            mentions=tuple(
                MentionRef(mention.user_id, mention.name, mention.is_bot)
                for mention in message.mentions
            ),
            metadata={"legacy_message": message},
        )

    async def execute_response_plan(
        context: MessageContext,
        plan: ResponsePlan,
    ) -> None:
        async def send_action(action: MessageAction) -> None:
            if action.action == "update":
                if action.message_id:
                    await delivery.update_card(action.message_id, dict(action.content))
                return
            await delivery.send(
                action.chat_id or context.chat_id,
                dict(action.content),
                dict(action.options) or None,
            )

        for action in plan.messages:
            await send_action(action)
        if not plan.tasks:
            return
        scheduler = services.work_scheduler
        if scheduler is None:
            for task in plan.tasks:
                for action in task.fallback_messages:
                    await send_action(action)
            return
        for task in plan.tasks:
            result = await scheduler.submit(
                task.lane,
                lambda _task_context, runner=task.run: runner(),
                TaskContext(
                    job=task.name,
                    message=context.message_id,
                    chat=context.chat_id,
                    user=context.sender_id,
                ),
            )
            if result.outcome is WorkOutcome.SUCCESS:
                if isinstance(result.value, ResponsePlan):
                    await execute_response_plan(context, result.value)
                continue
            if result.outcome in {
                WorkOutcome.DEGRADE_TEXT,
                WorkOutcome.DROP,
                WorkOutcome.FAILED,
                WorkOutcome.RETRY,
                WorkOutcome.TIMEOUT,
            }:
                if result.error:
                    LOGGER.warning(
                        "Feature task degraded name=%s outcome=%s error=%s",
                        task.name,
                        result.outcome,
                        type(result.error).__name__,
                    )
                for action in task.fallback_messages:
                    await send_action(action)

    feature_router: FeatureRouter | None = None

    def proactive_content_matches(context: MessageContext) -> bool:
        message = context.metadata["legacy_message"]
        return (
            not has_direct_image(message)
            and not context.mentioned_bot
            and context.chat_type != "p2p"
        )

    async def build_proactive_content_plan(
        context: MessageContext,
    ) -> ResponsePlan:
        message = context.metadata["legacy_message"]

        async def run_proactive() -> ResponsePlan | None:
            # 普通聊天最多触发一种主动内容，避免一条消息连发视频和奶图。
            if not await maybe_douyin_video_push(message):
                await maybe_random_push(message)
            return None

        return ResponsePlan(
            tasks=(AsyncTask(
                name="proactive_content",
                lane="proactive",
                run=run_proactive,
            ),),
            handled=True,
        )

    async def handle_media_message(message) -> None:
        target = await resolve_image_target(channel, message)
        if target.reason == "quote_unavailable":
            if message.mentioned_bot:
                await reply(
                    message,
                    "暂时无法读取被回复的历史消息，请直接发送图片后再试。",
                )
            return
        if not target.file_key:
            if message.mentioned_bot or message.chat_type == "p2p":
                text = message.body_text or message.content_text
                await reply_usage(message, unknown_command=bool(text.strip()))
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
        recognition_bytes = await asyncio.to_thread(gif_recognition_frame, image_bytes)
        try:
            doubao_prediction = (
                await inference_dispatcher.detect_doubao_bytes(
                    recognition_bytes,
                    message_id=target.message_id or message.message_id,
                    chat_id=message.chat_id,
                    user_id=message.sender_id,
                )
                if doubao_detector else None
            )
            prediction = await inference_dispatcher.predict_bytes(
                recognition_bytes,
                message_id=target.message_id or message.message_id,
                chat_id=message.chat_id,
                user_id=message.sender_id,
            )
        except InferenceTaskDegraded:
            if reply_error:
                await reply(message, "当前识图任务较多，文字功能正常，请稍后再试。")
            return
        if doubao_prediction and doubao_prediction.detected:
            LOGGER.warning(
                "Rejected Doubao image chat_hash=%s message_id=%s confidence=%.3f",
                store.identity_hash(message.chat_id),
                target.message_id or message.message_id,
                doubao_prediction.confidence,
            )
            if reply_error:
                await reply(
                    message,
                    f"{format_prediction(prediction)}\n"
                    f"{format_concentration(prediction)}\n"
                    "豆包过滤器认为这张图疑似含豆包，本次不收录图鉴。",
                )
            return
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
                await continue_chain(
                    message, observation, prediction, include_prediction=True,
                )
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
                await continue_chain(
                    message, observation, prediction, include_prediction=True,
                )
            elif message.mentioned_bot:
                await reply_mention_result(message, observation, prediction)
            else:
                await reply(message, format_prediction(prediction))
        elif response_mode == "chain":
            await continue_chain(message, observation, prediction)

    async def build_media_plan(context: MessageContext) -> ResponsePlan:
        await handle_media_message(context.metadata["legacy_message"])
        return ResponsePlan(handled=True)

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
            context = build_message_context(message)
            if feature_router is None:
                raise RuntimeError("Feature router is not initialized")
            plan = await feature_router.route(context)
            await execute_response_plan(context, plan)
        except Exception:
            LOGGER.exception("Message handling failed")
            if not has_direct_image(message) or should_reply_media_error(
                mentioned_bot=message.mentioned_bot,
                chat_type=message.chat_type,
            ):
                await reply(message, "消息处理失败，请稍后重试。")

    async def on_error(error) -> None:
        LOGGER.error("Adapter error instance=%s: %s", channel.scope.instance_id, error)

    async def initialize_group(chat_id: str) -> None:
        """手动登记全群成员，并识别最近历史消息里的表情包。"""
        await asyncio.to_thread(store.record_chat, chat_id, "group")
        try:
            members = await channel.get_chat_members(chat_id)
            await asyncio.to_thread(
                store.record_milk_family_members,
                [(member.id, member.name) for member in members],
            )
        except Exception:
            LOGGER.warning(
                "New group member enrollment failed instance=%s",
                channel.scope.instance_id,
                exc_info=True,
            )
        try:
            stickers = await channel.get_recent_stickers(chat_id)
            for sticker in stickers:
                try:
                    image_bytes = await download_target(
                        channel,
                        ImageTarget(
                            sticker.message_id,
                            sticker.file_key,
                            "sticker",
                            "resolved",
                        ),
                    )
                    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
                        continue
                    recognition_bytes = await asyncio.to_thread(
                        gif_recognition_frame, image_bytes,
                    )
                    if doubao_detector:
                        guard = await inference_dispatcher.detect_doubao_bytes(
                            recognition_bytes,
                            message_id=f"{sticker.message_id}:guard",
                            chat_id=chat_id,
                            user_id=sticker.sender_id,
                        )
                        if guard.detected:
                            continue
                    prediction = await inference_dispatcher.predict_bytes(
                        recognition_bytes,
                        message_id=sticker.message_id,
                        chat_id=chat_id,
                        user_id=sticker.sender_id,
                    )
                    if not prediction.accepted or prediction.label not in {"nailong", "naiwa"}:
                        continue
                    observation = await asyncio.to_thread(
                        store.record_observation,
                        image_bytes=image_bytes,
                        source_message_id=sticker.message_id,
                        chat_id=chat_id,
                        chat_type="group",
                        sender_id=sticker.sender_id,
                        predicted_label=prediction.label,
                        confidence=prediction.confidence,
                        accepted=True,
                        promote_seed=True,
                    )
                    if observation.is_new_asset:
                        services.notify_catalog_changed()
                except Exception:
                    LOGGER.warning(
                        "Historical sticker analysis failed instance=%s",
                        channel.scope.instance_id,
                        exc_info=True,
                    )
        except Exception:
            LOGGER.warning(
                "New group history initialization failed instance=%s",
                channel.scope.instance_id,
                exc_info=True,
            )
        await send_milk_name_speech(chat_id, "我进来了！")

    command_capabilities = (
        ("enterprise_emoji", 9_900),
        ("random_milk_image", 9_800),
        ("milk_text", 9_700),
        ("milk_name_for_other", 9_600),
        ("milk_name", 9_500),
        ("milk_fortune", 9_400),
        ("qq_milk_family", 9_300),
        ("milk_family_lookup", 9_200),
        ("douyin_topic", 9_100),
        ("hdr_milk_image", 9_000),
        ("weird_image", 8_900),
        ("image_correction", 8_800),
        ("daily_stats", 8_700),
        ("catalog", 8_600),
        ("guess_game", 8_500),
        ("hunter_ranking", 8_400),
    )
    command_features = tuple(
        CallbackFeature(
            capability,
            priority,
            lambda context, name=capability: command_capability_matches(context, name),
            lambda context, name=capability: build_command_capability_plan(context, name),
        )
        for capability, priority in command_capabilities
    )
    feature_router = FeatureRouter(FeatureRegistry((
        HelpFeature(build_help_plan),
        InitializationFeature(initialize_group),
        *command_features,
        SocialImportFeature(find_social_link, build_social_plan),
        AdminFeature(build_admin_plan),
        SpeechFeature(build_speech_plan),
        CallbackFeature(
            "proactive_content",
            -9_000,
            proactive_content_matches,
            build_proactive_content_plan,
            workload=Workload.BACKGROUND,
        ),
        CallbackFeature(
            "media_recognition",
            -10_000,
            lambda _context: True,
            build_media_plan,
            workload=Workload.BACKGROUND,
        ),
    )))

    async def handle_user_added(event: MembersAdded) -> None:
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
            welcome_voice = secrets.choice(("nailong", "naiwa"))
            text_result, audio_result = await responses.send_text_with_optional_speech(
                ResponseBundle(
                    message_id=f"welcome:{secrets.token_hex(8)}",
                    chat_id=chat_id,
                    user_id=claimed_ids[0] if claimed_ids else "",
                    text_content={"text": format_milk_name_welcome(entries)},
                    speech_text=format_milk_name_welcome_speech(
                        entries, voice=welcome_voice,
                    ),
                    voice=welcome_voice,
                    audio_source="opus",
                )
            )
            sent = text_result.success
            if not sent:
                LOGGER.error(
                    "Milk name welcome failed instance=%s members=%d error_code=%s",
                    channel.scope.instance_id,
                    len(entries),
                    getattr(text_result.error, "code", None),
                )
                return
            LOGGER.info(
                "Milk name welcome sent instance=%s members=%d voice_sent=%s",
                channel.scope.instance_id,
                len(entries),
                bool(audio_result and audio_result.success),
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

    async def handle_card_action(event: CardAction) -> None:
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
        result = await delivery.update_card(event.message_id, updated)
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

    event_router = FeatureRouter(FeatureRegistry((
        CallbackFeature(
            "card_correction",
            10_000,
            lambda context: context.event_type == "card_action",
            lambda context: build_event_plan(context, handle_card_action),
        ),
        CallbackFeature(
            "member_welcome",
            9_000,
            lambda context: context.event_type == "members_added",
            lambda context: build_event_plan(context, handle_user_added),
        ),
    )))

    async def on_card_action(event: CardAction) -> None:
        context = MessageContext(
            platform=channel.scope.platform,
            message_id=event.message_id,
            chat_id="",
            sender_id="",
            event_type="card_action",
            metadata={"legacy_event": event},
        )
        plan = await event_router.route(context)
        await execute_response_plan(context, plan)

    async def on_user_added(event: MembersAdded) -> None:
        context = MessageContext(
            platform=channel.scope.platform,
            message_id=f"members-added:{secrets.token_hex(8)}",
            chat_id=event.chat_id,
            sender_id="",
            chat_type="group",
            event_type="members_added",
            metadata={"legacy_event": event},
        )
        plan = await event_router.route(context)
        await execute_response_plan(context, plan)

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
