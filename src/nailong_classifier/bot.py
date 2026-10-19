from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, time
from pathlib import Path
from typing import Any, Literal

from lark_channel import FeishuChannel, PolicyConfig, SecurityConfig
from lark_channel.channel.types import MediaSource, ReplyRef
from lark_channel.core import LogLevel

from .catalog_document import CatalogDocumentClient
from .contact import ContactResolver
from .image_fingerprint import find_duplicate, try_content_hash, try_file_content_hash
from .model import NailongPredictor, Prediction
from .puzzle import try_make_puzzle_image
from .rarity import Rarity
from .storage import (
    LOCAL_TIMEZONE,
    AssetRecord,
    BotStore,
    CatalogStats,
    DailyStats,
    GuessRanking,
    GuessResult,
    GuessRound,
    ObservationRecord,
)

LOGGER = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
DISPLAY_LABELS = {
    "nailong": "奶龙",
    "naiwa": "奶蛙",
    "other": "其他",
}
CORRECTION_PATTERN = re.compile(r"^(?:纠正(?:为|成)?|改成|这是)\s*(奶龙|奶蛙|其他)$")
COMMAND_LABELS = {
    "奶龙": "nailong",
    "奶蛙": "naiwa",
    "其他": "other",
}
# 猜图抢答：只在奶龙/奶蛙之间二选一，所以答案词表不含“其他”。
GUESS_ANSWER_LABELS = {
    "奶龙": "nailong",
    "奶蛙": "naiwa",
}
DEFAULT_DAILY_REPORT_TIME = time(hour=23, minute=55)
DEFAULT_CATALOG_SYNC_RETRY_SECONDS = 60
# 每日播报的轮询间隔：每分钟看一眼是否过了播报点，错过或重启都能靠下一次轮询补上。
DAILY_REPORT_POLL_SECONDS = 60


@dataclass(frozen=True)
class ImageTarget:
    message_id: str | None = None
    file_key: str | None = None
    resource_type: str | None = None
    reason: str = "missing"


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
    raw = message.raw if isinstance(message.raw, dict) else {}
    nested = raw.get("message")
    if isinstance(nested, dict) and nested.get("parent_id"):
        return nested["parent_id"]
    parent_id = raw.get("parent_id")
    return parent_id if isinstance(parent_id, str) and parent_id else None


async def resolve_image_target(channel: FeishuChannel, message) -> ImageTarget:
    direct_images = [resource for resource in message.resources if resource.type == "image"]
    if direct_images:
        return ImageTarget(
            message_id=message.message_id,
            file_key=direct_images[0].file_key,
            resource_type="image",
        )
    direct_stickers = [resource for resource in message.resources if resource.type == "sticker"]
    if direct_stickers:
        return ImageTarget(
            message_id=message.message_id,
            file_key=direct_stickers[0].file_key,
            resource_type="sticker",
        )
    parent_id = message.reply_to_message_id or _raw_parent_id(message)
    if not parent_id:
        return ImageTarget()

    quoted_message = message
    if not message.reply_to_message_id:
        quoted_message = replace(message, reply=ReplyRef(message_id=parent_id))
    resolutions = await channel.resolve_quoted_contexts(
        [quoted_message],
        chat_mode=message.chat_type,
    )
    resolution = resolutions.get(message.message_id)
    if not resolution or resolution.decision != "resolved" or not resolution.context:
        return ImageTarget(reason="quote_unavailable")
    if resolution.context.content_type == "sticker":
        file_key = _find_file_key(_decode_message_content(resolution.context.raw))
        return ImageTarget(
            message_id=resolution.context.message_id,
            file_key=file_key,
            resource_type="sticker",
            reason="missing" if file_key else "quoted_message_has_no_image",
        )

    image_key = _find_image_key(_decode_message_content(resolution.context.raw))
    if not image_key:
        return ImageTarget(reason="quoted_message_has_no_image")
    return ImageTarget(
        message_id=resolution.context.message_id,
        file_key=image_key,
        resource_type="image",
    )


async def download_target(channel: FeishuChannel, target: ImageTarget) -> bytes | None:
    if not target.file_key:
        return None
    attempts: list[tuple[str | None, str, str]] = [
        (target.message_id, "image", "message_image")
    ]
    if target.resource_type == "sticker":
        attempts = [
            (target.message_id, "image", "message_image"),
            (target.message_id, "file", "message_file"),
            (None, "image", "standalone_image"),
            (None, "file", "standalone_file"),
        ]

    for message_id, resource_type, method in attempts:
        image_bytes = await channel.download_resource(
            target.file_key,
            resource_type=resource_type,
            message_id=message_id,
        )
        if image_bytes:
            LOGGER.info(
                "Downloaded resource kind=%s via=%s bytes=%d",
                target.resource_type,
                method,
                len(image_bytes),
            )
            return image_bytes
    return None


def format_prediction(prediction: Prediction) -> str:
    display_label = DISPLAY_LABELS.get(prediction.label, prediction.label)
    confidence = f"{prediction.confidence:.1%}"
    if not prediction.accepted:
        return f"我不太确定。最高候选是「{display_label}」，置信度 {confidence}。"
    if prediction.label == "other":
        return f"识别结果：这张图不像奶龙或奶蛙。置信度 {confidence}。"
    return f"识别结果：{display_label}。置信度 {confidence}。"


def milk_concentration(prediction: Prediction) -> int:
    """把「奶龙 + 奶蛙」两类概率之和当作这张图的奶含量，取 0~100 的整数。

    用两类之和而不是单一置信度，是为了让“偏奶但两类拿不准”的图也能有高浓度读数。
    """
    milk = prediction.scores.get("nailong", 0.0) + prediction.scores.get("naiwa", 0.0)
    return max(0, min(100, round(milk * 100)))


def format_concentration(prediction: Prediction) -> str:
    """奶龙浓度检测仪：把识别置信度包装成一句分级吐槽，仅用于 @机器人 的娱乐回复。"""
    percent = milk_concentration(prediction)
    display_label = DISPLAY_LABELS.get(prediction.label, prediction.label)
    if percent >= 90:
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

    识别结论沿用既有文案（已确认图/模型不确定提示都不变），浓度和稀有度作为附加娱乐信息。
    """
    lines = [format_existing_observation(observation, prediction), format_concentration(prediction)]
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


def is_help_command(text: str) -> bool:
    return text.strip().lower() in {"help", "帮助", "使用帮助", "奶龙帮助"}


def is_help_request(text: str, *, mentioned_bot: bool) -> bool:
    return mentioned_bot and is_help_command(text)


def is_guess_start_command(text: str) -> bool:
    return text.strip() in {"猜图", "盲盒猜图", "来一局", "猜一猜"}


def is_guess_reveal_command(text: str) -> bool:
    return text.strip() in {"公布答案", "揭晓答案", "公布"}


def is_guess_ranking_command(text: str) -> bool:
    return text.strip() in {"猜图榜", "猜图排行榜", "猜图积分榜"}


GUESS_ANSWER_PATTERN = re.compile(r"^(?:猜|是|答案是?)?\s*(奶龙|奶蛙)$")


def parse_guess_answer(text: str) -> str | None:
    """把「奶龙」「猜奶蛙」「是奶龙」这类抢答解析成内部标签，其它文本返回 None。"""
    match = GUESS_ANSWER_PATTERN.fullmatch(text.strip())
    return GUESS_ANSWER_LABELS[match.group(1)] if match else None


def format_guess_prompt() -> str:
    return (
        "🎲 盲盒猜图开始！\n"
        "这是打了码的局部，猜猜它是奶龙还是奶蛙？"
        "直接发「奶龙」或「奶蛙」抢答，首答对者得 1 分。"
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


def format_help_message(report_time: time) -> str:
    return (
        "我是奶龙，可以这样玩：\n"
        "1. 直接发送图片或表情：自动接龙，不用 @。\n"
        "2. 回复图片并 @我：查看识别结果、奶龙浓度和稀有度。\n"
        "3. 回复原图或接龙图，发送“纠正为奶龙/奶蛙/其他”：修正分类。\n"
        "4. 发送“奶龙图鉴”：查看最近收录的奶龙图。\n"
        "5. 发送“今日含奶量”：立即查看；"
        f"每天 {report_time.strftime('%H:%M')} 也会自动播报。\n"
        "6. 发送“猜图”：开一局盲盒猜图，抢答“奶龙/奶蛙”得分；"
        "“公布答案”揭晓本局，“猜图榜”看积分排行。"
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
        rows.append(
            [
                {
                    "tag": "text",
                    "text": (
                        f"#{asset.id:04d} · {display_label} · "
                        f"出现 {asset.seen_count} 次"
                    ),
                }
            ]
        )
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
    parser = argparse.ArgumentParser(description="Run the Feishu image classifier bot")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("MODEL_PATH", "artifacts/model.pt")),
    )
    parser.add_argument("--device", default=os.getenv("MODEL_DEVICE", "cpu"))
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
    return parser.parse_args()


async def run_bot(args: argparse.Namespace) -> None:
    app_id = os.getenv("LARK_APP_ID")
    app_secret = os.getenv("LARK_APP_SECRET")
    if not app_id or not app_secret:
        raise RuntimeError("LARK_APP_ID and LARK_APP_SECRET must be set")

    predictor = NailongPredictor(
        args.model,
        device=args.device,
        threshold=args.threshold,
    )
    store = BotStore(args.state_dir)
    channel = FeishuChannel(
        app_id=app_id,
        app_secret=app_secret,
        log_level=LogLevel.ERROR,
        policy=PolicyConfig(
            dm_policy="open",
            group_policy="open",
            require_mention=False,
        ),
        security=SecurityConfig(mode="audit"),
    )
    chain_lock = asyncio.Lock()
    guess_lock = asyncio.Lock()
    contact_resolver = ContactResolver(channel.client)
    catalog_sync_event = asyncio.Event()
    catalog_sync_lock = asyncio.Lock()
    catalog_document: CatalogDocumentClient | None = None
    if bool(args.catalog_document_id) != bool(args.catalog_section_block_id):
        raise RuntimeError(
            "CATALOG_DOCUMENT_ID and CATALOG_SECTION_BLOCK_ID must be set together"
        )
    if args.catalog_document_id and args.catalog_section_block_id:
        catalog_document = CatalogDocumentClient(
            channel.client,
            document_id=args.catalog_document_id,
            section_block_id=args.catalog_section_block_id,
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

    async def reply_catalog(
        message,
        assets: list[AssetRecord],
        stats: CatalogStats,
    ) -> None:
        uploaded_assets: list[AssetRecord] = []
        image_keys: list[str] = []
        for asset in assets:
            try:
                image_key = await channel.upload_media(
                    MediaSource(kind="file", path=str(asset.path)),
                    kind="image",
                )
            except Exception:
                LOGGER.exception("Failed to upload catalog asset id=%s", asset.id)
                continue
            uploaded_assets.append(asset)
            image_keys.append(image_key)

        if not uploaded_assets:
            await reply(message, "图鉴图片上传失败，请稍后重试。")
            return

        result = await channel.send(
            message.chat_id,
            {"post": build_catalog_post(uploaded_assets, image_keys, stats)},
            {"reply_to": message.message_id},
        )
        if not result.success:
            LOGGER.error("Failed to reply with catalog: %s", result.error)

    async def handle_command(message) -> bool:
        if message.raw_content_type not in {"text", "post"}:
            return False
        text = message.body_text or message.content_text

        if is_help_request(text, mentioned_bot=message.mentioned_bot):
            await reply(message, format_help_message(args.daily_report_time))
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
            if catalog_document and correction.new_label in {"nailong", "naiwa"}:
                catalog_sync_event.set()
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
            await reply(message, format_daily_stats(stats))
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

        guess_label = parse_guess_answer(text)
        if guess_label and not message.mentioned_bot:
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
        if puzzle is not None:
            await reply_image(message, puzzle, format_guess_prompt())
        else:
            await reply_image(message, round_.asset.path, format_guess_prompt())

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
    ) -> None:
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
                    await reply(message, format_prediction(prediction))
                return

            result = await reply_image(
                message,
                reply_asset.path,
                format_chain_caption(observation),
            )
            if result.success and result.message_id:
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

    async def build_mention_reply(
        observation: ObservationRecord,
        prediction: Prediction,
    ) -> str:
        # 稀有度只对真正入典的奶龙/奶蛙图有意义，“其他”不算稀有度。
        rarity = None
        if observation.asset.label in {"nailong", "naiwa"}:
            rarity = await asyncio.to_thread(store.asset_rarity, observation.asset.id)
        return format_mention_reply(observation, prediction, rarity)

    async def send_pending_daily_reports(report_day: str) -> bool:
        targets = await asyncio.to_thread(
            store.pending_daily_report_targets,
            report_day,
        )
        all_sent = True
        for target in targets:
            try:
                stats = await asyncio.to_thread(
                    store.daily_stats,
                    target.chat_id,
                    day=target.report_day,
                )
                result = await channel.send(
                    target.chat_id,
                    {"text": f"每日含奶量播报\n{format_daily_stats(stats)}"},
                )
                if not result.success:
                    all_sent = False
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
        while not channel.connection_snapshot().ready:
            await asyncio.sleep(1)

        # 每分钟轮询：只要当天已过播报点就尝试发送。发没发过由数据库按
        # (群, 日期) 去重，所以重复轮询不会重发；重启或错过点都会在下一次轮询自动补发。
        while True:
            now = datetime.now(LOCAL_TIMEZONE)
            report_at = datetime.combine(
                now.date(),
                args.daily_report_time,
                tzinfo=LOCAL_TIMEZONE,
            )
            if now >= report_at:
                report_day = now.date().isoformat()
                try:
                    await send_pending_daily_reports(report_day)
                except Exception:
                    LOGGER.exception("Daily report polling failed report_day=%s", report_day)
            await asyncio.sleep(DAILY_REPORT_POLL_SECONDS)

    async def on_message(message) -> None:
        if message.sender_is_bot:
            return
        LOGGER.info(
            "Received message chat_type=%s content_type=%s resource_types=%s "
            "quoted=%s mentioned=%s",
            message.chat_type,
            message.raw_content_type,
            [resource.type for resource in message.resources],
            bool(message.reply_to_message_id),
            message.mentioned_bot,
        )
        try:
            if await handle_command(message):
                return
            if not has_direct_image(message) and not message.mentioned_bot:
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
            prediction = await asyncio.to_thread(predictor.predict_bytes, image_bytes)
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
            )
            if catalog_document:
                catalog_sync_event.set()
            if not observation.is_new_observation:
                LOGGER.info(
                    "Skipping duplicate message source_message_id=%s",
                    target.message_id or message.message_id,
                )
                if message.mentioned_bot:
                    await reply(
                        message,
                        await build_mention_reply(observation, prediction),
                    )
                return

            response_mode = media_response_mode(
                mentioned_bot=message.mentioned_bot,
                chat_type=message.chat_type,
                label=observation.asset.label,
            )
            if response_mode == "prediction":
                await reply(
                    message,
                    await build_mention_reply(observation, prediction),
                )
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
        LOGGER.error("Feishu channel error: %s", error)

    async def on_reject(event) -> None:
        LOGGER.info("Message rejected reason=%s", event.reason)

    channel.on("message", on_message)
    channel.on("error", on_error)
    channel.on("reject", on_reject)
    LOGGER.info(
        "Starting bot with model=%s device=%s threshold=%.3f state_dir=%s "
        "daily_report_time=%s catalog_document_sync=%s",
        args.model,
        predictor.device,
        predictor.threshold,
        args.state_dir,
        args.daily_report_time.strftime("%H:%M"),
        bool(catalog_document),
    )
    report_task = asyncio.create_task(daily_report_loop())
    catalog_sync_task = (
        asyncio.create_task(catalog_document_sync_loop())
        if catalog_document
        else None
    )
    try:
        await channel.connect()
    finally:
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
