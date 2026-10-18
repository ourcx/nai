from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal

from lark_channel import FeishuChannel, PolicyConfig, SecurityConfig
from lark_channel.channel.types import MediaSource, ReplyRef
from lark_channel.core import LogLevel

from .model import NailongPredictor, Prediction
from .storage import (
    LOCAL_TIMEZONE,
    AssetRecord,
    BotStore,
    CatalogStats,
    DailyStats,
    ObservationRecord,
)

LOGGER = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
DISPLAY_LABELS = {
    "nailong": "奶龙",
    "naiwa": "奶娃",
    "other": "其他",
}
CORRECTION_PATTERN = re.compile(r"^(?:纠正(?:为|成)?|改成|这是)\s*(奶龙|奶娃|其他)$")
COMMAND_LABELS = {
    "奶龙": "nailong",
    "奶娃": "naiwa",
    "其他": "other",
}
DEFAULT_DAILY_REPORT_TIME = time(hour=23, minute=55)


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
        return f"识别结果：这张图不像奶龙或奶娃。置信度 {confidence}。"
    return f"识别结果：{display_label}。置信度 {confidence}。"


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


def parse_daily_report_time(value: str) -> time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as error:
        raise argparse.ArgumentTypeError("daily report time must use HH:MM") from error


def has_direct_image(message) -> bool:
    return any(resource.type in {"image", "sticker"} for resource in message.resources)


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
        f"奶龙 {stats.nailong} · 奶娃 {stats.naiwa} · 其他 {stats.other}\n"
        f"今日共识别 {stats.total} 张图片/表情，新收录奶龙 {stats.new_nailong} 张。"
    )


def format_help_message(report_time: time) -> str:
    return (
        "我是奶龙，可以这样玩：\n"
        "1. 直接发送图片或表情：自动接龙，不用 @。\n"
        "2. 回复图片并 @我：查看文字识别结果。\n"
        "3. 回复原图或接龙图，发送“纠正为奶龙/奶娃/其他”：修正分类。\n"
        "4. 发送“奶龙图鉴”：查看最近收录的奶龙图。\n"
        "5. 发送“今日含奶量”：立即查看；"
        f"每天 {report_time.strftime('%H:%M')} 也会自动播报。"
    )


def format_catalog_caption(stats: CatalogStats, shown: int) -> str:
    return (
        f"共收录 {stats.total} 张：奶龙 {stats.nailong} · 奶娃 {stats.naiwa} · "
        f"待确认 {stats.pending}。今日新增 {stats.today_new} 张，"
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
        display_label = DISPLAY_LABELS.get(asset.label, "待确认")
        if asset.label == "other":
            display_label = "待确认"
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
    return "奶娃接上了。"


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
                await reply(message, "请回复原图片或我的接龙图片，再发送“纠正为奶龙/奶娃/其他”。")
                return True
            correction = await asyncio.to_thread(
                store.correct_label,
                reference_id,
                corrected_label,
            )
            if not correction:
                await reply(message, "没有找到这张图的识别记录，请重新发送图片后再纠正。")
                return True
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

        return False

    async def continue_chain(
        message,
        observation: ObservationRecord,
        prediction: Prediction,
    ) -> None:
        reply_asset = await asyncio.to_thread(
            store.random_asset,
            observation.asset.label,
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

    async def daily_report_loop() -> None:
        while not channel.connection_snapshot().ready:
            await asyncio.sleep(1)

        while True:
            now = datetime.now(LOCAL_TIMEZONE)
            report_at = datetime.combine(
                now.date(),
                args.daily_report_time,
                tzinfo=LOCAL_TIMEZONE,
            )
            if now < report_at:
                await asyncio.sleep((report_at - now).total_seconds())
                continue

            report_day = now.date().isoformat()
            if not await send_pending_daily_reports(report_day):
                await asyncio.sleep(300)
                continue

            next_report_at = report_at + timedelta(days=1)
            await asyncio.sleep((next_report_at - now).total_seconds())

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

            image_bytes = await download_target(channel, target)
            if not image_bytes:
                if target.resource_type == "sticker":
                    await reply(
                        message,
                        "这张表情暂时拿不到图像数据。请把它保存或转发成普通图片后再试。",
                    )
                else:
                    await reply(message, "图片下载失败，请重新发送。")
                return
            if len(image_bytes) > MAX_IMAGE_BYTES:
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
            if not observation.is_new_observation:
                LOGGER.info(
                    "Skipping duplicate message source_message_id=%s",
                    target.message_id or message.message_id,
                )
                if message.mentioned_bot:
                    await reply(
                        message,
                        format_existing_observation(observation, prediction),
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
                    format_existing_observation(observation, prediction),
                )
            elif response_mode == "chain":
                await continue_chain(message, observation, prediction)
        except Exception:
            LOGGER.exception("Message handling failed")
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
        "daily_report_time=%s",
        args.model,
        predictor.device,
        predictor.threshold,
        args.state_dir,
        args.daily_report_time.strftime("%H:%M"),
    )
    report_task = asyncio.create_task(daily_report_loop())
    try:
        await channel.connect()
    finally:
        report_task.cancel()
        with suppress(asyncio.CancelledError):
            await report_task


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
