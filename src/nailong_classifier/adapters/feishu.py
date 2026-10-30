from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import replace
from typing import Any

from lark_channel import FeishuChannel, PolicyConfig, SecurityConfig
from lark_channel.channel.types import MediaSource, ReplyRef
from lark_channel.core import LogLevel

from ..catalog_document import CatalogDocumentClient
from ..contact import ContactResolver
from .base import (
    CardAction,
    Handler,
    IdentityScope,
    ImageTarget,
    Member,
    MembersAdded,
    Mention,
    Message,
    Resource,
    SendError,
    SendResult,
)

LOGGER = logging.getLogger(__name__)


def _decode_message_content(raw: dict[str, Any]) -> Any:
    body = raw.get("body")
    content = body.get("content") if isinstance(body, dict) else raw.get("content")
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}
    return content or {}


def _find_key(value: Any, key: str) -> str | None:
    if isinstance(value, dict):
        result = value.get(key)
        if isinstance(result, str) and result:
            return result
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        return None
    for nested in children:
        found = _find_key(nested, key)
        if found:
            return found
    return None


def _find_image_key(value: Any) -> str | None:
    return _find_key(value, "image_key")


def _find_file_key(value: Any) -> str | None:
    return _find_key(value, "file_key")


def _raw_parent_id(message) -> str | None:
    raw = getattr(message, "raw", None)
    if not isinstance(raw, dict):
        return None
    nested = raw.get("message")
    parent_id = nested.get("parent_id") if isinstance(nested, dict) else None
    parent_id = parent_id or raw.get("parent_id")
    return parent_id if isinstance(parent_id, str) and parent_id else None


class FeishuAdapter:
    """将飞书 SDK 事件、媒体与返回值转换为统一协议。"""

    def __init__(self, scope: IdentityScope, *, app_id: str, app_secret: str) -> None:
        self.scope = scope
        self.channel = FeishuChannel(
            app_id=app_id,
            app_secret=app_secret,
            log_level=LogLevel.ERROR,
            policy=PolicyConfig(dm_policy="open", group_policy="open", require_mention=False),
            security=SecurityConfig(mode="audit"),
        )
        self.contacts = ContactResolver(self.channel.client)
        self._user_chats: dict[str, str] = {}
        self._member_cache: dict[str, tuple[float, tuple[Member, ...]]] = {}
        self._member_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def normalize_message(self, message) -> Message:
        # 平台事件里真实的 @ 显示名也可用于之后查询这个人，不从正文猜昵称。
        self.contacts.remember(message.sender_id, message.sender_name)
        if message.chat_type == "group" and message.sender_id:
            self._user_chats[message.sender_id] = message.chat_id
        for item in getattr(message, "mentions", ()):
            if item.open_id and not item.is_bot:
                self.contacts.remember(item.open_id, item.name)
                if message.chat_type == "group":
                    self._user_chats[item.open_id] = message.chat_id
        parent = message.reply_to_message_id or _raw_parent_id(message)
        return Message(
            message_id=self.scope.key("message", message.message_id),
            chat_id=self.scope.key("chat", message.chat_id),
            chat_type=message.chat_type,
            sender_id=self.scope.key("user", message.sender_id),
            sender_name=message.sender_name or self.contacts.cached_name(message.sender_id),
            sender_is_bot=message.sender_is_bot,
            mentioned_bot=message.mentioned_bot,
            body_text=message.body_text or "",
            content_text=message.content_text or "",
            raw_content_type=message.raw_content_type,
            reply_to_message_id=self.scope.key("message", parent) if parent else None,
            resources=tuple(Resource(item.type, item.file_key) for item in message.resources),
            mentions=tuple(
                Mention(self.scope.key("user", item.open_id), item.name, item.is_bot)
                for item in getattr(message, "mentions", ())
                if item.open_id
            ),
            native=message,
        )

    def on(self, event: str, handler: Handler) -> None:
        async def forward(native) -> None:
            if event == "message":
                payload = self.normalize_message(native)
            elif event == "members_added":
                payload = self.normalize_members_added(native)
                if payload is None:
                    return
            elif event == "card_action":
                payload = CardAction(
                    self.scope.key("message", native.message_id),
                    native.action.value,
                    native.action.tag,
                )
            else:
                payload = native
            # SDK 在自己的后台线程/事件循环中派发。把业务统一转回运行循环，
            # 才能让多个飞书实例、QQ 和定时任务安全共用 asyncio 锁。
            if self._loop is None:
                raise RuntimeError("Adapter has not connected")
            if asyncio.get_running_loop() is self._loop:
                await handler(payload)
            else:
                await asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(handler(payload), self._loop)
                )

        if event == "members_added":
            self.channel.on_raw_event("im.chat.member.user.added_v1", forward)
            LOGGER.info(
                "Feishu member event handler registered instance=%s",
                self.scope.instance_id,
            )
        else:
            self.channel.on(
                "cardAction" if event == "card_action" else event, forward
            )

    def normalize_members_added(self, native: Any) -> MembersAdded | None:
        """兼容嵌套 UserID 与旧扁平事件；昵称缺失留给业务按身份反查。

        入群事件先记录接收与结构，再解析，避免空名单被静默吞掉。诊断日志
        只记字段名和人数，不输出原始事件、群 ID 或成员姓名。
        """
        data = native.get("event", native) if isinstance(native, dict) else None
        LOGGER.info(
            "Feishu member event received instance=%s fields=%s",
            self.scope.instance_id,
            sorted(data) if isinstance(data, dict) else [],
        )
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("chat_id"), str)
            or not data["chat_id"]
            or not isinstance(data.get("users"), list)
        ):
            LOGGER.warning(
                "Invalid Feishu member event structure instance=%s",
                self.scope.instance_id,
            )
            return None
        members = []
        seen: set[str] = set()
        for user in data["users"]:
            if not isinstance(user, dict):
                continue
            user_id = user.get("open_id")
            if not isinstance(user_id, str) or not user_id:
                identity = user.get("user_id")
                # UserID 内的 user_id/union_id 与 open_id 不是同一种身份；
                # 必须取 open_id，才能与消息、通讯录和欢迎去重的身份保持一致。
                user_id = identity.get("open_id") if isinstance(identity, dict) else identity
            if not isinstance(user_id, str) or not user_id or user_id in seen:
                continue
            seen.add(user_id)
            name = user.get("nickname") or user.get("name")
            name = name.strip() if isinstance(name, str) else None
            self._user_chats[user_id] = data["chat_id"]
            self.contacts.remember(user_id, name)
            members.append(Member(self.scope.key("user", user_id), name or None))
        LOGGER.info(
            "Feishu member event normalized instance=%s received=%d members=%d missing_names=%d",
            self.scope.instance_id,
            len(data["users"]),
            len(members),
            sum(member.name is None for member in members),
        )
        return MembersAdded(self.scope.key("chat", data["chat_id"]), tuple(members))

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self.channel.connect()

    async def close(self) -> None:
        await self.channel.disconnect()

    async def resolve(self, user_id: str, *, fallback: str = "神秘奶友") -> str:
        raw_id = self.scope.raw("user", user_id)
        try:
            name = await asyncio.wait_for(self.contacts.resolve(raw_id, fallback=""), timeout=3)
        except Exception:
            name = self.contacts.cached_name(raw_id)
        if name:
            return name
        chat_id = self._user_chats.get(raw_id)
        if chat_id:
            try:
                await self.get_chat_members(self.scope.key("chat", chat_id))
            except Exception:
                # 群成员权限也不可用时不阻断起名，且失败缓存避免每次求名重复请求。
                LOGGER.warning("Feishu member name fallback unavailable instance=%s",
                               self.scope.instance_id)
        return self.contacts.cached_name(raw_id) or fallback

    async def get_chat_members(self, chat_id: str) -> list[Member]:
        raw_chat = self.scope.raw("chat", chat_id)
        async with self._member_lock:
            cached = self._member_cache.get(raw_chat)
            if cached and time.monotonic() < cached[0]:
                return list(cached[1])
            try:
                members = await asyncio.wait_for(
                    self.channel.get_chat_members(
                        raw_chat, page_size=100, max_pages=100, force=True
                    ),
                    timeout=5,
                )
            except Exception:
                self._member_cache[raw_chat] = (time.monotonic() + 60, ())
                raise
            normalized = []
            for member in members:
                self.contacts.remember(member.id, member.name)
                self._user_chats[member.id] = raw_chat
                normalized.append(Member(self.scope.key("user", member.id), member.name))
            self._member_cache[raw_chat] = (time.monotonic() + 300, tuple(normalized))
            return normalized

    async def resolve_image_target(self, message: Message) -> ImageTarget:
        for kind in ("image", "sticker"):
            resource = next((r for r in message.resources if r.type == kind), None)
            if resource:
                return ImageTarget(message.message_id, resource.file_key, kind, "resolved")
        if not message.reply_to_message_id:
            return ImageTarget()
        native = message.native
        if native is None:
            return ImageTarget(reason="quote_unavailable")
        if not native.reply_to_message_id:
            native = replace(
                native,
                reply=ReplyRef(
                    message_id=self.scope.raw("message", message.reply_to_message_id)
                ),
            )
        resolutions = await self.channel.resolve_quoted_contexts(
            [native], chat_mode=message.chat_type
        )
        resolution = resolutions.get(native.message_id)
        if not resolution or resolution.decision != "resolved" or not resolution.context:
            return ImageTarget(reason="quote_unavailable")
        context = resolution.context
        kind = "sticker" if context.content_type == "sticker" else "image"
        body = _decode_message_content(context.raw)
        key = _find_file_key(body) if kind == "sticker" else _find_image_key(body)
        return ImageTarget(
            self.scope.key("message", context.message_id),
            key,
            kind,
            "resolved" if key else "quoted_message_has_no_image",
        )

    async def download_target(self, target: ImageTarget) -> bytes | None:
        if not target.file_key:
            return None
        message_id = (
            self.scope.raw("message", target.message_id) if target.message_id else None
        )
        attempts = [(message_id, "image")]
        if target.resource_type == "sticker":
            attempts.extend([(message_id, "file"), (None, "image"), (None, "file")])
        for source_id, kind in attempts:
            content = await self.channel.download_resource(
                target.file_key, resource_type=kind, message_id=source_id
            )
            if content:
                return content
        return None

    def _result(self, result) -> SendResult:
        error = (
            SendError(
                getattr(result.error.code, "value", str(result.error.code)),
                str(getattr(result.error, "message", "")),
            )
            if result.error
            else None
        )
        return SendResult(
            result.success,
            self.scope.key("message", result.message_id) if result.message_id else None,
            error,
        )

    async def send(
        self, chat_id: str, content: dict[str, Any], options: dict[str, Any] | None = None
    ) -> SendResult:
        payload = {key: value for key, value in content.items() if key != "fallback_text"}
        if "gallery" in payload:
            rows = [[{"tag": "text", "text": payload.get("caption", "")}]]
            for item in payload["gallery"]:
                try:
                    key = await self.channel.upload_media(
                        MediaSource(kind="file", path=str(item["source"])), kind="image"
                    )
                except Exception:
                    LOGGER.exception("Failed to upload gallery image")
                    continue
                rows.extend([
                    [{"tag": "img", "image_key": key}],
                    [{"tag": "text", "text": item["caption"]}],
                ])
            if len(rows) == 1:
                return SendResult(False, error=SendError("upload_failed"))
            payload = {"post": {"zh_cn": {"title": "奶龙图鉴", "content": rows}}}
        native_options = dict(options or {})
        if native_options.get("reply_to"):
            native_options["reply_to"] = self.scope.raw(
                "message", native_options["reply_to"]
            )
        return self._result(
            await self.channel.send(
                self.scope.raw("chat", chat_id), payload, native_options or None
            )
        )

    async def update_card(self, message_id: str, card: dict[str, Any]) -> SendResult:
        return self._result(
            await self.channel.update_card(self.scope.raw("message", message_id), card)
        )

    def catalog_document(self, document_id: str, section_block_id: str):
        return CatalogDocumentClient(
            self.channel.client,
            document_id=document_id,
            section_block_id=section_block_id,
        )
