from __future__ import annotations

import asyncio
import base64
import io
import itertools
import json
import logging
import re
from collections import OrderedDict
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageDraw, ImageOps
from websockets.asyncio.client import connect

from .base import (
    ErrorCode,
    Handler,
    IdentityScope,
    ImageTarget,
    Member,
    Mention,
    Message,
    Resource,
    SendError,
    SendResult,
)

LOGGER = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 20_971_520
MESSAGE_CACHE_SIZE = 1024
GROUP_AND_C2C_INTENTS = 33_554_432
MESSAGE_EVENTS = {"GROUP_AT_MESSAGE_CREATE", "GROUP_MESSAGE_CREATE", "C2C_MESSAGE_CREATE"}


class QQAPIError(RuntimeError):
    """仅保留状态码，不将请求凭证或平台返回的原始内容写入业务日志。"""

    def __init__(self, status: int, code: Any = None) -> None:
        self.status = status
        self.code = code
        super().__init__(f"QQ API failed (HTTP {status}, code={code})")


class QQAdapter:
    """腾讯官方 QQ 机器人：WebSocket 收事件，API v2 收发群聊/C2C 消息。

    AppID 是开放平台应用号，不是 QQ 账号。每个实例独立管理 Token、会话与
    回复序号；图片、模型和业务功能仍由共用后端提供。
    """

    def __init__(
        self,
        scope: IdentityScope,
        *,
        app_id: str,
        app_secret: str,
        allow_proactive: bool = False,
        sandbox: bool = False,
    ) -> None:
        if not app_id.isdigit() or not app_secret:
            raise ValueError("QQ requires a numeric AppID and an AppSecret")
        self.scope = scope
        self.app_id = app_id
        self._app_secret = app_secret
        self.allow_proactive = allow_proactive
        domain = "sandbox.api.sgroup.qq.com" if sandbox else "api.bot.qq.com"
        self._api = httpx.AsyncClient(base_url=f"https://{domain}/", timeout=20)
        self._auth = httpx.AsyncClient(base_url="https://api.bot.qq.com/", timeout=20)
        # CDN 下载不能携带 QQ 的 Authorization 或 AppSecret。
        self._media = httpx.AsyncClient(timeout=20, follow_redirects=True)
        self._token = ""
        self._expires_at = datetime.min.replace(tzinfo=UTC)
        self._token_lock = asyncio.Lock()
        self._handlers: dict[str, Handler] = {}
        self._socket = None
        self._closed = asyncio.Event()
        self._session_id: str | None = None
        self._sequence: int | None = None
        self._heartbeat_acked = True
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(maxsize=256)
        self._messages: OrderedDict[str, Message] = OrderedDict()
        self._received_at: OrderedDict[str, datetime] = OrderedDict()
        self._last_messages: OrderedDict[str, str] = OrderedDict()
        self._reply_sequences: OrderedDict[str, Any] = OrderedDict()
        self._sent_sources: OrderedDict[str, bytes] = OrderedDict()
        self._names: OrderedDict[str, str] = OrderedDict()
        self._user_aliases: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._reference_ids: OrderedDict[tuple[str, str], str] = OrderedDict()
        # 业务可用这个能力区别群 @抢答与飞书不点名抢答，不必解析平台协议。
        self.guess_answers_with_mention = True

    def on(self, event: str, handler: Handler) -> None:
        self._handlers[event] = handler

    async def _emit(self, event: str, payload: Any) -> None:
        handler = self._handlers.get(event)
        if handler:
            await handler(payload)

    @staticmethod
    def _cache(cache: OrderedDict, key: str, value: Any, *, limit: int = MESSAGE_CACHE_SIZE):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    async def _access_token(self, *, rejected_token: str | None = None) -> str:
        async with self._token_lock:
            now = datetime.now(UTC)
            # 并发请求只刷新一次；另一个请求已经换过 Token 时直接使用新值。
            if self._token and now < self._expires_at and rejected_token != self._token:
                return self._token
            response = await self._auth.post(
                "/app/getAppAccessToken",
                json={"appId": self.app_id, "clientSecret": self._app_secret},
            )
            if response.is_error:
                raise QQAPIError(response.status_code)
            data = response.json()
            token = data.get("access_token")
            if not isinstance(token, str) or not token:
                raise QQAPIError(response.status_code, data.get("code"))
            self._token = token
            # 官方有效期单位是秒；提前一分钟刷新，避免发送中途到期。
            ttl = max(
                getcontext().subtract(Decimal(str(data.get("expires_in", 7200))), Decimal(60)),
                Decimal(1),
            )
            self._expires_at = now + timedelta(seconds=float(ttl))
            return token

    async def _call(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        token = await self._access_token()
        response = await self._api.request(
            method, path, headers={"Authorization": f"QQBot {token}"}, **kwargs
        )
        if response.status_code == 401:
            token = await self._access_token(rejected_token=token)
            response = await self._api.request(
                method, path, headers={"Authorization": f"QQBot {token}"}, **kwargs
            )
        data = response.json()
        if response.is_error:
            raise QQAPIError(response.status_code, data.get("code"))
        if not isinstance(data, dict):
            raise QQAPIError(response.status_code)
        return data

    def _message_key(self, raw_chat: str, message_id: str) -> str:
        # 群范围 openid 和消息 ID 不可当成跨会话、跨实例的全局 ID。
        return self.scope.key("message", json.dumps([raw_chat, message_id]))

    @staticmethod
    def _scene_indexes(data: dict[str, Any]) -> dict[str, str]:
        """仅提取引用索引，不保存 message_scene 中的 auth_token。"""
        indexes = {}
        scene = data.get("message_scene") or {}
        for entry in scene.get("ext") or []:
            if not isinstance(entry, str):
                continue
            key, separator, value = entry.partition("=")
            if separator and value and key in {"msg_idx", "ref_msg_idx"}:
                indexes[key] = value
        return indexes

    @staticmethod
    def _image_resources(attachments: list[dict[str, Any]]) -> tuple[Resource, ...]:
        resources = []
        for attachment in attachments:
            if not str(attachment.get("content_type", "")).startswith("image/"):
                continue
            url = attachment.get("url")
            if isinstance(url, str) and url:
                if url.startswith("//"):
                    url = "https:" + url
                if urlparse(url).scheme in {"http", "https"}:
                    resources.append(Resource("image", url))
        return tuple(resources)

    @classmethod
    def _quoted_resources(cls, elements: list[dict[str, Any]]) -> tuple[Resource, ...]:
        # 新版引用事件可直接附带图片，深度有界遍历，避免把原始平台载荷交给业务。
        pending = [(elements, range(8))]
        while pending:
            nodes, levels = pending.pop()
            for node in nodes:
                resources = cls._image_resources(node.get("attachments") or [])
                if resources:
                    return resources
                if levels and node.get("msg_elements"):
                    pending.append((node["msg_elements"], levels[1:]))
        return ()

    def normalize_message(self, event_type: str, data: dict[str, Any]) -> Message:
        group = event_type != "C2C_MESSAGE_CREATE"
        kind = "group" if group else "p2p"
        author = data.get("author") or {}
        raw_user = str(author["member_openid"] if group else author["user_openid"])
        raw_chat = f"{kind}:{data['group_openid'] if group else raw_user}"
        user_identity = [raw_chat, raw_user] if group else ["p2p", raw_user]
        sender_id = self.scope.key("user", json.dumps(user_identity))
        resources = self._image_resources(data.get("attachments") or [])
        mentions = []
        for user in data.get("mentions") or []:
            # 文档中的 mentions 不含机器人自身；旧载荷仍显式排除 bot。
            user_id = user.get("member_openid" if group else "user_openid") or user.get("id")
            if not isinstance(user_id, str) or not user_id or user.get("bot"):
                continue
            identity = [raw_chat, user_id] if group else ["p2p", user_id]
            mentions.append(Mention(
                self._user_aliases.get(
                    (raw_chat, user_id), self.scope.key("user", json.dumps(identity))
                ),
                user.get("username") or None,
            ))
        # QQ @事件中的 mention 语法属于平台边界，不让业务去剥离。
        content = str(data.get("content") or "")
        bot_mention = re.search(rf"<@!?{re.escape(self.app_id)}>", content) is not None
        body = re.sub(rf"<@!?{re.escape(self.app_id)}>", "", content).strip()
        # 某些群事件只在正文保留 <@ID>，没有 mentions 数组。先用该群实际
        # 收到的 author.id/member_openid 别名映射回统一身份，避免昵称当 ID。
        seen_mentions = {mention.user_id for mention in mentions}
        for raw_id in re.findall(r"<@!?([^<>\s]+)>", body):
            identity = [raw_chat, raw_id] if group else ["p2p", raw_id]
            user_id = self._user_aliases.get(
                (raw_chat, raw_id), self.scope.key("user", json.dumps(identity))
            )
            if raw_id in {self.app_id, "all", "everyone"} or user_id in seen_mentions:
                continue
            seen_mentions.add(user_id)
            mentions.append(Mention(user_id, self._names.get(user_id)))
        reference = data.get("message_reference") or {}
        parent = reference.get("message_id")
        parent_id = self._message_key(raw_chat, str(parent)) if parent else None
        indexes = self._scene_indexes(data)
        if not parent_id and indexes.get("ref_msg_idx"):
            index = indexes["ref_msg_idx"]
            parent_id = self._reference_ids.get(
                (raw_chat, index), self._message_key(raw_chat, f"reference-index:{index}")
            )
        if not parent_id and data.get("message_type") == 103:
            parent_id = self._message_key(raw_chat, f"quoted:{data['id']}")
        quoted = self._quoted_resources(data.get("msg_elements") or []) if parent_id else ()
        return Message(
            message_id=self._message_key(raw_chat, str(data["id"])),
            chat_id=self.scope.key("chat", raw_chat),
            chat_type=kind,
            sender_id=sender_id,
            sender_name=author.get("username") or None,
            sender_is_bot=bool(author.get("bot")),
            mentioned_bot=(
                event_type in {"GROUP_AT_MESSAGE_CREATE", "C2C_MESSAGE_CREATE"} or bot_mention
            ),
            body_text=body,
            content_text=body,
            raw_content_type="text" if body or not resources else "image",
            reply_to_message_id=parent_id,
            resources=resources,
            mentions=tuple(mentions),
            native=quoted,
        )

    async def handle_event(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type not in MESSAGE_EVENTS:
            return
        message = self.normalize_message(event_type, data)
        if message.message_id in self._received_at:
            return
        # 线上只记录协议结构，定位 QQ 实际投递与文档差异；正文、昵称、
        # 用户/群标识、图片下载地址和 message_scene 的 auth_token 均不回显。
        def shape(element: dict[str, Any]) -> dict[str, Any]:
            return {
                "fields": sorted(element),
                "message_type": element.get("message_type"),
                "author_fields": sorted(element.get("author") or {}),
                "attachment_types": [
                    item.get("content_type") for item in element.get("attachments") or []
                ],
                "children": len(element.get("msg_elements") or []),
                "has_text": bool(str(element.get("content") or "").strip()),
            }

        LOGGER.info(
            "QQ message schema instance=%s event=%s shape=%s mention_fields=%s "
            "scene_keys=%s normalized_mentions=%d direct_images=%d quote_images=%d "
            "elements=%s parallel_shape=%s text_tags=%s",
            self.scope.instance_id,
            event_type,
            shape(data),
            [sorted(user) for user in data.get("mentions") or []],
            sorted(self._scene_indexes(data)),
            len(message.mentions),
            len(message.resources),
            len(message.native),
            [shape(element) for element in (data.get("msg_elements") or [])[:5]],
            {
                "type": type(data.get("parallel_message")).__name__,
                "fields": sorted(data["parallel_message"])
                if isinstance(data.get("parallel_message"), dict) else [],
            },
            re.findall(r"<([A-Za-z][A-Za-z_-]*|@!?)[^>]*>", str(data.get("content") or "")),
        )
        self._cache(self._messages, message.message_id, message)
        self._cache(self._received_at, message.message_id, datetime.now(UTC))
        self._cache(self._last_messages, message.chat_id, message.message_id)
        raw_chat = self.scope.raw("chat", message.chat_id)
        for field in ("id", "member_openid" if message.chat_type == "group" else "user_openid"):
            alias = (data.get("author") or {}).get(field)
            if isinstance(alias, str) and alias:
                self._cache(self._user_aliases, (raw_chat, alias), message.sender_id)
        index = self._scene_indexes(data).get("msg_idx")
        if index:
            self._cache(self._reference_ids, (raw_chat, index), message.message_id)
        for user_id, name in [
            (message.sender_id, message.sender_name),
            *((mention.user_id, mention.name) for mention in message.mentions),
        ]:
            if user_id and isinstance(name, str) and name.strip():
                self._cache(self._names, user_id, name.strip())
        await self._emit("message", message)

    async def _consume_events(self) -> None:
        # 接收与业务分开：图片推理、下载和发送不能阻挡心跳 ACK。
        while True:
            event_type, data = await self._queue.get()
            try:
                await self.handle_event(event_type, data)
            except Exception:
                LOGGER.exception("QQ event handling failed instance=%s", self.scope.instance_id)
            finally:
                self._queue.task_done()

    async def _heartbeat(self, socket, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            if not self._heartbeat_acked:
                LOGGER.warning("QQ heartbeat ACK missing; reconnecting")
                await socket.close(code=4000)
                return
            self._heartbeat_acked = False
            await socket.send(json.dumps({"op": 1, "d": self._sequence}))

    async def _session(self, socket) -> None:
        heartbeat = None
        try:
            async for raw in socket:
                frame = json.loads(raw)
                op = frame.get("op")
                data = frame.get("d") or {}
                if op == 10:
                    interval = float(
                        getcontext().divide(Decimal(str(data["heartbeat_interval"])), Decimal(1000))
                    )
                    if interval <= 0:
                        raise ValueError("Invalid QQ heartbeat interval")
                    token = await self._access_token()
                    if self._session_id is not None and self._sequence is not None:
                        identify = {
                            "op": 6,
                            "d": {
                                "token": f"QQBot {token}",
                                "session_id": self._session_id,
                                "seq": self._sequence,
                            },
                        }
                    else:
                        identify = {
                            "op": 2,
                            "d": {
                                "token": f"QQBot {token}",
                                "intents": GROUP_AND_C2C_INTENTS,
                                "shard": [0, 1],
                            },
                        }
                    await socket.send(json.dumps(identify))
                    self._heartbeat_acked = True
                    if heartbeat:
                        heartbeat.cancel()
                        with suppress(asyncio.CancelledError):
                            await heartbeat
                    heartbeat = asyncio.create_task(self._heartbeat(socket, interval))
                elif op == 11:
                    self._heartbeat_acked = True
                elif op == 1:
                    await socket.send(json.dumps({"op": 1, "d": self._sequence}))
                elif op == 0:
                    event_type = frame.get("t")
                    if event_type == "READY":
                        self._session_id = data["session_id"]
                        LOGGER.info("QQ connected app_id=%s instance=%s", self.app_id,
                                    self.scope.instance_id)
                    elif event_type in MESSAGE_EVENTS:
                        # 队列满时重连并从最后成功入队的 seq 恢复，不悄悄漏事件。
                        self._queue.put_nowait((event_type, data))
                    if isinstance(frame.get("s"), int):
                        self._sequence = frame["s"]
                elif op == 7:
                    return
                elif op == 9:
                    # 无效会话需要重新 Identify，不能无限重复 Resume。
                    self._session_id = None
                    self._sequence = None
                    return
        finally:
            if heartbeat:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await heartbeat

    async def connect(self) -> None:
        consumer = asyncio.create_task(self._consume_events())
        delays = itertools.chain((1, 2, 4, 8, 16), itertools.repeat(30))
        try:
            while not self._closed.is_set():
                try:
                    gateway = await self._call("GET", "/gateway")
                    if urlparse(gateway["url"]).scheme != "wss":
                        raise ValueError("QQ gateway must use WSS")
                    async with connect(
                        gateway["url"], max_size=4_194_304, open_timeout=20
                    ) as socket:
                        self._socket = socket
                        await self._session(socket)
                except Exception:
                    if not self._closed.is_set():
                        # 不回显 Token 或包含服务端鉴权信息的错误详情。
                        LOGGER.warning("QQ connection lost; reconnecting instance=%s",
                                       self.scope.instance_id)
                finally:
                    self._socket = None
                if not self._closed.is_set():
                    with suppress(TimeoutError):
                        await asyncio.wait_for(self._closed.wait(), timeout=next(delays))
        finally:
            consumer.cancel()
            with suppress(asyncio.CancelledError):
                await consumer

    async def close(self) -> None:
        self._closed.set()
        if self._socket:
            await self._socket.close()
        await self._api.aclose()
        await self._auth.aclose()
        await self._media.aclose()

    async def resolve(self, user_id: str, *, fallback: str = "神秘奶友") -> str:
        self.scope.raw("user", user_id)
        # 无独立昵称查询接口，但新版事件可携带昵称；旧事件无昵称时才兜底。
        return self._names.get(user_id, fallback)

    async def get_chat_members(self, chat_id: str) -> list[Member]:
        self.scope.raw("chat", chat_id)
        # 官方群机器人不提供普通群成员列表，也不投递普通成员入群通知。
        return []

    async def resolve_image_target(self, message: Message) -> ImageTarget:
        if message.resources:
            resource = message.resources[0]
            return ImageTarget(message.message_id, resource.file_key, "image", "resolved")
        if not message.reply_to_message_id:
            return ImageTarget()
        parent = self._messages.get(message.reply_to_message_id)
        if parent and parent.resources:
            resource = parent.resources[0]
            if (
                self.scope.owns("message", resource.file_key)
                and resource.file_key not in self._sent_sources
            ):
                if message.native:
                    resource = message.native[0]
                    return ImageTarget(parent.message_id, resource.file_key, "image", "resolved")
                return ImageTarget(reason="quote_unavailable")
            return ImageTarget(parent.message_id, resource.file_key, "image", "resolved")
        if message.native:
            resource = message.native[0]
            return ImageTarget(
                message.reply_to_message_id, resource.file_key, "image", "resolved"
            )
        return ImageTarget(reason="not_image" if parent else "quote_unavailable")

    async def download_target(self, target: ImageTarget) -> bytes | None:
        if target.file_key in self._sent_sources:
            return self._sent_sources[target.file_key]
        if not target.file_key or urlparse(target.file_key).scheme not in {"http", "https"}:
            return None
        try:
            async with self._media.stream("GET", target.file_key) as response:
                response.raise_for_status()
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > MAX_IMAGE_BYTES:
                        return None
                return bytes(chunks)
        except httpx.HTTPError:
            LOGGER.warning("QQ image download failed")
            return None

    @staticmethod
    def _png(source: bytes | str | Path) -> bytes:
        data = source if isinstance(source, bytes) else Path(source).read_bytes()
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("QQ image exceeds 20 MB")
        with Image.open(io.BytesIO(data)) as image:
            output = io.BytesIO()
            # 官方富媒体支持 PNG/JPG；GIF/WebP 取首帧，去掉不支持的 HDR 元数据。
            ImageOps.exif_transpose(image).convert("RGB").save(output, format="PNG")
        result = output.getvalue()
        if len(result) > MAX_IMAGE_BYTES:
            raise ValueError("QQ converted image exceeds 20 MB")
        return result

    @classmethod
    def _gallery(cls, items: list[dict[str, Any]]) -> bytes:
        import numpy as np

        if not items or len(items) > 9:
            raise ValueError("QQ gallery requires 1–9 images")
        tiles = []
        for item in items:
            with Image.open(io.BytesIO(cls._png(item["source"]))) as image:
                tile = Image.new("RGB", (320, 250), "white")
                tile.paste(ImageOps.pad(image, (320, 220), color="white"), (0, 0))
                # 图内显示素材编号，中文说明随同一条图片消息发送。
                label = re.search(r"#\d+", item.get("caption", ""))
                if label:
                    ImageDraw.Draw(tile).text((12, 226), label[0], fill="black")
                tiles.append(np.asarray(tile))
        blank = np.asarray(Image.new("RGB", (320, 250), "white"))
        iterator = iter(tiles)
        rows = itertools.zip_longest(*itertools.repeat(iterator, 3), fillvalue=blank)
        merged = np.concatenate([np.concatenate(row, axis=1) for row in rows], axis=0)
        output = io.BytesIO()
        Image.fromarray(merged).save(output, format="PNG")
        return output.getvalue()

    def _reply_context(self, chat_id: str, options: dict[str, Any]) -> dict[str, Any]:
        message_id = options.get("reply_to") or self._last_messages.get(chat_id)
        if message_id:
            raw_chat, raw_message = json.loads(self.scope.raw("message", message_id))
            if raw_chat != self.scope.raw("chat", chat_id):
                raise ValueError("QQ reply target belongs to another chat")
            received_at = self._received_at.get(message_id)
            # 回复机器人自己发出的图片时，msg_id 必须是新收到的用户消息，
            # 业务传入的 reply_to 始终是触发消息；不能拿图片 ID 当被动授权。
            if received_at and datetime.now(UTC) < received_at + timedelta(minutes=5):
                counter = self._reply_sequences.get(message_id)
                if counter is None:
                    counter = itertools.count(1)
                    self._cache(self._reply_sequences, message_id, counter)
                return {"msg_id": raw_message, "msg_seq": next(counter)}
            if options.get("reply_to"):
                raise ValueError("QQ passive reply window expired or trigger not received")
        if self.allow_proactive and self.scope.raw("chat", chat_id).startswith("group:"):
            return {}
        raise NotImplementedError("QQ needs a recent trigger or enabled group proactive messages")

    async def send(
        self, chat_id: str, content: dict[str, Any], options: dict[str, Any] | None = None
    ) -> SendResult:
        uploading = False
        try:
            kind, raw_id = self.scope.raw("chat", chat_id).split(":", 1)
            if kind not in {"group", "p2p"}:
                raise ValueError("Unsupported QQ target")
            route = f"/v2/{'groups' if kind == 'group' else 'users'}/{raw_id}"
            payload = self._reply_context(chat_id, options or {})
            payload["msg_type"] = 0
            text = (
                content.get("text") or content.get("fallback_text") or content.get("caption") or ""
            )
            source = None
            if "gallery" in content:
                uploading = True
                source = await asyncio.to_thread(self._gallery, content["gallery"])
                text = "\n".join(filter(None, [
                    text, *(item.get("caption", "") for item in content["gallery"])
                ]))
            elif "image" in content:
                uploading = True
                source = await asyncio.to_thread(self._png, content["image"]["source"])
            if source is not None:
                uploading = True
                file = await self._call(
                    "POST", f"{route}/files",
                    json={
                        "file_type": 1,
                        "file_data": base64.b64encode(source).decode(),
                        "srv_send_msg": False,
                    },
                )
                payload.update(msg_type=7, media={"file_info": file["file_info"]})
                uploading = False
            if text:
                payload["content"] = text
            if not text and source is None:
                raise NotImplementedError("QQ needs text or an image")
            result = await self._call("POST", f"{route}/messages", json=payload)
            message_id = self._message_key(f"{kind}:{raw_id}", str(result["id"]))
            resources = ()
            if source is not None:
                self._cache(self._sent_sources, message_id, source, limit=32)
                while sum(map(len, self._sent_sources.values())) > 33_554_432:
                    self._sent_sources.popitem(last=False)
                resources = (Resource("image", message_id),)
            self._cache(self._messages, message_id, Message(
                message_id=message_id, chat_id=chat_id, chat_type=kind,
                sender_id=self.scope.key("user", "bot"), sender_is_bot=True,
                body_text=text, content_text=text, resources=resources,
            ))
            return SendResult(True, message_id=message_id)
        except httpx.TimeoutException:
            return SendResult(False, error=SendError(
                ErrorCode.UPLOAD_FAILED if uploading else ErrorCode.SEND_TIMEOUT
            ))
        except QQAPIError as error:
            # 不把缺权限、限流、内容审核等错误一律当成“机器人已被移出群”。
            return SendResult(False, error=SendError(
                ErrorCode.UPLOAD_FAILED if uploading else ErrorCode.UNKNOWN, str(error)
            ))
        except NotImplementedError as error:
            return SendResult(False, error=SendError(ErrorCode.NOT_SUPPORTED, str(error)))
        except Exception:
            LOGGER.warning("QQ send failed instance=%s", self.scope.instance_id)
            return SendResult(False, error=SendError(
                ErrorCode.UPLOAD_FAILED if uploading else ErrorCode.UNKNOWN
            ))

    async def update_card(self, message_id: str, card: dict[str, Any]) -> SendResult:
        return SendResult(False, error=SendError(ErrorCode.NOT_SUPPORTED))

    def catalog_document(self, document_id: str, section_block_id: str):
        raise NotImplementedError("QQ has no Feishu catalog document")
