from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class ErrorCode(StrEnum):
    TARGET_REVOKED = "target_revoked"
    UPLOAD_FAILED = "upload_failed"
    NOT_CONNECTED = "not_connected"
    NOT_SUPPORTED = "not_supported"
    SEND_TIMEOUT = "send_timeout"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SendError:
    code: str
    message: str = ""


@dataclass(frozen=True)
class SendResult:
    success: bool
    message_id: str | None = None
    error: SendError | None = None


@dataclass(frozen=True)
class Resource:
    type: str
    file_key: str


@dataclass(frozen=True)
class Mention:
    user_id: str
    name: str | None = None
    is_bot: bool = False

    @property
    def open_id(self) -> str:
        """兼容已有点名格式化函数；业务身份本身不再依赖飞书 open_id。"""
        return self.user_id


@dataclass(frozen=True)
class Message:
    message_id: str
    chat_id: str
    chat_type: str
    sender_id: str
    sender_name: str | None = None
    sender_is_bot: bool = False
    mentioned_bot: bool = False
    body_text: str = ""
    content_text: str = ""
    raw_content_type: str = "text"
    reply_to_message_id: str | None = None
    resources: tuple[Resource, ...] = ()
    mentions: tuple[Mention, ...] = ()
    # 原始 SDK 对象只供 adapter 读取引用消息；业务不能解析 native。
    native: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ImageTarget:
    message_id: str | None = None
    file_key: str | None = None
    resource_type: str | None = None
    reason: str = "missing"


@dataclass(frozen=True)
class Member:
    id: str
    name: str | None = None


@dataclass(frozen=True)
class MembersAdded:
    chat_id: str
    members: tuple[Member, ...]


@dataclass(frozen=True)
class CardAction:
    message_id: str
    value: dict[str, Any]
    tag: str = ""


Handler = Callable[[Any], Awaitable[None]]


class IdentityScope:
    """库存素材共享，聊天状态按平台和机器人实例隔离。

    仅迁移前的飞书实例允许 legacy=True，沿用原 ID，使旧日报、纠正引用
    和欢迎记录继续可用。其他实例的 ID 带版本、平台、实例及实体类型。
    """

    marker = "bot:v1:"

    def __init__(self, platform: str, instance_id: str, *, legacy: bool = False) -> None:
        self.platform = platform
        self.instance_id = instance_id
        self.legacy = legacy
        self.prefix = f"{self.marker}{platform}:{instance_id}:"

    def key(self, kind: str, raw_id: str) -> str:
        if self.legacy:
            return raw_id
        encoded = base64.urlsafe_b64encode(raw_id.encode()).decode()
        return f"{self.prefix}{kind}:{encoded}"

    def owns(self, kind: str, value: str) -> bool:
        if self.legacy:
            return not value.startswith(self.marker)
        return value.startswith(f"{self.prefix}{kind}:")

    def raw(self, kind: str, value: str) -> str:
        if not self.owns(kind, value):
            raise ValueError("Identifier belongs to another adapter")
        if self.legacy:
            return value
        encoded = value.removeprefix(f"{self.prefix}{kind}:")
        return base64.urlsafe_b64decode(encoded).decode()


class BotAdapter(Protocol):
    """平台接口：SDK、身份、下载、发送和事件转换都在此边界内。

    send 的 text/image/gallery 是统一内容；card 是可选飞书展示，
    不支持卡片的平台使用同一请求中的 fallback_text。
    """

    scope: IdentityScope

    def on(self, event: str, handler: Handler) -> None: ...
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def resolve_image_target(self, message: Message) -> ImageTarget: ...
    async def download_target(self, target: ImageTarget) -> bytes | None: ...
    async def resolve(self, user_id: str, *, fallback: str = "神秘奶友") -> str: ...
    async def get_chat_members(self, chat_id: str) -> list[Member]: ...
    async def send(
        self, chat_id: str, content: dict[str, Any], options: dict[str, Any] | None = None
    ) -> SendResult: ...
    async def update_card(self, message_id: str, card: dict[str, Any]) -> SendResult: ...
    def catalog_document(self, document_id: str, section_block_id: str) -> Any: ...
