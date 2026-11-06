from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ResourceRef:
    """平台适配层提取出的资源引用。"""

    type: str
    key: str


@dataclass(frozen=True)
class MentionRef:
    user_id: str
    name: str | None = None
    is_bot: bool = False


@dataclass(frozen=True)
class MessageContext:
    """功能层使用的标准消息，不包含任何平台 SDK 对象。"""

    platform: str
    message_id: str
    chat_id: str
    sender_id: str
    text: str = ""
    chat_type: str = ""
    content_type: str = "text"
    sender_name: str | None = None
    mentioned_bot: bool = False
    reply_to_message_id: str | None = None
    resources: tuple[ResourceRef, ...] = ()
    mentions: tuple[MentionRef, ...] = ()
    event_type: str = "message"
    metadata: Mapping[str, Any] = field(default_factory=dict)
