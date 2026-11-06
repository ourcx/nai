from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class MessageAction:
    """由执行层发送或更新的一条消息。"""

    content: Mapping[str, Any]
    action: Literal["send", "update"] = "send"
    chat_id: str | None = None
    message_id: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


TaskRunner = Callable[[], Awaitable["ResponsePlan | None"]]


@dataclass(frozen=True)
class AsyncTask:
    """延后执行的任务；返回值可继续产生响应计划。"""

    name: str
    run: TaskRunner
    lane: str = "background"
    fallback_messages: tuple[MessageAction, ...] = ()


@dataclass(frozen=True)
class ResponsePlan:
    """一个功能产生的消息动作和异步任务。"""

    messages: tuple[MessageAction, ...] = ()
    tasks: tuple[AsyncTask, ...] = ()
    handled: bool = False
