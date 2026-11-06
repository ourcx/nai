from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from ..context import MessageContext
from ..response import ResponsePlan


class Workload(StrEnum):
    INLINE = "inline"
    BACKGROUND = "background"


class FeatureHandler(Protocol):
    """一个可独立匹配并处理消息的主功能。"""

    name: str
    priority: int
    workload: Workload

    def matches(self, context: MessageContext) -> bool: ...

    async def handle(self, context: MessageContext) -> ResponsePlan: ...
