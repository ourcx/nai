from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from ..context import MessageContext
from ..response import ResponsePlan
from .base import Workload


AdminAction = Literal["quiet", "resume"]
HandleAdmin = Callable[[MessageContext, AdminAction], Awaitable[ResponsePlan]]


@dataclass(frozen=True)
class AdminRequest:
    action: AdminAction


class AdminFeature:
    name = "admin"
    priority = 7_000
    workload = Workload.INLINE

    def __init__(self, handle_admin: HandleAdmin) -> None:
        self.handle_admin = handle_admin

    def _request_for(self, context: MessageContext) -> AdminRequest | None:
        if not context.mentioned_bot:
            return None
        normalized = context.text.strip().replace(" ", "")
        if normalized in {"安静一点", "少说话", "闭嘴"}:
            return AdminRequest("quiet")
        if normalized in {"开始说话", "活跃一点", "别安静"}:
            return AdminRequest("resume")
        return None

    def matches(self, context: MessageContext) -> bool:
        return self._request_for(context) is not None

    async def handle(self, context: MessageContext) -> ResponsePlan:
        request = self._request_for(context)
        if request is None:
            return ResponsePlan()
        return await self.handle_admin(context, request.action)
