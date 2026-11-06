from __future__ import annotations

from collections.abc import Callable

from ..context import MessageContext
from ..response import ResponsePlan
from .base import Workload


BuildPlan = Callable[[MessageContext], ResponsePlan]


class HelpFeature:
    name = "help"
    priority = 100_000
    workload = Workload.INLINE

    def __init__(self, build_plan: BuildPlan) -> None:
        self.build_plan = build_plan

    def matches(self, context: MessageContext) -> bool:
        return (
            context.content_type in {"text", "post"}
            and (context.mentioned_bot or context.chat_type == "p2p")
            and context.text.strip().lower()
            in {"help", "/help", "帮助", "/帮助", "使用帮助", "奶龙帮助"}
        )

    async def handle(self, context: MessageContext) -> ResponsePlan:
        return self.build_plan(context)
