from __future__ import annotations

import re
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
        text = context.text.strip()
        remaining = text
        for mention in context.mentions:
            if (
                (mention.is_bot or (
                    context.mentioned_bot and len(context.mentions) == 1
                ))
                and mention.name
            ):
                remaining = remaining.replace(f"@{mention.name}", "").replace(
                    mention.name, "",
                )
        if context.mentioned_bot:
            remaining = remaining.replace("@我是奶龙", "").replace("我是奶龙", "")
        mention_only = (
            context.mentioned_bot
            and not re.sub(r"[\s@，,：:]+", "", remaining)
        )
        return (
            context.content_type in {"text", "post"}
            and (context.mentioned_bot or context.chat_type == "p2p")
            and (
                mention_only
                or re.fullmatch(
                    r"(?:/?help|/?帮助|使用帮助|奶龙帮助)(?:[\s：:].*)?",
                    text,
                    flags=re.IGNORECASE,
                ) is not None
            )
        )

    async def handle(self, context: MessageContext) -> ResponsePlan:
        return self.build_plan(context)
