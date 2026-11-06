from __future__ import annotations

from collections.abc import Callable

from ..context import MessageContext
from ..response import ResponsePlan
from .base import Workload


FindLink = Callable[[str], str | None]
BuildPlan = Callable[[MessageContext, str], ResponsePlan]


class SocialImportFeature:
    name = "social_import"
    priority = 8_000
    workload = Workload.BACKGROUND

    def __init__(self, find_link: FindLink, build_plan: BuildPlan) -> None:
        self.find_link = find_link
        self.build_plan = build_plan

    def matches(self, context: MessageContext) -> bool:
        return (
            context.mentioned_bot or context.chat_type == "p2p"
        ) and self.find_link(context.text) is not None

    async def handle(self, context: MessageContext) -> ResponsePlan:
        url = self.find_link(context.text)
        if url is None:
            return ResponsePlan()
        return self.build_plan(context, url)
