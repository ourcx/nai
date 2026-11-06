from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..context import MessageContext
from ..response import AsyncTask, MessageAction, ResponsePlan
from .base import Workload


InitializeGroup = Callable[[str], Awaitable[None]]


class InitializationFeature:
    name = "group_initialization"
    priority = 10_000
    workload = Workload.BACKGROUND

    def __init__(self, initialize_group: InitializeGroup) -> None:
        self.initialize_group = initialize_group

    def matches(self, context: MessageContext) -> bool:
        normalized = context.text.strip().replace(" ", "")
        return (
            normalized in {"初始化", "初始化群聊"}
            and context.platform == "feishu"
            and context.chat_type == "group"
            and context.mentioned_bot
        )

    async def handle(self, context: MessageContext) -> ResponsePlan:
        async def initialize() -> ResponsePlan | None:
            await self.initialize_group(context.chat_id)
            return None

        return ResponsePlan(
            messages=(
                MessageAction(
                    content={"text": "正在登记群成员并分析历史表情包，稍等一下。"},
                    options={"reply_to": context.message_id},
                ),
            ),
            tasks=(
                AsyncTask(
                    name="initialize_group",
                    lane="initialization",
                    run=initialize,
                    fallback_messages=(
                        MessageAction(
                            content={"text": "初始化任务正在忙，文字功能正常，请稍后再试。"},
                            options={"reply_to": context.message_id},
                        ),
                    ),
                ),
            ),
            handled=True,
        )
