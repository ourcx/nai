from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from ..context import MessageContext
from ..response import ResponsePlan
from .base import Workload


SpeechVoice = Literal["nailong", "naiwa"]


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    voice: SpeechVoice


HandleRequest = Callable[[MessageContext, SpeechRequest], Awaitable[ResponsePlan]]


def _parse_request(text: str) -> SpeechRequest | None:
    stripped = text.strip()
    if stripped == "奶蛙大笑":
        return SpeechRequest(text="哈哈哈哈哈哈哈哈！", voice="naiwa")

    naiwa_match = re.fullmatch(r"奶蛙说(?:[\s：:]+(.*))?", stripped, flags=re.DOTALL)
    if naiwa_match:
        return SpeechRequest(text=(naiwa_match.group(1) or "").strip(), voice="naiwa")

    nailong_match = re.fullmatch(r"奶龙说(?:[\s：:]+(.*))?", stripped, flags=re.DOTALL)
    if nailong_match:
        return SpeechRequest(text=(nailong_match.group(1) or "").strip(), voice="nailong")
    return None


class SpeechFeature:
    name = "speech"
    priority = 100
    workload = Workload.INLINE

    def __init__(self, handle_request: HandleRequest) -> None:
        self.handle_request = handle_request

    def _request_for(self, context: MessageContext) -> SpeechRequest | None:
        if context.platform not in {"feishu", "qq"}:
            return None
        if context.chat_type != "p2p" and not context.mentioned_bot:
            return None
        return _parse_request(context.text)

    def matches(self, context: MessageContext) -> bool:
        return self._request_for(context) is not None

    async def handle(self, context: MessageContext) -> ResponsePlan:
        request = self._request_for(context)
        if request is None:
            return ResponsePlan()
        return await self.handle_request(context, request)
