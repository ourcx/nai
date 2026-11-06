from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..context import MessageContext
from ..response import ResponsePlan
from .base import Workload


Matcher = Callable[[MessageContext], bool]
Handler = Callable[[MessageContext], Awaitable[ResponsePlan]]


class CallbackFeature:
    """用统一协议注册已经存在的业务处理函数。"""

    def __init__(
        self,
        name: str,
        priority: int,
        matcher: Matcher,
        handler: Handler,
        *,
        workload: Workload = Workload.INLINE,
    ) -> None:
        self.name = name
        self.priority = priority
        self.matches = matcher
        self.handle = handler
        self.workload = workload
