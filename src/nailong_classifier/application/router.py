from __future__ import annotations

from collections.abc import Iterable

from .context import MessageContext
from .features import FeatureHandler
from .response import ResponsePlan


class FeatureRegistry:
    """按优先级选择首个匹配的主功能。"""

    def __init__(self, handlers: Iterable[FeatureHandler] = ()) -> None:
        self._handlers = sorted(handlers, key=lambda handler: handler.priority, reverse=True)

    @property
    def handlers(self) -> tuple[FeatureHandler, ...]:
        return tuple(self._handlers)

    def register(self, handler: FeatureHandler) -> None:
        self._handlers.append(handler)
        self._handlers.sort(key=lambda item: item.priority, reverse=True)

    def select(self, context: MessageContext) -> FeatureHandler | None:
        return next((handler for handler in self._handlers if handler.matches(context)), None)


class FeatureRouter:
    def __init__(self, registry: FeatureRegistry) -> None:
        self.registry = registry

    async def route(self, context: MessageContext) -> ResponsePlan:
        handler = self.registry.select(context)
        if handler is None:
            return ResponsePlan()
        return await handler.handle(context)
