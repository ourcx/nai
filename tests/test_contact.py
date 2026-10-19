from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from nailong_classifier.contact import ContactResolver


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.raw = SimpleNamespace(content=json.dumps(payload).encode("utf-8"))


class FakeClient:
    """记录请求次数的假 client，按预设 payload 返回。"""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    async def arequest(self, request):  # noqa: ARG002 - 签名对齐真实 client
        self.calls += 1
        return FakeResponse(self.payload)


def test_contact_resolver_returns_name_and_caches() -> None:
    client = FakeClient({"code": 0, "data": {"user": {"name": "邵思博"}}})
    resolver = ContactResolver(client)

    first = asyncio.run(resolver.resolve("ou_123"))
    second = asyncio.run(resolver.resolve("ou_123"))

    assert first == "邵思博"
    assert second == "邵思博"
    # 第二次命中缓存，不再打接口。
    assert client.calls == 1


def test_contact_resolver_falls_back_without_permission() -> None:
    client = FakeClient({"code": 99991672, "msg": "permission denied"})
    resolver = ContactResolver(client)

    name = asyncio.run(resolver.resolve("ou_x", fallback="神秘奶友"))

    assert name == "神秘奶友"


def test_contact_resolver_empty_open_id_returns_fallback() -> None:
    client = FakeClient({"code": 0, "data": {"user": {"name": "不该被调用"}}})
    resolver = ContactResolver(client)

    assert asyncio.run(resolver.resolve("", fallback="神秘奶友")) == "神秘奶友"
    assert client.calls == 0
