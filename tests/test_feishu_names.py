from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from nailong_classifier.adapters.base import IdentityScope
from nailong_classifier.adapters.feishu import FeishuAdapter


def make_adapter(monkeypatch, *, members=(), error=None):
    calls = []

    class Channel:
        client = SimpleNamespace()

        def __init__(self, **kwargs):
            pass

        async def get_chat_members(self, chat_id, **kwargs):
            calls.append(chat_id)
            if error:
                raise error
            return list(members)

    async def denied(request):
        return SimpleNamespace(raw=SimpleNamespace(
            content=json.dumps({"code": 41050, "msg": "no user authority"}).encode(),
        ))

    monkeypatch.setattr("nailong_classifier.adapters.feishu.FeishuChannel", Channel)
    adapter = FeishuAdapter(IdentityScope("feishu", "test"), app_id="dummy", app_secret="dummy")
    adapter.channel.client.arequest = denied
    return adapter, calls


def message(sender="user", *, name=None, mentions=()):
    return SimpleNamespace(
        message_id="message", chat_id="chat", chat_type="group", sender_id=sender,
        sender_name=name, sender_is_bot=False, mentioned_bot=True, body_text="求赐名",
        content_text="求赐名", raw_content_type="text", reply_to_message_id=None,
        resources=[], mentions=mentions, raw={},
    )


def test_unreadable_contact_uses_group_member_name_and_shared_cache(monkeypatch):
    adapter, calls = make_adapter(monkeypatch, members=(
        SimpleNamespace(id="user", name="王小明"),
        SimpleNamespace(id="other", name="李雷"),
    ))
    first = adapter.normalize_message(message())
    second = adapter.normalize_message(message("other"))

    async def exercise():
        names = await asyncio.gather(
            adapter.resolve(first.sender_id), adapter.resolve(second.sender_id),
        )
        assert names == ["王小明", "李雷"]
        assert calls == ["chat"]
        assert adapter.normalize_message(message()).sender_name == "王小明"

    asyncio.run(exercise())


def test_mention_name_resolves_before_contact_or_member_queries(monkeypatch):
    adapter, calls = make_adapter(monkeypatch)
    adapter.normalize_message(message("someone", mentions=(
        SimpleNamespace(open_id="user", name="王小明", is_bot=False),
        SimpleNamespace(open_id="bot", name="机器人", is_bot=True),
    )))
    normalized = adapter.normalize_message(message())
    assert normalized.sender_name == "王小明"
    assert asyncio.run(adapter.resolve(normalized.sender_id)) == "王小明"
    assert not calls
    assert adapter.contacts.cached_name("bot") is None


@pytest.mark.parametrize("error", [RuntimeError("permission denied"), TimeoutError()])
def test_unavailable_member_api_returns_fallback_and_does_not_repeat_requests(monkeypatch, error):
    adapter, calls = make_adapter(monkeypatch, error=error)
    normalized = adapter.normalize_message(message())

    async def exercise():
        assert await adapter.resolve(normalized.sender_id, fallback="") == ""
        assert await adapter.resolve(normalized.sender_id, fallback="神秘奶友") == "神秘奶友"
        assert calls == ["chat"]
        adapter.normalize_message(message(name="李雷"))
        assert await adapter.resolve(normalized.sender_id) == "李雷"
        assert calls == ["chat"]

    asyncio.run(exercise())


def test_newcomer_without_contact_access_can_get_group_member_name(monkeypatch):
    adapter, calls = make_adapter(monkeypatch, members=(SimpleNamespace(id="user", name="李雷"),))
    added = adapter.normalize_members_added({
        "chat_id": "chat", "users": [{"user_id": {"open_id": "user"}}],
    })
    assert asyncio.run(adapter.resolve(added.members[0].id)) == "李雷"
    assert calls == ["chat"]
