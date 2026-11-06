from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nailong_classifier.adapters.base import IdentityScope
from nailong_classifier.adapters.feishu import FeishuAdapter


def image_adapter(monkeypatch, *, fail_upload=False):
    uploads, sends = [], []

    async def upload(source, *, kind):
        uploads.append((source, kind))
        if fail_upload:
            raise RuntimeError("upload rejected")
        return "image-key"

    async def send(chat_id, payload, options):
        sends.append((chat_id, payload, options))
        return SimpleNamespace(success=True, message_id="reply", error=None)

    channel = SimpleNamespace(client=SimpleNamespace(), upload_media=upload, send=send)
    monkeypatch.setattr("nailong_classifier.adapters.feishu.FeishuChannel", lambda **_: channel)
    adapter = FeishuAdapter(IdentityScope("feishu", "test"), app_id="test", app_secret="test")
    return adapter, uploads, sends


def test_explicit_image_card_preserves_bytes_and_reply_target(monkeypatch):
    adapter, uploads, sends = image_adapter(monkeypatch)
    content = {"image": {"source": b"original-image-bytes", "presentation": "card"}}
    result = asyncio.run(adapter.send(
        adapter.scope.key("chat", "chat-a"), content,
        {"reply_to": adapter.scope.key("message", "source-a")},
    ))
    assert result.success
    assert len(uploads) == 1 and uploads[0][0].buffer == b"original-image-bytes"
    assert uploads[0][1] == "image"
    chat, payload, options = sends[0]
    assert chat == "chat-a" and options == {"reply_to": "source-a"}
    assert payload["card"]["schema"] == "2.0"
    assert payload["card"]["body"]["elements"] == [{
        "tag": "img", "img_key": "image-key",
        "alt": {"tag": "plain_text", "content": "图片"},
        "scale_type": "fit_horizontal", "preview": True,
    }]
    assert content == {"image": {"source": b"original-image-bytes", "presentation": "card"}}


def test_regular_image_keeps_existing_sdk_contract(monkeypatch):
    adapter, uploads, sends = image_adapter(monkeypatch)
    content = {"image": {"source": b"normal-image"}, "caption": "接龙说明"}
    result = asyncio.run(adapter.send(adapter.scope.key("chat", "chat-a"), content))
    assert result.success and uploads == []
    assert sends == [("chat-a", content, None)]


def test_card_upload_failure_does_not_send_a_broken_card(monkeypatch):
    adapter, uploads, sends = image_adapter(monkeypatch, fail_upload=True)
    result = asyncio.run(adapter.send(
        adapter.scope.key("chat", "chat-a"),
        {"image": {"source": b"image", "presentation": "card"}},
    ))
    assert not result.success and result.error.code == "upload_failed"
    assert len(uploads) == 1 and sends == []


@pytest.mark.parametrize("source", [None, 123, True])
def test_invalid_card_source_is_not_uploaded(monkeypatch, source):
    adapter, uploads, sends = image_adapter(monkeypatch)
    result = asyncio.run(adapter.send(
        adapter.scope.key("chat", "chat-a"),
        {"image": {"source": source, "presentation": "card"}},
    ))
    assert not result.success and result.error.code == "invalid_image"
    assert uploads == sends == []
