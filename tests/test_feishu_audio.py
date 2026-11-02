from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nailong_classifier.adapters.base import IdentityScope
from nailong_classifier.adapters.feishu import FeishuAdapter


def adapter_with_upload(monkeypatch, *, upload_code=0):
    uploads, sends = [], []

    async def upload(request):
        body = request.request_body
        uploads.append(
            {
                "type": body.file_type,
                "name": body.file_name,
                "duration": body.duration,
                "bytes": body.file.read(),
            }
        )
        return SimpleNamespace(code=upload_code, data=SimpleNamespace(file_key="file-a"))

    async def send(chat_id, payload, options):
        sends.append((chat_id, payload, options))
        return SimpleNamespace(success=True, message_id="message-a", error=None)

    channel = SimpleNamespace(
        client=SimpleNamespace(
            im=SimpleNamespace(v1=SimpleNamespace(file=SimpleNamespace(acreate=upload)))
        ),
        send=send,
    )
    monkeypatch.setattr("nailong_classifier.adapters.feishu.FeishuChannel", lambda **_: channel)
    adapter = FeishuAdapter(IdentityScope("feishu", "test"), app_id="test", app_secret="test")
    return adapter, uploads, sends


def test_audio_upload_carries_milliseconds_and_preserves_reply_routing(monkeypatch):
    adapter, uploads, sends = adapter_with_upload(monkeypatch)
    result = asyncio.run(
        adapter.send(
            adapter.scope.key("chat", "chat-a"),
            {"audio": {"source": b"OggS-test", "duration_ms": 3410}},
            {"reply_to": adapter.scope.key("message", "source-a")},
        )
    )
    assert result.success
    assert result.message_id == adapter.scope.key("message", "message-a")
    assert uploads == [
        {"type": "opus", "name": "nailong.opus", "duration": 3410, "bytes": b"OggS-test"}
    ]
    chat, payload, options = sends[0]
    assert chat == "chat-a" and options == {"reply_to": "source-a"}
    assert payload["audio"]["source"].kind == "key"
    assert payload["audio"]["source"].key == "file-a"


def test_upload_failure_does_not_send_empty_or_wrong_duration_audio(monkeypatch):
    adapter, uploads, sends = adapter_with_upload(monkeypatch, upload_code=999)
    result = asyncio.run(
        adapter.send(
            adapter.scope.key("chat", "chat-a"),
            {"audio": {"source": b"OggS-test", "duration_ms": 3410}},
        )
    )
    assert not result.success and result.error.code == "upload_failed"
    assert len(uploads) == 1 and sends == []


@pytest.mark.parametrize("duration", [0, -1, True, "3410", 100000])
def test_invalid_duration_is_not_uploaded(monkeypatch, duration):
    adapter, uploads, sends = adapter_with_upload(monkeypatch)
    result = asyncio.run(
        adapter.send(
            adapter.scope.key("chat", "chat-a"),
            {"audio": {"source": b"OggS-test", "duration_ms": duration}},
        )
    )
    assert not result.success
    assert uploads == [] and sends == []


def test_audio_without_duration_keeps_existing_sdk_contract(monkeypatch):
    adapter, uploads, sends = adapter_with_upload(monkeypatch)
    payload = {"audio": {"source": b"other-caller"}}
    result = asyncio.run(adapter.send(adapter.scope.key("chat", "chat-a"), payload))
    assert result.success and uploads == []
    assert sends == [("chat-a", payload, None)]
