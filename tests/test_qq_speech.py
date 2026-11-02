from __future__ import annotations

import asyncio
import base64
import io
import wave
from datetime import time
from types import SimpleNamespace

import httpx
import pytest
from test_qq_adapter import event, mock_adapter

from nailong_classifier.adapters.base import ErrorCode
from nailong_classifier.application import bot
from nailong_classifier.application.services import BackendServices, SerializedPredictor
from nailong_classifier.integrations.speech import SpeechAudio
from nailong_classifier.persistence.storage import BotStore


def voice_wav(*, rate=32000):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\x12\x34" * (rate // 2))
    return output.getvalue()


@pytest.mark.parametrize("kind", ["group", "p2p"])
def test_qq_voice_uses_native_media_and_original_passive_reply(kind):
    async def run():
        adapter, calls = await mock_adapter()
        event_type = "GROUP_AT_MESSAGE_CREATE" if kind == "group" else "C2C_MESSAGE_CREATE"
        incoming = event("奶龙说 你好")
        message = adapter.normalize_message(event_type, incoming)
        wav = voice_wav()
        try:
            await adapter.handle_event(event_type, incoming)
            result = await adapter.send(
                message.chat_id,
                {"audio": {"source": wav, "duration_ms": 500}},
                {"reply_to": message.message_id},
            )
            assert result.success
            route = "/v2/groups/group-a" if kind == "group" else "/v2/users/user-a"
            assert [path for path, _ in calls] == [route + "/files", route + "/messages"]
            upload = calls[0][1]
            assert upload["file_type"] == 3 and upload["srv_send_msg"] is False
            assert base64.b64decode(upload["file_data"]) == wav
            assert calls[1][1] == {
                "msg_type": 7,
                "media": {"file_info": "test-file"},
                "msg_id": "source",
                "msg_seq": 1,
            }
            assert adapter._messages[result.message_id].resources[0].type == "audio"
            assert result.message_id not in adapter._sent_sources
            quoted = adapter.normalize_message(event_type, event(id="quote", parent="reply-1"))
            target = await adapter.resolve_image_target(quoted)
            assert target.file_key is None and target.reason == "not_image"
        finally:
            await adapter.close()

    asyncio.run(run())


@pytest.mark.parametrize("source", [b"OggS-opus", b"RIFF", voice_wav(rate=16000), None])
def test_qq_rejects_invalid_voice_before_upload(source):
    async def run():
        adapter, calls = await mock_adapter()
        try:
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", event())
            result = await adapter.send(
                adapter.scope.key("chat", "group:group-a"), {"audio": {"source": source}}
            )
            assert not result.success and result.error.code == ErrorCode.UPLOAD_FAILED
            assert calls == []
        finally:
            await adapter.close()

    asyncio.run(run())


def test_qq_voice_upload_rejection_does_not_send_or_retry():
    async def run():
        seen = []

        def api(request):
            seen.append(request.url.path)
            return httpx.Response(400, json={"code": 40034001})

        adapter, _ = await mock_adapter(api=api)
        try:
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", event())
            result = await adapter.send(
                adapter.scope.key("chat", "group:group-a"),
                {"audio": {"source": voice_wav()}},
            )
            assert result.error.code == ErrorCode.UPLOAD_FAILED
            assert seen == ["/v2/groups/group-a/files"]
        finally:
            await adapter.close()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["group", "p2p", "unmentioned"])
def test_qq_speech_command_and_cooldown_use_shared_business_handler(tmp_path, monkeypatch, kind):
    monkeypatch.setattr("sys.argv", ["nailong-bot"])
    monkeypatch.delenv("BOT_ADAPTERS_CONFIG", raising=False)
    args = bot.parse_args()
    args.state_dir = tmp_path
    args.catalog_document_id = args.catalog_section_block_id = None
    args.random_push_probability = args.douyin_video_push_probability = 0
    generated = []
    wav = voice_wav()

    async def synthesize(text):
        generated.append(text)
        return SpeechAudio(b"OggS-not-for-QQ", 500, wav)

    async def run():
        adapter, calls = await mock_adapter()
        event_type = {
            "group": "GROUP_AT_MESSAGE_CREATE",
            "p2p": "C2C_MESSAGE_CREATE",
            "unmentioned": "GROUP_MESSAGE_CREATE",
        }[kind]

        async def connect():
            for index in range(2):
                incoming = event("奶龙说 你好", id=f"command-{index}")
                if kind != "group":
                    incoming["content"] = "奶龙说 你好"
                await adapter.handle_event(event_type, incoming)

        monkeypatch.setattr(adapter, "connect", connect)
        predictor = SerializedPredictor(SimpleNamespace(device="cpu", threshold=0.5))
        services = BackendServices(
            predictor, BotStore(tmp_path), speech_client=SimpleNamespace(synthesize=synthesize)
        )
        try:
            await bot.run_adapter(args, adapter, services)
            if kind == "unmentioned":
                assert generated == [] and calls == []
            else:
                assert generated == ["你好"]
                uploads = [data for path, data in calls if path.endswith("/files")]
                assert len(uploads) == 1 and uploads[0]["file_type"] == 3
                assert base64.b64decode(uploads[0]["file_data"]) == wav
                messages = [data for path, data in calls if path.endswith("/messages")]
                assert messages[0]["msg_type"] == 7
                assert messages[0]["msg_id"] == "command-0"
                assert "30 秒" in messages[1]["content"]
                assert messages[1]["msg_id"] == "command-1"
        finally:
            await adapter.close()

    asyncio.run(run())


def test_qq_help_only_advertises_speech_when_backend_enabled():
    assert "奶龙说" in bot.format_help_message(time(23, 55), platform="qq", speech_enabled=True)
    assert "奶龙说" not in bot.format_help_message(time(23, 55), platform="qq")
