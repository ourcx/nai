from __future__ import annotations

import asyncio
import io
import wave
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from test_hdr_bot import run_mock_bot

from nailong_classifier.application import bot
from nailong_classifier.integrations import speech
from nailong_classifier.integrations.speech import NailongSpeechClient, SpeechAudio, SpeechError

HTTP_CLIENT = httpx.AsyncClient


def wav_content():
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(32000)
        samples = np.round(np.sin(np.arange(16000) * 2 * np.pi * 440 / 32000) * 8000).astype("<i2")
        audio.writeframes(samples.tobytes())
    return output.getvalue()


def test_opus_is_decodable_and_preserves_audio_duration():
    av = pytest.importorskip("av")
    encoded = speech.encode_feishu_opus(wav_content())
    assert encoded.duration_ms == 850
    with wave.open(io.BytesIO(encoded.wav)) as preserved:
        assert preserved.getframerate() == 32000
        assert preserved.getnframes() == 27200
        pcm_bytes = preserved.readframes(preserved.getnframes())
        assert pcm_bytes[: 11200 * 2] == b"\0" * (11200 * 2)
        with wave.open(io.BytesIO(wav_content())) as original:
            assert pcm_bytes[11200 * 2 :] == original.readframes(original.getnframes())
    assert encoded.opus.startswith(b"OggS")
    with av.open(io.BytesIO(encoded.opus)) as container:
        assert container.streams.audio[0].codec_context.name == "opus"
        frames = list(container.decode(audio=0))
        pcm = np.concatenate([frame.to_ndarray().reshape(-1) for frame in frames])
        assert len(pcm) / 48000 == pytest.approx(0.85, abs=0.02)
        assert np.max(np.abs(pcm[: round(0.30 * 48000)])) < 0.002
        # 静音后仍保留完整的原波形，不能用裁掉句首换取播放缓冲。
        assert np.sqrt(np.mean(pcm[round(0.40 * 48000) :] ** 2)) > 0.1


def mock_http(monkeypatch, handler):
    monkeypatch.setattr(
        speech.httpx,
        "AsyncClient",
        lambda **kwargs: HTTP_CLIENT(transport=httpx.MockTransport(handler), **kwargs),
    )


def test_client_posts_to_configured_endpoint_and_validates_audio(monkeypatch):
    sent = []

    def handle(request):
        sent.append(request)
        return httpx.Response(200, headers={"content-type": "audio/wav"}, content=wav_content())

    mock_http(monkeypatch, handle)
    monkeypatch.setattr(speech, "encode_feishu_opus", lambda content: SpeechAudio(content, 500))
    client = NailongSpeechClient("http://127.0.0.1:19880", "private-test-token")
    audio = asyncio.run(client.synthesize("奶家人你好"))
    assert audio.duration_ms == 500
    assert str(sent[0].url) == "http://127.0.0.1:19880/tts"
    assert sent[0].headers["authorization"] == "Bearer private-test-token"
    assert "private-test-token" not in repr(client)


@pytest.mark.parametrize(
    "status,expected", [(429, "正在说话"), (400, "中文台词"), (503, "暂时不可用")]
)
def test_client_failure_is_actionable_without_backend_details(monkeypatch, status, expected):
    mock_http(monkeypatch, lambda request: httpx.Response(status, text="private backend details"))
    with pytest.raises(SpeechError, match=expected) as error:
        asyncio.run(NailongSpeechClient("http://127.0.0.1:19880", "token").synthesize("你好"))
    assert "private backend details" not in str(error.value)


def test_client_refuses_oversized_and_non_audio_responses(monkeypatch):
    for headers, content in [
        ({"content-type": "application/json"}, b"{}"),
        ({"content-type": "audio/wav"}, b"x" * (speech.MAX_WAV_BYTES + 1)),
    ]:
        mock_http(
            monkeypatch,
            lambda request, h=headers, c=content: httpx.Response(200, headers=h, content=c),
        )
        with pytest.raises(SpeechError):
            asyncio.run(NailongSpeechClient("http://127.0.0.1:19880", "token").synthesize("你好"))


@pytest.mark.parametrize(
    "value,expected",
    [
        ("奶龙说 你好", "你好"),
        (" 奶龙说：你好！ ", "你好！"),
        ("奶龙说", ""),
        ("奶龙说话很好笑", None),
        ("你好", None),
    ],
)
def test_speech_command_is_explicit(value, expected):
    assert bot.parse_speech_command(value) == expected


@pytest.mark.parametrize("chat_type,mentioned", [("group", True), ("p2p", False)])
def test_existing_feishu_bot_replies_with_audio(tmp_path, monkeypatch, chat_type, mentioned):
    calls = []
    uploads = []

    async def synthesize(text):
        calls.append(text)
        return SpeechAudio(b"OggS-test", 1234)

    monkeypatch.setattr(
        bot.NailongSpeechClient, "from_environment", lambda: SimpleNamespace(synthesize=synthesize)
    )
    sent, _ = run_mock_bot(
        tmp_path,
        monkeypatch,
        text="奶龙说 你好！",
        mentioned_bot=mentioned,
        chat_type=chat_type,
        random_push_probability=0,
        audio_uploads=uploads,
    )
    assert calls == ["你好！"]
    assert uploads == [(b"OggS-test", 1234)]
    assert len(sent) == 1
    assert sent[0][0]["audio"]["source"].key == "test-audio"
    assert sent[0][1] == {"reply_to": "source-id"}


def test_normal_group_chat_does_not_generate_speech(tmp_path, monkeypatch):
    async def forbidden(_text):
        raise AssertionError("normal group chat triggered speech")

    monkeypatch.setattr(
        bot.NailongSpeechClient, "from_environment", lambda: SimpleNamespace(synthesize=forbidden)
    )
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text="奶龙说 你好", mentioned_bot=False, random_push_probability=0
    )
    assert sent == []


def test_speech_service_failure_replies_to_same_message(tmp_path, monkeypatch):
    async def busy(_text):
        raise SpeechError("奶龙正在说话，等几秒再叫我。")

    monkeypatch.setattr(
        bot.NailongSpeechClient, "from_environment", lambda: SimpleNamespace(synthesize=busy)
    )
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text="奶龙说 你好", mentioned_bot=True, random_push_probability=0
    )
    assert sent == [({"text": "奶龙正在说话，等几秒再叫我。"}, {"reply_to": "source-id"})]


def test_speech_limiter_blocks_user_across_groups_and_chat_bursts():
    now = [0.0]
    limiter = speech.SpeechRequestLimiter(clock=lambda: now[0])
    with limiter.admit(chat_id="群一", user_id="甲"):
        assert limiter.pending == 1
    assert limiter.pending == 0
    with pytest.raises(SpeechError, match="30 秒"), limiter.admit(chat_id="群二", user_id="甲"):
        pytest.fail("same user bypassed cooldown via another group")
    with pytest.raises(SpeechError, match="5 秒"), limiter.admit(chat_id="群一", user_id="乙"):
        pytest.fail("same group bypassed cooldown via another user")
    now[0] = 5
    with limiter.admit(chat_id="群一", user_id="乙"):
        pass
    now[0] = 30
    with limiter.admit(chat_id="群二", user_id="甲"):
        pass


def test_speech_limiter_bounds_pending_and_recovers_after_failure():
    limiter = speech.SpeechRequestLimiter()
    with ExitStack() as stack:
        for index in range(4):
            stack.enter_context(limiter.admit(chat_id=str(index), user_id=str(index)))
        with (
            pytest.raises(SpeechError, match="队列已满"),
            limiter.admit(chat_id="第五群", user_id="第五人"),
        ):
            pytest.fail("pending limit exceeded")
    assert limiter.pending == 0
    with pytest.raises(RuntimeError), limiter.admit(chat_id="失败群", user_id="失败人"):
        raise RuntimeError("backend failed")
    assert limiter.pending == 0
    with limiter.admit(chat_id="新群", user_id="新人"):
        pass


def test_bot_replies_to_rate_limit_without_calling_backend(tmp_path, monkeypatch):
    async def forbidden(_text):
        raise AssertionError("rate limited request reached backend")

    @contextmanager
    def blocked(*_args, **_kwargs):
        raise SpeechError("每人 30 秒内只能生成一次语音，稍等一下。")
        yield

    monkeypatch.setattr(speech.SpeechRequestLimiter, "admit", blocked)
    monkeypatch.setattr(
        bot.NailongSpeechClient, "from_environment", lambda: SimpleNamespace(synthesize=forbidden)
    )
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text="奶龙说 你好", mentioned_bot=True, random_push_probability=0
    )
    assert sent == [
        ({"text": "每人 30 秒内只能生成一次语音，稍等一下。"}, {"reply_to": "source-id"})
    ]
