"""调用独立语音后端，并将受限的 WAV 响应转为飞书语音格式。"""

from __future__ import annotations

import asyncio
import io
import os
import wave
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import monotonic
from urllib.parse import urlsplit

import httpx

MAX_SPEECH_CHARS = 200
MAX_WAV_BYTES = 8 * 1024 * 1024
PLAYBACK_LEAD_MS = 350


class SpeechError(Exception):
    """可直接回复给用户的语音错误，不包含后端密钥或响应正文。"""


@dataclass(frozen=True)
class SpeechAudio:
    opus: bytes
    duration_ms: int
    wav: bytes = b""


class SpeechRequestLimiter:
    """机器人进程共享的语音入口限制，避免群聊刷屏变成大量后端连接。"""

    def __init__(self, *, clock: Callable[[], float] = monotonic) -> None:
        self.clock = clock
        self.pending = 0
        self.users: dict[str, float] = {}
        self.chats: dict[str, float] = {}

    @contextmanager
    def admit(self, *, chat_id: str, user_id: str):
        now = self.clock()
        # 只记最近半分钟，群成员数量增长时也不会一直保留历史身份。
        self.users = {key: value for key, value in self.users.items() if now - value < 30}
        self.chats = {key: value for key, value in self.chats.items() if now - value < 5}
        if self.pending >= 4:
            raise SpeechError("奶龙的语音队列已满，等一会儿再叫我。")
        if user_id in self.users:
            raise SpeechError("每人 30 秒内只能生成一次语音，稍等一下。")
        if chat_id in self.chats:
            raise SpeechError("这个群刚收到语音请求，请间隔 5 秒再试。")
        self.users[user_id], self.chats[chat_id] = now, now
        self.pending += 1
        try:
            yield
        finally:
            self.pending -= 1


def encode_feishu_opus(content: bytes) -> SpeechAudio:
    import av

    with wave.open(io.BytesIO(content), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise SpeechError("语音服务返回了不支持的音频格式。")
        rate, frames = source.getframerate(), source.getnframes()
        if rate != 32000 or not 0 < frames <= rate * 90:
            raise SpeechError("语音服务返回了不支持的音频时长或采样率。")
        # 校验完整 PCM 数据，避免把截断的 WAV 当作成功结果发送。
        pcm = source.readframes(frames)
        if len(pcm) != frames * 2:
            raise SpeechError("语音服务返回的音频不完整，请稍后重试。")
    # 移动端刚取得音频焦点时可能截掉句首；只加播放缓冲，不裁剪模型生成的音节。
    padding_frames = rate * PLAYBACK_LEAD_MS // 1000
    padded = io.BytesIO()
    with wave.open(padded, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(b"\0" * (padding_frames * 2) + pcm)
    output = io.BytesIO()
    with (
        av.open(io.BytesIO(padded.getvalue())) as source,
        av.open(output, "w", format="ogg") as target,
    ):
        stream = target.add_stream("libopus", rate=48000)
        stream.layout = "mono"
        stream.bit_rate = 32000
        for frame in source.decode(audio=0):
            for packet in stream.encode(frame):
                target.mux(packet)
        for packet in stream.encode(None):
            target.mux(packet)
    opus = output.getvalue()
    if not opus.startswith(b"OggS") or b"OpusHead" not in opus[:128]:
        raise SpeechError("语音转换失败，请稍后重试。")
    return SpeechAudio(opus, round((frames + padding_frames) / rate * 1000), padded.getvalue())


@dataclass(frozen=True)
class NailongSpeechClient:
    base_url: str
    token: str = field(repr=False)

    @classmethod
    def from_environment(cls) -> NailongSpeechClient | None:
        base_url, token = os.getenv("NAILONG_TTS_URL", ""), os.getenv("NAILONG_TTS_TOKEN", "")
        if not base_url and not token:
            return None
        parsed = urlsplit(base_url)
        if (
            not base_url
            or not token
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Set valid NAILONG_TTS_URL and NAILONG_TTS_TOKEN together")
        # 在开通功能时就检查编码依赖，避免用户发出第一条指令后才发现缺包。
        import av

        if "libopus" not in av.codecs_available:
            raise ValueError("Install the speech extra with a libopus-enabled PyAV wheel")
        return cls(base_url.rstrip("/"), token)

    async def synthesize(self, text: str, *, voice: str = "nailong") -> SpeechAudio:
        if not text.strip() or len(text) > MAX_SPEECH_CHARS:
            raise SpeechError(f"请提供 1～{MAX_SPEECH_CHARS} 字的中文台词。")
        if voice not in {"nailong", "naiwa"}:
            raise ValueError("Unsupported speech voice")
        try:
            async with (
                httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5), trust_env=False) as client,
                client.stream(
                    "POST",
                    self.base_url + "/tts",
                    headers={"Authorization": "Bearer " + self.token},
                    json={"text": text, "voice": voice},
                ) as response,
            ):
                if response.status_code == 429:
                    raise SpeechError("奶龙正在说话，队列已满或等待超时，请稍后重试。")
                if response.status_code == 400:
                    raise SpeechError(f"目前只支持不超过 {MAX_SPEECH_CHARS} 字的中文台词。")
                if response.status_code != 200:
                    raise SpeechError("语音服务暂时不可用，请稍后重试。")
                if response.headers.get("content-type", "").split(";", 1)[0] != "audio/wav":
                    raise SpeechError("语音服务返回了不支持的音频格式。")
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_WAV_BYTES:
                        raise SpeechError("这段语音太长了，请把台词缩短一些。")
            return await asyncio.to_thread(encode_feishu_opus, bytes(content))
        except SpeechError:
            raise
        except (httpx.HTTPError, wave.Error, EOFError, ValueError) as error:
            raise SpeechError("语音生成失败，请稍后重试。") from error
