from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Any

from ..adapters.base import BotAdapter, SendResult
from ..integrations.speech import NailongSpeechClient, SpeechAudio, SpeechError

LOGGER = logging.getLogger(__name__)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class AsyncTaskDegraded(SpeechError):
    """昂贵异步任务被限流或熔断，调用方应立即降级为文字。"""


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    base_delay: float = 60
    max_delay: float = 300
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    opened_at: float = 0
    open_count: int = 0
    probe_in_flight: bool = False

    @property
    def retry_delay(self) -> float:
        return min(self.base_delay * (2 ** max(0, self.open_count - 1)), self.max_delay)

    def admit(self, now: float) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if now - self.opened_at < self.retry_delay or self.probe_in_flight:
                return False
            self.state = CircuitState.HALF_OPEN
            self.probe_in_flight = True
            return True
        return False

    def succeed(self) -> None:
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.open_count = 0
        self.probe_in_flight = False

    def fail(self, now: float) -> None:
        self.failures += 1
        if self.state == CircuitState.HALF_OPEN or self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = now
            self.open_count += 1
        self.probe_in_flight = False


class PermitPool:
    """一次性预留一个或两个发送位，避免成对消息被其他任务拆开。"""

    def __init__(self, capacity: int) -> None:
        if capacity < 2:
            raise ValueError("Delivery capacity must be at least two")
        self.capacity = capacity
        self.available = capacity
        self.condition = asyncio.Condition()

    @asynccontextmanager
    async def reserve(self, count: int):
        if not 0 < count <= self.capacity:
            raise ValueError("Invalid delivery permit count")
        async with self.condition:
            await self.condition.wait_for(lambda: self.available >= count)
            self.available -= count
        try:
            yield
        finally:
            async with self.condition:
                self.available += count
                self.condition.notify_all()


class DeliveryGateway:
    """所有平台发送共用的并发和每群顺序控制。"""

    def __init__(self, channel: BotAdapter, *, concurrency: int = 8) -> None:
        self.channel = channel
        self.permits = PermitPool(concurrency)
        self.chat_locks: dict[str, asyncio.Lock] = {}

    def _chat_lock(self, chat_id: str) -> asyncio.Lock:
        return self.chat_locks.setdefault(chat_id, asyncio.Lock())

    async def send(
        self,
        chat_id: str,
        content: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> SendResult:
        async with self._chat_lock(chat_id), self.permits.reserve(1):
            return await self.channel.send(chat_id, content, options)

    async def send_pair(
        self,
        chat_id: str,
        first: dict[str, Any],
        second: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[SendResult, SendResult]:
        async with self._chat_lock(chat_id), self.permits.reserve(2):
            first_result, second_result = await asyncio.gather(
                self.channel.send(chat_id, first, options),
                self.channel.send(chat_id, second, options),
            )
            return first_result, second_result

    async def update_card(
        self,
        message_id: str,
        card: dict[str, Any],
    ) -> SendResult:
        async with self.permits.reserve(1):
            return await self.channel.update_card(message_id, card)


@dataclass
class SpeechJob:
    message_id: str
    chat_id: str
    user_id: str
    voice: str
    text: str
    result: asyncio.Future[SpeechAudio]
    circuit_probe: bool = False
    created_at: float = field(default_factory=monotonic)


@dataclass(frozen=True)
class ResponseBundle:
    message_id: str
    chat_id: str
    user_id: str
    text_content: dict[str, Any]
    speech_text: str
    voice: str
    audio_source: str
    options: dict[str, Any] | None = None


class SpeechDispatcher:
    """奶龙、奶蛙各自单消费者；队满或熔断时立即让文字降级接管。"""

    def __init__(
        self,
        client: NailongSpeechClient,
        *,
        queue_capacity: int = 3,
    ) -> None:
        self.client = client
        self.queues = {
            voice: asyncio.Queue[SpeechJob](maxsize=queue_capacity)
            for voice in ("nailong", "naiwa")
        }
        self.breakers = {voice: CircuitBreaker() for voice in self.queues}
        self.tasks: list[asyncio.Task] = []
        self.active_results: set[asyncio.Future[SpeechAudio]] = set()
        self.closed = False

    def start(self) -> None:
        if self.closed:
            raise RuntimeError("Speech dispatcher is closed")
        if self.tasks:
            return
        self.tasks = [
            asyncio.create_task(self._worker(voice), name=f"speech:{voice}")
            for voice in self.queues
        ]

    @property
    def busy(self) -> bool:
        return any(queue.full() for queue in self.queues.values()) or any(
            breaker.state != CircuitState.CLOSED
            for breaker in self.breakers.values()
        )

    async def close(self) -> None:
        self.closed = True
        for queue in self.queues.values():
            while not queue.empty():
                job = queue.get_nowait()
                if not job.result.done():
                    job.result.set_exception(AsyncTaskDegraded("语音服务正在恢复，已保留文字回复。"))
                queue.task_done()
        for result in self.active_results:
            if not result.done():
                result.set_exception(AsyncTaskDegraded("语音服务正在关闭，已保留文字回复。"))
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with suppress(asyncio.CancelledError):
                await task
        self.tasks.clear()

    async def generate(
        self,
        *,
        message_id: str,
        chat_id: str,
        user_id: str,
        voice: str,
        text: str,
    ) -> SpeechAudio:
        if self.closed:
            raise AsyncTaskDegraded("语音服务正在关闭，已保留文字回复。")
        if voice not in self.queues:
            raise ValueError("Unsupported speech voice")
        breaker = self.breakers[voice]
        now = monotonic()
        if not breaker.admit(now):
            raise AsyncTaskDegraded("语音服务正在恢复，已保留文字回复。")
        queue = self.queues[voice]
        if queue.full():
            if breaker.state == CircuitState.HALF_OPEN:
                breaker.fail(now)
            raise AsyncTaskDegraded("语音请求过多，已保留文字回复。")
        result = asyncio.get_running_loop().create_future()
        circuit_probe = breaker.state == CircuitState.HALF_OPEN
        queue.put_nowait(SpeechJob(
            message_id=message_id,
            chat_id=chat_id,
            user_id=user_id,
            voice=voice,
            text=text,
            result=result,
            circuit_probe=circuit_probe,
        ))
        return await result

    async def _worker(self, voice: str) -> None:
        queue = self.queues[voice]
        breaker = self.breakers[voice]
        while True:
            job = await queue.get()
            try:
                if job.result.cancelled():
                    if job.circuit_probe:
                        breaker.fail(monotonic())
                    continue
                self.active_results.add(job.result)
                audio = await self.client.synthesize(job.text, voice=voice)
                breaker.succeed()
                if not job.result.done():
                    job.result.set_result(audio)
            except asyncio.CancelledError:
                if not job.result.done():
                    job.result.set_exception(
                        AsyncTaskDegraded("语音服务正在关闭，已保留文字回复。")
                    )
                raise
            except Exception as error:
                breaker.fail(monotonic())
                if not job.result.done():
                    job.result.set_exception(error)
                LOGGER.warning(
                    "Speech worker failed voice=%s state=%s",
                    voice,
                    breaker.state,
                )
            finally:
                self.active_results.discard(job.result)
                queue.task_done()


class ResponseCoordinator:
    """健康时成对发送；队列或模型异常时只保留文字。"""

    def __init__(
        self,
        gateway: DeliveryGateway,
        speech: SpeechDispatcher | None,
        *,
        pair_timeout: float = 15,
    ) -> None:
        self.gateway = gateway
        self.speech = speech
        self.pair_timeout = pair_timeout

    async def send_text_with_optional_speech(
        self,
        bundle: ResponseBundle,
    ) -> tuple[SendResult, SendResult | None]:
        if self.speech is None:
            return await self.gateway.send(
                bundle.chat_id, bundle.text_content, bundle.options,
            ), None
        try:
            audio = await asyncio.wait_for(
                self.speech.generate(
                    message_id=bundle.message_id,
                    chat_id=bundle.chat_id,
                    user_id=bundle.user_id,
                    voice=bundle.voice,
                    text=bundle.speech_text,
                ),
                timeout=self.pair_timeout,
            )
        except (AsyncTaskDegraded, SpeechError, TimeoutError):
            return await self.gateway.send(
                bundle.chat_id, bundle.text_content, bundle.options,
            ), None
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Speech bundle degraded to text voice=%s", bundle.voice)
            return await self.gateway.send(
                bundle.chat_id, bundle.text_content, bundle.options,
            ), None
        payload = {
            "audio": {
                "source": audio.wav if bundle.audio_source == "wav" else audio.opus,
                "duration_ms": audio.duration_ms,
            }
        }
        text_result, audio_result = await self.gateway.send_pair(
            bundle.chat_id, bundle.text_content, payload, bundle.options,
        )
        return text_result, audio_result
