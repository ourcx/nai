from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Generic, Mapping, Optional, TypeVar


T = TypeVar("T")
AsyncWork = Callable[["TaskContext"], Awaitable[T]]


class OverloadPolicy(str, Enum):
    DROP = "DROP"
    RETRY = "RETRY"
    DEGRADE_TEXT = "DEGRADE_TEXT"


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class WorkOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    DROP = "DROP"
    RETRY = "RETRY"
    DEGRADE_TEXT = "DEGRADE_TEXT"


@dataclass(frozen=True)
class LaneConfig:
    concurrency: int
    queue_capacity: int
    timeout: float
    overload_policy: OverloadPolicy = OverloadPolicy.RETRY
    circuit_failure_threshold: int = 5
    circuit_reset_timeout: float = 30.0
    circuit_max_reset_timeout: float = 300.0
    recovery_step: int = 1

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if self.queue_capacity < 1:
            raise ValueError("queue_capacity must be at least 1")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.circuit_failure_threshold < 1:
            raise ValueError("circuit_failure_threshold must be at least 1")
        if self.circuit_reset_timeout <= 0:
            raise ValueError("circuit_reset_timeout must be positive")
        if self.circuit_max_reset_timeout < self.circuit_reset_timeout:
            raise ValueError("circuit_max_reset_timeout must not be smaller than reset timeout")
        if self.recovery_step < 1:
            raise ValueError("recovery_step must be at least 1")


@dataclass(frozen=True)
class TaskContext:
    job: Any = None
    message: Any = None
    chat: Any = None
    user: Any = None


@dataclass(frozen=True)
class WorkResult(Generic[T]):
    outcome: WorkOutcome
    context: TaskContext
    value: Optional[T] = None
    error: Optional[BaseException] = None
    reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.outcome is WorkOutcome.SUCCESS


@dataclass(frozen=True)
class _CircuitPermit:
    half_open: bool


class _CircuitBreaker:
    def __init__(self, config: LaneConfig) -> None:
        self._config = config
        self.state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_limit = 1
        self._half_open_in_flight = 0
        self._open_count = 0

    def accepts_submission(self) -> bool:
        self._advance_if_ready()
        return self.state is not CircuitState.OPEN

    def acquire(self) -> Optional[_CircuitPermit]:
        self._advance_if_ready()
        if self.state is CircuitState.OPEN:
            return None
        if self.state is CircuitState.HALF_OPEN:
            if self._half_open_in_flight >= self._half_open_limit:
                return None
            self._half_open_in_flight += 1
            return _CircuitPermit(half_open=True)
        return _CircuitPermit(half_open=False)

    def succeed(self, permit: _CircuitPermit) -> None:
        if permit.half_open:
            self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
            if self.state is not CircuitState.HALF_OPEN:
                return
            self._half_open_limit = min(
                self._config.concurrency,
                self._half_open_limit + self._config.recovery_step,
            )
            if self._half_open_limit >= self._config.concurrency:
                self.state = CircuitState.CLOSED
                self._failures = 0
                self._open_count = 0
            return
        if self.state is CircuitState.CLOSED:
            self._failures = 0

    def fail(self, permit: _CircuitPermit) -> None:
        if permit.half_open:
            self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
            self._open()
            return
        if self.state is not CircuitState.CLOSED:
            return
        self._failures += 1
        if self._failures >= self._config.circuit_failure_threshold:
            self._open()

    def _advance_if_ready(self) -> None:
        if (
            self.state is CircuitState.OPEN
            and time.monotonic() - self._opened_at
            >= self._reset_timeout()
        ):
            self.state = CircuitState.HALF_OPEN
            self._half_open_limit = 1
            self._half_open_in_flight = 0

    def _open(self) -> None:
        self.state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._open_count += 1
        self._half_open_limit = 1
        self._half_open_in_flight = 0

    def _reset_timeout(self) -> float:
        return min(
            self._config.circuit_reset_timeout * (2 ** max(0, self._open_count - 1)),
            self._config.circuit_max_reset_timeout,
        )


@dataclass
class _QueuedWork(Generic[T]):
    work: AsyncWork[T]
    context: TaskContext
    future: "asyncio.Future[WorkResult[T]]"


class _Lane:
    def __init__(self, config: LaneConfig) -> None:
        self.config = config
        self.queue: "asyncio.Queue[_QueuedWork[Any]]" = asyncio.Queue(
            maxsize=config.queue_capacity
        )
        self.breaker = _CircuitBreaker(config)
        self.workers: list["asyncio.Task[None]"] = []


class WorkScheduler:
    def __init__(self, lanes: Mapping[str, LaneConfig]) -> None:
        if not lanes:
            raise ValueError("at least one lane is required")
        self._lanes: Dict[str, _Lane] = {
            name: _Lane(config) for name, config in lanes.items()
        }
        self._started = False
        self._closed = False
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._start_lock:
            if self._closed:
                raise RuntimeError("scheduler is closed")
            if self._started:
                return
            for name, lane in self._lanes.items():
                lane.workers = [
                    asyncio.create_task(
                        self._worker(name, lane),
                        name=f"work-scheduler:{name}:{index}",
                    )
                    for index in range(lane.config.concurrency)
                ]
            self._started = True

    async def submit(
        self,
        lane_name: str,
        work: AsyncWork[T],
        context: TaskContext,
    ) -> WorkResult[T]:
        lane = self._get_lane(lane_name)
        if self._closed:
            return self._overload_result(lane, context, "scheduler_closed")
        await self.start()
        if not lane.breaker.accepts_submission():
            return self._overload_result(lane, context, "circuit_open")

        future: "asyncio.Future[WorkResult[T]]" = (
            asyncio.get_running_loop().create_future()
        )
        item = _QueuedWork(work=work, context=context, future=future)
        try:
            lane.queue.put_nowait(item)
        except asyncio.QueueFull:
            return self._overload_result(lane, context, "queue_full")
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    def circuit_state(self, lane_name: str) -> CircuitState:
        lane = self._get_lane(lane_name)
        lane.breaker.accepts_submission()
        return lane.breaker.state

    def busy(self, lane_name: str) -> bool:
        lane = self._get_lane(lane_name)
        return lane.queue.full() or not lane.breaker.accepts_submission()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for lane in self._lanes.values():
            while True:
                try:
                    item = lane.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not item.future.done():
                    item.future.set_result(
                        self._overload_result(
                            lane, item.context, "scheduler_closed"
                        )
                    )
                lane.queue.task_done()
            for worker in lane.workers:
                worker.cancel()
        workers = [
            worker for lane in self._lanes.values() for worker in lane.workers
        ]
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)

    async def __aenter__(self) -> "WorkScheduler":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _worker(self, lane_name: str, lane: _Lane) -> None:
        while True:
            item = await lane.queue.get()
            try:
                if item.future.cancelled():
                    continue
                permit = lane.breaker.acquire()
                if permit is None:
                    reason = (
                        "circuit_open"
                        if lane.breaker.state is CircuitState.OPEN
                        else "circuit_recovering"
                    )
                    item.future.set_result(
                        self._overload_result(lane, item.context, reason)
                    )
                    continue
                execution = asyncio.create_task(
                    asyncio.wait_for(
                        item.work(item.context),
                        timeout=lane.config.timeout,
                    ),
                    name=f"work-execution:{lane_name}",
                )
                try:
                    await asyncio.wait(
                        {execution, item.future},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if item.future.cancelled():
                        execution.cancel()
                        await asyncio.gather(execution, return_exceptions=True)
                        continue
                    value = await execution
                except asyncio.CancelledError:
                    execution.cancel()
                    await asyncio.gather(execution, return_exceptions=True)
                    if not item.future.done():
                        item.future.set_result(
                            self._overload_result(
                                lane, item.context, "scheduler_closed"
                            )
                        )
                    raise
                except asyncio.TimeoutError as error:
                    lane.breaker.fail(permit)
                    if not item.future.done():
                        item.future.set_result(
                            WorkResult(
                                outcome=WorkOutcome.TIMEOUT,
                                context=item.context,
                                error=error,
                                reason=f"{lane_name}_timeout",
                            )
                        )
                except Exception as error:
                    lane.breaker.fail(permit)
                    if not item.future.done():
                        item.future.set_result(
                            WorkResult(
                                outcome=WorkOutcome.FAILED,
                                context=item.context,
                                error=error,
                            )
                        )
                else:
                    lane.breaker.succeed(permit)
                    if not item.future.done():
                        item.future.set_result(
                            WorkResult(
                                outcome=WorkOutcome.SUCCESS,
                                context=item.context,
                                value=value,
                            )
                        )
            finally:
                lane.queue.task_done()

    def _get_lane(self, lane_name: str) -> _Lane:
        try:
            return self._lanes[lane_name]
        except KeyError as error:
            raise KeyError(f"unknown lane: {lane_name}") from error

    @staticmethod
    def _overload_result(
        lane: _Lane,
        context: TaskContext,
        reason: str,
    ) -> WorkResult[Any]:
        return WorkResult(
            outcome=WorkOutcome(lane.config.overload_policy.value),
            context=context,
            reason=reason,
        )
