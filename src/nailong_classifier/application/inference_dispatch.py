from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar


PredictionT = TypeVar("PredictionT", covariant=True)
InferenceMethod = Literal["predict_bytes", "predict_path"]
InferenceOperation = Literal["classify", "detect_doubao"]


class Predictor(Protocol[PredictionT]):
    """与现有分类器、豆包检测器一致的最小预测接口。"""

    def predict_bytes(self, data: bytes) -> PredictionT: ...

    def predict_path(self, path: Path) -> PredictionT: ...


class InferenceTaskDegraded(RuntimeError):
    """识图任务未被执行，调用方可以立即降级。"""


class InferenceQueueFull(InferenceTaskDegraded):
    """识图等待队列已满。"""


class InferenceDispatcherClosed(InferenceTaskDegraded):
    """识图调度器已经关闭。"""


@dataclass
class InferenceJob:
    message_id: str
    chat_id: str
    user_id: str
    operation: InferenceOperation
    predictor: Predictor[Any]
    method: InferenceMethod
    source: bytes | Path
    result: asyncio.Future[Any]


class InferenceDispatcher:
    """分类和豆包检测共用一个有界队列，保证模型调用严格串行。"""

    def __init__(
        self,
        predictor: Predictor[Any],
        doubao_detector: Predictor[Any] | None = None,
        *,
        queue_capacity: int = 3,
    ) -> None:
        if queue_capacity < 1:
            raise ValueError("Inference queue capacity must be positive")
        self.predictor = predictor
        self.doubao_detector = doubao_detector
        self.queue: asyncio.Queue[InferenceJob] = asyncio.Queue(maxsize=queue_capacity)
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def busy(self) -> bool:
        return self.queue.full()

    def start(self) -> None:
        if self._closed:
            raise InferenceDispatcherClosed("识图调度器已经关闭。")
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._worker(),
                name="inference-dispatch",
            )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while not self.queue.empty():
            job = self.queue.get_nowait()
            if not job.result.done():
                job.result.set_exception(InferenceDispatcherClosed("识图调度器已经关闭。"))
            self.queue.task_done()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def predict_bytes(
        self,
        data: bytes,
        *,
        message_id: str,
        chat_id: str,
        user_id: str,
    ) -> Any:
        return await self._submit(
            operation="classify",
            predictor=self.predictor,
            method="predict_bytes",
            source=data,
            message_id=message_id,
            chat_id=chat_id,
            user_id=user_id,
        )

    async def predict_path(
        self,
        path: Path,
        *,
        message_id: str,
        chat_id: str,
        user_id: str,
    ) -> Any:
        return await self._submit(
            operation="classify",
            predictor=self.predictor,
            method="predict_path",
            source=path,
            message_id=message_id,
            chat_id=chat_id,
            user_id=user_id,
        )

    async def detect_doubao_bytes(
        self,
        data: bytes,
        *,
        message_id: str,
        chat_id: str,
        user_id: str,
    ) -> Any:
        detector = self._require_doubao_detector()
        return await self._submit(
            operation="detect_doubao",
            predictor=detector,
            method="predict_bytes",
            source=data,
            message_id=message_id,
            chat_id=chat_id,
            user_id=user_id,
        )

    async def detect_doubao_path(
        self,
        path: Path,
        *,
        message_id: str,
        chat_id: str,
        user_id: str,
    ) -> Any:
        detector = self._require_doubao_detector()
        return await self._submit(
            operation="detect_doubao",
            predictor=detector,
            method="predict_path",
            source=path,
            message_id=message_id,
            chat_id=chat_id,
            user_id=user_id,
        )

    def _require_doubao_detector(self) -> Predictor[Any]:
        if self.doubao_detector is None:
            raise ValueError("Doubao detector is not configured")
        return self.doubao_detector

    async def _submit(
        self,
        *,
        operation: InferenceOperation,
        predictor: Predictor[Any],
        method: InferenceMethod,
        source: bytes | Path,
        message_id: str,
        chat_id: str,
        user_id: str,
    ) -> Any:
        if self._closed:
            raise InferenceDispatcherClosed("识图调度器已经关闭。")
        if self._task is None or self._task.done():
            self.start()

        result = asyncio.get_running_loop().create_future()
        job = InferenceJob(
            message_id=message_id,
            chat_id=chat_id,
            user_id=user_id,
            operation=operation,
            predictor=predictor,
            method=method,
            source=source,
            result=result,
        )
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as error:
            result.cancel()
            raise InferenceQueueFull("识图请求过多，请立即降级处理。") from error

        try:
            return await result
        except asyncio.CancelledError:
            result.cancel()
            raise

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            try:
                if job.result.cancelled():
                    continue
                predict = getattr(job.predictor, job.method)
                prediction = await asyncio.to_thread(predict, job.source)
                if not job.result.done():
                    job.result.set_result(prediction)
            except asyncio.CancelledError:
                if not job.result.done():
                    job.result.set_exception(
                        InferenceDispatcherClosed("识图调度器已经关闭。")
                    )
                raise
            except Exception as error:
                if not job.result.done():
                    job.result.set_exception(error)
            finally:
                self.queue.task_done()
