"""奶龙语音 HTTP 服务：固定声线、单次推理，不暴露模型管理接口。"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import math
import os
import re
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from time import monotonic

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

LOGGER = logging.getLogger("nailong.tts")
MAX_BODY_BYTES = 4096
MAX_TEXT_CHARS = 200
DEFAULT_SEED = 20261103


def create_app(
    synthesize: Callable[[str, int, str], tuple[bytes, float, float]],
    *,
    token: str,
    release: str,
    max_waiting: int = 3,
    queue_timeout: float = 30,
    rate_limit: int = 12,
    rate_window: float = 60,
    clock: Callable[[], float] = monotonic,
) -> Starlette:
    if len(token) < 32:
        raise ValueError("TTS_API_TOKEN must contain at least 32 characters")
    if max_waiting < 0 or queue_timeout <= 0 or rate_limit < 1 or rate_window <= 0:
        raise ValueError("Invalid speech queue or rate limit configuration")
    waiting = deque()
    admissions = deque()
    wake = asyncio.Event()
    active = False

    async def worker() -> None:
        nonlocal active
        while True:
            await wake.wait()
            # 被唤醒后，唯一的等待请求可能已取消；不能因此终止整个 worker。
            if not waiting:
                wake.clear()
                continue
            job = waiting.popleft()
            if not waiting:
                wake.clear()
            active = True
            text, seed, voice, started, result = job
            started.set()
            try:
                # 推理归后台 worker 所有；调用方取消请求也不能启动第二次推理。
                audio = await run_in_threadpool(synthesize, text, seed, voice)
                if not result.done():
                    result.set_result(audio)
            except Exception:
                LOGGER.exception("Speech synthesis failed")
                if not result.done():
                    result.set_result(None)
            finally:
                active = False

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(worker(), name="speech-inference-worker")
        try:
            yield
        finally:
            for job in waiting:
                job[4].cancel()
            waiting.clear()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def remove_waiting(job) -> None:
        with suppress(ValueError):
            waiting.remove(job)
        if not waiting:
            wake.clear()
        job[4].cancel()

    async def health(_request: Request) -> Response:
        return JSONResponse(
            {
                "status": "ok",
                "model": "nailong-naiwa-gpt-sovits-v2proplus",
                "release": release,
                "busy": active,
                "queued": len(waiting),
                "queue_capacity": max_waiting,
                "rate_limit_per_window": rate_limit,
                "rate_window_seconds": rate_window,
            }
        )

    async def tts(request: Request) -> Response:
        expected = f"Bearer {token}".encode()
        provided = request.headers.get("authorization", "").encode()
        if not hmac.compare_digest(provided, expected):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY_BYTES:
                return JSONResponse({"error": "request_too_large"}, status_code=413)
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError):
            return JSONResponse({"error": "invalid_json"}, status_code=400)
        if not isinstance(payload, dict) or set(payload) - {"text", "seed", "voice"}:
            return JSONResponse(
                {"error": "expected_text_with_optional_seed_and_voice"}, status_code=400
            )
        text = payload.get("text")
        seed = payload.get("seed", DEFAULT_SEED)
        voice = payload.get("voice", "nailong")
        if not isinstance(text, str) or not text.strip():
            return JSONResponse({"error": "text_required"}, status_code=400)
        text = text.strip()
        if len(text) > MAX_TEXT_CHARS:
            return JSONResponse(
                {"error": "text_too_long", "max_chars": MAX_TEXT_CHARS}, status_code=400
            )
        if re.search(r"[A-Za-z]", text) or not re.search(r"[0-9\u4e00-\u9fff]", text):
            return JSONResponse({"error": "chinese_text_required"}, status_code=400)
        if type(seed) is not int or not 0 <= seed < 2**32:
            return JSONResponse({"error": "invalid_seed"}, status_code=400)
        if voice not in {"nailong", "naiwa"}:
            return JSONResponse({"error": "invalid_voice"}, status_code=400)
        # 只有一个模型实例。一个执行位加有限等待位，队满时立即拒绝，不堆积任务。
        if len(waiting) + int(active) >= max_waiting + 1:
            return JSONResponse(
                {"error": "queue_full"}, status_code=429, headers={"Retry-After": "5"}
            )
        now = clock()
        while admissions and admissions[0] <= now - rate_window:
            admissions.popleft()
        if len(admissions) >= rate_limit:
            retry = max(1, math.ceil(admissions[0] + rate_window - now))
            return JSONResponse(
                {"error": "rate_limited"}, status_code=429, headers={"Retry-After": str(retry)}
            )
        admissions.append(now)
        started = asyncio.Event()
        result = asyncio.get_running_loop().create_future()
        job = (text, seed, voice, started, result)
        waiting.append(job)
        wake.set()
        try:
            try:
                await asyncio.wait_for(started.wait(), timeout=queue_timeout)
            except TimeoutError:
                # 临界时刻已被 worker 取走的任务继续等待，避免留下无人持有的推理。
                if not started.is_set():
                    remove_waiting(job)
                    return JSONResponse(
                        {"error": "queue_timeout"},
                        status_code=429,
                        headers={"Retry-After": "5"},
                    )
            queue_elapsed = clock() - now
            audio = await asyncio.shield(result)
            if audio is None:
                return JSONResponse({"error": "synthesis_failed"}, status_code=500)
            wav, duration, elapsed = audio
            return Response(
                wav,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": 'inline; filename="nailong.wav"',
                    "Cache-Control": "no-store",
                    "X-Audio-Duration-Seconds": f"{duration:.3f}",
                    "X-Generation-Seconds": f"{elapsed:.3f}",
                    "X-Model-Release": release,
                    "X-Queue-Wait-Seconds": f"{queue_elapsed:.3f}",
                },
            )
        except asyncio.CancelledError:
            # 排队阶段取消便腾出位置；执行阶段只丢弃结果，仍等模型结束再取下一项。
            remove_waiting(job)
            raise

    return Starlette(
        routes=[Route("/healthz", health), Route("/tts", tts, methods=["POST"])],
        lifespan=lifespan,
    )


def main() -> None:
    import uvicorn
    from tts_engine import GptSoVitsEngine

    root = Path(os.environ["TTS_RELEASE_DIR"]).resolve()
    token = os.environ["TTS_API_TOKEN"]
    if len(token) < 32:
        raise ValueError("TTS_API_TOKEN must contain at least 32 characters")
    engine = GptSoVitsEngine(root, Path(os.environ["TTS_STATE_DIR"]))
    app = create_app(
        engine.synthesize,
        token=token,
        release=engine.release_id,
        max_waiting=int(os.getenv("TTS_MAX_WAITING", "3")),
        queue_timeout=float(os.getenv("TTS_QUEUE_TIMEOUT", "30")),
        rate_limit=int(os.getenv("TTS_RATE_LIMIT", "12")),
    )
    uvicorn.run(
        app,
        host=os.environ.get("TTS_HOST", "127.0.0.1"),
        port=int(os.environ.get("TTS_PORT", "9880")),
        workers=1,
        access_log=False,
    )


if __name__ == "__main__":
    main()
