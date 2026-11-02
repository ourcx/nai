from __future__ import annotations

import asyncio
import importlib.util
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

spec = importlib.util.spec_from_file_location(
    "nailong_tts_service", Path(__file__).parents[1] / "deploy/tts/service.py"
)
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)
TOKEN = "test-token-" * 4
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def client_for(synthesize, **options):
    return TestClient(
        service.create_app(synthesize, token=TOKEN, release="test-release", **options)
    )


def test_auth_and_input_validation_do_not_enter_model():
    def forbidden(*_args):
        raise AssertionError("invalid request reached the model")

    client = client_for(forbidden)
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.post("/tts", json={"text": "我是奶龙"}).status_code == 401
    assert (
        client.post("/tts", headers={"Authorization": "Bearer wrong-token"}, json={}).status_code
        == 401
    )
    for payload in (
        {},
        {"text": " "},
        {"text": "奶" * 201},
        {"text": "hello"},
        {"text": "你好", "ref_audio_path": "/etc/passwd"},
        {"text": "你好", "seed": True},
        {"text": "你好", "seed": -1},
        {"text": "你好", "seed": 2**32},
        ["你好"],
    ):
        assert client.post("/tts", headers=HEADERS, json=payload).status_code == 400
    assert client.post("/tts", headers=HEADERS, content=b"{bad").status_code == 400
    assert client.post("/tts", headers=HEADERS, content=b" " * 4097).status_code == 413


def test_returns_wav_and_measurements_without_changing_text_or_seed():
    calls = []

    def generate(text, seed):
        calls.append((text, seed))
        return b"RIFF-test-wave", 4.2, 1.9

    with client_for(generate) as client:
        response = client.post("/tts", headers=HEADERS, json={"text": " 我是奶龙！ ", "seed": 42})
    assert response.status_code == 200
    assert response.content == b"RIFF-test-wave"
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-model-release"] == "test-release"
    assert response.headers["x-audio-duration-seconds"] == "4.200"
    assert response.headers["cache-control"] == "no-store"
    assert calls == [("我是奶龙！", 42)]


def test_concurrent_request_is_rejected_and_health_remains_responsive():
    entered, finish = threading.Event(), threading.Event()

    def generate(_text, _seed):
        entered.set()
        assert finish.wait(5)
        return b"RIFF", 1.0, 1.0

    with client_for(generate, max_waiting=0) as client, ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, "/tts", headers=HEADERS, json={"text": "第一句"})
        try:
            assert entered.wait(5)
            assert client.get("/healthz").json()["busy"] is True
            second = client.post("/tts", headers=HEADERS, json={"text": "第二句"})
            assert second.status_code == 429
            assert second.headers["retry-after"] == "5"
        finally:
            finish.set()
        assert first.result().status_code == 200
        assert client.get("/healthz").json()["busy"] is False


def test_model_failure_releases_lock_and_hides_details():
    calls = []

    def generate(_text, _seed):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("private runtime details")
        return b"RIFF", 1.0, 1.0

    with client_for(generate) as client:
        first = client.post("/tts", headers=HEADERS, json={"text": "你好"})
        assert first.status_code == 500
        assert "private runtime details" not in first.text
        assert client.post("/tts", headers=HEADERS, json={"text": "你好"}).status_code == 200


def test_refuses_missing_authentication_configuration():
    with pytest.raises(ValueError, match="TTS_API_TOKEN"):
        service.create_app(lambda *_: None, token="", release="test")


def wait_for_queue(client, size):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if client.get("/healthz").json()["queued"] == size:
            return
        time.sleep(0.005)
    raise AssertionError("Queue did not reach expected size")


def test_fifo_queue_is_bounded_and_only_one_model_call_runs():
    entered, finish = threading.Event(), threading.Event()
    calls = []
    running = 0

    def generate(text, _seed):
        nonlocal running
        assert running == 0
        running += 1
        calls.append(text)
        if text == "第一句":
            entered.set()
            assert finish.wait(5)
        running -= 1
        return b"RIFF", 1.0, 1.0

    with client_for(generate, max_waiting=2) as client, ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(client.post, "/tts", headers=HEADERS, json={"text": "第一句"})
        try:
            assert entered.wait(3)
            second = pool.submit(client.post, "/tts", headers=HEADERS, json={"text": "第二句"})
            wait_for_queue(client, 1)
            third = pool.submit(client.post, "/tts", headers=HEADERS, json={"text": "第三句"})
            wait_for_queue(client, 2)
            overflow = client.post("/tts", headers=HEADERS, json={"text": "第四句"})
            assert overflow.status_code == 429
            assert overflow.json()["error"] == "queue_full"
            assert calls == ["第一句"]
        finally:
            finish.set()
        assert [item.result().status_code for item in [first, second, third]] == [200, 200, 200]
        assert calls == ["第一句", "第二句", "第三句"]
        assert client.get("/healthz").json()["queued"] == 0


def test_expired_job_is_removed_and_never_enters_model():
    entered, finish = threading.Event(), threading.Event()
    calls = []

    def generate(text, _seed):
        calls.append(text)
        entered.set()
        assert finish.wait(5)
        return b"RIFF", 1.0, 1.0

    with client_for(generate, queue_timeout=0.05) as client, ThreadPoolExecutor(1) as pool:
        first = pool.submit(client.post, "/tts", headers=HEADERS, json={"text": "第一句"})
        try:
            assert entered.wait(3)
            expired = client.post("/tts", headers=HEADERS, json={"text": "过期台词"})
            assert expired.status_code == 429
            assert expired.json()["error"] == "queue_timeout"
            assert client.get("/healthz").json()["queued"] == 0
        finally:
            finish.set()
        assert first.result().status_code == 200
        assert calls == ["第一句"]


def test_global_rate_limit_expires_without_busy_loop_or_retries():
    now = [0.0]
    with client_for(lambda *_: (b"RIFF", 1.0, 1.0), rate_limit=2, clock=lambda: now[0]) as client:
        for _ in range(2):
            assert client.post("/tts", headers=HEADERS, json={"text": "你好"}).status_code == 200
        blocked = client.post("/tts", headers=HEADERS, json={"text": "你好"})
        assert blocked.status_code == 429
        assert blocked.json()["error"] == "rate_limited"
        assert blocked.headers["retry-after"] == "60"
        now[0] = 60.0
        assert client.post("/tts", headers=HEADERS, json={"text": "你好"}).status_code == 200


def test_cancelling_waiter_frees_slot_but_cancelling_running_request_keeps_model_owned():
    entered, finish = threading.Event(), threading.Event()
    calls = []

    def generate(text, _seed):
        calls.append(text)
        if text == "第一句":
            entered.set()
            assert finish.wait(5)
        return b"RIFF", 1.0, 1.0

    async def run():
        app = service.create_app(generate, token=TOKEN, release="test")
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):

            async def until(predicate):
                async with asyncio.timeout(3):
                    while not await predicate():
                        await asyncio.sleep(0.005)

            async def queued():
                return (await client.get("/healthz")).json()["queued"] == 1

            async def running():
                return entered.is_set()

            first = asyncio.create_task(
                client.post("/tts", headers=HEADERS, json={"text": "第一句"})
            )
            try:
                await until(running)
                second = asyncio.create_task(
                    client.post("/tts", headers=HEADERS, json={"text": "取消台词"})
                )
                await until(queued)
                second.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await second
                assert (await client.get("/healthz")).json()["queued"] == 0
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                third = asyncio.create_task(
                    client.post("/tts", headers=HEADERS, json={"text": "第三句"})
                )
                await until(queued)
                assert calls == ["第一句"]
                assert (await client.get("/healthz")).json()["busy"] is True
                finish.set()
                assert (await third).status_code == 200
                assert calls == ["第一句", "第三句"]
            finally:
                finish.set()

    asyncio.run(run())
