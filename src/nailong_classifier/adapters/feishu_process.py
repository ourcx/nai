from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import uuid
from collections import OrderedDict
from contextlib import suppress
from dataclasses import replace
from typing import Any

from .base import ErrorCode, Handler, IdentityScope, Message, SendError, SendResult

LOGGER = logging.getLogger(__name__)
CATALOG_METHODS = {
    "resolve_target_column", "list_section_image_tokens", "download_media",
    "create_image_block", "upload_image", "replace_image",
}


def _worker(connection, scope: IdentityScope, credentials: dict[str, str]) -> None:
    """每个进程只加载一个飞书长连接，规避 SDK 模块级全局 loop 的争用。"""
    # spawn 不继承主进程日志配置，显式启用才能看到入群事件的接收诊断。
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("Lark").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    from .feishu import FeishuAdapter

    async def serve() -> None:
        adapter = FeishuAdapter(scope, **credentials)
        send_lock = asyncio.Lock()
        natives: OrderedDict[str, Any] = OrderedDict()
        acknowledgements: dict[str, asyncio.Future] = {}
        documents: dict[tuple[str, str], Any] = {}
        commands: set[asyncio.Task] = set()

        async def send(frame) -> None:
            async with send_lock:
                await asyncio.to_thread(connection.send, frame)

        def subscribe(event: str) -> None:
            async def forward(payload) -> None:
                if isinstance(payload, Message):
                    natives[payload.message_id] = payload.native
                    if len(natives) > 1024:
                        natives.popitem(last=False)
                    payload = replace(payload, native=None)
                elif event == "error":
                    # SDK 异常对象可能带不可序列化的线程/客户端引用。
                    payload = SendError("unknown", "Feishu SDK error")
                elif event == "reject":
                    payload = {"reason": str(payload.reason)}
                event_id = uuid.uuid4().hex
                acknowledgement = asyncio.get_running_loop().create_future()
                acknowledgements[event_id] = acknowledgement
                try:
                    await send(("event", event_id, event, payload))
                    await acknowledgement
                finally:
                    acknowledgements.pop(event_id, None)
            adapter.on(event, forward)

        for event in ("message", "members_added", "card_action", "error", "reject"):
            subscribe(event)

        async def execute(request_id, method, args, kwargs) -> None:
            try:
                if method == "catalog":
                    document_id, section_id, operation, *parameters = args
                    if operation not in CATALOG_METHODS:
                        raise ValueError("Unsupported catalog operation")
                    key = (document_id, section_id)
                    if key not in documents:
                        documents[key] = adapter.catalog_document(*key)
                    result = await getattr(documents[key], operation)(*parameters, **kwargs)
                else:
                    if method not in {
                        "connect", "resolve", "get_chat_members", "send",
                        "resolve_image_target", "download_target", "update_card", "close",
                    }:
                        raise ValueError("Unsupported adapter operation")
                    if method == "resolve_image_target":
                        message = args[0]
                        args = (replace(message, native=natives.get(message.message_id)),)
                    result = await getattr(adapter, method)(*args, **kwargs)
                await send(("result", request_id, True, result))
            except Exception:
                # 不把 SDK 请求、密钥或原始异常内容序列化回主进程。
                await send(("result", request_id, False, f"Feishu {method} failed"))

        try:
            while True:
                frame = await asyncio.to_thread(connection.recv)
                if frame[0] == "ack":
                    pending = acknowledgements.get(frame[1])
                    if pending and not pending.done():
                        pending.set_result(None)
                    continue
                _, request_id, method, args, kwargs = frame
                if method == "close":
                    await execute(request_id, method, args, kwargs)
                    break
                task = asyncio.create_task(execute(request_id, method, args, kwargs))
                commands.add(task)
                task.add_done_callback(commands.discard)
        except (EOFError, OSError):
            pass
        finally:
            for task in commands:
                task.cancel()
            await adapter.close()
            connection.close()

    asyncio.run(serve())


class RemoteCatalogDocument:
    """文档同步仍由飞书 SDK 执行；后端只使用既有文档操作接口。"""

    def __init__(self, adapter, document_id: str, section_id: str) -> None:
        self.adapter = adapter
        self.document_id = document_id
        self.section_id = section_id

    def client_token(self, asset_sha256: str, operation: str) -> str:
        # 与 CatalogDocumentClient 相同的 UUID5 契约；不需导入 SDK。
        namespace = uuid.UUID("ea1d63de-e022-4c79-86d9-0935f6c77ceb")
        return str(uuid.uuid5(namespace, f"{self.document_id}:{asset_sha256}:{operation}"))

    def __getattr__(self, operation: str):
        if operation not in CATALOG_METHODS:
            raise AttributeError(operation)

        async def invoke(*args, **kwargs):
            return await self.adapter._rpc(
                "catalog", self.document_id, self.section_id, operation, *args, **kwargs
            )
        return invoke


class ProcessFeishuAdapter:
    """多飞书实例时的 SDK 进程代理，模型与业务状态始终留在主进程。"""

    def __init__(self, scope: IdentityScope, *, app_id: str, app_secret: str) -> None:
        self.scope = scope
        self._credentials = {"app_id": app_id, "app_secret": app_secret}
        self._connection = None
        self._process = None
        self._reader = None
        self._send_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._handlers: dict[str, Handler] = {}
        self._pending: dict[str, asyncio.Future] = {}
        self._events: set[asyncio.Task] = set()

    def on(self, event: str, handler: Handler) -> None:
        self._handlers[event] = handler

    async def _start(self) -> None:
        async with self._start_lock:
            if self._process is not None:
                return
            context = multiprocessing.get_context("spawn")
            connection, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(child, self.scope, self._credentials),
                name=f"feishu:{self.scope.instance_id}",
                daemon=True,
            )
            process.start()
            child.close()
            self._connection = connection
            self._process = process
            self._reader = asyncio.create_task(self._receive())

    async def _send(self, frame) -> None:
        async with self._send_lock:
            await asyncio.to_thread(self._connection.send, frame)

    async def _dispatch(self, event_id: str, event: str, payload: Any) -> None:
        try:
            handler = self._handlers.get(event)
            if handler:
                if event == "reject":
                    # 兼容业务现有的 reason 属性。
                    from types import SimpleNamespace

                    payload = SimpleNamespace(**payload)
                await handler(payload)
        except Exception:
            LOGGER.exception("Feishu event handling failed instance=%s", self.scope.instance_id)
        finally:
            with suppress(EOFError, OSError):
                await self._send(("ack", event_id))

    async def _receive(self) -> None:
        try:
            while True:
                frame = await asyncio.to_thread(self._connection.recv)
                if frame[0] == "event":
                    task = asyncio.create_task(self._dispatch(*frame[1:]))
                    self._events.add(task)
                    task.add_done_callback(self._events.discard)
                else:
                    _, request_id, success, result = frame
                    future = self._pending.get(request_id)
                    if future and not future.done():
                        if success:
                            future.set_result(result)
                        else:
                            future.set_exception(RuntimeError(result))
        except (EOFError, OSError):
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("Feishu SDK worker disconnected"))

    async def _rpc(self, method: str, *args, **kwargs):
        await self._start()
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send(("call", request_id, method, args, kwargs))
            # connect 是长连接生命周期，其余 API 有界等待。
            if method == "connect":
                return await future
            return await asyncio.wait_for(future, timeout=60)
        finally:
            self._pending.pop(request_id, None)

    async def connect(self) -> None:
        await self._rpc("connect")

    async def resolve(self, user_id: str, *, fallback: str = "神秘奶友") -> str:
        return await self._rpc("resolve", user_id, fallback=fallback)

    async def get_chat_members(self, chat_id: str):
        return await self._rpc("get_chat_members", chat_id)

    async def resolve_image_target(self, message: Message):
        return await self._rpc("resolve_image_target", message)

    async def download_target(self, target):
        return await self._rpc("download_target", target)

    async def send(self, chat_id: str, content: dict, options: dict | None = None):
        try:
            return await self._rpc("send", chat_id, content, options)
        except TimeoutError:
            return SendResult(False, error=SendError(ErrorCode.SEND_TIMEOUT))

    async def update_card(self, message_id: str, card: dict):
        return await self._rpc("update_card", message_id, card)

    def catalog_document(self, document_id: str, section_block_id: str):
        return RemoteCatalogDocument(self, document_id, section_block_id)

    async def close(self) -> None:
        if self._process is None:
            return
        with suppress(Exception):
            await asyncio.wait_for(self._rpc("close"), timeout=8)
        self._connection.close()
        await asyncio.to_thread(self._process.join, 2)
        if self._process.is_alive():
            self._process.terminate()
            await asyncio.to_thread(self._process.join, 2)
        if self._process.is_alive():
            self._process.kill()
            await asyncio.to_thread(self._process.join, 2)
        for task in self._events:
            task.cancel()
        if self._events:
            await asyncio.gather(*self._events, return_exceptions=True)
        if self._reader:
            self._reader.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader
