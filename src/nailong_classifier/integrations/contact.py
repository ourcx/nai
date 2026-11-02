from __future__ import annotations

import asyncio
import json
import logging
import time

from lark_channel.client import Client
from lark_channel.core.enum import AccessTokenType, HttpMethod
from lark_channel.core.model import BaseRequest

LOGGER = logging.getLogger(__name__)

# 群消息事件里 sender_name 常为空，退而求其次调通讯录按 open_id 反查真名。
# 查过的名字缓存起来，避免同一个人每次抢答都打一次接口。
FALLBACK_DISPLAY_NAME = "神秘奶友"


class ContactResolver:
    """按 open_id 反查飞书用户昵称，带内存缓存与优雅兜底。

    需要机器人应用开通 `contact:user.base:readonly`（或等价）权限。
    缺权限/查不到时返回兜底昵称，并记一次日志，不中断主流程。
    """

    def __init__(self, client: Client) -> None:
        self.client = client
        self._cache: dict[str, str] = {}
        self._failed_until: dict[str, float] = {}
        self._lock = asyncio.Lock()
        # 没权限时只告警一次，避免刷屏日志。
        self._warned_permission = False

    def remember(self, open_id: str, name: str | None) -> None:
        """缓存消息/@/群成员接口提供的昵称，拿到真实显示名后清除失败缓存。"""
        name = name.strip() if isinstance(name, str) else ""
        if open_id and name:
            self._cache[open_id] = name
            self._failed_until.pop(open_id, None)

    def cached_name(self, open_id: str) -> str | None:
        return self._cache.get(open_id)

    async def resolve(self, open_id: str, *, fallback: str = FALLBACK_DISPLAY_NAME) -> str:
        """返回这个 open_id 对应的昵称；查不到或没权限时返回 fallback。"""
        if not open_id:
            return fallback
        cached = self._cache.get(open_id)
        if cached is not None:
            return cached
        async with self._lock:
            # 拿到锁后再查一次缓存，避免并发重复请求。
            cached = self._cache.get(open_id)
            if cached is not None:
                return cached
            if time.monotonic() < self._failed_until.get(open_id, 0):
                return fallback
            name = await self._fetch_name(open_id)
            # 网络查询期间也可能收到带昵称的消息，优先使用刚收到的显示名。
            cached = self._cache.get(open_id)
            if cached is not None:
                return cached
            if name:
                self.remember(open_id, name)
                return name
            self._failed_until[open_id] = time.monotonic() + 300
        return fallback

    async def _fetch_name(self, open_id: str) -> str | None:
        try:
            response = await self.client.arequest(self._request(open_id))
            payload = json.loads(response.raw.content.decode("utf-8"))
        except Exception:
            LOGGER.exception("Contact lookup failed open_id_suffix=%s", open_id[-6:])
            return None

        code = payload.get("code")
        if code != 0:
            # 1000 系列多为权限不足；只在第一次提醒，后续静默。
            if not self._warned_permission:
                self._warned_permission = True
                LOGGER.warning(
                    "Contact lookup returned code=%s msg=%s; "
                    "检查 contact:user.base:readonly 权限与通讯录可见范围；"
                    "外部成员可能无法通过通讯录读取",
                    code,
                    payload.get("msg"),
                )
            return None
        user = (payload.get("data") or {}).get("user") or {}
        name = user.get("nickname") or user.get("name")
        return str(name) if name else None

    @staticmethod
    def _request(open_id: str) -> BaseRequest:
        request = BaseRequest()
        request.http_method = HttpMethod.GET
        request.uri = "/open-apis/contact/v3/users/:user_id"
        request.token_types = {AccessTokenType.TENANT}
        request.paths = {"user_id": open_id}
        request.add_query("user_id_type", "open_id")
        return request
