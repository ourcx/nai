from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .storage import BotStore


class SerializedPredictor:
    """多个接入实例共用一份模型，串行前向计算，避免同时争抢 CPU/GPU。"""

    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor
        self._lock = threading.Lock()

    @property
    def device(self):
        return self.predictor.device

    @property
    def threshold(self):
        return self.predictor.threshold

    def predict_bytes(self, data: bytes):
        with self._lock:
            return self.predictor.predict_bytes(data)

    def predict_path(self, path: Path):
        with self._lock:
            return self.predictor.predict_path(path)


@dataclass(frozen=True)
class BackendServices:
    """所有 adapter 调用同一组后端能力，不随机器人重复加载模型或图库。

    当前部署在同一进程内；统一协议不依赖 HTTP，后续可以替换为远程服务客户端。
    BotStore 自身的事务继续负责抢答、纠正、日报和资源去重。
    """

    predictor: SerializedPredictor
    store: BotStore
    doubao_detector: SerializedPredictor | None = None
    catalog_sync_events: list[asyncio.Event] = field(default_factory=list)

    def notify_catalog_changed(self) -> None:
        """任何平台收录或纠正图片后，唤醒所有配置了图鉴同步的接入实例。"""
        for event in self.catalog_sync_events:
            event.set()
