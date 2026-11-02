from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .data import DOUBAO_DETECTOR_CLASSES, evaluation_transform
from .model import load_checkpoint, select_device


@dataclass(frozen=True)
class DoubaoPrediction:
    confidence: float
    detected: bool
    scores: dict[str, float]


class DoubaoDetector:
    """独立 CNN：只判断画面中是否出现豆包，不参与奶龙/奶蛙分类。"""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "auto",
        threshold: float | None = None,
    ) -> None:
        self.device = select_device(device)
        (
            self.model,
            class_names,
            self.image_size,
            checkpoint_threshold,
            self.metadata,
        ) = load_checkpoint(model_path, self.device)
        if class_names != DOUBAO_DETECTOR_CLASSES:
            raise ValueError(
                f"Doubao detector classes must be {DOUBAO_DETECTOR_CLASSES}, got {class_names}"
            )
        self.threshold = checkpoint_threshold if threshold is None else threshold
        self.transform = evaluation_transform(self.image_size)
        self._lock = threading.Lock()

    def predict_bytes(self, image_bytes: bytes) -> DoubaoPrediction:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return self.predict_image(image)

    def predict_path(self, image_path: Path) -> DoubaoPrediction:
        with Image.open(image_path) as image:
            return self.predict_image(image)

    def predict_image(self, image: Image.Image) -> DoubaoPrediction:
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        batch = self.transform(prepared).unsqueeze(0).to(self.device)
        with self._lock, torch.inference_mode():
            probabilities = torch.softmax(self.model(batch), dim=1)[0].cpu()
        scores = {
            label: float(probabilities[index].item())
            for index, label in enumerate(DOUBAO_DETECTOR_CLASSES)
        }
        confidence = scores["doubao"]
        return DoubaoPrediction(
            confidence=confidence,
            detected=confidence >= self.threshold,
            scores=scores,
        )
