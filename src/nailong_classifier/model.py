from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .data import evaluation_transform
from .milk import yellow_pixel_ratio

ARCHITECTURE = "mobilenet_v3_small"


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    accepted: bool
    scores: dict[str, float]
    # 与分类概率分开保存，避免颜色占比改变模型置信度或入典门槛；旧调用可不提供。
    yellow_ratio: float | None = None


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, *, pretrained: bool) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, num_classes)
    return model


def configure_for_finetuning(model: nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.features[-2:].parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    class_names: list[str],
    image_size: int,
    threshold: float,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": ARCHITECTURE,
            "class_names": class_names,
            "image_size": image_size,
            "threshold": threshold,
            "model_state": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[nn.Module, list[str], int, float, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Unsupported architecture: {checkpoint.get('architecture')}")

    class_names = list(checkpoint["class_names"])
    image_size = int(checkpoint["image_size"])
    threshold = float(checkpoint["threshold"])
    model = build_model(len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, class_names, image_size, threshold, dict(checkpoint.get("metadata", {}))


class NailongPredictor:
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
            self.class_names,
            self.image_size,
            checkpoint_threshold,
            self.metadata,
        ) = load_checkpoint(model_path, self.device)
        self.threshold = checkpoint_threshold if threshold is None else threshold
        self.transform = evaluation_transform(self.image_size)
        self._lock = threading.Lock()

    def predict_bytes(self, image_bytes: bytes) -> Prediction:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return self.predict_image(image)

    def predict_path(self, image_path: Path) -> Prediction:
        with Image.open(image_path) as image:
            return self.predict_image(image)

    def predict_image(self, image: Image.Image) -> Prediction:
        oriented = ImageOps.exif_transpose(image)
        yellow_ratio = yellow_pixel_ratio(oriented)
        prepared = oriented.convert("RGB")
        batch = self.transform(prepared).unsqueeze(0).to(self.device)
        with self._lock, torch.inference_mode():
            probabilities = torch.softmax(self.model(batch), dim=1)[0].cpu()
        index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[index].item())
        scores = {
            class_name: float(probabilities[class_index].item())
            for class_index, class_name in enumerate(self.class_names)
        }
        return Prediction(
            label=self.class_names[index],
            confidence=confidence,
            accepted=confidence >= self.threshold,
            scores=scores,
            yellow_ratio=yellow_ratio,
        )
