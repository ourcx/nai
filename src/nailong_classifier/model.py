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
REGULARIZED_ARCHITECTURE = "mobilenet_v3_small_regularized"
SUPPORTED_ARCHITECTURES = (ARCHITECTURE, REGULARIZED_ARCHITECTURE)


class RegularizedMobileNet(nn.Module):
    """复用预训练特征，用卷积瓶颈和池化代替容量较大的全连接头。"""

    def __init__(self, features: nn.Sequential, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.features = features
        # 深度卷积处理局部形状，1×1 卷积压缩通道；不引入高参数量的普通 576 通道卷积。
        # GroupNorm 不依赖小批次统计，避免新增随机初始化层污染预训练 BatchNorm。
        self.adapter = nn.Sequential(
            nn.Conv2d(576, 576, kernel_size=3, padding=1, groups=576, bias=False),
            nn.GroupNorm(16, 576),
            nn.Hardswish(),
            nn.Conv2d(576, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.Hardswish(),
            nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.adapter(self.features(images))
        return self.classifier(torch.flatten(self.avgpool(features), 1))


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


def build_model(
    num_classes: int,
    *,
    pretrained: bool,
    architecture: str = ARCHITECTURE,
    dropout: float = 0.2,
) -> nn.Module:
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if num_classes < 2 or not 0.0 <= dropout < 1.0:
        raise ValueError("num_classes must be at least 2 and dropout must be in [0, 1)")
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    if architecture == REGULARIZED_ARCHITECTURE:
        model = RegularizedMobileNet(model.features, num_classes, dropout)
    else:
        input_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(input_features, num_classes)
        model.classifier[2].p = dropout
    model.architecture = architecture
    model.model_config = {"dropout": dropout}
    return model


def configure_for_finetuning(model: nn.Module, *, unfreeze_blocks: int = 2) -> None:
    if not 0 <= unfreeze_blocks <= len(model.features):
        raise ValueError("unfreeze_blocks must be within the number of feature blocks")
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if unfreeze_blocks:
        for parameter in model.features[-unfreeze_blocks:].parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if hasattr(model, "adapter"):
        for parameter in model.adapter.parameters():
            parameter.requires_grad = True


def freeze_frozen_batchnorm(model: nn.Module) -> None:
    """model.train() 后调用：冻结参数的 BN 也必须冻结均值/方差，后段 BN 仍可学习。"""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
            parameter.requires_grad for parameter in module.parameters()
        ):
            module.eval()


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
            "architecture": getattr(model, "architecture", ARCHITECTURE),
            "model_config": getattr(model, "model_config", {"dropout": 0.2}),
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
    if checkpoint.get("architecture") not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {checkpoint.get('architecture')}")

    class_names = list(checkpoint["class_names"])
    image_size = int(checkpoint["image_size"])
    threshold = float(checkpoint["threshold"])
    # 老 checkpoint 没有 model_config，使用与旧模型完全一致的结构和 Dropout 默认值。
    model = build_model(
        len(class_names),
        pretrained=False,
        architecture=checkpoint["architecture"],
        **checkpoint.get("model_config", {}),
    )
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
