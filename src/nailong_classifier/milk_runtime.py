"""已验收奶蛙优先模型的生产推理，不依赖实验脚本、训练库或在线下载。"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import math
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms

from .data import DOUBAO_DETECTOR_CLASSES, IMAGENET_MEAN, IMAGENET_STD, evaluation_transform
from .doubao import DoubaoPrediction
from .milk import yellow_pixel_ratio
from .model import Prediction, load_checkpoint, select_device

LOGGER = logging.getLogger(__name__)
CLASSES = ["nailong", "naiwa", "other"]


class WholeView:
    """保留整张图的主体与文字，与验收时的 336 像素补边完全一致。"""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.pad(image, (self.size, self.size), color=(124, 116, 104))


def verified_weight(root: Path, info: dict) -> Path:
    path = (root / info["file"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Release weight must be inside the release directory")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != info["sha256"]:
        raise ValueError(f"Release checkpoint hash mismatch: {path.name}")
    return path


def load_linear(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as saved:
        return {
            key: torch.from_numpy(saved[key]).float().to(device)
            for key in ("coef", "intercept", "mean", "scale")
        }


def encode(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    tokens = model.forward_features(images)
    return torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1)


class MilkPriorityPredictor:
    """主分类与豆包过滤共用一组权重，并缓存同一图片的两次调用结果。

    机器人先过滤豆包、再识别奶图；缓存只留摘要和预测，不留聊天图片，
    避免重复计算两个 DINO 底座，也避免跨接入实例重复加载模型。
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        device: str = "cpu",
        threshold: float | None = None,
        guard_threshold: float | None = None,
    ) -> None:
        import timm
        from safetensors.torch import load_file
        from timm.models.vision_transformer import checkpoint_filter_fn

        self.device = select_device(device)
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest["format"] != "milk-priority-v1":
            raise ValueError("Unsupported milk release format")
        self.threshold = self.manifest["threshold"]
        self.guard_threshold = self.manifest["guard_threshold"]
        # 阈值也是验收版本的一部分。拒绝被旧环境悄悄覆盖，防止上线成绩失真。
        for override, expected in (
            (threshold, self.threshold), (guard_threshold, self.guard_threshold)
        ):
            if override is not None and not math.isclose(override, expected):
                raise ValueError("Threshold override differs from the verified milk release")
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
        paths = {
            name: verified_weight(manifest_path.parent, info)
            for name, info in self.manifest["weights"].items()
        }
        self.online, classes, size, _, _ = load_checkpoint(paths["online"], self.device)
        self.guard, guard_classes, guard_size, _, _ = load_checkpoint(
            paths["old_guard"], self.device
        )
        if classes != CLASSES or guard_classes != DOUBAO_DETECTOR_CLASSES:
            raise ValueError("Release class order differs from the verified models")
        if size != guard_size or size != 224:
            raise ValueError("Release MobileNet image size must be 224")
        self.guard_index = guard_classes.index("doubao")
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False, num_classes=0, img_size=336,
        )
        self.backbone.load_state_dict(
            checkpoint_filter_fn(load_file(str(paths["backbone"])), self.backbone), strict=True
        )
        self.backbone.eval().requires_grad_(False)
        # 微调增量只替换末两层；深拷贝保留与验收完全相同的位置编码插值结果。
        self.fine_backbone = copy.deepcopy(self.backbone)
        saved = torch.load(paths["finetune"], map_location="cpu", weights_only=True)
        if saved["architecture"] != "mlp128" or saved["size"] != 336:
            raise ValueError("Unexpected finetuned head architecture")
        state = self.fine_backbone.state_dict()
        if not set(saved["backbone_delta"]).issubset(state):
            raise ValueError("Finetuned delta has unknown backbone parameters")
        state.update(saved["backbone_delta"])
        self.fine_backbone.load_state_dict(state, strict=True)
        self.fine_head = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU(),
            nn.Dropout(.5), nn.Linear(128, 3),
        )
        self.fine_head.load_state_dict({
            key.removeprefix("layers."): value for key, value in saved["head_state"].items()
        })
        del saved, state
        self.backbone.to(self.device)
        self.fine_backbone.eval().to(self.device)
        self.fine_head.eval().to(self.device)
        self.joint_head = load_linear(paths["joint"], self.device)
        self.guard_head = load_linear(paths["new_guard"], self.device)
        self.center_transform = evaluation_transform(224)
        self.whole_transform = transforms.Compose([
            WholeView(336), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[Prediction, DoubaoPrediction]] = OrderedDict()
        self.doubao_detector = SharedMilkGuard(self)
        LOGGER.info(
            "Milk release loaded release=%s manifest=%s device=%s threshold=%.3f "
            "doubao_threshold=%.3f weights_verified=%d threads=%d",
            self.manifest["release"], manifest_path, self.device, self.threshold,
            self.guard_threshold, len(paths), torch.get_num_threads(),
        )

    def forward(self, center: torch.Tensor, whole: torch.Tensor) -> dict[str, np.ndarray]:
        """按验收顺序执行混合、旧奶蛙保护和普通图兜底，过滤概率独立返回。"""
        with torch.inference_mode():
            center, whole = center.to(self.device), whole.to(self.device)
            mobile = self.online.avgpool(self.online.features(center)).flatten(1)
            online = torch.softmax(self.online.classifier(mobile), 1)
            dino = encode(self.backbone, whole)
            features = torch.cat([
                nn.functional.normalize(mobile, dim=1),
                nn.functional.normalize(dino, dim=1),
            ], dim=1)
            joint = self.joint_head
            scaled = torch.div(torch.sub(features, joint["mean"]), joint["scale"])
            joint_probabilities = torch.softmax(
                nn.functional.linear(scaled, joint["coef"], joint["intercept"]), 1
            )
            fine_features = nn.functional.normalize(encode(self.fine_backbone, whole), dim=1)
            foundation = torch.softmax(self.fine_head(fine_features), 1)
            config = self.manifest["selection"]
            selected = torch.stack([
                torch.mul(foundation, config["weights"][0]),
                torch.mul(joint_probabilities, config["weights"][1]),
            ]).sum(0)
            policy = config["preserve_policy"]
            preserve = torch.logical_and(
                torch.logical_and(
                    online.argmax(1).eq(1), online[:, 1].ge(policy["online_confidence"])
                ),
                selected[:, 1].ge(policy["mixed_naiwa_minimum"]),
            )
            selected = torch.where(preserve.unsqueeze(1), online, selected)
            guard_mobile = self.guard.avgpool(self.guard.features(center)).flatten(1)
            old_guard = torch.softmax(self.guard.classifier(guard_mobile), 1)[:, self.guard_index]
            guard_features = torch.cat([guard_mobile, dino], dim=1)
            guard = self.guard_head
            scaled = torch.div(torch.sub(guard_features, guard["mean"]), guard["scale"])
            logits = torch.add(nn.functional.linear(scaled, guard["coef"]), guard["intercept"])
            new_guard = torch.sigmoid(logits[:, 0])
            if config["unknown_naiwa_fallback"]:
                fallback = torch.logical_and(
                    selected.argmax(1).eq(2),
                    torch.logical_and(
                        torch.logical_and(online.argmax(1).eq(1), online[:, 1].ge(.5)),
                        old_guard.lt(.3),
                    ),
                )
                selected = torch.where(fallback.unsqueeze(1), online, selected)
            return {
                "selected": selected.cpu().numpy(), "online": online.cpu().numpy(),
                "old_guard": old_guard.cpu().numpy(), "new_guard": new_guard.cpu().numpy(),
            }

    def _predict_image(self, image: Image.Image) -> tuple[Prediction, DoubaoPrediction]:
        oriented = ImageOps.exif_transpose(image)
        prepared = oriented.convert("RGB")
        values = self.forward(
            self.center_transform(prepared).unsqueeze(0),
            self.whole_transform(prepared).unsqueeze(0),
        )
        probabilities = values["selected"][0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        guard_probability = float(values["new_guard"][0])
        return (
            Prediction(
                label=CLASSES[index], confidence=confidence,
                accepted=confidence >= self.threshold and guard_probability < self.guard_threshold,
                scores=dict(zip(CLASSES, probabilities.tolist(), strict=True)),
                yellow_ratio=yellow_pixel_ratio(oriented),
            ),
            DoubaoPrediction(
                confidence=guard_probability, detected=guard_probability >= self.guard_threshold,
                scores={
                    "doubao": guard_probability,
                    "clean": float(np.subtract(1., guard_probability)),
                },
            ),
        )

    def _predict_bytes(self, data: bytes) -> tuple[Prediction, DoubaoPrediction]:
        key = hashlib.sha256(data).hexdigest()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
            with Image.open(io.BytesIO(data)) as image:
                result = self._predict_image(image)
            self._cache[key] = result
            if len(self._cache) > 8:
                self._cache.popitem(last=False)
            return result

    def predict_bytes(self, data: bytes) -> Prediction:
        return self._predict_bytes(data)[0]

    def predict_path(self, path: Path) -> Prediction:
        return self.predict_bytes(path.read_bytes())

    def predict_image(self, image: Image.Image) -> Prediction:
        with self._lock:
            return self._predict_image(image)[0]


class SharedMilkGuard:
    """复用同次神经网络推理的豆包概率，保留机器人现有过滤调用协议。"""

    def __init__(self, predictor: MilkPriorityPredictor) -> None:
        self.predictor = predictor
        self.device = predictor.device
        self.threshold = predictor.guard_threshold

    def predict_bytes(self, data: bytes) -> DoubaoPrediction:
        return self.predictor._predict_bytes(data)[1]

    def predict_path(self, path: Path) -> DoubaoPrediction:
        return self.predict_bytes(path.read_bytes())
