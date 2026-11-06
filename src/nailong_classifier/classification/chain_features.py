from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


class DinoEncoder:
    """DINO 只提取接龙检索特征，分类仍沿用现有模型；多平台共用一份权重。"""

    # 固定模型、输入尺寸与预处理版本，防止不同特征空间混用缓存。
    model_id = "timm1.0.22:vit_small_patch14_dinov2.lvd142m:224:white-exif:v1"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._transform = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def encode(self, path: Path) -> np.ndarray:
        import timm
        import torch
        from timm.data import create_transform, resolve_model_data_config

        with self._lock:
            if self._model is None:
                model = timm.create_model(
                    "vit_small_patch14_dinov2.lvd142m",
                    pretrained=True, num_classes=0, img_size=224,
                ).eval()
                config = resolve_model_data_config(model)
                config["input_size"] = (3, 224, 224)
                self._transform = create_transform(**config, is_training=False)
                self._model = model
            with Image.open(path) as image:
                image.seek(0)
                frame = ImageOps.exif_transpose(image).convert("RGBA")
                background = Image.new("RGBA", frame.size, "white")
                background.alpha_composite(frame)
                tensor = self._transform(background.convert("RGB")).unsqueeze(0)
            with torch.inference_mode():
                feature = self._model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(feature)
            if not np.isfinite(feature).all() or norm <= 0:
                raise ValueError("Invalid DINO feature")
            return np.divide(feature, norm)
