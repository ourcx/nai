from __future__ import annotations

from collections.abc import Mapping

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

# 模型判断承担主要权重，颜色补充漏识别线索；加权读数用于娱乐，不是分类置信度。
MODEL_MILK_WEIGHT = 0.6
YELLOW_WEIGHT = 0.4
YELLOW_SAMPLE_SIZE = (256, 256)
# Pillow 的 HSV 三通道均为 0~255。此色相范围约为 35°~78°，覆盖金黄和浅黄；
# 饱和度、亮度门槛排除白灰色、低饱和米色及暗色，避免只因偏暖就判为黄色。
YELLOW_HUE_MIN = 25
YELLOW_HUE_MAX = 55
YELLOW_SATURATION_MIN = 64
YELLOW_VALUE_MIN = 64


def yellow_pixel_ratio(image: Image.Image) -> float:
    """检测当前帧可见像素中的黄色占比；缩略全图保留边缘，不跟随模型中心裁剪。

    透明像素可能藏有黄色 RGB，所以用 alpha 对分子、分母加权，完全透明区域不计入。
    不遍历 GIF 后续帧，与当前分类器识别同一帧。
    """
    sampled = image.convert("RGBA")
    sampled.thumbnail(YELLOW_SAMPLE_SIZE, Image.Resampling.BOX)
    alpha = pil_to_tensor(sampled)[3].to(dtype=torch.float64)
    visible = torch.sum(alpha)
    if visible.item() == 0:
        return 0.0

    hue, saturation, value = pil_to_tensor(sampled.convert("RGB").convert("HSV"))
    yellow = (
        (hue >= YELLOW_HUE_MIN)
        & (hue <= YELLOW_HUE_MAX)
        & (saturation >= YELLOW_SATURATION_MIN)
        & (value >= YELLOW_VALUE_MIN)
    )
    yellow_area = torch.sum(torch.mul(yellow, alpha))
    return float(torch.div(yellow_area, visible).clamp(0.0, 1.0).item())


def combined_milk_concentration(
    scores: Mapping[str, float], yellow_ratio: float | None
) -> int:
    """合成 0~100 的奶含量；旧调用没有颜色信息时沿用模型读数，不把未知当零。

    计算复用现有 PyTorch，权重固定为模型 60%、黄色 40%，不修改三分类概率。
    """
    milk_scores = torch.tensor(
        [scores.get("nailong", 0.0), scores.get("naiwa", 0.0)], dtype=torch.float64
    )
    model_milk = torch.sum(milk_scores).clamp(0.0, 1.0)
    if yellow_ratio is None:
        combined = model_milk
    else:
        components = torch.stack(
            [model_milk, torch.tensor(yellow_ratio, dtype=torch.float64).clamp(0.0, 1.0)]
        )
        weights = torch.tensor([MODEL_MILK_WEIGHT, YELLOW_WEIGHT], dtype=torch.float64)
        combined = torch.sum(torch.mul(components, weights))
    return int(torch.round(torch.mul(combined, 100)).clamp(0, 100).item())
