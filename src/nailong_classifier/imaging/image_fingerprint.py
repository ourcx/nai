from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path

from PIL import Image, ImageOps

# 16x16 dHash 共 256 位。8x8（64 位）在“字节跳动/字节奶动”这类相似异图上距离只有个位数，区分不开。
HASH_SIZE = 16
# 线上文档实测：同一张图换尺寸/压缩/静态与 GIF 首帧，距离 0~7；最相近的异图距离 14。
# 取 10 作为“内容相同”的上限，给压缩噪声留余量，同时不误合相似异图。
DUPLICATE_DISTANCE = 10


def content_hash(image_bytes: bytes) -> str:
    """把图片归一化后计算内容指纹，让同一张图的不同尺寸、格式和压缩版本得到相近的值。

    归一化步骤：只取首帧（GIF 和静态图按首帧比较），按 EXIF 摆正方向，透明底合成到白底，
    转灰度后缩放到 17x16，再比较每行相邻像素的明暗得到 dHash。
    """
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.seek(0)
        frame = ImageOps.exif_transpose(image)
        if frame.mode in {"RGBA", "LA", "P"}:
            frame = frame.convert("RGBA")
            background = Image.new("RGBA", frame.size, "white")
            background.alpha_composite(frame)
            frame = background
        pixels = (
            frame.convert("L")
            .resize((HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS)
            .load()
        )
    bits = 0
    for row in range(HASH_SIZE):
        for column in range(HASH_SIZE):
            bits = bits << 1 | (pixels[column, row] > pixels[column + 1, row])
    return f"{bits:0{HASH_SIZE * HASH_SIZE // 4}x}"


def try_content_hash(image_bytes: bytes) -> str | None:
    """同 content_hash，但图片解析不了时返回 None，调用方据此跳过比较而不是中断同步。"""
    try:
        return content_hash(image_bytes)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def try_file_content_hash(path: Path) -> str | None:
    try:
        return try_content_hash(path.read_bytes())
    except OSError:
        return None


def hash_distance(left: str, right: str) -> int:
    """两个指纹不同的位数，越小越像。"""
    return (int(left, 16) ^ int(right, 16)).bit_count()


def find_duplicate(candidate: str, known_hashes: Mapping[str, str]) -> str | None:
    """known_hashes 是「素材 token → 指纹」。

    返回与 candidate 内容相同的第一个 token，没有则返回 None。
    """
    for token, known in known_hashes.items():
        if hash_distance(candidate, known) <= DUPLICATE_DISTANCE:
            return token
    return None
