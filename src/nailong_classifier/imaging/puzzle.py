from __future__ import annotations

import io
import random

from PIL import Image, ImageFilter, ImageOps

# 盲盒出题默认只露原图约 30% 宽高的一块局部，放大后发出来，让人只能靠细节猜，不再是一眼就能认出。
DEFAULT_REVEAL_FRACTION = 0.30
# 裁出的局部放大到这个边长再发送，避免小图在群里糊成一团看不清。
PUZZLE_OUTPUT_SIZE = 360
# 放大后再叠一层高斯模糊，连局部细节也抹掉，和裁剪一起把“一眼认出”的线索压到最低。
DEFAULT_BLUR_RADIUS = 9.0


def make_puzzle_image(
    image_bytes: bytes,
    *,
    fraction: float = DEFAULT_REVEAL_FRACTION,
    blur_radius: float = DEFAULT_BLUR_RADIUS,
    rng: random.Random | None = None,
) -> bytes:
    """把一张图裁成随机位置的局部、放大并高斯模糊，作为盲盒猜图的题面。

    归一化步骤和内容指纹一致：取首帧、按 EXIF 摆正、透明底合成白底，保证 GIF 和带透明通道的
    表情都能正常出题。裁剪框大小为原图的 fraction 倍，位置随机；放大到固定边长后再叠一层
    半径 blur_radius 的高斯模糊，裁剪藏轮廓、模糊抹细节，双重遮挡避免一眼认出。
    blur_radius 传 0 则只裁剪不模糊。传入 rng 可让测试得到确定结果。失败时由调用方回退发完整原图。
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if blur_radius < 0:
        raise ValueError("blur_radius must be non-negative")
    chooser = rng or random.Random()
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.seek(0)
        frame = ImageOps.exif_transpose(image)
        if frame.mode in {"RGBA", "LA", "P"}:
            frame = frame.convert("RGBA")
            background = Image.new("RGBA", frame.size, "white")
            background.alpha_composite(frame)
            frame = background
        frame = frame.convert("RGB")
        width, height = frame.size

        crop_w = max(1, int(width * fraction))
        crop_h = max(1, int(height * fraction))
        left = chooser.randint(0, max(0, width - crop_w))
        top = chooser.randint(0, max(0, height - crop_h))
        patch = frame.crop((left, top, left + crop_w, top + crop_h))
        # 放大到固定边长，保持长宽比，短边补白，既看得清又不暴露原图整体轮廓。
        patch = ImageOps.contain(patch, (PUZZLE_OUTPUT_SIZE, PUZZLE_OUTPUT_SIZE))
        canvas = Image.new("RGB", (PUZZLE_OUTPUT_SIZE, PUZZLE_OUTPUT_SIZE), "white")
        offset = (
            (PUZZLE_OUTPUT_SIZE - patch.width) // 2,
            (PUZZLE_OUTPUT_SIZE - patch.height) // 2,
        )
        canvas.paste(patch, offset)
        if blur_radius > 0:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    output = io.BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def try_make_puzzle_image(
    image_bytes: bytes,
    *,
    fraction: float = DEFAULT_REVEAL_FRACTION,
    blur_radius: float = DEFAULT_BLUR_RADIUS,
    rng: random.Random | None = None,
) -> bytes | None:
    """同 make_puzzle_image，但图片解析不了时返回 None，调用方据此回退发完整原图。"""
    try:
        return make_puzzle_image(
            image_bytes,
            fraction=fraction,
            blur_radius=blur_radius,
            rng=rng,
        )
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
