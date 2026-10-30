from __future__ import annotations

import argparse
import io
import logging
import math
import secrets
from pathlib import Path

import imagecodecs
import numpy as np
from PIL import Image, ImageCms, ImageOps

LOGGER = logging.getLogger(__name__)
DEFAULT_HDR_PROBABILITY = 0.2
DEFAULT_HDR_PEAK_NITS = 1000.0
SDR_WHITE_NITS = 203.0
HDR_MAX_SIZE = (1600, 1600)


def validate_hdr_settings(probability: float, peak_nits: float) -> None:
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("HDR probability must be between 0 and 1")
    if not math.isfinite(peak_nits) or not SDR_WHITE_NITS < peak_nits <= 10000.0:
        raise ValueError("HDR peak brightness must be above 203 and at most 10000 nits")


def make_hdr_image(image_bytes: bytes, *, peak_nits: float = DEFAULT_HDR_PEAK_NITS) -> bytes:
    """把奶图当前帧编码成真正的 Ultra HDR JPEG，包含 SDR 底图和 HDR 增益图。

    在线性 sRGB 中扩大亮度范围，黑色保持黑色；1.0 对应 203 尼特 SDR 参考白。
    nits 只是编码目标，实际亮度取决于屏幕、客户端和上传链路是否保留增益图。
    GIF 输出当前首帧静态图，不引入闪烁。只返回衍生图片，原图库文件不覆盖。
    """
    validate_hdr_settings(1.0, peak_nits)
    if not imagecodecs.ULTRAHDR.available:
        raise RuntimeError("Ultra HDR encoder is not available")
    with Image.open(io.BytesIO(image_bytes)) as image:
        oriented = ImageOps.exif_transpose(image)
        rgba = oriented.convert("RGBA")
        rgba.thumbnail(HDR_MAX_SIZE, Image.Resampling.LANCZOS)
        # JPEG 没有透明通道，用白底合成，避免隐藏 RGB 造成透明区域变黑。
        background = Image.new("RGBA", rgba.size, "white")
        prepared = Image.alpha_composite(background, rgba).convert("RGB")
        icc_profile = image.info.get("icc_profile")
        if icc_profile:
            prepared = ImageCms.profileToProfile(
                prepared,
                ImageCms.ImageCmsProfile(io.BytesIO(icc_profile)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )

    # libultrahdr 的 YUV 编码要求宽高为偶数；仅补一行/列边缘像素，不拉伸原图。
    pixels = np.asarray(prepared, dtype=np.float32)
    padding = (
        (0, int(np.remainder(prepared.height, 2))),
        (0, int(np.remainder(prepared.width, 2))),
        (0, 0),
    )
    pixels = np.pad(pixels, padding, mode="edge")
    srgb = np.divide(pixels, 255.0)
    linear = np.where(
        srgb <= 0.04045,
        np.divide(srgb, 12.92),
        np.power(np.divide(np.add(srgb, 0.055), 1.055), 2.4),
    )
    highlights = np.multiply(linear, np.divide(peak_nits, SDR_WHITE_NITS))
    alpha = np.ones((*highlights.shape[:2], 1), dtype=np.float32)
    hdr_pixels = np.concatenate([highlights, alpha], axis=2).astype(np.float16)
    encoded = imagecodecs.ultrahdr_encode(
        hdr_pixels,
        level=92,
        scale=1,
        gamut=imagecodecs.ULTRAHDR.CG.BT_709,
        transfer=imagecodecs.ULTRAHDR.CT.LINEAR,
        nits=peak_nits,
    )
    if not imagecodecs.ultrahdr_check(encoded):
        raise RuntimeError("Encoder output has no valid Ultra HDR gain map")
    return bytes(encoded)


def random_milk_image_content(
    source: Path,
    *,
    probability: float = DEFAULT_HDR_PROBABILITY,
    peak_nits: float = DEFAULT_HDR_PEAK_NITS,
) -> dict:
    """为主动推送选择原图或聊天内可播放的闪亮 GIF，始终只发图片。

    飞书会过滤或丢失 Ultra HDR 的 gain map，因此群聊不再直接上传 HDR JPEG。
    保留原配置名和触发概率，命中后生成普通 GIF，在客户端用亮度脉冲模拟闪亮。
    文件读取与编码由调用方放到工作线程中，避免阻塞群聊消息处理。
    """
    validate_hdr_settings(probability, peak_nits)
    content = {"image": {"source": str(source)}}
    if probability <= 0 or secrets.SystemRandom().random() >= probability:
        return content
    try:
        encoded = make_flash_gif(source.read_bytes())
    except Exception:
        LOGGER.exception("Flash GIF conversion failed, sending original image path=%s", source)
        return content
    LOGGER.info("Prepared flash milk GIF bytes=%d", len(encoded))
    return {"image": {"source": encoded}}


def make_flash_gif(image_bytes: bytes) -> bytes:
    """把一张奶图做成客户端可播放的短 GIF，模拟高光闪一下的效果。"""
    with Image.open(io.BytesIO(image_bytes)) as image:
        frame = ImageOps.exif_transpose(image).convert("RGBA")
        frame.thumbnail(HDR_MAX_SIZE, Image.Resampling.LANCZOS)
        background = Image.new("RGBA", frame.size, "white")
        base = Image.alpha_composite(background, frame).convert("RGB")

    # 三帧短循环：原图 -> 高亮闪白 -> 原图。使用 SDR GIF，避免飞书过滤 HDR gain map。
    flash = Image.blend(base, Image.new("RGB", base.size, "white"), 0.38)
    output = io.BytesIO()
    base.save(
        output,
        format="GIF",
        save_all=True,
        append_images=[flash, base],
        duration=[300, 120, 600],
        loop=0,
        disposal=2,
        optimize=True,
    )
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a milk image to Ultra HDR JPEG")
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--peak-nits", type=float, default=DEFAULT_HDR_PEAK_NITS)
    args = parser.parse_args()
    if args.image.resolve() == args.output.resolve():
        parser.error("Output must not overwrite the original image")
    encoded = make_hdr_image(args.image.read_bytes(), peak_nits=args.peak_nits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(f"Ultra HDR JPEG saved: {args.output} (target peak {args.peak_nits:g} nits)")


if __name__ == "__main__":
    main()
