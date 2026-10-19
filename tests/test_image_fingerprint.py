from __future__ import annotations

import io

from PIL import Image, ImageDraw

from nailong_classifier.image_fingerprint import (
    DUPLICATE_DISTANCE,
    content_hash,
    find_duplicate,
    hash_distance,
    try_content_hash,
)


def draw_sticker(size: int, *, variant: int = 0, mode: str = "RGB") -> Image.Image:
    # 纯色图的 dHash 全是 0，测不出区别，这里画一张有明暗结构的“表情”。
    background = (255, 255, 255, 0) if mode == "RGBA" else "white"
    image = Image.new(mode, (size, size), background)
    draw = ImageDraw.Draw(image)
    scale = size / 100
    draw.ellipse((10 * scale, 10 * scale, 90 * scale, 90 * scale), fill="gold")
    if variant == 0:
        draw.ellipse((28 * scale, 35 * scale, 40 * scale, 47 * scale), fill="black")
        draw.ellipse((60 * scale, 35 * scale, 72 * scale, 47 * scale), fill="black")
        draw.arc(
            (30 * scale, 50 * scale, 70 * scale, 80 * scale),
            0,
            180,
            fill="black",
            width=max(1, int(4 * scale)),
        )
    else:
        draw.rectangle((20 * scale, 20 * scale, 50 * scale, 80 * scale), fill="green")
        draw.polygon(
            [(55 * scale, 75 * scale), (85 * scale, 75 * scale), (70 * scale, 25 * scale)],
            fill="navy",
        )
    return image


def encode(image: Image.Image, image_format: str, **options) -> bytes:
    output = io.BytesIO()
    image.save(output, format=image_format, **options)
    return output.getvalue()


def test_same_sticker_with_different_size_and_format_is_duplicate() -> None:
    # 模拟同一张表情被转发压缩：原图缩放后换格式重新编码，字节 hash 完全不同。
    original = draw_sticker(240)
    original_hash = content_hash(encode(original, "PNG"))
    variants = [
        encode(original.resize((480, 480)), "PNG"),
        encode(original.resize((128, 128)).convert("RGB"), "JPEG", quality=85),
        encode(original.resize((160, 160)), "WEBP", quality=80),
    ]

    for variant in variants:
        assert hash_distance(original_hash, content_hash(variant)) <= DUPLICATE_DISTANCE


def test_gif_is_compared_by_first_frame() -> None:
    still = content_hash(encode(draw_sticker(120), "PNG"))
    frames = [draw_sticker(120), draw_sticker(120, variant=1)]
    animated = content_hash(
        encode(frames[0], "GIF", save_all=True, append_images=frames[1:], duration=100)
    )

    assert hash_distance(still, animated) <= DUPLICATE_DISTANCE


def test_transparent_background_matches_white_background() -> None:
    transparent = content_hash(encode(draw_sticker(120, mode="RGBA"), "PNG"))
    white = content_hash(encode(draw_sticker(120), "JPEG", quality=90))

    assert hash_distance(transparent, white) <= DUPLICATE_DISTANCE


def test_different_stickers_are_not_merged() -> None:
    face = content_hash(encode(draw_sticker(120), "PNG"))
    shapes = content_hash(encode(draw_sticker(120, variant=1), "PNG"))

    assert hash_distance(face, shapes) > DUPLICATE_DISTANCE
    assert find_duplicate(shapes, {"face-token": face}) is None
    assert find_duplicate(face, {"shapes": shapes, "face-token": face}) == "face-token"


def test_unreadable_image_has_no_hash() -> None:
    assert try_content_hash(b"not an image") is None
