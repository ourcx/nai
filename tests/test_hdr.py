from __future__ import annotations

import io
import random
from pathlib import Path

import imagecodecs
import numpy as np
import pytest
from PIL import Image

from nailong_classifier.hdr import (
    make_flash_gif,
    make_hdr_image,
    random_milk_image_content,
    validate_hdr_settings,
)


def image_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def test_hdr_contains_gain_map_and_real_values_above_sdr_white() -> None:
    source = Image.new("RGB", (64, 32), "white")
    source.paste("black", (0, 0, 32, 32))

    encoded = make_hdr_image(image_bytes(source), peak_nits=1000)

    assert imagecodecs.ultrahdr_check(encoded)
    with Image.open(io.BytesIO(encoded)) as compatible:
        # 增益图作为第二幅 JPEG 存储，Pillow 会把容器识别为 MPO。
        assert compatible.format in {"JPEG", "MPO"}
        assert compatible.size == source.size
        assert compatible.convert("RGB").getpixel((48, 16))[0] > 200
        assert compatible.convert("RGB").getpixel((16, 16))[0] < 10
    sdr = imagecodecs.ultrahdr_decode(encoded, boost=1.0)
    hdr = imagecodecs.ultrahdr_decode(encoded, boost=5.0)
    assert float(np.max(sdr[..., :3])) <= 1.1
    assert float(np.mean(hdr[8:24, 40:56, :3])) > 3.0
    assert float(np.max(hdr[8:24, 8:24, :3])) < 0.05


def test_hdr_output_strength_follows_target_peak() -> None:
    source = image_bytes(Image.new("RGB", (32, 32), "yellow"))
    low = imagecodecs.ultrahdr_decode(make_hdr_image(source, peak_nits=400), boost=50)
    high = imagecodecs.ultrahdr_decode(make_hdr_image(source, peak_nits=2000), boost=50)
    assert float(np.mean(high[..., :3])) > float(np.mean(low[..., :3]))


def test_hdr_pads_odd_dimensions_and_composites_transparency() -> None:
    source = Image.new("RGBA", (31, 25), (0, 0, 0, 0))

    encoded = make_hdr_image(image_bytes(source))

    with Image.open(io.BytesIO(encoded)) as image:
        assert image.size == (32, 26)
        assert image.convert("RGB").getpixel((16, 16))[0] > 200


def test_hdr_limits_size_and_does_not_stretch_aspect() -> None:
    source = Image.new("RGB", (3200, 800), "gold")
    encoded = make_hdr_image(image_bytes(source))
    with Image.open(io.BytesIO(encoded)) as image:
        assert image.size == (1600, 400)


def test_hdr_gif_uses_first_frame_and_grayscale_works() -> None:
    first = Image.new("RGB", (32, 32), "red")
    output = io.BytesIO()
    first.save(
        output,
        format="GIF",
        save_all=True,
        append_images=[Image.new("RGB", (32, 32), "blue")],
        duration=100,
    )
    encoded = make_hdr_image(output.getvalue())
    with Image.open(io.BytesIO(encoded)) as image:
        red, _, blue = image.convert("RGB").getpixel((16, 16))
        assert red > blue
        # MPO 的第二幅是增益图，Pillow 也会标记 is_animated，不能据此判为 GIF。
        assert image.format in {"JPEG", "MPO"}
        assert "duration" not in image.info
    linear = imagecodecs.ultrahdr_decode(encoded, boost=5)
    assert float(np.mean(linear[..., 0])) > float(np.mean(linear[..., 2]))
    gray = make_hdr_image(image_bytes(Image.new("L", (32, 32), 180)))
    assert imagecodecs.ultrahdr_check(gray)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probability_is_rejected(probability: float) -> None:
    with pytest.raises(ValueError):
        validate_hdr_settings(probability, 1000)


@pytest.mark.parametrize("peak_nits", [0, 203, 10001, float("nan"), float("inf")])
def test_invalid_brightness_is_rejected(peak_nits: float) -> None:
    with pytest.raises(ValueError):
        make_hdr_image(image_bytes(Image.new("RGB", (32, 32), "gold")), peak_nits=peak_nits)


def test_random_flash_push_only_contains_animated_gif_and_keeps_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "milk.png"
    original = image_bytes(Image.new("RGB", (32, 32), "yellow"))
    source.write_bytes(original)
    monkeypatch.setattr("nailong_classifier.hdr.secrets.SystemRandom", lambda: random.Random(1))

    content = random_milk_image_content(source, probability=1.0, peak_nits=1000)

    assert set(content) == {"image"}
    assert set(content["image"]) == {"source"}
    with Image.open(io.BytesIO(content["image"]["source"])) as image:
        assert image.format == "GIF"
        assert image.n_frames == 3
        assert image.info["loop"] == 0
    assert source.read_bytes() == original


def test_disabled_flash_does_not_read_or_convert_missing_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_rng():
        raise AssertionError("关闭 HDR 后不应抽概率或读取原图")

    monkeypatch.setattr("nailong_classifier.hdr.secrets.SystemRandom", unexpected_rng)
    path = tmp_path / "missing.png"
    assert random_milk_image_content(path, probability=0) == {"image": {"source": str(path)}}


def test_probability_miss_preserves_original_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("nailong_classifier.hdr.secrets.SystemRandom", lambda: random.Random(2))
    path = tmp_path / "milk.png"
    assert random_milk_image_content(path, probability=0.2) == {"image": {"source": str(path)}}


@pytest.mark.parametrize("failure", ["bad-image", "missing", "gif"])
def test_flash_conversion_failure_falls_back_to_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = tmp_path / "milk.png"
    if failure == "bad-image":
        path.write_bytes(b"not an image")
    elif failure == "gif":
        path.write_bytes(image_bytes(Image.new("RGB", (32, 32), "gold")))

        def broken_gif(*args, **kwargs):
            raise RuntimeError("GIF encoder failure")

        monkeypatch.setattr("nailong_classifier.hdr.make_flash_gif", broken_gif)

    assert random_milk_image_content(path, probability=1) == {"image": {"source": str(path)}}


def test_sdr_only_encoder_output_is_not_sent_as_hdr(monkeypatch: pytest.MonkeyPatch) -> None:
    source = image_bytes(Image.new("RGB", (32, 32), "yellow"))
    sdr = image_bytes(Image.new("RGB", (32, 32), "yellow"), image_format="JPEG")
    monkeypatch.setattr("nailong_classifier.hdr.imagecodecs.ultrahdr_encode", lambda *a, **k: sdr)
    with pytest.raises(RuntimeError, match="no valid Ultra HDR"):
        make_hdr_image(source)


def test_flash_gif_has_a_brighter_middle_frame() -> None:
    source = image_bytes(Image.new("RGB", (32, 32), (120, 120, 0)))

    with Image.open(io.BytesIO(make_flash_gif(source))) as gif:
        first = gif.convert("RGB").getpixel((16, 16))[0]
        gif.seek(1)
        flash = gif.convert("RGB").getpixel((16, 16))[0]

    assert flash > first
