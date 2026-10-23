from __future__ import annotations

import io
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from nailong_classifier.bot import (
    OTHER_MILKY_REACTIONS,
    build_prediction_card,
    format_concentration,
    format_prediction,
    is_milky_other,
    milk_concentration,
)
from nailong_classifier.milk import combined_milk_concentration, yellow_pixel_ratio
from nailong_classifier.model import NailongPredictor, Prediction


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        ("yellow", 1.0),
        ("gold", 1.0),
        ("#FFEB80", 1.0),
        ("white", 0.0),
        ("gray", 0.0),
        ("black", 0.0),
        ("red", 0.0),
        ("lime", 0.0),
        ("blue", 0.0),
        ("#FFFDEB", 0.0),
        ("#303000", 0.0),
    ],
)
def test_yellow_detector_excludes_neutral_dark_and_other_hues(color: str, expected: float) -> None:
    assert yellow_pixel_ratio(Image.new("RGB", (32, 32), color)) == expected


def test_yellow_detector_measures_full_image_not_center_crop() -> None:
    image = Image.new("RGB", (1024, 512), "blue")
    image.paste("yellow", (0, 0, 256, 512))
    image.paste("yellow", (768, 0, 1024, 512))

    assert yellow_pixel_ratio(image) == pytest.approx(0.5)


def test_transparent_yellow_does_not_count_as_visible_yellow() -> None:
    transparent = Image.new("RGBA", (32, 32), (255, 255, 0, 0))
    assert yellow_pixel_ratio(transparent) == 0.0
    transparent.paste((0, 0, 255, 255), (0, 0, 16, 32))
    assert yellow_pixel_ratio(transparent) == 0.0
    transparent.paste((255, 255, 0, 255), (16, 0, 32, 32))
    assert yellow_pixel_ratio(transparent) == 0.5


def test_yellow_detector_weights_partial_opacity() -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 255, 255))
    image.paste((255, 255, 0, 85), (0, 0, 16, 32))

    assert yellow_pixel_ratio(image) == pytest.approx(0.25)


def test_palette_transparency_and_grayscale_are_supported() -> None:
    palette = Image.new("P", (32, 32), 0)
    palette.putpalette([255, 255, 0, 0, 0, 255])
    palette.info["transparency"] = 0
    palette.paste(1, (0, 0, 16, 32))

    assert yellow_pixel_ratio(palette) == 0.0
    assert yellow_pixel_ratio(Image.new("L", (32, 32), 255)) == 0.0


@pytest.mark.parametrize(
    ("scores", "yellow", "expected"),
    [
        ({"nailong": 0.04, "naiwa": 0.06}, 1.0, 46),
        ({"nailong": 0.04, "naiwa": 0.06}, 0.0, 6),
        ({"nailong": 0.04, "naiwa": 0.06}, 0.5, 26),
        ({"nailong": 0.5, "naiwa": 0.4}, 1.0, 94),
        ({"nailong": 0.5, "naiwa": 0.4}, 0.0, 54),
        ({"nailong": 0.5, "naiwa": 0.4}, None, 90),
        ({}, 1.0, 40),
        ({}, None, 0),
        ({"nailong": 1.0, "naiwa": 1.0}, 2.0, 100),
        ({"nailong": -1.0}, -1.0, 0),
    ],
)
def test_milk_weights_and_legacy_fallback(
    scores: dict, yellow: float | None, expected: int
) -> None:
    assert combined_milk_concentration(scores, yellow) == expected


class FixedOtherClassifier(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.tensor([[0.04, 0.06, 0.90]]))


def make_predictor() -> NailongPredictor:
    # 用固定概率验证图片入口、颜色检测与回复之间的完整连接，不依赖模型权重或重训。
    predictor = NailongPredictor.__new__(NailongPredictor)
    predictor.device = torch.device("cpu")
    predictor.model = FixedOtherClassifier()
    predictor.class_names = ["nailong", "naiwa", "other"]
    predictor.threshold = 0.65
    predictor.transform = pil_to_tensor
    predictor._lock = threading.Lock()
    return predictor


def test_yellow_other_triggers_suspicion_in_text_and_card_without_changing_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictor = make_predictor()
    image = Image.new("RGB", (48, 48), "yellow")
    image_path = tmp_path / "yellow.png"
    image.save(image_path)
    prediction = predictor.predict_path(image_path)
    plain = predictor.predict_image(Image.new("RGB", (48, 48), "blue"))
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        from_bytes = predictor.predict_bytes(output.getvalue())

    assert prediction == from_bytes
    assert prediction.yellow_ratio == 1.0
    assert milk_concentration(prediction) == 46
    assert is_milky_other(prediction)
    assert milk_concentration(plain) == 6
    assert not is_milky_other(plain)
    assert prediction.label == plain.label == "other"
    assert prediction.accepted is plain.accepted is True
    assert prediction.scores == plain.scores
    assert prediction.confidence == plain.confidence == pytest.approx(0.9)

    monkeypatch.setattr("nailong_classifier.bot.secrets.choice", lambda choices: choices[0])
    assert format_prediction(prediction).startswith(OTHER_MILKY_REACTIONS[0])
    concentration = format_concentration(prediction)
    assert "本图奶含量 46%" in concentration
    assert "可能是没认出来" in concentration
    observation = SimpleNamespace(id=1, asset=SimpleNamespace(corrected=False, label="other"))
    card = build_prediction_card(observation, prediction, None)
    assert card["config"]["summary"]["content"] == (
        f"{OTHER_MILKY_REACTIONS[0]}，置信度 90.0%，奶含量 46%"
    )
    assert "可能漏识别了奶龙或奶蛙" in str(card)


def test_predictor_keeps_alpha_for_detection_and_uses_current_gif_frame() -> None:
    predictor = make_predictor()
    transparent = predictor.predict_image(Image.new("RGBA", (32, 32), (255, 255, 0, 0)))
    assert transparent.yellow_ratio == 0.0

    output = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(
        output,
        format="GIF",
        save_all=True,
        append_images=[Image.new("RGB", (32, 32), "yellow")],
        duration=100,
    )
    assert predictor.predict_bytes(output.getvalue()).yellow_ratio == 0.0
    with Image.open(io.BytesIO(output.getvalue())) as gif:
        gif.seek(1)
        assert predictor.predict_image(gif).yellow_ratio == 1.0


def test_missing_yellow_measurement_keeps_existing_concentration() -> None:
    prediction = Prediction(
        label="nailong",
        confidence=0.61,
        accepted=True,
        scores={"nailong": 0.61, "naiwa": 0.30, "other": 0.09},
    )
    assert prediction.yellow_ratio is None
    assert milk_concentration(prediction) == 91
