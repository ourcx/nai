from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from nailong_classifier.bot import (
    _decode_message_content,
    _find_file_key,
    _find_image_key,
    _raw_parent_id,
    format_prediction,
)
from nailong_classifier.data import (
    ANIMATED_SAMPLE_WEIGHT,
    build_train_val_loaders,
    validate_image_tree,
)
from nailong_classifier.model import (
    NailongPredictor,
    Prediction,
    build_model,
    load_checkpoint,
    save_checkpoint,
)


def test_dataset_layout_is_valid() -> None:
    expected_classes = {"nailong", "naiwa", "other"}
    train = validate_image_tree(Path("data/train"), expected_classes)
    test = validate_image_tree(Path("data/test"), expected_classes)

    assert set(train.class_counts) == expected_classes
    assert set(test.class_counts) == expected_classes
    assert all(count >= 12 for count in train.class_counts.values())
    assert all(count >= 3 for count in test.class_counts.values())


def test_gif_hard_negatives_are_loaded_only_for_training() -> None:
    train_loader, val_loader, _ = build_train_val_loaders(
        Path("data/train"),
        image_size=224,
        batch_size=8,
        val_fraction=0.25,
        seed=20261017,
        num_workers=0,
    )
    train_subset = train_loader.dataset
    val_subset = val_loader.dataset
    train_paths = [
        Path(train_subset.dataset.samples[index][0])
        for index in train_subset.indices
    ]
    val_paths = [
        Path(val_subset.dataset.samples[index][0])
        for index in val_subset.indices
    ]

    assert any(path.suffix.lower() == ".gif" for path in train_paths)
    assert all(path.suffix.lower() != ".gif" for path in val_paths)
    assert float(train_loader.sampler.weights.max().item()) == ANIMATED_SAMPLE_WEIGHT
    images, targets = next(iter(train_loader))
    assert images.shape[1:] == (3, 224, 224)
    assert images.shape[0] == targets.shape[0]


def test_yellow_kangaroo_hard_negative_is_other() -> None:
    predictor = NailongPredictor(Path("artifacts/model.pt"), device="cpu")

    prediction = predictor.predict_path(
        Path("data/train/other/v3_0016h_1ed02b0a-b69c-4a62-b947-3dc46dc0b9cg.gif")
    )

    assert predictor.threshold == 0.60
    assert prediction.label == "other"
    assert prediction.accepted is True
    assert prediction.scores["naiwa"] < predictor.threshold


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    class_names = ["nailong", "naiwa", "other"]
    model = build_model(len(class_names), pretrained=False)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        class_names=class_names,
        image_size=224,
        threshold=0.65,
        metadata={"test": True},
    )

    loaded, loaded_classes, image_size, threshold, metadata = load_checkpoint(
        checkpoint, torch.device("cpu")
    )

    assert loaded.training is False
    assert loaded_classes == class_names
    assert image_size == 224
    assert threshold == 0.65
    assert metadata == {"test": True}


def test_prediction_message_handles_rejection() -> None:
    prediction = Prediction(
        label="naiwa",
        confidence=0.52,
        accepted=False,
        scores={"nailong": 0.31, "naiwa": 0.52, "other": 0.17},
    )

    assert format_prediction(prediction) == "我不太确定。最高候选是「奶蛙」，置信度 52.0%。"


def test_quoted_image_key_is_found_in_message_body() -> None:
    raw = {
        "body": {
            "content": (
                '{"content":[[{"tag":"text","text":"quoted"},'
                '{"tag":"img","image_key":"img_test"}]]}'
            )
        }
    }

    assert _find_image_key(_decode_message_content(raw)) == "img_test"


def test_invalid_quoted_content_has_no_image_key() -> None:
    assert _find_image_key(_decode_message_content({"body": {"content": "not-json"}})) is None


def test_sticker_file_key_is_found_in_message_body() -> None:
    raw = {"body": {"content": '{"file_key":"file_sticker"}'}}

    assert _find_file_key(_decode_message_content(raw)) == "file_sticker"


def test_raw_parent_id_recovers_root_reply_dropped_by_sdk() -> None:
    message = SimpleNamespace(
        raw={"parent_id": "om_parent", "root_id": "om_parent"},
    )

    assert _raw_parent_id(message) == "om_parent"
