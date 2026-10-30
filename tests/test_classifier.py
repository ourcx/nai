from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from nailong_classifier.bot import (
    _decode_message_content,
    _find_file_key,
    _find_image_key,
    _raw_parent_id,
    format_prediction,
)
from nailong_classifier.data import (
    ANIMATED_SAMPLE_WEIGHT,
    DEFAULT_CLASS_BALANCE_POWER,
    DOUBAO_DETECTOR_CLASSES,
    build_doubao_detector_evaluation_loader,
    build_doubao_detector_train_val_loaders,
    build_train_val_loaders,
    validate_image_tree,
)
from nailong_classifier.doubao import DoubaoDetector
from nailong_classifier.model import (
    NailongPredictor,
    Prediction,
    build_model,
    load_checkpoint,
    save_checkpoint,
)


def test_dataset_layout_is_valid() -> None:
    expected_classes = {"doubao", "nailong", "naiwa", "other"}
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

    # 默认让每个类别的总采样权重相同；同一类别内 GIF 的权重仍是静态图的
    # ANIMATED_SAMPLE_WEIGHT 倍。
    weights = train_loader.sampler.weights.tolist()
    targets_by_position = [
        train_subset.dataset.targets[index] for index in train_subset.indices
    ]
    class_totals: dict[int, float] = {}
    class_counts: dict[int, int] = {}
    for target, weight in zip(targets_by_position, weights, strict=True):
        class_totals[target] = class_totals.get(target, 0.0) + weight
        class_counts[target] = class_counts.get(target, 0) + 1
    normalized_totals = [
        total / class_counts[target] ** (1.0 - DEFAULT_CLASS_BALANCE_POWER)
        for target, total in class_totals.items()
    ]
    assert max(normalized_totals) - min(normalized_totals) < 1e-9
    for target in class_totals:
        gif_weights = {
            round(weight, 12)
            for path, weight, item_target in zip(
                train_paths, weights, targets_by_position, strict=True
            )
            if item_target == target and path.suffix.lower() == ".gif"
        }
        static_weights = {
            round(weight, 12)
            for path, weight, item_target in zip(
                train_paths, weights, targets_by_position, strict=True
            )
            if item_target == target and path.suffix.lower() != ".gif"
        }
        if gif_weights and static_weights:
            assert len(gif_weights) == 1 and len(static_weights) == 1
            ratio = gif_weights.pop() / static_weights.pop()
            assert abs(ratio - ANIMATED_SAMPLE_WEIGHT) < 1e-9
    images, targets = next(iter(train_loader))
    assert images.shape[1:] == (3, 224, 224)
    assert images.shape[0] == targets.shape[0]


def test_yellow_kangaroo_hard_negative_is_other() -> None:
    predictor = NailongPredictor(Path("artifacts/model.pt"), device="cpu", threshold=0.65)

    prediction = predictor.predict_path(
        Path("data/train/other/v3_0016h_1ed02b0a-b69c-4a62-b947-3dc46dc0b9cg.gif")
    )

    assert predictor.threshold == 0.65
    assert prediction.label == "other"
    assert prediction.accepted is True
    assert prediction.scores["naiwa"] < predictor.threshold


@pytest.mark.parametrize("confidence,accepted", [(0.49, False), (0.50, True), (0.55, True)])
def test_majority_candidate_is_accepted_at_relaxed_threshold(
    monkeypatch, confidence, accepted,
) -> None:
    class CandidateModel(torch.nn.Module):
        def forward(self, images):
            remaining = torch.tensor(1.0).sub(confidence).div(2.0)
            probabilities = torch.stack((remaining, torch.tensor(confidence), remaining))
            return probabilities.log().unsqueeze(0).expand(images.shape[0], -1)

    monkeypatch.setattr(
        "nailong_classifier.model.load_checkpoint",
        lambda path, device: (CandidateModel(), ["nailong", "naiwa", "other"], 32, 0.5, {}),
    )
    predictor = NailongPredictor(Path("unused.pt"), device="cpu")
    prediction = predictor.predict_image(Image.new("RGB", (32, 32), "yellow"))
    assert prediction.label == "naiwa"
    assert prediction.accepted is accepted
    if accepted:
        assert "识别结果：奶蛙" in format_prediction(prediction)


@pytest.mark.parametrize("mismatched_classes", [False, True])
def test_explicit_validation_keeps_all_training_samples_out_of_holdout(
    tmp_path, mismatched_classes,
) -> None:
    classes = ["nailong", "naiwa", "other"]
    for split in ("train", "validation"):
        for label in classes:
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            Image.new("RGB", (32, 32), "yellow" if split == "train" else "blue").save(
                folder / f"{split}.png"
            )
    if mismatched_classes:
        (tmp_path / "validation" / "other").rename(tmp_path / "validation" / "wrong")
        with pytest.raises(RuntimeError, match="Class order"):
            build_train_val_loaders(
                tmp_path / "train", val_dir=tmp_path / "validation", image_size=32,
                batch_size=3, val_fraction=0.25, seed=20261030, num_workers=0,
            )
        return
    train_loader, val_loader, actual_classes = build_train_val_loaders(
        tmp_path / "train", val_dir=tmp_path / "validation", image_size=32,
        batch_size=3, val_fraction=0.25, seed=20261030, num_workers=0,
    )
    assert actual_classes == classes
    assert len(train_loader.dataset) == 3
    assert len(val_loader.dataset) == 3
    assert all(
        Path(path).is_relative_to(tmp_path / "validation") for path, _ in val_loader.dataset.samples
    )
    assert all(
        Path(path).is_relative_to(tmp_path / "train")
        for path, _ in train_loader.dataset.dataset.samples
    )


def test_doubao_detector_loaders_map_all_other_classes_to_clean() -> None:
    train_loader, val_loader, class_names = build_doubao_detector_train_val_loaders(
        Path("data/train"),
        image_size=224,
        batch_size=8,
        val_fraction=0.25,
        seed=20261026,
        num_workers=0,
    )
    test_loader = build_doubao_detector_evaluation_loader(
        Path("data/test"),
        image_size=224,
        batch_size=8,
        num_workers=0,
    )

    assert class_names == DOUBAO_DETECTOR_CLASSES
    train_targets = {
        train_loader.dataset.dataset[index][1] for index in train_loader.dataset.indices
    }
    val_targets = {
        val_loader.dataset.dataset[index][1] for index in val_loader.dataset.indices
    }
    assert train_targets == {0, 1}
    assert val_targets == {0, 1}
    test_targets = [target for _, target in test_loader.dataset]
    assert test_targets.count(1) == 10
    assert test_targets.count(0) == 43


def test_doubao_detector_rejects_non_detector_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model=build_model(3, pretrained=False),
        class_names=["nailong", "naiwa", "other"],
        image_size=224,
        threshold=0.8,
        metadata={},
    )

    with pytest.raises(ValueError, match="Doubao detector classes"):
        DoubaoDetector(checkpoint, device="cpu")


@pytest.mark.parametrize(
    "class_names",
    [["nailong", "naiwa", "other"], ["doubao", "nailong", "naiwa", "other"]],
)
def test_checkpoint_round_trip(tmp_path: Path, class_names: list[str]) -> None:
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
