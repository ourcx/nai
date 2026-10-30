from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nailong_classifier.data import (
    ROBUST_AUGMENTATION,
    RandomChatDegradation,
    build_train_val_loaders,
    evaluation_transform,
    training_transform,
)
from nailong_classifier.model import (
    ARCHITECTURE,
    REGULARIZED_ARCHITECTURE,
    build_model,
    configure_for_finetuning,
    load_checkpoint,
    save_checkpoint,
)
from nailong_classifier.train import (
    build_optimizer,
    parse_args,
    run_epoch,
    train,
    validate_training_args,
)


@pytest.mark.parametrize("architecture", [ARCHITECTURE, REGULARIZED_ARCHITECTURE])
@pytest.mark.parametrize("num_classes", [2, 3, 4])
def test_checkpoint_preserves_logits_and_dropout(tmp_path, architecture, num_classes):
    model = build_model(
        num_classes, pretrained=False, architecture=architecture, dropout=0.45
    ).eval()
    images = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        expected = model(images)
    checkpoint = tmp_path / "candidate.pt"
    save_checkpoint(
        checkpoint, model=model, class_names=[str(i) for i in range(num_classes)],
        image_size=64, threshold=0.5, metadata={"test": True},
    )
    loaded, classes, size, threshold, metadata = load_checkpoint(
        checkpoint, torch.device("cpu")
    )
    assert loaded.architecture == architecture
    assert loaded.classifier[2].p == 0.45
    assert (len(classes), size, threshold, metadata) == (num_classes, 64, 0.5, {"test": True})
    with torch.inference_mode():
        torch.testing.assert_close(loaded(images), expected, rtol=0, atol=0)


def test_preexisting_checkpoint_without_model_config_still_loads(tmp_path):
    model = build_model(2, pretrained=False).eval()
    checkpoint = tmp_path / "legacy.pt"
    torch.save({
        "architecture": ARCHITECTURE, "class_names": ["clean", "doubao"],
        "image_size": 64, "threshold": 0.3, "model_state": model.state_dict(),
    }, checkpoint)
    loaded, *_ = load_checkpoint(checkpoint, torch.device("cpu"))
    assert loaded.classifier[2].p == 0.2
    assert list(loaded.state_dict()) == list(model.state_dict())
    images = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(images), model(images), rtol=0, atol=0)


def test_real_step_freezes_old_bn_and_trains_new_convolutions():
    model = build_model(3, pretrained=False, architecture=REGULARIZED_ARCHITECTURE)
    configure_for_finetuning(model)
    optimizer = build_optimizer(model, parse_args([]))
    parameter_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {id(p) for p in model.parameters() if p.requires_grad}
    for group in optimizer.param_groups:
        assert all(p.ndim > 1 for p in group["params"]) if group["weight_decay"] else all(
            p.ndim == 1 for p in group["params"]
        )
    frozen_bn = model.features[0][1]
    active_bn = model.features[-1][1]
    old_frozen_mean = frozen_bn.running_mean.clone()
    old_active_mean = active_bn.running_mean.clone()
    old_weights = model.adapter[0].weight.detach().clone()
    old_frozen_weights = model.features[0][0].weight.detach().clone()
    loader = DataLoader(TensorDataset(
        torch.randn(6, 3, 64, 64), torch.tensor([0, 1, 2, 0, 1, 2])
    ), batch_size=3)
    loss, accuracy = run_epoch(
        model, loader, criterion=nn.CrossEntropyLoss(), device=torch.device("cpu"),
        optimizer=optimizer, freeze_batchnorm=True, max_grad_norm=1.0,
    )
    assert np.isfinite(loss) and 0 <= accuracy <= 1
    assert torch.equal(frozen_bn.running_mean, old_frozen_mean)
    assert torch.equal(model.features[0][0].weight, old_frozen_weights)
    assert not torch.equal(active_bn.running_mean, old_active_mean)
    assert not torch.equal(model.adapter[0].weight, old_weights)
    gradient_norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(p.grad) for p in model.parameters() if p.grad is not None
    ]))
    assert gradient_norm <= 1.00001


def _image_tree(root: Path) -> None:
    generator = np.random.default_rng(20261030)
    for split in ("train", "val", "test"):
        for label in ("nailong", "naiwa", "other"):
            folder = root / split / label
            folder.mkdir(parents=True)
            Image.fromarray(generator.integers(
                0, 256, size=(48, 64, 3), dtype=np.uint8
            )).save(folder / "sample.png")


def test_robust_augmentation_preserves_files_and_evaluation_is_deterministic(
    tmp_path, monkeypatch,
):
    _image_tree(tmp_path)
    source = tmp_path / "train" / "nailong" / "sample.png"
    original = source.read_bytes()
    monkeypatch.setattr("nailong_classifier.data.random.random", lambda: 0.0)
    with Image.open(source) as image:
        degraded = RandomChatDegradation()(image.convert("RGB"))
        assert degraded.size == image.size
        assert not np.array_equal(np.asarray(degraded), np.asarray(image))
        for _ in range(5):
            tensor = training_transform(64, profile=ROBUST_AUGMENTATION)(image)
            assert tensor.shape == (3, 64, 64) and torch.isfinite(tensor).all()
        transform = evaluation_transform(64)
        torch.testing.assert_close(transform(image), transform(image), rtol=0, atol=0)
    train_loader, val_loader, _ = build_train_val_loaders(
        tmp_path / "train", val_dir=tmp_path / "val", image_size=64, batch_size=3,
        val_fraction=0.25, seed=1, num_workers=0, augmentation_profile=ROBUST_AUGMENTATION,
    )
    assert source.read_bytes() == original
    assert len(train_loader.dataset) == 3
    assert len(val_loader.dataset) == 3
    assert not any(
        isinstance(transform, RandomChatDegradation)
        for transform in val_loader.dataset.transform.transforms
    )


@pytest.mark.parametrize("architecture", [ARCHITECTURE, REGULARIZED_ARCHITECTURE])
def test_training_smoke_migrates_backbone_and_serializes_configuration(
    tmp_path, monkeypatch, architecture,
):
    _image_tree(tmp_path)
    initial = tmp_path / "initial.pt"
    source_model = build_model(3, pretrained=False)
    frozen_weight = source_model.features[0][0].weight.detach().clone()
    save_checkpoint(
        initial, model=source_model, class_names=["nailong", "naiwa", "other"],
        image_size=64, threshold=0.5, metadata={"trained_at": "source"},
    )
    args = parse_args([
        "--data-dir", str(tmp_path), "--validation-dir", str(tmp_path / "val"),
        "--output-dir", str(tmp_path / "output"), "--initial-model", str(initial),
        "--architecture", architecture, "--epochs", "1", "--batch-size", "3",
        "--image-size", "64", "--device", "cpu", "--skip-test-evaluation",
    ])
    # 留出测试集只在候选固定后使用，烟测也验证这一门禁。
    real_evaluate = __import__(
        "nailong_classifier.train", fromlist=["evaluate_loader"]
    ).evaluate_loader

    def validation_only(model, loader, **kwargs):
        assert Path(loader.dataset.root) == tmp_path / "val"
        return real_evaluate(model, loader, **kwargs)

    monkeypatch.setattr("nailong_classifier.train.evaluate_loader", validation_only)
    checkpoint = train(args)
    loaded, _, _, _, metadata = load_checkpoint(checkpoint, torch.device("cpu"))
    assert loaded.architecture == architecture
    torch.testing.assert_close(loaded.features[0][0].weight, frozen_weight, rtol=0, atol=0)
    assert metadata["training_config"]["weight_decay"] == args.weight_decay
    assert metadata["training_config"]["dropout"] == args.dropout
    assert metadata["augmentation_profile"] == ROBUST_AUGMENTATION
    assert metadata["frozen_batchnorm_statistics"] is True
    metrics = json.loads((tmp_path / "output" / "metrics.json").read_text())
    assert metrics["test"] is None
    assert len(metrics["history"]) == 1


@pytest.mark.parametrize("arguments", [
    ["--epochs", "0"], ["--max-grad-norm", "0"], ["--weight-decay", "-1"],
    ["--dropout", "1"], ["--label-smoothing", "nan"], ["--unfreeze-blocks", "-1"],
])
def test_bad_training_configuration_is_rejected(arguments):
    with pytest.raises(ValueError):
        validate_training_args(parse_args(arguments))


def test_source_balance_preserves_class_mass_and_normalizes_missing_sources(tmp_path):
    _image_tree(tmp_path)
    extra = tmp_path / "train" / "naiwa" / "new.png"
    Image.new("RGB", (64, 48), "green").save(extra)
    sources = {
        str(path.resolve()): "new" if path == extra else "old"
        for path in (tmp_path / "train").glob("*/*")
    }
    loader, _, _ = build_train_val_loaders(
        tmp_path / "train", val_dir=tmp_path / "val", image_size=64, batch_size=3,
        val_fraction=0.25, seed=1, num_workers=0, sampling_sources=sources,
        source_weights={"old": 0.6, "new": 0.4, "absent": 1.0},
    )
    weights = loader.sampler.weights.numpy()
    samples = loader.dataset.dataset.samples
    for target in (0, 1, 2):
        np.testing.assert_allclose(
            np.sum([w for w, (_, t) in zip(weights, samples, strict=True) if t == target]),
            1.0,
        )
    np.testing.assert_allclose(weights[1:3], [0.4, 0.6])
    with pytest.raises(ValueError, match="Missing source"):
        build_train_val_loaders(
            tmp_path / "train", val_dir=tmp_path / "val", image_size=64, batch_size=3,
            val_fraction=0.25, seed=1, num_workers=0, sampling_sources={},
            source_weights={"old": 1.0},
        )


def test_freezing_all_bn_preserves_statistics_while_last_block_learns():
    model = build_model(3, pretrained=False)
    configure_for_finetuning(model, unfreeze_blocks=1)
    optimizer = build_optimizer(model, parse_args([]))
    old_mean = model.features[-1][1].running_mean.clone()
    old_weight = model.features[-1][0].weight.detach().clone()
    loader = DataLoader(TensorDataset(
        torch.randn(6, 3, 64, 64), torch.tensor([0, 1, 2, 0, 1, 2])
    ), batch_size=3)
    run_epoch(
        model, loader, criterion=nn.CrossEntropyLoss(), device=torch.device("cpu"),
        optimizer=optimizer, freeze_batchnorm=True, freeze_all_batchnorm=True,
    )
    assert torch.equal(model.features[-1][1].running_mean, old_mean)
    assert not torch.equal(model.features[-1][0].weight, old_weight)
