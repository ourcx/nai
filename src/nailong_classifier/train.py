from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .data import (
    DEFAULT_CLASS_BALANCE_POWER,
    build_evaluation_loader,
    build_train_val_loaders,
    validate_image_tree,
)
from .metrics import evaluate_loader
from .model import (
    build_model,
    configure_for_finetuning,
    load_checkpoint,
    save_checkpoint,
    select_device,
)

EXPECTED_CLASSES = {"nailong", "naiwa", "other"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Nailong and Naiwa CNN classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--initial-model",
        type=Path,
        help="Fine-tune an existing checkpoint instead of starting from ImageNet weights",
    )
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--feature-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--class-balance-power",
        type=float,
        default=DEFAULT_CLASS_BALANCE_POWER,
        help="0 uses natural class frequency, 1 gives every class equal sampling weight",
    )
    # 交叉验证中 0.65 比 0.60 少收进几张截图等非奶龙图，代价是少收录少量真奶龙/奶蛙。
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    correct_total = 0
    sample_total = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            loss_total += float(loss.item()) * batch_size
            correct_total += int((logits.argmax(dim=1) == targets).sum().item())
            sample_total += batch_size

    return loss_total / sample_total, correct_total / sample_total


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    train_dir = args.data_dir / "train"
    test_dir = args.data_dir / "test"
    train_summary = validate_image_tree(train_dir, EXPECTED_CLASSES)
    test_summary = validate_image_tree(test_dir, EXPECTED_CLASSES)
    print(json.dumps({"train": train_summary.__dict__, "test": test_summary.__dict__}, indent=2))

    train_loader, val_loader, class_names = build_train_val_loaders(
        train_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        class_balance_power=args.class_balance_power,
    )
    test_loader = build_evaluation_loader(
        test_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        expected_classes=class_names,
    )

    device = select_device(args.device)
    print(f"device={device} classes={class_names}")
    initial_metadata: dict[str, object] = {}
    if args.initial_model is None:
        model = build_model(len(class_names), pretrained=True)
    else:
        (
            model,
            initial_classes,
            initial_image_size,
            _initial_threshold,
            initial_metadata,
        ) = load_checkpoint(args.initial_model, device)
        if initial_classes != class_names:
            raise ValueError(
                f"Initial model classes {initial_classes} do not match dataset {class_names}"
            )
        if initial_image_size != args.image_size:
            raise ValueError(
                f"Initial model image size {initial_image_size} does not match "
                f"requested {args.image_size}"
            )
    configure_for_finetuning(model)
    model.to(device)

    feature_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("features.") and parameter.requires_grad
    ]
    classifier_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("classifier.") and parameter.requires_grad
    ]
    optimizer = AdamW(
        [
            {"params": feature_parameters, "lr": args.feature_learning_rate},
            {"params": classifier_parameters, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    checkpoint_path = args.output_dir / "model.pt"
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        scheduler.step()
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics))

        improved = val_accuracy > best_accuracy or (
            np.isclose(val_accuracy, best_accuracy) and val_loss < best_loss
        )
        if improved:
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path,
                model=model,
                class_names=class_names,
                image_size=args.image_size,
                threshold=args.threshold,
                metadata={
                    "trained_at": datetime.now(UTC).isoformat(),
                    "best_epoch": best_epoch,
                    "val_accuracy": best_accuracy,
                    "val_loss": best_loss,
                    "seed": args.seed,
                    "label_smoothing": args.label_smoothing,
                    "class_balance_power": args.class_balance_power,
                    "initial_model": (
                        args.initial_model.name if args.initial_model is not None else None
                    ),
                    "fine_tuned_from_trained_at": initial_metadata.get("trained_at"),
                    "augmentation_profile": "color_invariant_v2",
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch}")
                break

    best_model, class_names, image_size, threshold, metadata = load_checkpoint(
        checkpoint_path, device
    )
    val_result = evaluate_loader(
        best_model,
        val_loader,
        device=device,
        class_names=class_names,
        threshold=threshold,
    )
    test_result = evaluate_loader(
        best_model,
        test_loader,
        device=device,
        class_names=class_names,
        threshold=threshold,
    )
    metrics = {
        "architecture": "mobilenet_v3_small",
        "image_size": image_size,
        "threshold": threshold,
        "metadata": metadata,
        "dataset": {
            "train": train_summary.__dict__,
            "test": test_summary.__dict__,
        },
        "validation": val_result.as_dict(),
        "test": test_result.as_dict(),
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"checkpoint": str(checkpoint_path), "test": test_result.as_dict()}, indent=2))
    return checkpoint_path


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
