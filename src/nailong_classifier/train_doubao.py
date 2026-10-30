from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import (
    DOUBAO_DETECTOR_CLASSES,
    build_doubao_detector_evaluation_loader,
    build_doubao_detector_train_val_loaders,
    validate_image_tree,
)
from .metrics import evaluate_loader
from .model import build_model, configure_for_finetuning, save_checkpoint, select_device
from .train import run_epoch, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the standalone Doubao presence CNN")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/doubao-detector"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--feature-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Doubao probability required to block collection and replies.",
    )
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def train(args: argparse.Namespace) -> Path:
    if not 0 < args.threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    seed_everything(args.seed)
    train_dir = args.data_dir / "train"
    test_dir = args.data_dir / "test"
    train_summary = validate_image_tree(
        train_dir, {"doubao", "nailong", "naiwa", "other"}
    )
    test_summary = validate_image_tree(
        test_dir, {"doubao", "nailong", "naiwa", "other"}
    )
    train_loader, val_loader, class_names = build_doubao_detector_train_val_loaders(
        train_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    test_loader = build_doubao_detector_evaluation_loader(
        test_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    device = select_device(args.device)
    print(
        json.dumps(
            {
                "device": str(device),
                "classes": class_names,
                "train": train_summary.__dict__,
                "test": test_summary.__dict__,
            },
            ensure_ascii=False,
        )
    )
    model = build_model(len(DOUBAO_DETECTOR_CLASSES), pretrained=True)
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
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion=criterion, device=device, optimizer=optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion=criterion, device=device, optimizer=None
        )
        scheduler.step()
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(metrics)
        print(json.dumps(metrics))
        improved = val_accuracy > best_accuracy or (
            np.isclose(val_accuracy, best_accuracy) and val_loss < best_loss
        )
        if improved:
            best_accuracy = val_accuracy
            best_loss = val_loss
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path,
                model=model,
                class_names=DOUBAO_DETECTOR_CLASSES,
                image_size=args.image_size,
                threshold=args.threshold,
                metadata={
                    "task": "doubao_presence_detection",
                    "trained_at": datetime.now(UTC).isoformat(),
                    "best_epoch": epoch,
                    "val_accuracy": best_accuracy,
                    "val_loss": best_loss,
                    "seed": args.seed,
                    "positive_class": "doubao",
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stop epoch={epoch}")
                break

    from .model import load_checkpoint

    best_model, _, image_size, threshold, metadata = load_checkpoint(checkpoint_path, device)
    result = {
        "architecture": "mobilenet_v3_small",
        "task": "doubao_presence_detection",
        "image_size": image_size,
        "threshold": threshold,
        "metadata": metadata,
        "dataset": {"train": train_summary.__dict__, "test": test_summary.__dict__},
        "validation_best_accuracy": best_accuracy,
        "test": evaluate_loader(
            best_model,
            test_loader,
            device=device,
            class_names=DOUBAO_DETECTOR_CLASSES,
            threshold=threshold,
        ).as_dict(),
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"checkpoint": str(checkpoint_path), "test": result["test"]},
            ensure_ascii=False,
        )
    )
    return checkpoint_path


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
