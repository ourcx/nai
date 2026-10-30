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
    AUGMENTATION_PROFILES,
    DEFAULT_CLASS_BALANCE_POWER,
    ROBUST_AUGMENTATION,
    build_evaluation_loader,
    build_train_val_loaders,
    validate_image_tree,
)
from .metrics import evaluate_loader
from .model import (
    ARCHITECTURE,
    SUPPORTED_ARCHITECTURES,
    build_model,
    configure_for_finetuning,
    freeze_frozen_batchnorm,
    load_checkpoint,
    save_checkpoint,
    select_device,
)

EXPECTED_CLASSES = {"nailong", "naiwa", "other"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Nailong and Naiwa CNN classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--validation-dir",
        type=Path,
        help="Use a separately curated holdout instead of splitting the training directory",
    )
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
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.15)
    parser.add_argument(
        "--architecture", choices=SUPPORTED_ARCHITECTURES, default=ARCHITECTURE
    )
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--augmentation-profile", choices=AUGMENTATION_PROFILES, default=ROBUST_AUGMENTATION
    )
    parser.add_argument(
        "--selection-metric", choices=("macro_f1", "accuracy"), default="macro_f1"
    )
    parser.add_argument(
        "--skip-test-evaluation", action="store_true",
        help="Compare candidates on validation only; evaluate test after choosing a candidate",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--class-balance-power",
        type=float,
        default=DEFAULT_CLASS_BALANCE_POWER,
        help="0 uses natural class frequency, 1 gives every class equal sampling weight",
    )
    # 最高候选达到一半概率就接受，减少奶图因旧 0.65 门槛被搁置；可显式覆盖。
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def validate_training_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.image_size < 32:
        raise ValueError("epochs, batch_size, patience must be positive; image_size must be >=32")
    if args.num_workers < 0 or args.unfreeze_blocks < 0:
        raise ValueError("num_workers and unfreeze_blocks must be nonnegative")
    if not 0.0 <= args.dropout < 1.0 or not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("dropout and label_smoothing must be in [0, 1)")
    if not 0.0 <= args.threshold <= 1.0 or not 0.0 <= args.class_balance_power <= 1.0:
        raise ValueError("threshold and class_balance_power must be in [0, 1]")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if not all(np.isfinite(value) and value > 0 for value in (
        args.learning_rate, args.feature_learning_rate, args.max_grad_norm
    )) or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError(
            "learning rates/gradient limit must be positive; decay must be nonnegative"
        )


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> AdamW:
    """特征小步微调，新增头正常学习；仅对卷积/线性权重施加 AdamW 惩罚。"""
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups.setdefault(
                (name.startswith("features."), parameter.ndim > 1), []
            ).append(parameter)
    return AdamW([
        {
            "params": parameters,
            "lr": args.feature_learning_rate if features else args.learning_rate,
            "weight_decay": args.weight_decay if decay else 0.0,
        }
        for (features, decay), parameters in groups.items()
    ])


def parameter_summary(model: nn.Module) -> dict[str, int | float]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(parameter.detach()) for parameter in trainable
    ]))
    return {
        "total_parameters": int(np.sum([p.numel() for p in model.parameters()])),
        "trainable_parameters": int(np.sum([p.numel() for p in trainable])),
        "trainable_parameter_norm": float(norm.item()),
    }


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
    freeze_batchnorm: bool = False,
    max_grad_norm: float | None = None,
    freeze_all_batchnorm: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    if training and freeze_batchnorm:
        freeze_frozen_batchnorm(model)
    if training and freeze_all_batchnorm:
        # 来源比例改变时，小批次的统计量容易偏离旧群聊分布；独立开关只冻结
        # BN 均值/方差，仍允许未冻结层的卷积权重和 BN 仿射参数接收梯度。
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
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
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                        error_if_nonfinite=True,
                    )
                optimizer.step()

            batch_size = targets.size(0)
            loss_total += float(loss.item()) * batch_size
            correct_total += int((logits.argmax(dim=1) == targets).sum().item())
            sample_total += batch_size

    return loss_total / sample_total, correct_total / sample_total


def train(args: argparse.Namespace) -> Path:
    validate_training_args(args)
    seed_everything(args.seed)
    train_dir = args.data_dir / "train"
    test_dir = args.data_dir / "test"
    train_summary = validate_image_tree(train_dir)
    dataset_classes = set(train_summary.class_counts)
    # 豆包独立成类后允许四分类；保留旧三分类数据的训练入口，测试集必须与训练集一致。
    if dataset_classes not in (EXPECTED_CLASSES, EXPECTED_CLASSES | {"doubao"}):
        raise ValueError(
            f"Dataset classes {sorted(dataset_classes)} must be "
            f"{sorted(EXPECTED_CLASSES)} or {sorted(EXPECTED_CLASSES | {'doubao'})}"
        )
    test_summary = validate_image_tree(test_dir, dataset_classes)
    validation_dir = getattr(args, "validation_dir", None)
    validation_summary = (
        validate_image_tree(validation_dir, dataset_classes) if validation_dir else None
    )
    print(json.dumps({
        "train": train_summary.__dict__,
        "validation": validation_summary.__dict__ if validation_summary else None,
        "test": test_summary.__dict__,
    }, indent=2))

    train_loader, val_loader, class_names = build_train_val_loaders(
        train_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        class_balance_power=args.class_balance_power,
        val_dir=validation_dir,
        augmentation_profile=args.augmentation_profile,
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
        model = build_model(
            len(class_names), pretrained=True,
            architecture=args.architecture, dropout=args.dropout,
        )
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
        if model.architecture != args.architecture:
            # 跨结构只迁移兼容的骨干；新卷积和分类头重新学习，不混用旧头的权重。
            source_model = model
            model = build_model(
                len(class_names), pretrained=False,
                architecture=args.architecture, dropout=args.dropout,
            )
            model.features.load_state_dict(source_model.features.state_dict())
        else:
            model.classifier[2].p = args.dropout
            model.model_config = {"dropout": args.dropout}
    configure_for_finetuning(model, unfreeze_blocks=args.unfreeze_blocks)
    model.to(device)
    initial_parameters = parameter_summary(model)
    print(json.dumps({"architecture": args.architecture, **initial_parameters}))
    optimizer = build_optimizer(model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    checkpoint_path = args.output_dir / "model.pt"
    best_score = -1.0
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
            freeze_batchnorm=True,
            max_grad_norm=args.max_grad_norm,
        )
        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        val_result = evaluate_loader(
            model, val_loader, device=device, class_names=class_names, threshold=args.threshold
        )
        val_macro_f1 = float(val_result.classification_report["macro avg"]["f1-score"])
        scheduler.step()
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "trainable_parameter_norm": parameter_summary(model)["trainable_parameter_norm"],
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics))

        score = val_macro_f1 if args.selection_metric == "macro_f1" else val_accuracy
        improved = score > best_score or (
            np.isclose(score, best_score) and val_loss < best_loss
        )
        if improved:
            best_score = score
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
                    "val_accuracy": val_accuracy,
                    "val_macro_f1": val_macro_f1,
                    "selection_metric": args.selection_metric,
                    "val_loss": best_loss,
                    "seed": args.seed,
                    "label_smoothing": args.label_smoothing,
                    "class_balance_power": args.class_balance_power,
                    "initial_model": (
                        args.initial_model.name if args.initial_model is not None else None
                    ),
                    "fine_tuned_from_trained_at": initial_metadata.get("trained_at"),
                    "augmentation_profile": args.augmentation_profile,
                    "training_config": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "initial_parameters": initial_parameters,
                    "best_parameters": parameter_summary(model),
                    "frozen_batchnorm_statistics": True,
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
    test_result = None if args.skip_test_evaluation else evaluate_loader(
        best_model,
        test_loader,
        device=device,
        class_names=class_names,
        threshold=threshold,
    )
    metrics = {
        "architecture": args.architecture,
        "image_size": image_size,
        "threshold": threshold,
        "metadata": metadata,
        "dataset": {
            "train": train_summary.__dict__,
            "validation": validation_summary.__dict__ if validation_summary else None,
            "test": test_summary.__dict__,
        },
        "validation": val_result.as_dict(),
        "test": test_result.as_dict() if test_result else None,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "validation": val_result.as_dict(),
        "test": metrics["test"],
    }, indent=2))
    return checkpoint_path


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
