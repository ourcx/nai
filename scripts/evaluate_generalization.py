"""先按验证集固定候选，再对原图与固定群聊退化视图做一次对比评估。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

from nailong_classifier.data import build_evaluation_loader, evaluation_transform
from nailong_classifier.model import load_checkpoint, select_device


class FixedChatDegradation:
    """每张图使用相同退化强度，方便不同模型逐图对照；这些视图不是新增独立样本。"""

    def __call__(self, image: Image.Image) -> Image.Image:
        dimensions = np.maximum(1, np.rint(np.multiply(image.size, 0.5))).astype(int)
        resized = image.resize(tuple(dimensions), Image.Resampling.BILINEAR).resize(
            image.size, Image.Resampling.BILINEAR
        ).filter(ImageFilter.GaussianBlur(radius=0.6))
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=55)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


def score(
    model_path: Path, split_dir: Path, *,
    device: torch.device, guard_path: Path, degradation: bool = False,
) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    model, classes, size, _, metadata = load_checkpoint(model_path, device)
    guard, guard_classes, guard_size, _, _ = load_checkpoint(guard_path, device)
    if size != guard_size:
        raise ValueError("This paired evaluation requires equal classifier/guard image sizes")
    loader = build_evaluation_loader(
        split_dir, image_size=size, batch_size=16, num_workers=0, expected_classes=classes
    )
    if degradation:
        loader.dataset.transform = transforms.Compose([
            FixedChatDegradation(), *evaluation_transform(size).transforms,
        ])
    expected, predictions, confidences, blocked = [], [], [], []
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            probabilities = torch.softmax(model(images), dim=1).cpu()
            confidence, predicted = probabilities.max(dim=1)
            guard_scores = torch.softmax(guard(images), dim=1).cpu()
            expected.extend(targets.tolist())
            predictions.extend(predicted.tolist())
            confidences.extend(confidence.tolist())
            blocked.extend((guard_scores[:, guard_classes.index("doubao")] >= 0.3).tolist())
    truth = np.asarray(expected)
    predicted = np.asarray(predictions)
    accepted = np.asarray(confidences) >= 0.5
    milk_indices = [classes.index("nailong"), classes.index("naiwa")]
    milk = np.isin(truth, milk_indices)
    emitted = np.logical_and.reduce((
        accepted, np.isin(predicted, milk_indices), np.logical_not(blocked),
    ))
    return {
        "count": len(expected),
        "accuracy": float(accuracy_score(truth, predicted)),
        "classification_report": classification_report(
            truth, predicted, labels=list(range(len(classes))), target_names=classes,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=list(range(len(classes)))
        ).tolist(),
        "threshold": 0.5,
        "doubao_threshold": 0.3,
        "coverage": float(np.mean(accepted)),
        "correct_milk_images": int(np.count_nonzero(
            np.logical_and(emitted, predicted == truth)
        )),
        "false_milk_images": int(np.count_nonzero(np.logical_and(emitted, ~milk))),
        "milk_support": int(np.count_nonzero(milk)),
        "other_support": int(np.count_nonzero(~milk)),
        "metadata": metadata,
    }


def select_candidate(experiment_dir: Path) -> tuple[str, dict]:
    candidates = {
        name: json.loads((experiment_dir / name / "metrics.json").read_text())
        for name in ("control", "conv-pool")
    }
    # 两个训练器都按验证宏 F1 留存最佳轮次；跨结构仍按宏 F1，再以 loss 打破平局。
    selected = max(candidates, key=lambda name: (
        candidates[name]["validation"]["classification_report"]["macro avg"]["f1-score"],
        float(np.negative(candidates[name]["metadata"]["val_loss"])),
    ))
    return selected, candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--doubao-model", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = select_device("auto")
    selected, candidates = select_candidate(args.experiment_dir)
    selected_path = args.experiment_dir / selected / "model.pt"
    result = {
        "selected_by_validation": selected,
        "validation": {
            name: {
                "accuracy": metrics["validation"]["accuracy"],
                "macro_f1": metrics["validation"]["classification_report"]["macro avg"]["f1-score"],
                "val_loss": metrics["metadata"]["val_loss"],
                "best_epoch": metrics["metadata"]["best_epoch"],
                "epochs_run": len(metrics["history"]),
                "parameters": metrics["metadata"]["initial_parameters"],
            }
            for name, metrics in candidates.items()
        },
        "checkpoint_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("baseline", args.baseline), ("selected", selected_path))
        },
        "views": {},
    }
    # 在首次测试推理之前落盘选择，避免事后根据测试结果改挑模型。
    selection_path = args.experiment_dir / "selection.json"
    if selection_path.exists():
        raise FileExistsError(f"Evaluation already started: {selection_path}")
    selection_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"selected_by_validation": selected}), flush=True)
    for view, degradation in (("original", False), ("chat_degradation", True)):
        result["views"][view] = {
            name: score(
                path, args.data_dir / "test", device=device,
                guard_path=args.doubao_model, degradation=degradation,
            )
            for name, path in (("baseline", args.baseline), ("selected", selected_path))
        }
    (args.experiment_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
