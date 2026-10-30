"""仅按验证集选候选，再固定评估多个来源的测试集和群聊退化视图。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from nailong_classifier.data import evaluation_image_loader, evaluation_transform
from nailong_classifier.model import load_checkpoint, select_device
from scripts.evaluate_generalization import FixedChatDegradation


class EvaluationImages(Dataset):
    """固定类别编码，允许某个来源的测试集缺少某一类，并保留逐图对照路径。"""

    def __init__(self, folder: Path, classes: list[str], size: int, degraded: bool):
        self.samples = [
            (path, target)
            for target, label in enumerate(classes)
            for path in sorted((folder / label).glob("*"))
            if path.is_file()
        ]
        if not self.samples:
            raise ValueError(f"No test images: {folder}")
        self.transform = evaluation_transform(size)
        if degraded:
            self.transform = transforms.Compose([
                FixedChatDegradation(), *self.transform.transforms,
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        return self.transform(evaluation_image_loader(str(path))), target


def wilson_interval(correct: int, total: int) -> list[float]:
    """二项比例的 95% Wilson 区间；近似图的残余相关性仍需另行解释。"""
    if not total:
        return [0.0, 1.0]
    z = 1.959963984540054
    proportion = np.divide(correct, total)
    z2 = np.square(z)
    denominator = np.add(1, np.divide(z2, total))
    center = np.divide(np.add(proportion, np.divide(z2, np.multiply(2, total))), denominator)
    variance = np.add(
        np.divide(np.multiply(proportion, np.subtract(1, proportion)), total),
        np.divide(z2, np.multiply(4, np.square(total))),
    )
    radius = np.divide(np.multiply(z, np.sqrt(variance)), denominator)
    return [float(np.subtract(center, radius)), float(np.add(center, radius))]


def score(model_path: Path, folder: Path, guard_path: Path, device, degraded: bool):
    model, classes, size, _, _ = load_checkpoint(model_path, device)
    guard, guard_classes, guard_size, _, _ = load_checkpoint(guard_path, device)
    if guard_size != size:
        raise ValueError("Classifier and guard must use the same input size")
    dataset = EvaluationImages(folder, classes, size, degraded)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    truth, predictions, confidence, blocked = [], [], [], []
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            probabilities = torch.softmax(model(images), dim=1).cpu()
            values, indices = probabilities.max(dim=1)
            guard_scores = torch.softmax(guard(images), dim=1).cpu()
            truth.extend(targets.tolist())
            predictions.extend(indices.tolist())
            confidence.extend(values.tolist())
            blocked.extend((guard_scores[:, guard_classes.index("doubao")] >= 0.3).tolist())
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    accepted = np.asarray(confidence) >= 0.5
    milk_classes = [classes.index("nailong"), classes.index("naiwa")]
    is_milk = np.isin(truth, milk_classes)
    emitted = np.logical_and.reduce((
        accepted, np.isin(predictions, milk_classes), np.logical_not(blocked),
    ))
    effective = np.where(emitted, predictions, classes.index("other"))
    correct = int(np.count_nonzero(truth == predictions))
    metrics = {
        "count": len(truth),
        "correct": correct,
        "accuracy": float(accuracy_score(truth, predictions)),
        "accuracy_wilson_95": wilson_interval(correct, len(truth)),
        "classification_report": classification_report(
            truth, predictions, labels=list(range(len(classes))), target_names=classes,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=list(range(len(classes)))
        ).tolist(),
        "bot_accuracy": float(accuracy_score(truth, effective)),
        "correct_milk_images": int(np.count_nonzero(
            np.logical_and(emitted, truth == predictions)
        )),
        "false_milk_images": int(np.count_nonzero(np.logical_and(emitted, ~is_milk))),
        "milk_support": int(np.count_nonzero(is_milk)),
        "other_support": int(np.count_nonzero(~is_milk)),
        "threshold": 0.5,
        "doubao_threshold": 0.3,
    }
    records = [{
        "file": str(path),
        "label": classes[int(target)],
        "prediction": classes[int(prediction)],
        "confidence": probability,
        "guard_blocked": guard_blocked,
        "bot_prediction": classes[int(bot_prediction)],
    } for (path, target), prediction, probability, guard_blocked, bot_prediction in zip(
        dataset.samples, predictions, confidence, blocked, effective, strict=True,
    )]
    return metrics, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--previous-control", type=Path, required=True)
    parser.add_argument("--guard", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = select_device("auto")
    candidates = {
        name: json.loads((args.experiment_dir / name / "metrics.json").read_text())
        for name in ("strong", "mild")
    }
    selected = max(candidates, key=lambda name: (
        candidates[name]["validation"]["classification_report"]["macro avg"]["f1-score"],
        float(np.negative(candidates[name]["metadata"]["val_loss"])),
    ))
    paths = {
        "baseline": args.baseline,
        "previous_control": args.previous_control,
        "selected": args.experiment_dir / selected / "model.pt",
    }
    result = {
        "selected_by_validation": selected,
        "validation": {
            name: {
                "accuracy": data["validation"]["accuracy"],
                "macro_f1": data["validation"]["classification_report"]["macro avg"]["f1-score"],
                "best_epoch": data["metadata"]["best_epoch"],
                "epochs_run": len(data["history"]),
            }
            for name, data in candidates.items()
        },
        "checkpoint_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
        },
        "sets": {},
    }
    # 此文件在第一次测试推理前创建；禁止测试后换候选或覆盖第一次选择。
    selection_path = args.experiment_dir / "selection.json"
    with selection_path.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    for name in ("test", "test-online", "test-public", "test-document"):
        result["sets"][name] = {}
        for view, degraded in (("original", False), ("chat_degradation", True)):
            result["sets"][name][view] = {}
            for model_name, model_path in paths.items():
                metrics, rows = score(
                    model_path, args.data_dir / name, args.guard, device, degraded,
                )
                result["sets"][name][view][model_name] = metrics
                output = args.experiment_dir / f"predictions-{name}-{view}-{model_name}.csv"
                with output.open("w") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(json.dumps({
                    "set": name, "view": view, "model": model_name,
                    "count": metrics["count"], "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["classification_report"]["macro avg"]["f1-score"],
                    "correct_milk": metrics["correct_milk_images"],
                    "false_milk": metrics["false_milk_images"],
                }), flush=True)
        (args.experiment_dir / "comparison.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )


if __name__ == "__main__":
    main()
