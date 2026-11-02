from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    coverage: float
    accepted_accuracy: float | None
    mean_confidence: float
    class_names: list[str]
    confusion_matrix: list[list[int]]
    classification_report: dict

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    class_names: list[str],
    threshold: float,
) -> EvaluationResult:
    model.eval()
    expected: list[int] = []
    predicted: list[int] = []
    confidences: list[float] = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            probabilities = torch.softmax(model(images), dim=1).cpu()
            confidence, predictions = torch.max(probabilities, dim=1)
            expected.extend(targets.tolist())
            predicted.extend(predictions.tolist())
            confidences.extend(confidence.tolist())

    label_indices = list(range(len(class_names)))
    report = classification_report(
        expected,
        predicted,
        labels=label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(expected, predicted, labels=label_indices)
    expected_array = np.asarray(expected)
    predicted_array = np.asarray(predicted)
    accepted = np.asarray(confidences) >= threshold
    accepted_accuracy = (
        float(accuracy_score(expected_array[accepted], predicted_array[accepted]))
        if np.any(accepted)
        else None
    )
    return EvaluationResult(
        accuracy=float(accuracy_score(expected, predicted)),
        coverage=float(np.mean(accepted)),
        accepted_accuracy=accepted_accuracy,
        mean_confidence=float(np.mean(confidences)),
        class_names=class_names,
        confusion_matrix=matrix.tolist(),
        classification_report=report,
    )
