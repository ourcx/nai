from __future__ import annotations

import json
import sys

import pytest
import torch

from scripts import evaluate_generalization as evaluation


def _candidates(root):
    for name, f1, accuracy, misleading_test in (
        ("control", 0.75, 0.90, 1.0),
        ("conv-pool", 0.80, 0.85, 0.1),
    ):
        folder = root / name
        folder.mkdir()
        (folder / "model.pt").write_bytes(name.encode())
        (folder / "metrics.json").write_text(json.dumps({
            "validation": {
                "accuracy": accuracy,
                "classification_report": {"macro avg": {"f1-score": f1}},
            },
            "metadata": {"val_loss": 0.5, "best_epoch": 2, "initial_parameters": {}},
            "history": [{}, {}],
            "test": {"accuracy": misleading_test},
        }))


def test_candidate_choice_ignores_test_and_majority_class_accuracy(tmp_path):
    _candidates(tmp_path)
    selected, _ = evaluation.select_candidate(tmp_path)
    assert selected == "conv-pool"


def test_selection_is_recorded_before_first_test_inference(tmp_path, monkeypatch):
    _candidates(tmp_path)
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"baseline")
    monkeypatch.setattr(sys, "argv", [
        "evaluate_generalization.py", "--experiment-dir", str(tmp_path),
        "--data-dir", str(tmp_path / "data"), "--baseline", str(baseline),
        "--doubao-model", str(tmp_path / "guard.pt"),
    ])
    monkeypatch.setattr(evaluation, "select_device", lambda _: torch.device("cpu"))

    def first_inference(*args, **kwargs):
        selection = json.loads((tmp_path / "selection.json").read_text())
        assert selection["selected_by_validation"] == "conv-pool"
        assert not selection["views"]
        raise RuntimeError("stop after checking the persisted selection")

    monkeypatch.setattr(evaluation, "score", first_inference)
    with pytest.raises(RuntimeError, match="persisted selection"):
        evaluation.main()
