from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from scripts import build_expanded_dataset as dataset
from scripts import evaluate_multisets as evaluation


def _fingerprint(key, hashes):
    return {"id": key, "sha256": key, "dhashes": hashes, "phashes": hashes}


def test_cross_source_bridge_keeps_all_near_duplicates_in_one_cluster():
    # 首尾距离大，最后加入的桥接图同时接近两者，必须合并整个连通簇。
    rows = [
        _fingerprint("online", ["0"]),
        _fingerprint("document", ["fffff"]),
        _fingerprint("web", ["3ff"]),
    ]
    assert not dataset.nearby(rows[0], rows[1])
    assert len(dataset.clusters(rows)) == 1


def test_animation_overlap_checks_more_than_first_frame():
    left = _fingerprint("old-animation", ["ffffffffffffffff", "0"])
    right = _fingerprint("new-still", ["3ff"])
    assert dataset.nearby(left, right)


def test_animation_suffix_uses_decoded_frames_without_source_metadata(tmp_path):
    source = tmp_path / "opaque-download"
    frames = [Image.new("RGB", (32, 32), color) for color in ("red", "blue")]
    frames[0].save(source, format="GIF", save_all=True, append_images=frames[1:])
    assert dataset.image_suffix(source) == ".gif"


def test_relative_output_path_is_resolved_before_manifest_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    historical = tmp_path / "old.png"
    Image.new("RGB", (32, 32), "yellow").save(historical)
    (tmp_path / "historical.csv").write_text(
        "file,split,label,sha256,origin,source\n"
        "old.png,train,nailong,old,fixture,local\n"
    )
    reviewed = [
        {
            **_fingerprint(name, [fingerprint]),
            "source": "document", "source_group": name, "label": "naiwa",
            "path": str(historical), "annotation": "fixture",
        }
        for name, fingerprint in (("section-a", "0"), ("section-b", "ffffffffffffffff"))
    ]
    (tmp_path / "reviewed.json").write_text(json.dumps(reviewed))
    monkeypatch.setattr(sys, "argv", [
        "build_expanded_dataset.py", "--reviewed", "reviewed.json",
        "--historical-manifest", "historical.csv", "--workspace", str(tmp_path),
        "--output", "expanded",
    ])
    dataset.main()
    audit = json.loads((tmp_path / "expanded/audit.json").read_text())
    assert audit["historical_count"] == 1 and audit["added_count"] == 2
    manifest = (tmp_path / "expanded/manifest.csv").read_text()
    assert str(tmp_path) not in manifest
    assert Path("expanded/train/nailong/old.png").exists()


def test_source_evaluation_preserves_class_index_when_a_class_is_absent(tmp_path):
    (tmp_path / "naiwa").mkdir()
    Image.new("RGB", (64, 64), "yellow").save(tmp_path / "naiwa" / "sample.png")
    images = evaluation.EvaluationImages(
        tmp_path, ["nailong", "naiwa", "other"], 64, False,
    )
    tensor, label = images[0]
    assert tensor.shape == (3, 64, 64)
    assert label == 1


def test_small_perfect_test_does_not_prove_95_percent_population_accuracy():
    lower, upper = evaluation.wilson_interval(20, 20)
    assert lower < 0.95
    assert upper == pytest.approx(1.0)


def test_multiple_test_sets_are_hidden_until_validation_selection_is_persisted(
    tmp_path, monkeypatch,
):
    for name, f1, test_accuracy in (("strong", 0.8, 0.1), ("mild", 0.7, 1.0)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "model.pt").write_bytes(name.encode())
        (folder / "metrics.json").write_text(json.dumps({
            "validation": {
                "accuracy": 0.8,
                "classification_report": {"macro avg": {"f1-score": f1}},
            },
            "metadata": {"val_loss": 0.5, "best_epoch": 2},
            "history": [{}, {}],
            "test": {"accuracy": test_accuracy},
        }))
    for name in ("baseline", "previous"):
        (tmp_path / f"{name}.pt").write_bytes(name.encode())
    monkeypatch.setattr(sys, "argv", [
        "evaluate_multisets.py", "--experiment-dir", str(tmp_path),
        "--data-dir", str(tmp_path / "data"), "--baseline", str(tmp_path / "baseline.pt"),
        "--previous-control", str(tmp_path / "previous.pt"),
        "--guard", str(tmp_path / "guard.pt"),
    ])
    monkeypatch.setattr(evaluation, "select_device", lambda _: torch.device("cpu"))

    def first_inference(*args, **kwargs):
        selection = json.loads((tmp_path / "selection.json").read_text())
        assert selection["selected_by_validation"] == "strong"
        assert selection["sets"] == {}
        raise RuntimeError("selection verified")

    monkeypatch.setattr(evaluation, "score", first_inference)
    with pytest.raises(RuntimeError, match="selection verified"):
        evaluation.main()
