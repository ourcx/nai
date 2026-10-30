"""把已验收选择冻结为可移植的最小部署包；不携带训练数据与实验依赖。"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_release(destination: Path) -> None:
    experiment = ROOT / "artifacts/cnn-milk-priority-20261031"
    foundation = ROOT / "artifacts/cnn-foundation-20261030"
    main = json.loads((experiment / "main-selection-target60-fallback.json").read_text())
    guard = json.loads((experiment / "guard-selection-crossval.json").read_text())
    fine = json.loads((foundation / "finetune-selection.json").read_text())
    backbone = main["protocol"]["main_protocol"]["backbone"]
    joint = main["protocol"]["joint_weights"]["joint-C0.01"]
    if main["selected_config"]["models"] != ["foundation", "joint-C0.01"]:
        raise ValueError("Release builder only supports the verified target60 fallback")
    sources = {
        "online": (ROOT / "artifacts/model.pt", main["protocol"]["main_protocol"]["online_sha256"]),
        "old_guard": (
            ROOT / "artifacts/doubao-detector-20261026/model.pt",
            guard["protocol"]["guard_sha256"],
        ),
        "backbone": (Path(backbone["weight_path"]), backbone["weight_sha256"]),
        "finetune": (Path(fine["selected_config"]["weights_file"]),
                     fine["selected_config"]["weights_sha256"]),
        "joint": (Path(joint["path"]), joint["sha256"]),
        "new_guard": (Path(guard["selected_config"]["weights_file"]),
                      guard["selected_config"]["weights_sha256"]),
    }
    destination.mkdir(parents=True, exist_ok=True)
    weights = {}
    for name, (source, expected) in sources.items():
        with source.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"Selected source hash mismatch: {source.name}")
        target = destination / f"{name}{source.suffix}"
        shutil.copyfile(source, target)
        weights[name] = {"file": target.name, "sha256": expected}
    keys = ("models", "weights", "preserve_policy", "unknown_naiwa_fallback")
    manifest = {
        "format": "milk-priority-v1",
        "release": "milk-priority-20261031-target60-fallback",
        "threshold": .5, "guard_threshold": .3,
        "selection": {key: main["selected_config"][key] for key in keys},
        "weights": weights,
        "source_selection_sha256": hashlib.sha256(
            (experiment / "main-selection-target60-fallback.json").read_bytes()
        ).hexdigest(),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"release": manifest["release"], "directory": str(destination),
                      "weights": len(weights)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/milk-release-20261031"
    )
    build_release(parser.parse_args().output)
