from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import build_evaluation_loader, validate_image_tree
from .metrics import evaluate_loader
from .model import load_checkpoint, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier")
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, class_names, image_size, threshold, metadata = load_checkpoint(args.model, device)
    summary = validate_image_tree(args.data_dir, set(class_names))
    loader = build_evaluation_loader(
        args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        expected_classes=class_names,
    )
    result = {
        "dataset": summary.__dict__,
        "threshold": threshold,
        "metadata": metadata,
        "metrics": evaluate_loader(
            model,
            loader,
            device=device,
            class_names=class_names,
        threshold=threshold,
        ).as_dict(),
    }
    content = json.dumps(result, ensure_ascii=False, indent=2)
    print(content)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
