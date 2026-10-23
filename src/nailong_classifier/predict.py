from __future__ import annotations

import argparse
import json
from pathlib import Path

from .milk import combined_milk_concentration
from .model import NailongPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = NailongPredictor(
        args.model,
        device=args.device,
        threshold=args.threshold,
    )
    prediction = predictor.predict_path(args.image)
    print(
        json.dumps(
            {
                "label": prediction.label,
                "confidence": prediction.confidence,
                "accepted": prediction.accepted,
                "scores": prediction.scores,
                "yellow_ratio": prediction.yellow_ratio,
                "milk_concentration": combined_milk_concentration(
                    prediction.scores, prediction.yellow_ratio
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
