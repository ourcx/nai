from __future__ import annotations

import argparse
import os
from pathlib import Path

from .data import SUPPORTED_SUFFIXES
from .storage import BotStore

SEED_LABELS = ("nailong", "naiwa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed the bot image reply pool")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.getenv("BOT_STATE_DIR", "runtime")),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    return parser.parse_args()


def seed(state_dir: Path, data_dir: Path) -> dict[str, int]:
    store = BotStore(state_dir)
    counts: dict[str, int] = {}
    for label in SEED_LABELS:
        label_dir = data_dir / label
        paths = sorted(
            path
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        for path in paths:
            store.register_seed(path.read_bytes(), label)
        counts[label] = len(paths)
    return counts


def main() -> None:
    args = parse_args()
    counts = seed(args.state_dir, args.data_dir)
    print(" ".join(f"{label}={count}" for label, count in counts.items()))


if __name__ == "__main__":
    main()

