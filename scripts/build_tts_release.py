"""打包已试听的 GPT-SoVITS 源码、权重和参考录音，不携带本机虚拟环境。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = ("service.py", "tts_engine.py", "requirements.txt", "nailong-tts.service")


def copy_file(source: str, target: str) -> str:
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "outputs/nailong-voice-20261102/gpt-sovits"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/tts-release")
    parser.add_argument(
        "--naiwa-source",
        type=Path,
        help="Optional directory containing base-model/ and naiwa-reference.wav",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite release directory: {output}")
    voice = json.loads((source / "author_model_metadata.json").read_text())
    verified = json.loads((source / "model_integrity.json").read_text())
    app_digest = hashlib.sha256()
    for name in sorted(RUNTIME_FILES):
        path = ROOT / "deploy/tts" / name
        app_digest.update(path.name.encode())
        app_digest.update(path.read_bytes())
    release_id = f"nailong-tts-20261102-{app_digest.hexdigest()[:10]}"
    output.mkdir(parents=True)
    shutil.copytree(
        source / "upstream",
        output / "upstream",
        copy_function=copy_file,
        ignore=shutil.ignore_patterns(".git", ".cache", "__pycache__", "*.pyc", ".DS_Store"),
    )
    (output / "weights").mkdir()
    weights = []
    for item in verified:
        original = source / item["file"]
        if item["file"].startswith("author_model/"):
            target = output / "weights" / original.name
            copy_file(str(original), str(target))
        else:
            target = output / item["file"]
        with target.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Weight hash mismatch: {item['file']}")
        weights.append({"file": str(target.relative_to(output)), "sha256": actual})
    reference = source.parent / "reference.wav"
    copy_file(str(reference), str(output / "reference.wav"))
    weights.append(
        {"file": "reference.wav", "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
    )
    voices = {
        "nailong": {
            "gpt_weights": "weights/NaiLong-e16.ckpt",
            "sovits_weights": "weights/NaiLong_e14_s504.pth",
            "reference": "reference.wav",
            "reference_text": "啊，我才不要这样，好害羞啊。",
        }
    }
    if args.naiwa_source:
        naiwa_source = args.naiwa_source.resolve()
        naiwa_files = {
            "weights/s1v3.ckpt": naiwa_source / "base-model/s1v3.ckpt",
            "weights/s2Gv2ProPlus.pth": naiwa_source / "base-model/v2Pro/s2Gv2ProPlus.pth",
            "naiwa-reference.wav": naiwa_source / "naiwa-reference.wav",
        }
        for relative, original in naiwa_files.items():
            if not original.is_file():
                raise FileNotFoundError(original)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_file(str(original), str(target))
            weights.append({
                "file": relative,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            })
        voices["naiwa"] = {
            "gpt_weights": "weights/s1v3.ckpt",
            "sovits_weights": "weights/s2Gv2ProPlus.pth",
            "reference": "naiwa-reference.wav",
            "reference_text": "我落地枪，不要慌，全靠运气全靠浪。",
        }
    for name in RUNTIME_FILES:
        shutil.copyfile(ROOT / "deploy/tts" / name, output / name)
    metadata = {
        "release_id": release_id,
        "author_model": voice,
        "upstream_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source / "upstream", text=True
        ).strip(),
        "reference_text": "啊，我才不要这样，好害羞啊。",
        "voices": voices,
        "weights": weights,
        "gpt_weights": "NaiLong-e16.ckpt",
        "sovits_weights": "NaiLong_e14_s504.pth",
    }
    (output / "release.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    with (output / "SHA256SUMS").open("w") as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                with path.open("rb") as data:
                    digest = hashlib.file_digest(data, "sha256").hexdigest()
                handle.write(f"{digest}  {path.relative_to(output)}\n")
    print(
        json.dumps(
            {
                "release_id": release_id,
                "directory": str(output),
                "bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
            }
        )
    )


if __name__ == "__main__":
    main()
