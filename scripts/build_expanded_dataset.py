"""将视觉确认的图片按近重复簇划分，保留旧训练/验证/测试分配。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CLASSES = ("nailong", "naiwa", "other")


def nearby(left: dict, right: dict) -> bool:
    """多帧 dHash 和 pHash 任一命中即归为一簇，防止跨集合出现同图变体。"""
    if left["sha256"] == right["sha256"]:
        return True
    return any(
        (int(a, 16) ^ int(b, 16)).bit_count() <= limit
        for key, limit in (("dhashes", 10), ("phashes", 6))
        for a in left[key] for b in right[key]
    )


def clusters(rows: list[dict]) -> list[list[dict]]:
    """用并查集保留传递关系；跨来源的桥接图也只能属于一个集合。"""
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, left in enumerate(rows):
        for other in range(index):
            if nearby(left, rows[other]):
                parents[root(index)] = root(other)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)
    return list(groups.values())


def export_image(source: Path, target: Path):
    """统一小图解码成本，动图保留五个均匀采样帧；仍按原图计一个样本。"""
    with Image.open(source) as image:
        count = getattr(image, "n_frames", 1)
        positions = np.unique(np.rint(np.linspace(
            0, np.subtract(count, 1), min(count, 5),
        )).astype(int))
        frames = []
        for position in positions:
            image.seek(int(position))
            frame = ImageOps.exif_transpose(image).convert("RGB")
            frame.thumbnail((512, 512), Image.Resampling.LANCZOS)
            frames.append(frame.copy())
        if len(frames) > 1:
            frames[0].save(
                target, format="GIF", save_all=True, append_images=frames[1:],
                duration=120, loop=0,
            )
        else:
            frames[0].save(target, format="PNG")


def image_suffix(source: Path) -> str:
    """使用实际解码帧数，避免来源缺元数据时把动图写成静态图后缀。"""
    with Image.open(source) as image:
        return ".gif" if getattr(image, "n_frames", 1) > 1 else ".png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # 清单路径统一相对于绝对工作区；命令行也允许传入相对输出目录。
    args.workspace = args.workspace.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("Dataset is fixed once created; choose a new output path")
    rows = json.loads(args.reviewed.read_text())
    groups = clusters(rows)
    conflicts, usable = [], []
    for group in groups:
        if len({row["label"] for row in group}) > 1:
            conflicts.extend(row["id"] for row in group)
        else:
            usable.append(group)

    document_sections = Counter(
        row.get("source_group", "intro")
        for group in usable for row in group if row["source"] == "document"
    )
    if len(document_sections) < 2:
        raise ValueError("Need multiple document sections for a genuine section holdout")
    target = np.multiply(sum(document_sections.values()), 0.25)
    # 仅根据来源规模选留出章节，完全不查看任何模型表现。
    held_section = min(document_sections, key=lambda key: abs(
        np.subtract(document_sections[key], target)
    ))
    assignments = []
    unassigned = defaultdict(list)
    for group in usable:
        key = min(row["sha256"] for row in group)
        if any(
            row["source"] == "online" and int(row["id"][1:]) >= 824 for row in group
        ):
            split = "test-online"
        elif any(
            row["source"] == "document" and row.get("source_group", "intro") == held_section
            for row in group
        ):
            split = "test-document"
        else:
            unassigned[group[0]["label"]].append((key, group))
            continue
        assignments.append((split, key, group))

    for candidates in unassigned.values():
        candidates.sort(key=lambda pair: hashlib.sha256(pair[0].encode()).hexdigest())
        public = [(key, group) for key, group in candidates if any(
            row["source"] == "web" for row in group
        )]
        public_count = int(np.ceil(np.multiply(len(public), 0.25)))
        public_keys = {key for key, _ in public[:public_count]}
        remaining = [(key, group) for key, group in candidates if key not in public_keys]
        validation_count = min(25, int(np.ceil(np.multiply(len(remaining), 0.15))))
        validation_keys = {key for key, _ in remaining[:validation_count]}
        for key, group in candidates:
            split = "test-public" if key in public_keys else (
                "val" if key in validation_keys else "train"
            )
            assignments.append((split, key, group))

    # 创建训练入口要求的旧 test，以及可能缺类的来源测试目录。
    for split in ("train", "val", "test", "test-online", "test-public", "test-document"):
        for label in CLASSES:
            (args.output / split / label).mkdir(parents=True)

    manifest = []
    with args.historical_manifest.open() as handle:
        historical = list(csv.DictReader(handle))
    for row in historical:
        source = args.workspace / row["file"]
        target_path = args.output / row["split"] / row["label"] / source.name
        shutil.copy2(source, target_path)
        manifest.append({
            "split": row["split"], "label": row["label"],
            "file": str(target_path.relative_to(args.workspace)),
            "source": "historical", "original_sha256": row["sha256"],
            "file_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
            "cluster_id": f"historical:{row['sha256']}",
            "annotation": row["origin"], "source_ref": row["source"],
        })
    excluded_duplicates = []
    for split, key, group in assignments:
        # 一簇仅保留一张，优先保留决定该来源留出边界的原始图。
        family = {
            "test-online": "online", "test-document": "document", "test-public": "web",
        }.get(split)
        group.sort(key=lambda row: (row["source"] != family, row["id"]))
        row = group[0]
        excluded_duplicates.extend(item["id"] for item in group[1:])
        suffix = image_suffix(Path(row["path"]))
        target_path = args.output / split / row["label"] / f"{row['sha256']}{suffix}"
        export_image(Path(row["path"]), target_path)
        source_ref = row.get("source_page") or row.get("url") or row["id"]
        manifest.append({
            "split": split, "label": row["label"],
            "file": str(target_path.relative_to(args.workspace)),
            "source": row["source"], "original_sha256": row["sha256"],
            "file_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
            "cluster_id": key, "annotation": row["annotation"], "source_ref": source_ref,
        })
    cluster_splits = defaultdict(set)
    for row in manifest:
        cluster_splits[row["cluster_id"]].add(row["split"])
    assert all(len(splits) == 1 for splits in cluster_splits.values())
    assert sum(row["source"] == "historical" for row in manifest) == len(historical)
    with (args.output / "manifest.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    audit = {
        "reviewed_count": len(rows),
        "added_count": len(assignments),
        "historical_count": len(historical),
        "label_conflicts_excluded": conflicts,
        "cross_source_duplicates_excluded": excluded_duplicates,
        "document_section_holdout": held_section,
        "document_section_sizes": dict(document_sections),
        "near_duplicate_clusters_crossing_splits": 0,
        "historical_filter": (
            "5-frame dHash256 <=10 or pHash64 <=6, including older four-class data"
        ),
        "normalization": "new images max512px; animations at most5 frames; one original=one sample",
        "seed_rule": "sha256 ordering; public25%, remaining validation15% capped25 per class",
        "counts": {
            split: dict(Counter(row["label"] for row in manifest if row["split"] == split))
            for split in sorted({row["split"] for row in manifest})
        },
        "new_source_counts": dict(Counter(
            row["source"] for row in manifest if row["source"] != "historical"
        )),
    }
    (args.output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    (args.output / "manifest.sha256").write_text(
        hashlib.sha256((args.output / "manifest.csv").read_bytes()).hexdigest()
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
