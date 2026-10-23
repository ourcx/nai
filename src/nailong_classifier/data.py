from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_SUFFIXES = {".gif", ".jpg", ".jpeg", ".png", ".webp"}
ANIMATED_SAMPLE_WEIGHT = 4.0
DEFAULT_CLASS_BALANCE_POWER = 1.0


@dataclass(frozen=True)
class DatasetSummary:
    split: str
    class_counts: dict[str, int]
    total: int


def training_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.72, 1.0),
                ratio=(0.8, 1.2),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.2),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.4,
                hue=0.08,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomAutocontrast(p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.15,
                scale=(0.02, 0.12),
                ratio=(0.5, 2.0),
                value="random",
            ),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def evaluation_transform(image_size: int) -> transforms.Compose:
    resize_size = int(image_size / 0.875)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def validate_image_tree(
    split_dir: Path,
    expected_classes: set[str] | None = None,
) -> DatasetSummary:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split does not exist: {split_dir}")

    class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    class_names = {path.name for path in class_dirs}
    if expected_classes is not None and class_names != expected_classes:
        raise ValueError(
            f"{split_dir} classes are {sorted(class_names)}, expected {sorted(expected_classes)}"
        )

    counts: Counter[str] = Counter()
    failures: list[str] = []
    for class_dir in class_dirs:
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                failures.append(f"{path}: {exc}")
                continue
            counts[class_dir.name] += 1

    if failures:
        raise ValueError("Unreadable images:\n" + "\n".join(failures))
    empty_classes = class_names - set(counts)
    if empty_classes:
        raise ValueError(f"Classes without readable images: {sorted(empty_classes)}")

    return DatasetSummary(
        split=split_dir.name,
        class_counts=dict(sorted(counts.items())),
        total=sum(counts.values()),
    )


def _is_supported_image(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


def _load_image(path: str, *, random_frame: bool) -> Image.Image:
    with open(path, "rb") as image_file, Image.open(image_file) as image:
        frame_count = getattr(image, "n_frames", 1)
        if random_frame and frame_count > 1:
            image.seek(random.randrange(frame_count))
        return ImageOps.exif_transpose(image).convert("RGB")


def training_image_loader(path: str) -> Image.Image:
    return _load_image(path, random_frame=True)


def evaluation_image_loader(path: str) -> Image.Image:
    return _load_image(path, random_frame=False)


def _image_folder(
    root: Path,
    *,
    transform: transforms.Compose | None = None,
    training: bool = False,
) -> datasets.ImageFolder:
    return datasets.ImageFolder(
        root,
        transform=transform,
        loader=training_image_loader if training else evaluation_image_loader,
        is_valid_file=_is_supported_image,
    )


def build_train_val_loaders(
    train_dir: Path,
    *,
    image_size: int,
    batch_size: int,
    val_fraction: float,
    seed: int,
    num_workers: int,
    class_balance_power: float = DEFAULT_CLASS_BALANCE_POWER,
) -> tuple[DataLoader, DataLoader, list[str]]:
    from sklearn.model_selection import train_test_split

    if not 0.0 <= class_balance_power <= 1.0:
        raise ValueError("class_balance_power must be between 0 and 1")

    index_dataset = _image_folder(train_dir)
    animated_indices = [
        index
        for index, (path, _) in enumerate(index_dataset.samples)
        if Path(path).suffix.lower() == ".gif"
    ]
    static_indices = [
        index for index in range(len(index_dataset)) if index not in animated_indices
    ]
    train_indices, val_indices = train_test_split(
        static_indices,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=[index_dataset.targets[index] for index in static_indices],
    )
    train_indices.extend(animated_indices)

    train_dataset = _image_folder(
        train_dir,
        transform=training_transform(image_size),
        training=True,
    )
    val_dataset = _image_folder(
        train_dir,
        transform=evaluation_transform(image_size),
    )
    class_names = list(index_dataset.classes)
    if train_dataset.classes != class_names or val_dataset.classes != class_names:
        raise RuntimeError("Class order changed while building datasets")

    # 线上收集的图里奶龙极少，完全按自然分布训练会忽略奶龙；但三类强制等概率也可能
    # 过度重复少数类，增加相似黄色角色的误报。power 可在两者间连续调节：
    # 0 为自然分布，1 为三类等概率，介于两者之间时保留部分真实类别分布。
    # GIF 权重只改变同一类别内部的抽样，不改变该类别的总采样权重。
    animated_index_set = set(animated_indices)
    raw_weights = [
        ANIMATED_SAMPLE_WEIGHT if index in animated_index_set else 1.0
        for index in train_indices
    ]
    class_raw_weight_totals: Counter[int] = Counter()
    class_sample_counts: Counter[int] = Counter()
    for index, weight in zip(train_indices, raw_weights, strict=True):
        target = index_dataset.targets[index]
        class_raw_weight_totals[target] += weight
        class_sample_counts[target] += 1
    sample_weights = [
        (
            weight
            / class_raw_weight_totals[index_dataset.targets[index]]
            * class_sample_counts[index_dataset.targets[index]]
            ** (1.0 - class_balance_power)
        )
        for index, weight in zip(train_indices, raw_weights, strict=True)
    ]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_indices),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, class_names


def build_evaluation_loader(
    split_dir: Path,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    expected_classes: list[str],
) -> DataLoader:
    dataset = _image_folder(
        split_dir,
        transform=evaluation_transform(image_size),
    )
    if list(dataset.classes) != expected_classes:
        raise ValueError(
            f"{split_dir} class order {dataset.classes} does not match model {expected_classes}"
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
