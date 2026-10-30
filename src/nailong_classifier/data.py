from __future__ import annotations

import io
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_SUFFIXES = {".gif", ".jpg", ".jpeg", ".png", ".webp"}
ANIMATED_SAMPLE_WEIGHT = 4.0
DEFAULT_CLASS_BALANCE_POWER = 1.0
DOUBAO_DETECTOR_CLASSES = ["clean", "doubao"]
LEGACY_AUGMENTATION = "color_invariant_v2"
ROBUST_AUGMENTATION = "chat_robust_v3"
AUGMENTATION_PROFILES = (LEGACY_AUGMENTATION, ROBUST_AUGMENTATION)


class RandomChatDegradation:
    """模拟群聊图片被缩小和 JPEG 重压缩；仅用于训练视图，不改动原始文件。"""

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.35:
            factor = random.uniform(0.45, 0.85)
            dimensions = np.maximum(1, np.rint(np.multiply(image.size, factor))).astype(int)
            image = image.resize(tuple(dimensions), Image.Resampling.BILINEAR).resize(
                image.size, Image.Resampling.BILINEAR
            )
        if random.random() < 0.35:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=random.randint(45, 90))
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                image = compressed.convert("RGB")
        return image


@dataclass(frozen=True)
class DatasetSummary:
    split: str
    class_counts: dict[str, int]
    total: int


def training_transform(
    image_size: int, *, profile: str = LEGACY_AUGMENTATION
) -> transforms.Compose:
    if profile not in AUGMENTATION_PROFILES:
        raise ValueError(f"Unsupported augmentation profile: {profile}")
    # 保留独立豆包训练器的旧默认配置；主分类器通过显式 profile 开启群聊退化增强。
    degradation = (
        [
            RandomChatDegradation(),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.2),
        ]
        if profile == ROBUST_AUGMENTATION
        else []
    )
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
            *degradation,
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
    val_dir: Path | None = None,
    augmentation_profile: str = LEGACY_AUGMENTATION,
    sampling_sources: dict[str, str] | None = None,
    source_weights: dict[str, float] | None = None,
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
    if val_dir is None:
        train_indices, val_indices = train_test_split(
            static_indices,
            test_size=val_fraction,
            random_state=seed,
            shuffle=True,
            stratify=[index_dataset.targets[index] for index in static_indices],
        )
        train_indices.extend(animated_indices)
    else:
        # 显式留出集由数据整理阶段按内容去重划分，全部训练样本进入训练。
        # 这样高置信伪标签只放 train 时，不会被随机抽到验证集冒充真值。
        train_indices = list(range(len(index_dataset)))
        val_indices = None

    train_dataset = _image_folder(
        train_dir,
        transform=training_transform(image_size, profile=augmentation_profile),
        training=True,
    )
    val_dataset = _image_folder(
        val_dir if val_dir is not None else train_dir,
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
    if sampling_sources is not None or source_weights is not None:
        if sampling_sources is None or not source_weights:
            raise ValueError("sampling_sources and source_weights must be provided together")
        if any(not np.isfinite(weight) or weight <= 0 for weight in source_weights.values()):
            raise ValueError("Source weights must be positive and finite")
        # 先在每类内分配来源份额，再在来源内分配 GIF/静图权重。缺少某个来源时
        # 只对该类实际存在的来源重新归一化，不改变此前设置的类别总抽样概率。
        sources = []
        source_totals: Counter[tuple[int, str]] = Counter()
        for index, weight in zip(train_indices, raw_weights, strict=True):
            path, target = index_dataset.samples[index]
            source = sampling_sources.get(str(Path(path).resolve()))
            if source not in source_weights:
                raise ValueError(f"Missing source or source weight for training image: {path}")
            sources.append(source)
            source_totals[target, source] += weight
        class_source_totals: Counter[int] = Counter()
        for target, source in source_totals:
            class_source_totals[target] += source_weights[source]
        sample_weights = [
            float(np.multiply(
                np.divide(weight, source_totals[index_dataset.targets[index], source]),
                np.multiply(
                    np.divide(source_weights[source],
                              class_source_totals[index_dataset.targets[index]]),
                    np.power(class_sample_counts[index_dataset.targets[index]],
                             np.subtract(1.0, class_balance_power)),
                ),
            ))
            for index, weight, source in zip(train_indices, raw_weights, sources, strict=True)
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
        Subset(val_dataset, val_indices) if val_indices is not None else val_dataset,
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


class _DoubaoDetectorDataset(Dataset):
    """把现有四分类样本映射为“未检测到豆包 / 含豆包”的独立检测任务。"""

    def __init__(
        self,
        samples: list[tuple[str, int]],
        *,
        doubao_index: int,
        transform: transforms.Compose,
        training: bool,
    ) -> None:
        self.samples = samples
        self.doubao_index = doubao_index
        self.transform = transform
        self.loader = training_image_loader if training else evaluation_image_loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, source_target = self.samples[index]
        image = self.loader(path)
        target = 1 if source_target == self.doubao_index else 0
        return self.transform(image), target


def _doubao_detector_samples(
    split_dir: Path,
) -> tuple[list[tuple[str, int]], int]:
    index_dataset = _image_folder(split_dir)
    if "doubao" not in index_dataset.class_to_idx:
        raise ValueError(f"{split_dir} has no doubao class")
    return index_dataset.samples, index_dataset.class_to_idx["doubao"]


def build_doubao_detector_train_val_loaders(
    train_dir: Path,
    *,
    image_size: int,
    batch_size: int,
    val_fraction: float,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """构建独立豆包检测器的平衡二分类训练集与验证集。"""
    from sklearn.model_selection import train_test_split

    samples, doubao_index = _doubao_detector_samples(train_dir)
    static_indices = [
        index for index, (path, _) in enumerate(samples) if Path(path).suffix.lower() != ".gif"
    ]
    animated_indices = [
        index for index, (path, _) in enumerate(samples) if Path(path).suffix.lower() == ".gif"
    ]
    static_targets = [
        1 if samples[index][1] == doubao_index else 0 for index in static_indices
    ]
    train_indices, val_indices = train_test_split(
        static_indices,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=static_targets,
    )
    train_indices.extend(animated_indices)
    train_dataset = _DoubaoDetectorDataset(
        samples,
        doubao_index=doubao_index,
        transform=training_transform(image_size),
        training=True,
    )
    val_dataset = _DoubaoDetectorDataset(
        samples,
        doubao_index=doubao_index,
        transform=evaluation_transform(image_size),
        training=False,
    )
    train_targets = [
        1 if samples[index][1] == doubao_index else 0 for index in train_indices
    ]
    class_counts = Counter(train_targets)
    sample_weights = [1.0 / class_counts[target] for target in train_targets]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_indices),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        DataLoader(
            Subset(train_dataset, train_indices),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,
        ),
        DataLoader(
            Subset(val_dataset, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        DOUBAO_DETECTOR_CLASSES,
    )


def build_doubao_detector_evaluation_loader(
    split_dir: Path,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    samples, doubao_index = _doubao_detector_samples(split_dir)
    dataset = _DoubaoDetectorDataset(
        samples,
        doubao_index=doubao_index,
        transform=evaluation_transform(image_size),
        training=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
