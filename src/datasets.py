from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


class SortedImageFolderDataset(Dataset):
    def __init__(self, image_dir: Path, transform=None) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.samples = sorted(self.image_dir.glob("*.jpg"), key=lambda p: int(p.stem))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, None, str(path)


def subset_dataset(dataset: Dataset, limit: Optional[int], seed: int) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def build_dogs_vs_cats_transforms(image_size: int, augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dogs_vs_cats_dataloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_augment: bool = True,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> Dict:
    data_root = Path(data_root)
    train_dataset = datasets.ImageFolder(
        root=data_root / "train",
        transform=build_dogs_vs_cats_transforms(image_size=image_size, augment=train_augment),
    )
    val_dataset = datasets.ImageFolder(
        root=data_root / "val",
        transform=build_dogs_vs_cats_transforms(image_size=image_size, augment=False),
    )
    test_dataset = SortedImageFolderDataset(
        image_dir=data_root / "test",
        transform=build_dogs_vs_cats_transforms(image_size=image_size, augment=False),
    )

    train_dataset = subset_dataset(train_dataset, train_limit, seed)
    val_dataset = subset_dataset(val_dataset, val_limit, seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_test_batch,
    )

    class_to_idx = {"cat": 0, "dog": 1}
    idx_to_class = {value: key for key, value in class_to_idx.items()}
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }


def collate_test_batch(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    paths = [item[2] for item in batch]
    return images, None, paths


def build_cifar10_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _targets_from_dataset(dataset: Dataset) -> List[int]:
    if isinstance(dataset, Subset):
        base_targets = _targets_from_dataset(dataset.dataset)
        return [base_targets[index] for index in dataset.indices]
    if hasattr(dataset, "targets"):
        return [int(item) for item in dataset.targets]
    raise ValueError("Dataset does not expose targets.")


def build_long_tail_indices(
    targets: Sequence[int],
    num_classes: int,
    imbalance_ratio: float,
    seed: int,
) -> List[int]:
    rng = np.random.default_rng(seed)
    class_indices: Dict[int, List[int]] = {class_id: [] for class_id in range(num_classes)}
    for index, target in enumerate(targets):
        class_indices[int(target)].append(index)

    max_samples = min(len(indices) for indices in class_indices.values())
    selected: List[int] = []

    for class_id in range(num_classes):
        retain_ratio = imbalance_ratio ** (class_id / max(1, num_classes - 1))
        class_count = max(1, int(round(max_samples * retain_ratio)))
        pool = np.array(class_indices[class_id])
        rng.shuffle(pool)
        selected.extend(pool[:class_count].tolist())

    rng.shuffle(selected)
    return selected


def class_distribution(targets: Sequence[int], num_classes: int) -> Dict[int, int]:
    counts = {class_id: 0 for class_id in range(num_classes)}
    for target in targets:
        counts[int(target)] += 1
    return counts


def build_weighted_sampler(targets: Sequence[int], num_classes: int) -> WeightedRandomSampler:
    counts = class_distribution(targets, num_classes)
    sample_weights = [1.0 / counts[int(target)] for target in targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def build_class_weights(targets: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = class_distribution(targets, num_classes)
    total = float(sum(counts.values()))
    weights = [total / (num_classes * max(1, counts[class_id])) for class_id in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def build_cifar10_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_size: int = 5000,
    imbalance: str = "none",
    imbalance_ratio: float = 0.1,
    mitigation: str = "none",
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> Dict:
    data_root = Path(data_root)
    train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=build_cifar10_transforms(True))
    eval_full = datasets.CIFAR10(root=data_root, train=True, download=False, transform=build_cifar10_transforms(False))
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=build_cifar10_transforms(False))

    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(train_full), generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]

    train_split = Subset(train_full, train_indices)
    eval_train_split = Subset(eval_full, train_indices)
    eval_val_split = Subset(eval_full, val_indices)

    original_targets = _targets_from_dataset(eval_train_split)
    if imbalance == "long_tail":
        chosen = build_long_tail_indices(original_targets, num_classes=10, imbalance_ratio=imbalance_ratio, seed=seed)
        train_split = Subset(train_split, chosen)
        eval_train_split = Subset(eval_train_split, chosen)

    train_targets = _targets_from_dataset(eval_train_split)
    train_distribution = class_distribution(train_targets, num_classes=10)

    sampler = None
    shuffle = True
    if mitigation == "weighted_sampler":
        sampler = build_weighted_sampler(train_targets, num_classes=10)
        shuffle = False

    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        eval_val_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_distribution": train_distribution,
        "class_names": test_dataset.classes,
        "train_targets": train_targets,
        "val_size": len(eval_val_split),
        "train_size": len(train_split),
        "test_size": len(test_dataset),
    }
