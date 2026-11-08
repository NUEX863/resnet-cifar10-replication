"""Download CIFAR-10 and compute the pixel mean for its official training split."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from torchvision.datasets import CIFAR10

from src.data import DEFAULT_DATA_DIR, MEAN_FILENAME


def check_split(dataset: CIFAR10, name: str, per_class: int) -> None:
    expected_count = 10 * per_class
    if dataset.data.shape != (expected_count, 32, 32, 3) or len(dataset.classes) != 10:
        raise ValueError(f"Unexpected {name} split: {dataset.data.shape}, {dataset.classes}")
    labels, counts = np.unique(dataset.targets, return_counts=True)
    if not np.array_equal(labels, np.arange(10)) or not np.all(counts == per_class):
        raise ValueError(f"Unexpected {name} class counts: {dict(zip(labels, counts))}")
    print(f"{name}: {expected_count:,} RGB images, 32 x 32, {per_class:,} per class")


def prepare(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    train = CIFAR10(root=data_dir, train=True, download=True)
    test = CIFAR10(root=data_dir, train=False)
    check_split(train, "train", per_class=5_000)
    check_split(test, "test", per_class=1_000)

    # Average the 50,000 unaugmented images along the sample axis only.
    mean = train.data.mean(axis=0, dtype=np.float64).astype(np.float32)
    target = data_dir / MEAN_FILENAME
    # Publish a complete file even if a previous preparation was interrupted.
    with TemporaryDirectory(dir=data_dir) as staging:
        temp_path = Path(staging) / MEAN_FILENAME
        np.save(temp_path, mean, allow_pickle=False)
        temp_path.replace(target)
    print(f"Saved training pixel mean: {target} (32 x 32 x 3, original RGB units)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    prepare(parser.parse_args().data_dir)
