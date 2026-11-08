"""CIFAR-10 inputs for the classification experiments in ResNet section 4.2."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import pil_to_tensor


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MEAN_FILENAME = "cifar10_train_mean.npy"


class SubtractPixelMean:
    def __init__(self, mean: torch.Tensor):
        self.mean = mean

    def __call__(self, image: Image.Image) -> torch.Tensor:
        # pil_to_tensor preserves the original RGB units (0 to 255).
        return pil_to_tensor(image).float() - self.mean


def make_loaders(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    batch_size: int = 128,
    seed: int = 1,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Return training and test loaders for the official 50k/10k split."""
    root = Path(data_dir)
    mean_path = root / MEAN_FILENAME
    if not mean_path.is_file():
        raise FileNotFoundError(
            f"Missing {mean_path}. Run python -m scripts.prepare_data "
            f'--data-dir "{root}" first.'
        )
    mean = np.load(mean_path, allow_pickle=False)
    if mean.shape != (32, 32, 3) or mean.dtype != np.float32:
        raise ValueError(f"Unexpected mean image in {mean_path}. Run data preparation again.")
    center = SubtractPixelMean(torch.from_numpy(mean).permute(2, 0, 1))
    augment = transforms.Compose([
        center,
        transforms.Pad(4, fill=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32),
    ])
    train = CIFAR10(root=root, train=True, transform=augment)
    test = CIFAR10(root=root, train=False, transform=center)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    return train_loader, test_loader
