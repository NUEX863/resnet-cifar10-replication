"""CIFAR-10 Plain/ResNet-20 and Plain/ResNet-56 for the degradation experiment.

Architecture follows He et al., section 4.2 (arXiv:1512.03385).
The two variants share all learned layers and differ only in shortcut addition.
The weighted depth is one stem convolution, 6n block convolutions and one
classifier, with n=3 for depth 20 and n=9 for depth 56.

Implementation reference: Yerlan Idelbayev's pytorch_resnet_cifar10
(https://github.com/akamaster/pytorch_resnet_cifar10), see LICENSE.akamaster.
This revision extends our existing src/resnet.py at project commit 5ba56e9.

Initialization uses He normal with fan_in and ReLU gain for convolution and
linear weights, zero linear bias, BN scale one and BN offset zero. The paper
cites He initialization but does not disclose every CIFAR-10 initialization
detail. fan_in and the classifier distribution follow the reference above.
BN uses PyTorch defaults (eps=1e-5, momentum=0.1, running mean=0, variance=1).

Use create_model("plain20", seed=1) or create_model("resnet20", seed=1) for
paired initialization. The factory creates a CPU model using the supplied
seed and restores the caller's CPU random state. Equal-depth Plain and
ResNet have identical initial state_dict values in the same environment.
This pairing is our experimental control. Seed data loading separately.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    """Two post-activation convolutions with an optional Option A shortcut."""

    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True):
        super().__init__()
        self.shortcut = shortcut
        self.downsample = in_channels != out_channels
        if self.downsample and out_channels != 2 * in_channels:
            raise ValueError("A stage transition must double the channel count")
        stride = 2 if self.downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))

        if self.shortcut:
            shortcut = x
            if self.downsample:
                # Option A halves spatial size and pads channels without learned weights.
                channels = x.size(1)
                shortcut = F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, channels // 2, channels - channels // 2),
                )
            residual = residual + shortcut
        return F.relu(residual)


def _stage(
    in_channels: int, out_channels: int, blocks: int, shortcut: bool
) -> nn.Sequential:
    return nn.Sequential(
        BasicBlock(in_channels, out_channels, shortcut),
        *[BasicBlock(out_channels, out_channels, shortcut) for _ in range(blocks - 1)],
    )


class CifarNet(nn.Module):
    """Map 32 x 32 RGB inputs to ten logits using depth 20 or 56."""

    def __init__(self, depth: int, shortcut: bool):
        super().__init__()
        if depth not in (20, 56):
            raise ValueError("Supported depths are 20 and 56")
        blocks = (depth - 2) // 6
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.stage1 = _stage(16, 16, blocks, shortcut)
        self.stage2 = _stage(16, 32, blocks, shortcut)
        self.stage3 = _stage(32, 64, blocks, shortcut)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, 10)

        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
                layer.reset_running_stats()

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(self.pool(x).flatten(1))


class ResNet20(CifarNet):
    """Retain the original class entry point and state_dict key layout."""

    def __init__(self):
        super().__init__(depth=20, shortcut=True)


def plain20() -> CifarNet:
    return CifarNet(depth=20, shortcut=False)


def plain56() -> CifarNet:
    return CifarNet(depth=56, shortcut=False)


def resnet20() -> ResNet20:
    return ResNet20()


def resnet56() -> CifarNet:
    return CifarNet(depth=56, shortcut=True)


def create_model(model_name: str, seed: int) -> CifarNet:
    """Create one of the four CPU models with reproducible initial weights.

    Supported names are plain20, plain56, resnet20 and resnet56. The seed
    must be a Python integer in [0, 2**64 - 1]. Call model.to(device) in the
    training program to move it to the GPU. This seed controls model
    initialization; data order and augmentation use the training program's
    random state and loader configuration.
    """
    constructors = {
        "plain20": plain20,
        "plain56": plain56,
        "resnet20": resnet20,
        "resnet56": resnet56,
    }
    if model_name not in constructors:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {tuple(constructors)}.")
    if type(seed) is not int:
        raise TypeError("seed must be a Python integer")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be between 0 and 2**64 - 1")

    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        # Seed only the CPU generator used to construct and initialize the model.
        torch.random.default_generator.manual_seed(seed)
        return constructors[model_name]()
