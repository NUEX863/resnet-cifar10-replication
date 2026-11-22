"""ImageNet Plain-34 and ResNet-34 with parameter-free Option A shortcuts.

Architecture: He et al., Table 1, Figure 3 and section 4.1 (Figure 4 / Table 2).
Source: https://arxiv.org/pdf/1512.03385
Stage blocks: (3, 4, 6, 3); stage channels: (64, 128, 256, 512).
Weighted depth: one stem + 32 block convolutions + one classifier = 34.
For 224 x 224 RGB input, the four stage sizes are 56, 28, 14 and 7.

Both variants use Conv-BN-ReLU-Conv-BN blocks followed by ReLU. Only ResNet
adds the shortcut before the final ReLU. Option A subsamples by stride 2
and zero-pads channels at stage transitions; it has no learned parameters.
Channel padding is split equally before and after the original channels.

The paper specifies He initialization. This implementation uses the project's
explicit convention: fan_in / ReLU He normal for Conv and Linear weights,
zero classifier bias, BN scale one and offset zero. BN uses PyTorch defaults
(eps=1e-5, momentum=0.1). The factories return logits, before softmax.
No pretrained weights are loaded. create_model pairs the two variants using
the same seed without changing the caller's CPU random-generator state.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class OptionAShortcut(nn.Module):
    """Identity, or spatial subsampling plus zero-filled extra channels."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        if (stride, out_channels) not in ((1, in_channels), (2, 2 * in_channels)):
            raise ValueError("Option A requires identity or stride-2 channel doubling")
        self.stride = stride
        extra = out_channels - in_channels
        self.pad_before = extra // 2
        self.pad_after = extra - self.pad_before

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            return x
        return F.pad(
            x[:, :, ::2, ::2],
            (0, 0, 0, 0, self.pad_before, self.pad_after),
        )


class BasicBlock34(nn.Module):
    """The original post-activation two-convolution building block."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, residual: bool
    ):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = OptionAShortcut(in_channels, out_channels, stride)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)
        return F.relu(out)


def _stage(
    in_channels: int, out_channels: int, blocks: int, stride: int, residual: bool
) -> nn.Sequential:
    return nn.Sequential(
        BasicBlock34(in_channels, out_channels, stride, residual),
        *[
            BasicBlock34(out_channels, out_channels, 1, residual)
            for _ in range(blocks - 1)
        ],
    )


class ImageNet34(nn.Module):
    """Plain/ResNet-34; the paper's ImageNet classifier has 1,000 classes."""

    def __init__(self, residual: bool, num_classes: int = 1000):
        super().__init__()
        if type(num_classes) is not int or num_classes < 1:
            raise ValueError("num_classes must be a positive Python integer")
        if type(residual) is not bool:
            raise TypeError("residual must be a bool")
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.stage1 = _stage(64, 64, 3, 1, residual)
        self.stage2 = _stage(64, 128, 4, 2, residual)
        self.stage3 = _stage(128, 256, 6, 2, residual)
        self.stage4 = _stage(256, 512, 3, 2, residual)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
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
        x = self.maxpool(self.stem(x))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.classifier(self.pool(x).flatten(1))


def plain34(num_classes: int = 1000) -> ImageNet34:
    return ImageNet34(residual=False, num_classes=num_classes)


def resnet34(num_classes: int = 1000) -> ImageNet34:
    return ImageNet34(residual=True, num_classes=num_classes)


def create_model(model_name: str, seed: int, num_classes: int = 1000) -> ImageNet34:
    """Create a seeded CPU model; matched seeds give identical learned weights.

    Model names: plain34, resnet34. Seeds are integers in [0, 2**64 - 1].
    The same seed and num_classes pair all parameters and BN buffers. The
    caller controls any later device transfer, preprocessing and data RNG.
    """
    constructors = {"plain34": plain34, "resnet34": resnet34}
    if model_name not in constructors:
        raise ValueError(f"Unknown model {model_name!r}; choose plain34 or resnet34")
    if type(seed) is not int:
        raise TypeError("seed must be a Python integer")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be between 0 and 2**64 - 1")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        return constructors[model_name](num_classes=num_classes)
