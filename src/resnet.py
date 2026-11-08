"""CIFAR-10 ResNet-20 from He et al., section 4.2 (arXiv:1512.03385).

Implementation reference: Yerlan Idelbayev's pytorch_resnet_cifar10
(https://github.com/akamaster/pytorch_resnet_cifar10).
"""

from torch import Tensor, nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    """Two convolutions with a parameter-free Option A shortcut."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
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

        shortcut = x
        if self.downsample:
            # Retain every second pixel and center the input in the wider channel axis.
            channels = x.size(1)
            shortcut = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, channels // 2, channels - channels // 2))
        return F.relu(residual + shortcut)


def _stage(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        BasicBlock(in_channels, out_channels),
        BasicBlock(out_channels, out_channels),
        BasicBlock(out_channels, out_channels),
    )


class ResNet20(nn.Module):
    """Map a batch of 32 x 32 RGB images to ten class logits per image."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.stage1 = _stage(16, 16)
        self.stage2 = _stage(16, 32)
        self.stage3 = _stage(32, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, 10)

        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(self.pool(x).flatten(1))


def resnet20() -> ResNet20:
    return ResNet20()
