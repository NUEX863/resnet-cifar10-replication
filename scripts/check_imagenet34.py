"""CPU structural checks for ImageNet Plain-34 and Option A ResNet-34.

Optional execution from the repository root:
    python scripts/check_imagenet34.py
    python -m scripts.check_imagenet34

Uses synthetic inputs and unittest; no dataset, training or weight download.
Checks cover the paper's stage layout, forward dimensions, shortcut behavior,
initial BN state, and equal seeded parameters for the Plain/ResNet pair.
"""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.resnet_imagenet import BasicBlock34, OptionAShortcut, create_model


class ImageNet34Checks(unittest.TestCase):
    def test_paper_architecture_and_forward_shapes(self):
        for name in ("plain34", "resnet34"):
            with self.subTest(model=name):
                model = create_model(name, seed=1).eval()
                convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
                self.assertEqual(len(convs), 33)
                self.assertEqual(len(linears), 1)
                self.assertEqual(model.stem[0].kernel_size, (7, 7))
                self.assertEqual(model.stem[0].stride, (2, 2))
                self.assertEqual(model.stem[0].padding, (3, 3))
                self.assertEqual(model.maxpool.kernel_size, 3)
                self.assertEqual(model.maxpool.stride, 2)
                self.assertEqual(model.maxpool.padding, 1)
                self.assertEqual(model.classifier.in_features, 512)
                self.assertEqual(model.classifier.out_features, 1000)
                self.assertTrue(all(layer.bias is None for layer in convs))
                self.assertEqual(
                    sum(isinstance(m, BasicBlock34) for m in model.modules()), 16
                )
                stages = (model.stage1, model.stage2, model.stage3, model.stage4)
                channels = (64, 128, 256, 512)
                self.assertEqual(tuple(len(stage) for stage in stages), (3, 4, 6, 3))
                for index, (stage, width) in enumerate(zip(stages, channels)):
                    for block_index, block in enumerate(stage):
                        stride = 2 if index > 0 and block_index == 0 else 1
                        self.assertEqual(block.conv1.stride, (stride, stride))
                        self.assertEqual(block.conv2.stride, (1, 1))
                        self.assertEqual(block.conv1.out_channels, width)
                        self.assertEqual(block.conv2.out_channels, width)
                        self.assertEqual(block.conv1.kernel_size, (3, 3))
                        self.assertEqual(block.conv2.kernel_size, (3, 3))
                        self.assertEqual(block.residual, name == "resnet34")
                        self.assertEqual(sum(p.numel() for p in block.shortcut.parameters()), 0)

                observed = {}

                def record(label):
                    def hook(_module, _inputs, output):
                        observed[label] = tuple(output.shape)
                    return hook

                modules = {"stem": model.stem, "maxpool": model.maxpool}
                modules.update({f"stage{i}": stage for i, stage in enumerate(stages, 1)})
                modules["pool"] = model.pool
                handles = [m.register_forward_hook(record(label)) for label, m in modules.items()]
                try:
                    generator = torch.Generator().manual_seed(7)
                    x = torch.randn(1, 3, 224, 224, generator=generator)
                    with torch.inference_mode():
                        logits = model(x)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(observed, {
                    "stem": (1, 64, 112, 112),
                    "maxpool": (1, 64, 56, 56),
                    "stage1": (1, 64, 56, 56),
                    "stage2": (1, 128, 28, 28),
                    "stage3": (1, 256, 14, 14),
                    "stage4": (1, 512, 7, 7),
                    "pool": (1, 512, 1, 1),
                })
                self.assertEqual(tuple(logits.shape), (1, 1000))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_option_a_preserves_sampled_values_and_adds_only_zeros(self):
        x = torch.arange(1, 1 + 4 * 5 * 5, dtype=torch.float32).reshape(1, 4, 5, 5)
        self.assertIs(OptionAShortcut(4, 4, 1)(x), x)
        shortcut = OptionAShortcut(4, 8, 2)
        actual = shortcut(x)
        self.assertEqual(tuple(actual.shape), (1, 8, 3, 3))
        self.assertTrue(torch.equal(actual[:, 2:6], x[:, :, ::2, ::2]))
        self.assertEqual(torch.count_nonzero(actual[:, :2]).item(), 0)
        self.assertEqual(torch.count_nonzero(actual[:, 6:]).item(), 0)
        self.assertEqual(len(shortcut.state_dict()), 0)

    def test_zero_main_branch_exposes_residual_addition(self):
        # Zero learned branches: Plain emits zero; ResNet emits ReLU(shortcut).
        x = torch.linspace(-2, 2, 4 * 5 * 5).reshape(1, 4, 5, 5)
        for width, stride in ((4, 1), (8, 2)):
            for residual in (False, True):
                with self.subTest(width=width, residual=residual):
                    block = BasicBlock34(4, width, stride, residual).eval()
                    with torch.no_grad():
                        block.conv1.weight.zero_()
                        block.conv2.weight.zero_()
                        actual = block(x)
                    expected = torch.zeros_like(actual)
                    if residual:
                        sampled = x if stride == 1 else x[:, :, ::2, ::2]
                        if stride == 1:
                            expected = sampled.relu()
                        else:
                            expected[:, 2:6] = sampled.relu()
                    self.assertTrue(torch.equal(actual, expected))

    def test_paired_initialization_and_cpu_rng_preservation(self):
        before = torch.random.get_rng_state().clone()
        plain = create_model("plain34", seed=1)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        residual = create_model("resnet34", seed=1)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        left, right = plain.state_dict(), residual.state_dict()
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), msg=key)
        self.assertEqual(
            sum(p.numel() for p in plain.parameters()),
            sum(p.numel() for p in residual.parameters()),
        )
        for layer in plain.modules():
            if isinstance(layer, nn.BatchNorm2d):
                self.assertTrue(torch.equal(layer.weight, torch.ones_like(layer.weight)))
                self.assertTrue(torch.equal(layer.bias, torch.zeros_like(layer.bias)))
                self.assertTrue(torch.equal(layer.running_mean, torch.zeros_like(layer.running_mean)))
                self.assertTrue(torch.equal(layer.running_var, torch.ones_like(layer.running_var)))
                self.assertEqual(layer.num_batches_tracked.item(), 0)
        self.assertEqual(torch.count_nonzero(plain.classifier.bias).item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
