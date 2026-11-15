# CIFAR-10 depth and residual connection comparison

This experiment uses the four unchanged model factories from main commit
`db3310f`: Plain-20, Plain-56, ResNet-20 and ResNet-56. Equal-depth pairs
have identical initial state dictionaries with seed 1. Data order and random
augmentation are seeded separately and reset for each model. The controlled
comparison uses one seed initially; the project's three-seed baseline and
independent extension remain separate work.

## Fixed protocol

- Official CIFAR-10 50,000 training images and 10,000 test images.
- Existing repository preprocessing: subtract the training per-pixel RGB mean
  in original 0–255 units, zero-pad 4 pixels, random horizontal flip and 32×32
  crop. Evaluation uses the original image with mean subtraction only.
- Batch size 128, SGD momentum 0.9, weight decay 0.0001, no dropout.
- Exactly 64,000 updates. Learning rate 0.1 for updates 1–32,000, 0.01 for
  32,001–48,000, and 0.001 for 48,001–64,000. Keep the smaller last batch.
- FP32, no AMP or TF32; deterministic PyTorch algorithms. All four models
  use identical settings. Reproducibility is limited to the recorded environment.
- Save the latest checkpoint at each epoch boundary. An interrupted epoch is
  replayed from the previous checkpoint, including optimizer and RNG states.
- Evaluate every epoch on the full unaugmented training and test splits in
  `eval()` mode. Also retain online training loss/error measured with augmentation
  and batch statistics; these are a different metric from clean training error.
- Compare the final fixed-budget checkpoints. Test errors do not control
  hyperparameters, stopping, seeds, or checkpoint selection.

The source is [He et al., section 4.2, Figures 1 and 6, Table 6](https://arxiv.org/pdf/1512.03385).
The paper's schedule was selected on a 45k/5k training/validation split and
used two GPUs. This replication applies that published schedule directly to
the official 50k training set on one GPU, with local PyTorch BN and evaluation
semantics. It does not claim bitwise reproduction of the original Caffe run.
Table 6 reports ResNet-20 test error 8.75% and ResNet-56 6.97%; those are
reference values, never substitutes for local measurements.

## Commands (PowerShell, repository root)

```powershell
$python = 'D:/codex/envs/resnet-cifar10/python.exe'
& $python -m scripts.prepare_data --data-dir D:/codex/datasets/cifar10
# One full epoch per model: pipeline and throughput check, not final evidence.
& $python -m scripts.train_comparison --output results/smoke-seed1 --stop-after-epochs 1
# Full comparison. --resume also works for a fresh directory.
& $python -m scripts.train_comparison --output results/seed1 --resume
# Optional: run the full suite, then commit/push only its result directory.
./scripts/run_comparison.ps1 -PushResults -GitProxy 'http://127.0.0.1:7897'
```

`--data-dir`, `--checkpoints`, `--workers`, `--models` and `--seed` are configurable.
Use a new output directory for a changed protocol. Do not run two processes
against the same output directory. Checkpoint loading is only for locally
generated, trusted files (`torch.load` uses pickle).

Each model produces `config.json`, `history.json` and, only after its budget
is exhausted, `complete.json`. The root produces `comparison.png`,
`summary.csv`, and `status.json`. Partial plots and summaries are explicitly
provisional. `RESULTS.md` is generated once all four models finish. Local
checkpoints and datasets are excluded from Git; only code, configuration,
metrics and figures are suitable for upload.

## Interpretation

The primary degradation signal is **higher training error for Plain-56 than
Plain-20**, accompanied by the test comparison. A test-only deterioration is
insufficient to identify optimization degradation. Lower ResNet-56 training
and test errors than ResNet-20 support the residual connection hypothesis in
this run. Report all four measured outcomes even if these inequalities fail.
A single seed supplies an example, not a statistical demonstration. Additional
seeds must be predeclared and all results retained.

## Local verification, 15 November 2026

All four unchanged model definitions passed a GPU forward/backward check;
equal-depth initial state dictionaries matched exactly for seed 1. Parameter
counts are 269,722 at depth 20 and 853,018 at depth 56. A synthetic-input
timing check measured approximately 0.018–0.020 seconds per update at depth
20 and 0.052–0.054 at depth 56. These timings exclude data loading and full
evaluation and are not classification results.

The existing environment has PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128,
NumPy 2.4.6 and Pillow 13.0.0; matplotlib 3.10.8 was added for plotting.
NumPy and Pillow therefore differ from `requirements.txt`; exact actual
versions are recorded in every run configuration.
