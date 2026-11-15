"""Paired CIFAR-10 experiments; run from the repository with python -m scripts.train_comparison."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision
import PIL
from src.data import make_loaders
from src.resnet import create_model

MODELS = ("plain20", "plain56", "resnet20", "resnet56")


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    wrong = torch.zeros((), device=device, dtype=torch.long)
    loss = torch.zeros((), device=device)
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        wrong += (logits.argmax(1) != y).sum()
        loss += nn.functional.cross_entropy(logits, y, reduction="sum")
        count += y.numel()
    return 100 * wrong.item() / count, loss.item() / count


def report(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    complete = all((root / n / "complete.json").exists() for n in MODELS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    finals = []
    for name in MODELS:
        path = root / name / "history.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not rows:
            continue
        ax = axes[0 if name.startswith("plain") else 1]
        color = "tab:blue" if name.endswith("20") else "tab:orange"
        for key, style, label in [("train_error", "--", "train (eval)"), ("test_error", "-", "test")]:
            ax.plot([r["step"] for r in rows], [r[key] for r in rows], style,
                    color=color, label=f"{name} {label}")
        finals.append({"model": name, **rows[-1]})
    for ax, title in zip(axes, ("Plain networks", "Residual networks")):
        ax.set(title=title, xlabel="SGD updates", ylabel="Error (%)", ylim=(0, 100))
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle("CIFAR-10: measured errors" + (" (completed budget)" if complete else " (partial runs)"))
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=180)
    plt.close(fig)
    if finals:
        with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(finals[0]))
            writer.writeheader()
            writer.writerows(finals)
    write_json(root / "status.json", {"complete": complete, "available_models": [r["model"] for r in finals]})
    if complete:
        values = {r["model"]: r for r in finals}
        lines = ["# Measured CIFAR-10 comparison", "", "Final fixed-budget checkpoints; no selection by test accuracy.", "",
                 "| Model | Train error (%) | Test error (%) |", "|---|---:|---:|"]
        lines += [f"| {r['model']} | {r['train_error']:.2f} | {r['test_error']:.2f} |" for r in finals]
        lines += ["", "Differences below are depth 56 minus depth 20, in percentage points."]
        for family in ("plain", "resnet"):
            for key in ("train_error", "test_error"):
                delta = values[family + "56"][key] - values[family + "20"][key]
                lines.append(f"- {family} {key}: {delta:+.2f}")
        lines += ["", "Positive Plain training difference supports degradation in this run. Negative ResNet differences support improvement with depth. A single seed does not establish robustness; report contradictory outcomes as measured."]
        (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(args, name):
    out = args.output / name
    out.mkdir(parents=True, exist_ok=True)
    run_id = hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()[:12]
    checkpoint = args.checkpoints / run_id / name / "last.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train_loader, test_loader = make_loaders(args.data_dir, batch_size=128, seed=args.seed, num_workers=args.workers)
    # Dedicated evaluation generator prevents evaluation from changing training RNG.
    clean = CIFAR10(args.data_dir, train=True, transform=test_loader.dataset.transform)
    clean_loader = DataLoader(clean, batch_size=256, num_workers=args.workers,
                              generator=torch.Generator().manual_seed(args.seed + 1))
    model = create_model(name, seed=args.seed).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=1e-4)
    config = dict(model=name, seed=args.seed, steps=args.steps, batch_size=128,
                  lr=.1, milestones=[32000, 48000], momentum=.9, weight_decay=1e-4,
                  workers=args.workers, precision="FP32; TF32 disabled", deterministic=True,
                  preprocessing="training pixel mean in RGB 0..255; zero pad 4, random flip/crop",
                  data_dir=str(args.data_dir.resolve()), train_size=50000, test_size=10000,
                  model_sha256=hashlib.sha256(Path("src/resnet.py").read_bytes()).hexdigest(),
                  trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  data_code_sha256=hashlib.sha256(Path("src/data.py").read_bytes()).hexdigest(),
                  mean_sha256=hashlib.sha256((args.data_dir / "cifar10_train_mean.npy").read_bytes()).hexdigest(),
                  torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__,
                  pillow=PIL.__version__,
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    step, epoch, history = 0, 0, []
    if checkpoint.exists():
        if not args.resume:
            raise RuntimeError(f"Existing checkpoint {checkpoint}; use --resume or a new output directory")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if {k: v for k, v in state["config"].items() if k != "commit"} != {k: v for k, v in config.items() if k != "commit"}:
            raise ValueError("Resume configuration differs from checkpoint")
        config = state["config"]
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, history = state["step"], state["epoch"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        train_loader.generator.set_state(state["loader_rng"])
    elif (out / "config.json").exists():
        raise RuntimeError("Output exists without a checkpoint; choose a new output directory")
    write_json(out / "config.json", config)
    torch.cuda.reset_peak_memory_stats()
    while step < args.steps:
        started = time.perf_counter()
        model.train()
        wrong, loss_sum, count = torch.zeros((), device=device), torch.zeros((), device=device), 0
        for x, y in train_loader:
            lr = .1 * (.1 ** sum(step >= m for m in (32000, 48000)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            wrong += (logits.detach().argmax(1) != y).sum()
            loss_sum += loss.detach() * y.numel()
            count += y.numel()
            step += 1
            if step >= args.steps:
                break
        train_error, train_loss = evaluate(model, clean_loader, device)
        test_error, test_loss = evaluate(model, test_loader, device)
        epoch += 1
        row = dict(epoch=epoch, step=step, lr=lr, online_train_error=100 * wrong.item() / count,
                   online_train_loss=loss_sum.item() / count, train_error=train_error,
                   train_loss=train_loss, test_error=test_error, test_loss=test_loss,
                   seconds=time.perf_counter()-started,
                   peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)
        if not all(np.isfinite(v) for v in row.values()):
            raise RuntimeError(f"Nonfinite metrics: {row}")
        history.append(row)
        state = dict(config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     epoch=epoch, step=step, history=history, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(),
                     numpy_rng=np.random.get_state(), loader_rng=train_loader.generator.get_state())
        temp = checkpoint.with_suffix(".tmp")
        torch.save(state, temp)
        temp.replace(checkpoint)
        write_json(out / "history.json", history)
        print(json.dumps({"model": name, **row}), flush=True)
        report(args.output)
        if args.stop_after_epochs and epoch >= args.stop_after_epochs and step < args.steps:
            return
    write_json(out / "complete.json", {"step": step, "final": history[-1]})
    report(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("D:/codex/datasets/cifar10"))
    parser.add_argument("--output", type=Path, default=Path("results/seed1"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=64000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epochs", type=int, help="Stop at an epoch boundary for timing/resume checks; does not mark a run complete")
    args = parser.parse_args()
    if args.steps <= 0 or not 0 <= args.seed < 2**32:
        parser.error("steps must be positive and seed must be in [0, 2**32)")
    for name in args.models:
        train(args, name)


if __name__ == "__main__":
    main()
