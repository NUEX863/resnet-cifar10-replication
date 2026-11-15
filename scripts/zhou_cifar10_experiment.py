"""Independent four-model training/evaluation for Zhou; never edits Li's scripts.

Run from the repository root: python -m scripts.zhou_cifar10_experiment --help
The published schedule is fixed at 64,000 updates. Test scores never select weights.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
import torchvision

from src.data import MEAN_FILENAME
from src.resnet import create_model
from scripts.zhou_plot_comparison import MODELS, report, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "zhou_seed1_20261115"
DEFAULT_CHECKPOINTS = ROOT / "checkpoints" / "zhou_seed1_20261115"
STEPS = 64000
BATCH = 128


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def learning_rate(completed_steps):
    return .1 * .1 ** sum(completed_steps >= milestone for milestone in (32000, 48000))


def augment_batch(images, mean, generator):
    """Original RGB units: mean subtraction -> zero pad -> per-image flip/crop.

    Random choices use an independent CPU generator. GPU tensor indexing only
    accelerates the same transforms; it does not rescale colors or add samples.
    """
    centered = images.float() - mean
    size = images.shape[0]
    flip = (torch.rand(size, generator=generator) < .5).to(images.device)
    top = torch.randint(0, 9, (size,), generator=generator).to(images.device)
    left = torch.randint(0, 9, (size,), generator=generator).to(images.device)
    padded = F.pad(centered, (4, 4, 4, 4))
    padded = torch.where(flip[:, None, None, None], padded.flip(-1), padded)
    axis = torch.arange(32, device=images.device)
    cropped = padded.permute(0, 2, 3, 1)[
        torch.arange(size, device=images.device)[:, None, None],
        top[:, None, None] + axis[None, :, None],
        left[:, None, None] + axis[None, None, :], :]
    return cropped.permute(0, 3, 1, 2).contiguous()


class Data:
    def __init__(self, directory, device):
        directory = Path(directory)
        mean_path = directory / MEAN_FILENAME
        mean = np.load(mean_path, allow_pickle=False)
        if mean.shape != (32, 32, 3) or mean.dtype != np.float32 or not np.isfinite(mean).all():
            raise ValueError("Invalid training pixel mean; run scripts.prepare_data first")
        self.mean_hash = sha256(mean_path)
        self.mean = torch.from_numpy(mean).permute(2, 0, 1).to(device)
        self.splits = {}
        for split, train, expected in (("train", True, 50000), ("test", False, 10000)):
            dataset = CIFAR10(directory, train=train, download=False)
            if len(dataset) != expected:
                raise ValueError(f"Expected full {split} split ({expected} images)")
            if train:
                actual_mean = dataset.data.mean(axis=0, dtype=np.float64).astype(np.float32)
                if not np.array_equal(actual_mean, mean):
                    raise ValueError("Mean file is not the pixel mean of this 50,000-image training set")
            images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous().to(device)
            labels = torch.tensor(dataset.targets, dtype=torch.long, device=device)
            self.splits[split] = (images, labels)


@torch.inference_mode()
def evaluate(model, images, labels, mean, batch_size=256):
    model.eval()
    wrong = torch.zeros((), dtype=torch.long, device=images.device)
    loss_sum = torch.zeros((), dtype=torch.float64, device=images.device)
    for start in range(0, len(labels), batch_size):
        y = labels[start:start + batch_size]
        logits = model(images[start:start + batch_size].float() - mean)
        wrong += (logits.argmax(1) != y).sum()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").double()
    count = len(labels)
    return {"count": count, "wrong": wrong.item(), "error": 100 * wrong.item() / count,
            "loss": loss_sum.item() / count}


def measure(model, data, batch_size):
    metrics = {}
    for split, (images, labels) in data.splits.items():
        result = evaluate(model, images, labels, data.mean, batch_size)
        metrics.update({f"{split}_{k}": v for k, v in result.items()})
    if not all(math.isfinite(v) for v in metrics.values()):
        raise RuntimeError("Nonfinite evaluation; refusing to publish a result")
    return metrics


def configure():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local training run")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device("cuda")


@contextmanager
def run_lock(directory):
    """OS lock is released even if a process crashes; no stale PID cleanup needed."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def protocol(args, data):
    return {"seed": args.seed, "steps": STEPS, "batch_size": BATCH, "learning_rate": .1,
            "milestones": [32000, 48000], "momentum": .9, "weight_decay": .0001,
            "precision": "FP32; TF32 disabled", "deterministic": True,
            "preprocessing": "training pixel mean in RGB 0..255; zero-pad 4, flip, random 32x32 crop",
            "augmentation_backend": "batched tensors, independent CPU generator per model",
            "evaluation": "full unaugmented 50000/10000 splits in eval mode",
            "eval_batch_size": args.eval_batch_size,
            "data_dir": str(args.data_dir.resolve()), "output": str(args.output.resolve()),
            "mean_sha256": data.mean_hash, "model_sha256": sha256(ROOT / "src/resnet.py"),
            "trainer_sha256": sha256(__file__), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "numpy": np.__version__,
            "python": platform.python_version(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name()}


def train(args, data, device):
    config = protocol(args, data)
    output = args.output
    if (output / "run.json").exists() and not args.resume:
        raise RuntimeError("Output already exists: use --resume or a new output/checkpoint pair")
    states = {}
    for name in MODELS:
        model = create_model(name, args.seed).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(args.seed)
        checkpoint = args.checkpoints / name / "last.pt"
        state = {"model": model, "optimizer": optimizer, "generator": generator,
                 "history": [], "step": 0, "epoch": 0}
        if checkpoint.exists():
            if not args.resume:
                raise RuntimeError(f"Checkpoint exists: {checkpoint}; use --resume")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["config"] != {**config, "model": name}:
                raise ValueError(f"Resume protocol/environment/source mismatch: {name}")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            generator.set_state(saved["data_rng"])
            for key in ("history", "step", "epoch"):
                state[key] = saved[key]
        elif (output / name / "history.json").exists():
            raise RuntimeError(f"History exists without checkpoint for {name}; refusing to restart silently")
        states[name] = state
        write_json(output / name / "config.json", {**config, "model": name})
        if state["history"]:
            write_json(output / name / "history.json", state["history"])
        if state["step"] == STEPS:
            write_json(output / name / "complete.json", {"step": STEPS})
    run_metadata = {"config": config, "pid": os.getpid(),
                    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(output / "run.json", run_metadata)
    train_images, train_labels = data.splits["train"]
    rounds = 0
    while any(state["step"] < STEPS for state in states.values()):
        # After interruption, lagging models catch up before any model advances again.
        next_epoch = min(s["epoch"] for s in states.values() if s["step"] < STEPS)
        for name, state in states.items():
            if state["step"] == STEPS or state["epoch"] != next_epoch:
                continue
            started = time.perf_counter()
            model, optimizer, generator = state["model"], state["optimizer"], state["generator"]
            def publish_progress(phase):
                write_json(output / "live_progress.json", {
                    "model": name, "phase": phase,
                    "epoch": state["epoch"] if phase == "saving results" else state["epoch"] + 1,
                    "steps": {n: s["step"] for n, s in states.items()},
                    "target_steps": STEPS, "pid": os.getpid(),
                    "updated_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            publish_progress("training")
            model.train()
            order = torch.randperm(len(train_labels), generator=generator).to(device)
            wrong = torch.zeros((), dtype=torch.long, device=device)
            loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            seen = 0
            torch.cuda.reset_peak_memory_stats()
            for start in range(0, len(order), BATCH):
                indices = order[start:start + BATCH]
                x = augment_batch(train_images[indices], data.mean, generator)
                y = train_labels[indices]
                lr = learning_rate(state["step"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                wrong += (logits.detach().argmax(1) != y).sum()
                loss_sum += loss.detach().double() * y.numel()
                seen += y.numel()
                state["step"] += 1
                if state["step"] % 25 == 0:
                    publish_progress("training")
                if state["step"] % 100 == 0:
                    print(f"{name}: update {state['step']}/{STEPS}", flush=True)
                if state["step"] == STEPS:
                    break
            publish_progress("evaluating full train/test splits")
            metrics = measure(model, data, args.eval_batch_size)
            state["epoch"] += 1
            row = {"epoch": state["epoch"], "step": state["step"], "lr": lr,
                   "online_train_count": seen, "online_train_error": 100 * wrong.item() / seen,
                   "online_train_loss": loss_sum.item() / seen, **metrics,
                   "seconds": time.perf_counter() - started,
                   "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1024**2}
            if not all(math.isfinite(v) for v in row.values()):
                raise RuntimeError(f"Nonfinite metrics for {name}")
            state["history"].append(row)
            publish_progress("saving results")
            save_checkpoint(args.checkpoints / name / "last.pt", {
                "config": {**config, "model": name}, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "data_rng": generator.get_state(),
                "step": state["step"], "epoch": state["epoch"], "history": state["history"]})
            write_json(output / name / "history.json", state["history"])
            if state["step"] == STEPS:
                write_json(output / name / "complete.json", {"step": STEPS, "final": row})
            complete = all(s["step"] == STEPS for s in states.values())
            remaining_seconds = sum((STEPS - s["step"]) *
                                    (s["history"][-1]["seconds"] / min(391, s["step"]))
                                    for s in states.values() if s["history"])
            write_json(output / "status.json", {"state": "complete" if complete else "running",
                       "complete": complete, "updated_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "steps": {n: s["step"] for n, s in states.items()},
                       "estimated_remaining_hours": remaining_seconds / 3600 if all(s["history"] for s in states.values()) else None})
            report(output, output / "figures", "Zhou RTX 3060 - seed " + str(args.seed))
            print(json.dumps({"model": name, **row}), flush=True)
            if (output / "STOP_AFTER_EPOCH").exists():
                write_json(output / "process_status.json", {"state": "paused", "reason": "STOP_AFTER_EPOCH"})
                return
        rounds += 1
        if args.max_rounds and rounds >= args.max_rounds:
            write_json(output / "process_status.json", {"state": "paused", "reason": "max_rounds"})
            return
    write_json(output / "process_status.json", {"state": "complete"})


def evaluate_checkpoints(args, data, device):
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Evaluation needs a new, empty output directory")
    for name in MODELS:
        path = args.checkpoints / name / "last.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["config"]["model"] != name or saved["config"]["mean_sha256"] != data.mean_hash:
            raise ValueError(f"Model or data mean mismatch: {path}")
        if saved["config"]["model_sha256"] != sha256(ROOT / "src/resnet.py"):
            raise ValueError(f"Model source differs from checkpoint: {path}")
        model = create_model(name, saved["config"]["seed"]).to(device)
        model.load_state_dict(saved["model"], strict=True)
        row = {"epoch": saved["epoch"], "step": saved["step"], **measure(model, data, args.eval_batch_size)}
        write_json(args.output / name / "history.json", [row])
        write_json(args.output / name / "provenance.json", {"checkpoint": str(path.resolve()),
                   "sha256": sha256(path), "config": saved["config"]})
        if saved["step"] == STEPS:
            write_json(args.output / name / "complete.json", {"step": STEPS})
        print(json.dumps({"model": name, **row}), flush=True)
        del model
    report(args.output, args.output / "figures", "Independent checkpoint re-evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=0, help="Pause after N rounds; 0 runs the full budget")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32 or args.eval_batch_size <= 0 or args.max_rounds < 0:
        parser.error("Invalid seed, batch size or round count")
    protected = (ROOT / "results" / "seed1").resolve()
    if args.output.resolve().is_relative_to(protected) or args.checkpoints.resolve().is_relative_to(protected):
        parser.error("Li's results/seed1 is read-only for this program; choose your own folders")
    if args.mode == "train" and (args.output / "STOP_AFTER_EPOCH").exists():
        parser.error("Remove your STOP_AFTER_EPOCH file before resuming")
    device = configure()
    with run_lock(args.checkpoints):
        data = Data(args.data_dir, device)
        if args.mode == "evaluate":
            evaluate_checkpoints(args, data, device)
        else:
            try:
                write_json(args.output / "process_status.json", {"state": "running", "pid": os.getpid()})
                train(args, data, device)
            except BaseException as exc:
                write_json(args.output / "process_status.json", {"state": "stopped", "reason": repr(exc)})
                raise


if __name__ == "__main__":
    main()
