"""Zhou's independent plots; accepts either teammate histories or local histories."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "checkpoints" / "zhou_mpl_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ("plain20", "plain56", "resnet20", "resnet56")
LABELS = {name: name.replace("plain", "Plain-").replace("resnet", "ResNet-") for name in MODELS}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # Windows readers can briefly hold a handle without delete-sharing. Keep
    # the previous complete JSON visible and retry the atomic publication.
    for attempt in range(100):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.03)


def read_histories(root):
    histories = {}
    for name in MODELS:
        path = root / name / "history.json"
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        steps = [row["step"] for row in rows]
        if steps != sorted(set(steps)):
            raise ValueError(f"Non-increasing or duplicate steps in {path}")
        for row in rows:
            for key in ("train_error", "test_error"):
                if not 0 <= row[key] <= 100:
                    raise ValueError(f"Invalid error rate in {path}: {row}")
        histories[name] = rows
    return histories


def draw_panels(histories, panels, destination, title, zoom=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    maximum_step = max((r["step"] for rows in histories.values() for r in rows), default=0)
    for ax, (panel_title, names, keys) in zip(axes, panels):
        for name in names:
            rows = histories[name]
            if not rows:
                continue
            color = "#2166ac" if name.endswith("20") else "#c43c39"
            x = [r["step"] / 10000 for r in rows]
            for key in keys:
                training = key == "train_error"
                ax.plot(x, [r[key] for r in rows], color=color,
                        linestyle="--" if training else "-", linewidth=1.3 if training else 2,
                        marker="o" if training else "s", markersize=5,
                        markerfacecolor="none" if training else color,
                        label=f"{LABELS[name]} {'train' if training else 'test'}")
        ax.set(title=panel_title, xlabel="Parameter updates (x 10,000)", ylabel="Error (%)")
        ax.set_xlim(0, max(maximum_step / 10000 * 1.03, .05))
        ax.set_ylim(0, 20 if zoom else 100)
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=9, loc="best")
            if zoom and not any(r[k] <= 20 for n in names for r in histories[n] for k in keys):
                ax.text(.5, .45, "All measured errors exceed 20%.\nSee the full-range figure.",
                        transform=ax.transAxes, ha="center", fontsize=10)
        else:
            ax.text(.5, .5, "No measured points yet", transform=ax.transAxes, ha="center")
    fig.suptitle(title, fontsize=12)
    fig.text(.5, .015, "Clean full-split evaluation; no smoothing. Markers are measured points."
             + (" Zoom: errors above 20% are outside this view." if zoom else ""),
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94))
    temporary = destination.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(destination)


def report(source, output, title="Zhou local run"):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    histories = read_histories(source)
    complete = all((source / n / "complete.json").exists() for n in MODELS)
    phase = "completed budget" if complete else "PARTIAL - not a final conclusion"
    figure6 = [("Plain networks", MODELS[:2], ("train_error", "test_error")),
               ("Residual networks", MODELS[2:], ("train_error", "test_error"))]
    draw_panels(histories, figure6, output / "figure6_comparison.png", f"{title}: {phase}")
    draw_panels(histories, figure6, output / "figure6_zoom_0_20.png", f"{title}: {phase}", zoom=True)
    figure1 = [("Plain: training error", MODELS[:2], ("train_error",)),
               ("Plain: test error", MODELS[:2], ("test_error",))]
    draw_panels(histories, figure1, output / "figure1_plain_degradation.png", f"{title}: {phase}")
    final_rows = [{"model": name, **rows[-1]} for name, rows in histories.items() if rows]
    if final_rows:
        fields = list(dict.fromkeys(k for row in final_rows for k in row))
        temporary = output / "summary.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(final_rows)
        temporary.replace(output / "summary.csv")
    common_steps = set.intersection(*[{row["step"] for row in histories[name]} for name in MODELS])
    common = max(common_steps) if common_steps else None
    lines = [f"# {title}", "", f"Status: {phase}.", "",
             "Source: " + str(source.resolve()), "",
             "Error (%) = wrong predictions / split size * 100. Training error uses all 50,000",
             "original training images in eval mode; test error uses all 10,000 test images.",
             "No random crops/flips during evaluation. This is not online augmented training error.", "",
             "Figure 1 compares Plain depths; Figure 6 separates Plain and ResNet, using dashed",
             "training lines and solid test lines. Only selected depths 20 and 56 are included.",
             "The full-range figure retains high initial errors; the zoom file intentionally shows 0-20%.", ""]
    if common is not None:
        measured = {name: next(row for row in histories[name] if row["step"] == common) for name in MODELS}
        lines += [f"## Latest equal-budget comparison: {common:,} updates", "",
                  "| Model | Train error (%) | Test error (%) |", "|---|---:|---:|"]
        lines += [f"| {LABELS[n]} | {r['train_error']:.3f} | {r['test_error']:.3f} |" for n, r in measured.items()]
        lines += ["", "Differences are percentage points; positive means the first model is worse."]
        for a, b in (("plain56", "plain20"), ("resnet56", "resnet20"),
                     ("resnet20", "plain20"), ("resnet56", "plain56")):
            lines += [f"- {LABELS[a]} minus {LABELS[b]}: train "
                      f"{measured[a]['train_error']-measured[b]['train_error']:+.3f}; test "
                      f"{measured[a]['test_error']-measured[b]['test_error']:+.3f}."]
        if complete:
            plain_degrades = measured["plain56"]["train_error"] > measured["plain20"]["train_error"]
            residual_depth_improves = all(measured["resnet56"][k] < measured["resnet20"][k]
                                          for k in ("train_error", "test_error"))
            lines += ["", f"Plain training degradation observed in this seed: {plain_degrades}.",
                      f"ResNet-56 improves both errors over ResNet-20 in this seed: {residual_depth_improves}.",
                      "These are observations from one seed, not guarantees or a statistical proof."]
        else:
            lines += ["", "Training is incomplete. These measurements do not establish the final degradation result."]
    else:
        lines += ["No common measured update across all four models yet; do not compare unequal budgets."]
    (output / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "plot_status.json", {"source": str(source.resolve()), "complete": complete,
               "common_step": common, "points": {n: len(r) for n, r in histories.items()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Replot of measured history")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Use a new output folder; the source histories will remain untouched.")
    report(args.source, args.output, args.title)
