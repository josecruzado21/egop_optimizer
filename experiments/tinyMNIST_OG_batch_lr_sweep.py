from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures on headless servers and local machines.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_ROOT = EXPERIMENTS_DIR / "logs" / "tinyMNIST_OG_batch_lr_sweep"
DATE_STR = datetime.now().strftime("%b-%d-%Y")
FIGURE_DIR = EXPERIMENTS_DIR / "figures" / DATE_STR

# ── Experiment configuration ──────────────────────────────────────────────
BATCH_SIZES = (30, 100, 300, 1000)
SOLVERS = ("Adam", "Adagrad")
LEARNING_RATES = (0.4, 0.2, 0.1, 0.08, 0.04, 0.02, 0.01, 0.008, 0.004, 0.002, 0.001)
SEEDS = tuple(range(100, 110))  # Ten trials: 100, ..., 110.
EPOCHS = 200

TRAIN_VAL_TEST_SPLIT = [None, 0.33, 0.66]
DATA_SPLIT_SEED = 0

INPUT_SIZE = 64
HIDDEN_SIZES = [32]
NUM_CLASSES = 10

SOLVER_COLORS = {"Adam": "tab:blue", "Adagrad": "tab:red"}

# Match the logging format consumed by plot_tinyMNIST_comparison.py.
# Accept 'nan'/'inf', which may occur at the largest learning rates; such
# values are not used for the maximum, but do count toward log completeness.
NUMBER = r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:nan|inf(?:inity)?))"
VAL_ACCURACY_RE = re.compile(
    rf"Epoch\s+(\d+):\s*Training Loss\s*=\s*{NUMBER},\s*"
    rf"Validation Loss\s*=\s*{NUMBER},\s*"
    rf"Training Acc\.\s*=\s*{NUMBER},\s*"
    rf"Validation Acc\.\s*=\s*({NUMBER})",
    flags=re.IGNORECASE,
)


def seed_everything(seed: int, torch) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible trials."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(solver: str, parameters, lr: float, torch):
    """Construct an optimizer with the same settings as tinyMNIST_OG.py."""
    if solver == "Adam":
        return torch.optim.Adam(parameters, lr=lr)
    if solver == "Adagrad":
        return torch.optim.Adagrad(parameters, lr=lr)
    raise ValueError(f"Unknown solver: {solver}")


def experiment_name(batch_size: int, solver: str, lr: float, seed: int) -> str:
    """Relative name passed to basic_train_loop (which writes to logs/)."""
    return (
        f"tinyMNIST_OG_batch_lr_sweep/batch_{batch_size}/"
        f"{solver}/LR_{lr:g}/trial_{seed}"
    )


def log_path(batch_size: int, solver: str, lr: float, seed: int) -> Path:
    """Absolute path to one training.log, independent of launch directory."""
    return (
        EXPERIMENTS_DIR
        / "logs"
        / experiment_name(batch_size, solver, lr, seed)
        / "training.log"
    )


def read_validation_log(path: Path) -> dict[int, float]:
    """Return {epoch: validation_accuracy} from an existing training.log."""
    results: dict[int, float] = {}
    if not path.exists():
        return results
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = VAL_ACCURACY_RE.search(line)
            if match:
                results[int(match.group(1))] = float(match.group(2))
    return results


def log_is_complete(path: Path, epochs: int) -> bool:
    """A complete log has at least one metric record for each epoch."""
    return len(read_validation_log(path)) >= epochs


def save_configuration(*, seeds: tuple[int, ...], epochs: int) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    config = {
        "batch_sizes": BATCH_SIZES,
        "optimizers": SOLVERS,
        "learning_rates": LEARNING_RATES,
        "seeds": seeds,
        "epochs": epochs,
        "train_val_test_split": TRAIN_VAL_TEST_SPLIT,
        "data_split_seed": DATA_SPLIT_SEED,
        "input_size": INPUT_SIZE,
        "hidden_sizes": HIDDEN_SIZES,
        "num_classes": NUM_CLASSES,
        "per_trial_score": "maximum finite validation accuracy across all epochs",
        "aggregate_score": "median of trial-wise maxima; 25th-75th percentile band",
        "best_lr_rule": "largest median trial-wise maximum; ties favor smaller LR",
    }
    (LOG_ROOT / "invocation.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


def run_experiments(*, seeds: tuple[int, ...], epochs: int, overwrite: bool) -> None:
    """Run OG models only; skip existing complete trial logs by default."""
    import torch

    from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader
    from egop_optimizer.engine.train import basic_train_loop
    from egop_optimizer.models.TinyMNISTClassifier import TinyMNISTClassifier

    # basic_train_loop writes under a relative 'logs/' folder.
    os.chdir(EXPERIMENTS_DIR)
    save_configuration(seeds=seeds, epochs=epochs)

    loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)
    total = len(BATCH_SIZES) * len(SOLVERS) * len(LEARNING_RATES) * len(seeds)
    current = 0

    for batch_size in BATCH_SIZES:
        # Reset the random state before making each batch-size-specific loader.
        # This preserves the data-split convention used by the batch ablation.
        seed_everything(DATA_SPLIT_SEED, torch)
        trainloader, valloader, _ = tinyMNIST_dataloader(
            batch_size=batch_size,
            train_val_test_split=TRAIN_VAL_TEST_SPLIT,
        )

        for solver in SOLVERS:
            for lr in LEARNING_RATES:
                for seed in seeds:
                    current += 1
                    name = experiment_name(batch_size, solver, lr, seed)
                    path = log_path(batch_size, solver, lr, seed)
                    if not overwrite and log_is_complete(path, epochs):
                        print(f"[{current}/{total}] Skip complete: {name}", flush=True)
                        continue

                    print(f"[{current}/{total}] Run: {name}", flush=True)
                    seed_everything(seed, torch)
                    model = TinyMNISTClassifier(
                        input_size=INPUT_SIZE,
                        hidden_sizes=HIDDEN_SIZES,
                        num_classes=NUM_CLASSES,
                        seed=seed,
                    )
                    model.reinitialize_seeded(seed)
                    optimizer = make_optimizer(solver, model.parameters(), lr, torch)

                    # Incomplete training is restarted, not resumed from weights;
                    # remove old metrics to avoid mixing two separate runs.
                    if path.exists():
                        path.unlink()

                    # Reset seed again to control minibatch shuffle randomness.
                    seed_everything(seed, torch)
                    basic_train_loop(
                        model=model,
                        trainloader=trainloader,
                        optimizer=optimizer,
                        epochs=epochs,
                        loss_method=loss_method,
                        LR_scheduler=None,
                        valloader=valloader,
                        experiment_name=name,
                        ten_crop=False,
                        report_validation_metrics=True,
                    )


def summarize(*, seeds: tuple[int, ...], epochs: int) -> list[dict]:
    """Collect trial-wise best accuracies and their median/IQR per LR."""
    rows: list[dict] = []
    for batch_size in BATCH_SIZES:
        for solver in SOLVERS:
            for lr in LEARNING_RATES:
                complete_trials = 0
                peaks = []
                for seed in seeds:
                    history = read_validation_log(
                        log_path(batch_size, solver, lr, seed)
                    )
                    if len(history) < epochs:
                        continue  # Do not give incomplete runs a biased peak score.
                    complete_trials += 1
                    finite = [
                        value for value in history.values() if math.isfinite(value)
                    ]
                    if finite:
                        peaks.append(max(finite))

                row = {
                    "batch_size": batch_size,
                    "optimizer": solver,
                    "lr": lr,
                    "n_complete": complete_trials,
                    "n_valid": len(peaks),
                    "median_peak_val_acc": (
                        float(np.median(peaks)) if peaks else math.nan
                    ),
                    "q25_peak_val_acc": (
                        float(np.percentile(peaks, 25)) if peaks else math.nan
                    ),
                    "q75_peak_val_acc": (
                        float(np.percentile(peaks, 75)) if peaks else math.nan
                    ),
                    "is_best_lr": False,
                }
                rows.append(row)

    return rows


def select_best_lrs(rows: list[dict], *, n_trials: int) -> dict[int, dict[str, dict]]:
    """Select LRs only after all candidates have all requested trials.

    A candidate needs finite peak accuracies in all trials to be eligible.
    Diverged runs with no finite validation score cannot win the sweep.
    """
    selection: dict[int, dict[str, dict]] = {}
    for batch_size in BATCH_SIZES:
        for solver in SOLVERS:
            group = [
                r
                for r in rows
                if r["batch_size"] == batch_size and r["optimizer"] == solver
            ]
            if len(group) != len(LEARNING_RATES) or any(
                r["n_complete"] < n_trials for r in group
            ):
                continue  # A partial sweep does not determine a final best LR.
            eligible = [r for r in group if r["n_valid"] == n_trials]
            if not eligible:
                continue
            best = max(eligible, key=lambda r: (r["median_peak_val_acc"], -r["lr"]))
            best["is_best_lr"] = True
            selection.setdefault(batch_size, {})[solver] = {
                "lr": best["lr"],
                "median_peak_val_acc": best["median_peak_val_acc"],
            }
    return selection


def save_summaries(rows: list[dict], best_lrs: dict[int, dict[str, dict]]) -> None:
    """Save full LR summary and a small ready-to-copy best-LR JSON."""
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = LOG_ROOT / "lr_sweep_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: ("" if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in row.items()
                }
            )
    print(f"Saved summary: {summary_path}")

    best_path = LOG_ROOT / "best_learning_rates.json"
    # Only finalized batch/solver selections are written to this file.
    best_mapping = {
        f"batch_{batch_size}": {
            solver: result["lr"] for solver, result in selections.items()
        }
        for batch_size, selections in sorted(best_lrs.items())
    }
    best_path.write_text(json.dumps(best_mapping, indent=2) + "\n", encoding="utf-8")
    print(f"Saved finalized LR selections: {best_path}")
    for batch_size, solvers in sorted(best_mapping.items()):
        print(
            f"  {batch_size}: " + ", ".join(f"{s}={lr:g}" for s, lr in solvers.items())
        )


def plot_results(
    rows: list[dict], best_lrs: dict[int, dict[str, dict]], *, n_trials: int
) -> None:
    """Four batch-size panels; one colored curve and winning star per solver."""
    if not any(r["n_valid"] for r in rows):
        print("No completed training logs yet; figure not generated.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    sorted_lrs = sorted(LEARNING_RATES)

    for ax, batch_size in zip(axes.flat, BATCH_SIZES):
        ax.set_title(f"Batch size = {batch_size}")
        for solver in SOLVERS:
            group = sorted(
                (
                    r
                    for r in rows
                    if r["batch_size"] == batch_size
                    and r["optimizer"] == solver
                    and r["n_valid"] > 0
                ),
                key=lambda r: r["lr"],
            )
            if not group:
                continue
            x = [r["lr"] for r in group]
            med = [r["median_peak_val_acc"] for r in group]
            q25 = [r["q25_peak_val_acc"] for r in group]
            q75 = [r["q75_peak_val_acc"] for r in group]
            color = SOLVER_COLORS[solver]
            ax.plot(
                x, med, "-o", color=color, markersize=4, linewidth=1.6, label=solver
            )
            ax.fill_between(x, q25, q75, color=color, alpha=0.17)

            best = best_lrs.get(batch_size, {}).get(solver)
            if best is not None:
                ax.plot(
                    best["lr"],
                    best["median_peak_val_acc"],
                    marker="*",
                    markersize=19,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    linestyle="None",
                    zorder=5,
                )

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(sorted_lrs))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{lr:g}" for lr in sorted_lrs]))
        ax.tick_params(axis="x", which="major", labelrotation=45, labelsize=8)
        ax.minorticks_off()
        ax.grid(True, which="major", alpha=0.27)

    fig.suptitle(
        f"TinyMNIST (original coordinates): learning-rate sweep ({n_trials} seeds)",
        fontsize=15,
    )
    fig.supxlabel("Learning rate", y=0.065)
    fig.supylabel(
        "Maximum validation accuracy over epochs\n(median and IQR across trials)"
    )
    legend_handles = [
        Line2D([], [], color=SOLVER_COLORS[s], marker="o", linewidth=1.6, label=s)
        for s in SOLVERS
    ] + [
        Line2D(
            [],
            [],
            color="black",
            marker="*",
            linestyle="None",
            markersize=12,
            label="Best LR (completed sweep)",
        )
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
    )
    fig.tight_layout(rect=(0.04, 0.08, 1.0, 0.95))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / "tinyMNIST_OG_batch_lr_sweep"
    for extension in ("pdf", "png"):
        filename = stem.with_suffix("." + extension)
        fig.savefig(filename, bbox_inches="tight", dpi=180)
        print(f"Saved figure: {filename}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots and summaries using existing training logs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Repeat all runs even if their training logs are complete.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=len(SEEDS),
        help=f"Number of seeds to use (1 to {len(SEEDS)}; default: {len(SEEDS)}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of epochs (default: {EPOCHS}).",
    )
    args = parser.parse_args()
    if not 1 <= args.trials <= len(SEEDS):
        parser.error(f"--trials must be between 1 and {len(SEEDS)}")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    seeds = SEEDS[: args.trials]

    try:
        if not args.plot_only:
            run_experiments(seeds=seeds, epochs=args.epochs, overwrite=args.overwrite)
    finally:
        # Also save partial figures and trial summaries after Ctrl+C.
        rows = summarize(seeds=seeds, epochs=args.epochs)
        best = select_best_lrs(rows, n_trials=len(seeds))
        save_summaries(rows, best)
        plot_results(rows, best, n_trials=len(seeds))


if __name__ == "__main__":
    main()
