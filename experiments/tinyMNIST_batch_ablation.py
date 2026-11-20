from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures on local machines and headless clusters.
import matplotlib.pyplot as plt
import numpy as np

# This script is intended to live alongside tinyMNIST_OG.py in experiments/.
EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_ROOT = EXPERIMENTS_DIR / "logs" / "tinyMNIST_batch_ablation"
DATE_STR = datetime.now().strftime("%b-%d-%Y")
FIGURE_DIR = EXPERIMENTS_DIR / "figures" / DATE_STR

# ── Experiment configuration ───────────────────────────────────────────
SOLVERS = ("Adam", "Adagrad")
BATCH_SIZES = (30, 100, 300, 1000)

# Selected learning rates for each training batch size and optimizer.
# TODO: Replace the current defaults with the winning values from the
# original-coordinate LR sweeps. Both OG and reparam use the same selected LR.
LR_BY_BATCH_SIZE = {
    30: {"Adam": 0.02, "Adagrad": 0.04},
    100: {"Adam": 0.02, "Adagrad": 0.04},
    300: {"Adam": 0.02, "Adagrad": 0.04},
    1000: {"Adam": 0.02, "Adagrad": 0.04},
}
SEEDS = tuple(range(1000, 1005))  # Five trials: 1000, ..., 1004.
EPOCHS = 200

TRAIN_VAL_TEST_SPLIT = [None, 0.33, 0.66]
DATA_SPLIT_SEED = 0  # Same randomly constructed split across batch sizes.

INPUT_SIZE = 64
HIDDEN_SIZES = [32]
NUM_CLASSES = 10

EGOP_OVERSAMPLING_FACTOR = 1.0
USE_RANDOMIZED_EVD = False
SOLVER_STYLE = {"Adam": "blue", "Adagrad": "red"}
COORDINATE_STYLE = {"OG": ("OG", "-"), "reparam": ("reparameterized", "--")}

# Allow scientific notation in logs (the original regex handled decimals only).
NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
METRIC_RE = re.compile(
    rf"Epoch (\d+): Training Loss = {NUMBER}, Validation Loss = {NUMBER}, "
    rf"Training Acc\. = {NUMBER}, Validation Acc\. = {NUMBER}"
)


def seed_everything(seed, torch):
    """Seed Python, NumPy and PyTorch before a run or data split."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(name, parameters, lr, torch):
    """Build an optimizer with the original scripts' settings."""
    if name == "Adam":
        return torch.optim.Adam(parameters, lr=lr)
    if name == "Adagrad":
        return torch.optim.Adagrad(parameters, lr=lr)
    raise ValueError(f"Unexpected optimizer: {name}")


def trial_name(batch_size, coordinate, solver, seed):
    """Relative experiment name understood by basic_train_loop."""
    lr = LR_BY_BATCH_SIZE[batch_size][solver]
    return (
        f"tinyMNIST_batch_ablation/batch_{batch_size}/{coordinate}/"
        f"{solver}/LR_{lr}/trial_{seed}"
    )


def log_path(batch_size, coordinate, solver, seed):
    """Path where basic_train_loop writes training.log."""
    return (
        EXPERIMENTS_DIR
        / "logs"
        / trial_name(batch_size, coordinate, solver, seed)
        / "training.log"
    )


def parse_training_log(path):
    """Read loss and accuracy records from a training.log file."""
    data = {k: [] for k in ("epoch", "train_loss", "val_loss", "train_acc", "val_acc")}
    if not path.exists():
        return {k: np.asarray(v) for k, v in data.items()}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = METRIC_RE.search(line)
            if match is None:
                continue
            epoch, train_loss, val_loss, train_acc, val_acc = match.groups()
            data["epoch"].append(int(epoch))
            for key, value in zip(
                ("train_loss", "val_loss", "train_acc", "val_acc"),
                (train_loss, val_loss, train_acc, val_acc),
            ):
                data[key].append(float(value))
    return {k: np.asarray(v) for k, v in data.items()}


def is_complete(path):
    """Determine whether an existing log has all requested epochs."""
    return len(parse_training_log(path)["epoch"]) >= EPOCHS


def validate_learning_rates():
    """Require one positive learning rate per batch size and optimizer."""
    for batch_size in BATCH_SIZES:
        if batch_size not in LR_BY_BATCH_SIZE:
            raise ValueError(f"Missing LR settings for batch size {batch_size}")
        for solver in SOLVERS:
            lr = LR_BY_BATCH_SIZE[batch_size].get(solver)
            if lr is None or not np.isfinite(lr) or lr <= 0:
                raise ValueError(
                    f"Batch {batch_size}, {solver}: expected a positive finite LR; got {lr}"
                )


def save_configuration():
    """Record the settings used by the experiment in its log directory."""
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    configuration = {
        "batch_sizes": BATCH_SIZES,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "learning_rates_by_batch_size": LR_BY_BATCH_SIZE,
        "train_val_test_split": TRAIN_VAL_TEST_SPLIT,
        "data_split_seed": DATA_SPLIT_SEED,
        "input_size": INPUT_SIZE,
        "hidden_sizes": HIDDEN_SIZES,
        "num_classes": NUM_CLASSES,
        "egop_oversampling_factor": EGOP_OVERSAMPLING_FACTOR,
        "egop_batch_size": "same as training batch size",
        "use_randomized_EVD": USE_RANDOMIZED_EVD,
    }
    (LOG_ROOT / "invocation.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )


def run_experiments(overwrite=False):
    """Run all 4 x 5 x 2 x 2 = 80 training trials."""
    import torch

    from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader
    from egop_optimizer.engine.train import basic_train_loop
    from egop_optimizer.models.TinyMNISTClassifier import (
        ReparamTinyMNISTClassifier,
        TinyMNISTClassifier,
    )
    from egop_optimizer.utils.EGOP_utils import (
        compute_V_by_layer,
        layerwise_reparam_init_equiv,
    )

    # The existing basic_train_loop uses a relative "logs/" destination.
    # This guarantees experiments/logs/ even when invoked from project root.
    os.chdir(EXPERIMENTS_DIR)
    save_configuration()

    loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    for batch_size in BATCH_SIZES:
        # Recreate the same train/validation split for every training batch size.
        seed_everything(DATA_SPLIT_SEED, torch)
        trainloader, valloader, _ = tinyMNIST_dataloader(
            batch_size=batch_size,
            train_val_test_split=TRAIN_VAL_TEST_SPLIT,
        )

        # The same loader (and therefore the same batch size) is used for
        # gradient sampling during EGOP estimation and model optimization.

        for seed in SEEDS:
            for solver in SOLVERS:
                lr = LR_BY_BATCH_SIZE[batch_size][solver]
                for coordinate in COORDINATE_STYLE:
                    experiment_name = trial_name(batch_size, coordinate, solver, seed)
                    path = log_path(batch_size, coordinate, solver, seed)
                    if not overwrite and is_complete(path):
                        print(f"Skipping completed: {experiment_name}", flush=True)
                        continue

                    print(f"Running: {experiment_name}", flush=True)
                    seed_everything(seed, torch)
                    og_model = TinyMNISTClassifier(
                        input_size=INPUT_SIZE,
                        hidden_sizes=HIDDEN_SIZES,
                        num_classes=NUM_CLASSES,
                        seed=seed,
                    )

                    if coordinate == "OG":
                        og_model.reinitialize_seeded(seed)
                        model = og_model
                    else:
                        # Follow tinyMNIST_reparam.py's initialization sequence.
                        V_dict = compute_V_by_layer(
                            model_OG=og_model,
                            k=int(
                                EGOP_OVERSAMPLING_FACTOR * og_model.count_parameters()
                            ),
                            data_loader=trainloader,
                            criterion=criterion,
                            use_randomized_svd=USE_RANDOMIZED_EVD,
                        )
                        reparam_model = ReparamTinyMNISTClassifier(
                            input_size=INPUT_SIZE,
                            hidden_sizes=HIDDEN_SIZES,
                            num_classes=NUM_CLASSES,
                            seed=seed,
                            V_by_layer_dict=V_dict,
                        )
                        og_model.reinitialize_seeded(seed)
                        _, model = layerwise_reparam_init_equiv(
                            EGOP_model=reparam_model, OG_model=og_model, seed=seed
                        )

                    optimizer = make_optimizer(solver, model.parameters(), lr, torch)

                    # Clear an incomplete log before rerunning rather than appending
                    # another set of epoch records to the same trial.
                    if path.exists():
                        path.unlink()
                    # Reset RNG after EGOP estimation so OG/reparam runs with the
                    # same seed have comparable shuffled mini-batch sequences.
                    seed_everything(seed, torch)
                    basic_train_loop(
                        model=model,
                        trainloader=trainloader,
                        optimizer=optimizer,
                        epochs=EPOCHS,
                        loss_method=loss_method,
                        LR_scheduler=None,
                        valloader=valloader,
                        experiment_name=experiment_name,
                        ten_crop=False,
                        report_validation_metrics=True,
                    )


def collect_trials(batch_size, coordinate, solver):
    """Load the available repetitions for one (batch, coordinate, solver)."""
    trials = []
    for seed in SEEDS:
        data = parse_training_log(log_path(batch_size, coordinate, solver, seed))
        if len(data["epoch"]) > 0:
            trials.append(data)
    return trials


def plot_metric(metric, ylabel, filename, *, log_scale=False):
    """Plot four batch-size panels with a single shared y-axis extent."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    found_data = False

    for ax, batch_size in zip(axes, BATCH_SIZES):
        ax.set_title(f"Batch size = {batch_size}")
        for solver in SOLVERS:
            for coordinate, (coord_label, linestyle) in COORDINATE_STYLE.items():
                trials = collect_trials(batch_size, coordinate, solver)
                if not trials:
                    continue
                found_data = True
                min_len = min(len(t[metric]) for t in trials)
                epochs = trials[0]["epoch"][:min_len]
                values = np.stack([t[metric][:min_len] for t in trials])
                median = np.median(values, axis=0)
                q25, q75 = np.percentile(values, [25, 75], axis=0)
                color = SOLVER_STYLE[solver]
                ax.plot(
                    epochs,
                    median,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    label=f"{solver} ({coord_label})",
                )
                ax.fill_between(epochs, q25, q75, color=color, alpha=0.15)
        ax.grid(True, which="both", alpha=0.3)

    if not found_data:
        plt.close(fig)
        print(f"No logs found for {metric}; skipping {filename}.")
        return

    if log_scale:
        axes[0].set_yscale("log")
        axes[0].set_ylim(bottom=1e-5)

    fig.supxlabel("Epochs")
    fig.supylabel(ylabel)
    fig.suptitle(f"TinyMNIST batch-size ablation: {ylabel}")

    # Build a single, common legend rather than repeating it in four panels.
    legend_entries = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_entries.setdefault(label, handle)
    fig.legend(
        legend_entries.values(),
        legend_entries.keys(),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.065, 1.0, 0.95))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved figure: {path}")
    plt.close(fig)


def make_figures():
    """Save the same two metrics plotted by plot_tinyMNIST_comparison.py."""
    plot_metric(
        "train_loss",
        "Training Loss",
        "tinyMNIST_batch_ablation_training_loss.pdf",
        log_scale=True,
    )
    plot_metric(
        "val_acc",
        "Validation Accuracy",
        "tinyMNIST_batch_ablation_validation_accuracy.pdf",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate plots from existing logs.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rerun even completed trials."
    )
    args = parser.parse_args()
    validate_learning_rates()

    try:
        if not args.plot_only:
            run_experiments(overwrite=args.overwrite)
    finally:
        # On KeyboardInterrupt, also plot whatever logs exist so far.
        make_figures()


if __name__ == "__main__":
    main()
