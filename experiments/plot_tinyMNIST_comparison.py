import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pdb

BASE_DIR = Path(__file__).resolve().parent

OG_LOG_DIR = Path(__file__).resolve().parent / "logs" / "tinyMNIST_OG"
REPARAM_LOG_DIR = Path(__file__).resolve().parent / "logs" / "tinyMNIST_reparam"

SOLVER_STYLE = {
    "Adam": {"color": "blue", "label": "Adam"},
    "Adagrad": {"color": "red", "label": "Adagrad"},
    "GD": {"color": "green", "label": "SGD"},
    "GD_w_momentum": {"color": "cyan", "label": "SGD with Momentum"},
}

# Regex to extract epoch metrics from a training.log line
METRIC_RE = re.compile(
    r"Epoch (\d+): Training Loss = ([\d.]+), Validation Loss = ([\d.]+), "
    r"Training Acc\. = ([\d.]+), Validation Acc\. = ([\d.]+)"
)


def parse_training_log(log_path):
    """Parse a single training.log and return arrays of metrics per epoch."""
    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []
    with open(log_path, "r") as f:
        for line in f:
            m = METRIC_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
                train_acc.append(float(m.group(4)))
                val_acc.append(float(m.group(5)))
    return {
        "epoch": np.array(epochs),
        "train_loss": np.array(train_loss),
        "val_loss": np.array(val_loss),
        "train_acc": np.array(train_acc),
        "val_acc": np.array(val_acc),
    }


def collect_solver_data(solver_name, log_dir):
    """Collect metrics across all LR/trial combos for a solver in one log folder."""
    solver_dir = Path(log_dir) / solver_name
    if not solver_dir.exists():
        return []
    trials = []
    for lr_dir in sorted(solver_dir.iterdir()):
        if not lr_dir.is_dir():
            continue
        for trial_dir in sorted(lr_dir.iterdir()):
            log_path = trial_dir / "training.log"
            if log_path.exists():
                data = parse_training_log(log_path)
                if len(data["epoch"]) > 0:
                    trials.append(data)
    return trials


def compute_median_and_iqr(trials, key):
    """Given a list of trial dicts, compute median and IQR for a metric key."""
    min_len = min(len(t[key]) for t in trials)
    matrix = np.stack([t[key][:min_len] for t in trials], axis=0)
    median = np.median(matrix, axis=0)
    q25 = np.percentile(matrix, 25, axis=0)
    q75 = np.percentile(matrix, 75, axis=0)
    epochs = trials[0]["epoch"][:min_len]
    return epochs, median, q25, q75


def plot_solver_traces(ax, key, og_log_dir, reparam_log_dir):
    """Plot solver medians and IQR bands for both coordinate systems."""
    coordinate_systems = [
        ("OG", og_log_dir, "-"),
        ("reparameterized", reparam_log_dir, "--"),
    ]
    for solver_name, style in SOLVER_STYLE.items():
        for coordinate_label, log_dir, linestyle in coordinate_systems:
            trials = collect_solver_data(solver_name, log_dir)
            if not trials:
                continue
            epochs, median, q25, q75 = compute_median_and_iqr(trials, key)
            ax.plot(
                epochs,
                median,
                color=style["color"],
                linestyle=linestyle,
                label=f"{style['label']} ({coordinate_label})",
                linewidth=1.5,
            )
            ax.fill_between(epochs, q25, q75, color=style["color"], alpha=0.15)


def plot_training_loss(log_dir=None, save_path=None, reparam_log_dir=None):
    """Plot training-loss medians and IQR bands for OG and reparameterized runs.

    Args:
        log_dir (str or Path, optional): Original-coordinate logs; defaults to OG_LOG_DIR.
        save_path (str or Path, optional): Where to save the figure; otherwise display it.
        reparam_log_dir (str or Path, optional): Reparameterized-coordinate logs;
            defaults to REPARAM_LOG_DIR.
    """
    og_log_dir = OG_LOG_DIR if log_dir is None else Path(log_dir)
    reparam_log_dir = (
        REPARAM_LOG_DIR if reparam_log_dir is None else Path(reparam_log_dir)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_solver_traces(ax, "train_loss", og_log_dir, reparam_log_dir)

    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-5)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Training Loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    return fig, ax


def plot_validation_accuracy(log_dir=None, save_path=None, reparam_log_dir=None):
    """Plot validation-accuracy medians and IQR bands for both coordinate systems.

    Args:
        log_dir (str or Path, optional): Original-coordinate logs; defaults to OG_LOG_DIR.
        save_path (str or Path, optional): Where to save the figure; otherwise display it.
        reparam_log_dir (str or Path, optional): Reparameterized-coordinate logs;
            defaults to REPARAM_LOG_DIR.
    """
    og_log_dir = OG_LOG_DIR if log_dir is None else Path(log_dir)
    reparam_log_dir = (
        REPARAM_LOG_DIR if reparam_log_dir is None else Path(reparam_log_dir)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_solver_traces(ax, "val_acc", og_log_dir, reparam_log_dir)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Validation Accuracy")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    return fig, ax


if __name__ == "__main__":

    print(f"Reparam log directory: {REPARAM_LOG_DIR}")
    print(f"Directory exists: {REPARAM_LOG_DIR.exists()}")

    for solver_name in SOLVER_STYLE:
        trials = collect_solver_data(solver_name, REPARAM_LOG_DIR)
        print(
            f"{solver_name}: {len(trials)} trials, "
            f"epoch counts: {[len(t['epoch']) for t in trials]}"
        )

    date_str = datetime.now().strftime("%b-%d-%Y")
    save_dir = Path(__file__).resolve().parent / "figures" / date_str
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_training_loss(save_path=save_dir / "tinyMNIST_comparison_training_loss.pdf")
    plot_validation_accuracy(
        save_path=save_dir / "tinyMNIST_comparison_validation_accuracy.pdf"
    )
