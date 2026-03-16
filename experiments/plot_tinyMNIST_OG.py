import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent / "logs" / "tinyMNIST_OG_new_train_method"

SOLVER_STYLE = {
    "Adam":           {"color": "blue",  "label": "Adam"},
    "Adagrad":        {"color": "red",   "label": "Adagrad"},
    "GD":             {"color": "green", "label": "SGD"},
    "GD_w_momentum":  {"color": "cyan",  "label": "SGD with Momentum"},
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


def collect_solver_data(solver_name):
    """Collect metrics across all LR/trial combos for a solver. Returns list of dicts."""
    solver_dir = LOG_DIR / solver_name
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


def plot_training_loss(log_dir=None, save_path=None):
    """Plot training loss curves for all solvers with mean +/- 1 std band."""
    if log_dir is not None:
        global LOG_DIR
        LOG_DIR = Path(log_dir)

    fig, ax = plt.subplots(figsize=(7, 5))

    for solver_name, style in SOLVER_STYLE.items():
        trials = collect_solver_data(solver_name)
        if not trials:
            continue
        epochs, median, q25, q75 = compute_median_and_iqr(trials, "train_loss")
        ax.plot(epochs, median, color=style["color"], label=style["label"], linewidth=1.5)
        ax.fill_between(epochs, q25, q75, color=style["color"], alpha=0.15)

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


def plot_validation_accuracy(log_dir=None, save_path=None):
    """Plot validation accuracy curves for all solvers."""
    if log_dir is not None:
        global LOG_DIR
        LOG_DIR = Path(log_dir)

    fig, ax = plt.subplots(figsize=(7, 5))

    for solver_name, style in SOLVER_STYLE.items():
        trials = collect_solver_data(solver_name)
        if not trials:
            continue
        epochs, median, q25, q75 = compute_median_and_iqr(trials, "val_acc")
        ax.plot(epochs, median, color=style["color"], label=style["label"], linewidth=1.5)
        ax.fill_between(epochs, q25, q75, color=style["color"], alpha=0.15)

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
    plot_training_loss()
