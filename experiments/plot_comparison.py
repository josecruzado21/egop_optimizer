"""
Compare FashionMNIST OG results between old repo (CSV) and new repo (logs).
Plots a 2x2 grid with median + IQR shading for both repos.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LOG_DIR = Path(__file__).resolve().parent / "logs" / "fashionMNIST_OG"
PLOT_DIR = Path(__file__).resolve().parent / "plots"

METRIC_RE = re.compile(
    r"Epoch \d+: Training Loss = ([\d.]+), Validation Loss = ([\d.]+), "
    r"Training Acc\. = ([\d.]+), Validation Acc\. = ([\d.]+)"
)


def parse_training_log(log_path):
    """Parse a single training.log into a history dict."""
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
    }
    with open(log_path, 'r') as f:
        for line in f:
            m = METRIC_RE.search(line)
            if m:
                history['train_loss'].append(float(m.group(1)))
                history['val_loss'].append(float(m.group(2)))
                history['train_acc'].append(float(m.group(3)))
                history['val_acc'].append(float(m.group(4)))
    return history


def collect_new_repo_histories(solver_name="AdamW"):
    """Collect all trial histories for a solver from new repo logs.
    Expects structure: logs/fashionMNIST_OG/{solver}/LR_{lr}/trial_{i}/training.log
    """
    solver_dir = LOG_DIR / solver_name
    if not solver_dir.exists():
        return []
    histories = []
    for lr_dir in sorted(solver_dir.iterdir()):
        if not lr_dir.is_dir():
            continue
        for trial_dir in sorted(lr_dir.iterdir()):
            log_path = trial_dir / "training.log"
            if log_path.exists():
                h = parse_training_log(str(log_path))
                if h['train_loss']:
                    histories.append(h)
    print(f"Loaded {len(histories)} runs from new repo logs ({solver_name})")
    return histories


def load_old_repo_histories(csv_path):
    """Load old repo CSV into list of history dicts (one per seed)."""
    df = pd.read_csv(csv_path)
    histories = []
    for seed, group in df.groupby('seed'):
        group = group.sort_values('epoch')
        histories.append({
            'train_loss': group['train_loss'].tolist(),
            'train_acc': group['train_acc'].tolist(),
            'val_loss': group['val_loss'].tolist(),
            'val_acc': group['val_acc'].tolist(),
        })
    print(f"Loaded {len(histories)} runs from old repo CSV")
    return histories


def plot_comparison(old_histories, new_histories, save_path):
    """Plot 2x2 comparison grid: old repo vs new repo with median + IQR."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    num_epochs_old = len(old_histories[0]['train_loss'])
    num_epochs_new = len(new_histories[0]['train_loss'])
    epochs_old = np.arange(1, num_epochs_old + 1)
    epochs_new = np.arange(1, num_epochs_new + 1)

    metrics = [
        ('train_loss', 'Training Loss', axes[0, 0], True),
        ('val_loss', 'Validation Loss', axes[0, 1], True),
        ('train_acc', 'Training Accuracy', axes[1, 0], False),
        ('val_acc', 'Validation Accuracy', axes[1, 1], False),
    ]

    for key, title, ax, log_scale in metrics:
        # Old repo
        old_curves = np.array([h[key] for h in old_histories])
        old_median = np.median(old_curves, axis=0)
        old_q25 = np.percentile(old_curves, 25, axis=0)
        old_q75 = np.percentile(old_curves, 75, axis=0)
        ax.plot(epochs_old, old_median, color='blue', linewidth=2, label='Old repo (median)')
        ax.fill_between(epochs_old, old_q25, old_q75, color='blue', alpha=0.15, label='Old repo (IQR)')

        # New repo
        new_curves = np.array([h[key] for h in new_histories])
        new_median = np.median(new_curves, axis=0)
        new_q25 = np.percentile(new_curves, 25, axis=0)
        new_q75 = np.percentile(new_curves, 75, axis=0)
        ax.plot(epochs_new, new_median, color='red', linewidth=2, label='New repo (median)')
        ax.fill_between(epochs_new, new_q25, new_q75, color='red', alpha=0.15, label='New repo (IQR)')

        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.spines[['right', 'top']].set_visible(False)
        ax.legend(fontsize=9)
        if log_scale:
            ax.set_yscale('log')

    fig.suptitle(
        f"FashionMNIST OG: Old Repo vs New Repo\n"
        f"({len(old_histories)} old runs, {len(new_histories)} new runs)",
        fontsize=16, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    old_csv_path = Path(r"E:\Explore\Project\Fast Adaptive\egop_optimization\experiments\results\fashionMNIST_og_only\fashionMNIST_og_only_20260315_180204.csv")

    old_histories = load_old_repo_histories(str(old_csv_path))
    new_histories = collect_new_repo_histories("AdamW")

    PLOT_DIR.mkdir(exist_ok=True)
    plot_comparison(old_histories, new_histories, str(PLOT_DIR / "fashionMNIST_OG_comparison.png"))
