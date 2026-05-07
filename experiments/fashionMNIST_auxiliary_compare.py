"""
Experiment: Original FashionMNIST classifier vs Auxiliary EGOP reparam.

Mirrors the old repo's reduced_and_auxiliary_example.py — runs num_iterations
trials, each with its own random seed. Per trial:
  1. Build OG model, init via reinitialize_seeded(seed)
  2. Compute EGOP basis V from OG using a single forward+backward
  3. Build auxiliary reparam model with V
  4. layerwise_reparam_init_equiv: makes auxiliary functionally equivalent to OG at t=0
  5. Train OG and auxiliary independently with AdamW (selective WD on weight_d/weight_r/bias)
  6. Save per-trial CSV (one row per epoch, includes initial state as epoch=0)

After all trials: aggregate CSVs (median + IQR) and produce 2x2 plot
(train_loss / val_loss / train_acc / val_acc).

This script does NOT modify any existing modules — it imports new repo's APIs
and supplies its own train_epoch / evaluate / aggregation / plotting helpers.

Defaults follow old repo's reduced_and_auxiliary_example.py.

Usage:
    python experiments/fashionMNIST_auxiliary_compare.py \\
        --rsvd_components 50 --epochs 30 --num_iterations 1
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from egop_optimizer.models.FashionMNISTClassfier import (
    AuxiliaryReparamFashionMNISTClassifier,
    FashionMNISTClassifier,
)
from egop_optimizer.dataloaders.fashionMNIST_dataloader import fashionMNIST_dataloader
from egop_optimizer.utils.EGOP_utils import (
    compute_V_by_layer,
    layerwise_reparam_init_equiv,
)
from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()
print(f"Using device: {DEVICE}")

# Windows multiprocessing workaround
NUM_WORKERS = 0 if sys.platform == "win32" else 2


# ============================================================================
# Tuned hyperparameters (from old repo: reduced_and_auxiliary_example.py)
# ============================================================================
# Format: {rsvd_components: {model_type: {'lr': lr, 'wd': wd}}}
TUNED_HYPERPARAMS = {
    50: {
        "og": {"lr": 0.005, "wd": 0.1},
        "aux_egop": {"lr": 0.005, "wd": 0.1},
    },
    100: {
        "og": {"lr": 0.005, "wd": 0.1},
        "aux_egop": {"lr": 0.005, "wd": 0.1},
    },
    200: {
        "og": {"lr": 0.005, "wd": 0.1},
        "aux_egop": {"lr": 0.005, "wd": 0.2},
    },
    400: {
        "og": {"lr": 0.005, "wd": 0.1},
        "aux_egop": {"lr": 0.005, "wd": 0.1},
    },
    800: {
        "og": {"lr": 0.005, "wd": 0.1},
        "aux_egop": {"lr": 0.005, "wd": 0.2},
    },
    1000: {
        "og": {"lr": 0.005, "wd": 0.1},
        "aux_egop": {"lr": 0.005, "wd": 0.1},
    },
    2000: {
        "og": {"lr": 0.005, "wd": 0.1},
        "aux_egop": {"lr": 0.005, "wd": 0.1},
    },
    4000: {
        "og": {"lr": 0.005, "wd": 0.1},
        "aux_egop": {"lr": 0.005, "wd": 0.2},
    },
    7940: {
        "og": {"lr": 0.005, "wd": 0.1},
        "aux_egop": {"lr": 0.005, "wd": 0.2},
    },
}

DEFAULT_CONFIG = {
    "hidden_size": 100,
    "num_classes": 10,
    "pool_factor": 1,
    "batch_size": 300,
    "egop_factor": 0.1,
    "weight_dist": "gaussian",
}


# ============================================================================
# Utility functions
# ============================================================================
def set_seed(seed: int) -> None:
    """Set random seeds across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_optimizer(
    model: nn.Module,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float = 0.01,
) -> optim.Optimizer:
    """
    Create AdamW optimizer with parameter groups.

    For auxiliary EGOP models, applies weight decay to weight_d and weight_r,
    but NOT to bias parameters. Mirrors old repo's create_optimizer.
    """
    if optimizer_type.lower() != "adamw":
        raise ValueError(f"Only AdamW is supported, got {optimizer_type}")

    weight_d_params = [p for n, p in model.named_parameters() if "weight_d" in n]
    weight_r_params = [p for n, p in model.named_parameters() if "weight_r" in n]
    bias_params = [p for n, p in model.named_parameters() if "bias" in n]

    if weight_d_params and weight_r_params:
        # Auxiliary model: selective weight decay
        aux_param_names = set()
        aux_param_names.update(
            [n for n, _ in model.named_parameters() if "weight_d" in n]
        )
        aux_param_names.update(
            [n for n, _ in model.named_parameters() if "weight_r" in n]
        )
        aux_param_names.update([n for n, _ in model.named_parameters() if "bias" in n])

        other_params = [
            p for n, p in model.named_parameters() if n not in aux_param_names
        ]

        return optim.AdamW(
            [
                {"params": weight_d_params, "weight_decay": weight_decay},
                {"params": weight_r_params, "weight_decay": weight_decay},
                {"params": bias_params, "weight_decay": 0.0},
                {"params": other_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
        )

    # Standard model
    return optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


# ============================================================================
# Training / evaluation (history-returning, not using basic_train_loop)
# ============================================================================
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """One training epoch. Returns (mean_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model. Returns (mean_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(dataloader), correct / total


def train_model(
    model: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    model_name: str = "Model",
    optimizer_type: str = "adamw",
    device: torch.device = DEVICE,
    verbose: bool = True,
) -> Dict:
    """
    Train model. Returns history dict containing per-epoch metrics PLUS the
    initial state recorded as epoch 0 (so training curves include the
    initialization point).

    history keys: train_loss, train_acc, val_loss, val_acc, time_per_epoch,
                  learning_rate, weight_decay
    The first entry of each list (index 0) corresponds to the initial state
    (before any training step). Indices 1..epochs are post-training-epoch values.
    """
    model = model.to(device)
    if hasattr(model, "move_bases_to_device"):
        model.move_bases_to_device(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, optimizer_type, learning_rate, weight_decay)

    history: Dict = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "time_per_epoch": [],
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }

    if verbose:
        print(
            f"    Training {model_name} | LR={learning_rate:.1e}, WD={weight_decay:.1e}"
        )

    # Record initial state (epoch 0)
    init_train_loss, init_train_acc = evaluate(model, trainloader, criterion, device)
    init_val_loss, init_val_acc = evaluate(model, valloader, criterion, device)
    history["train_loss"].append(init_train_loss)
    history["train_acc"].append(init_train_acc)
    history["val_loss"].append(init_val_loss)
    history["val_acc"].append(init_val_acc)
    history["time_per_epoch"].append(0.0)
    if verbose:
        print(
            f"    Epoch   0/{epochs} (init) | "
            f"Train: {init_train_loss:.4f}/{init_train_acc:.4f} | "
            f"Val: {init_val_loss:.4f}/{init_val_acc:.4f}"
        )

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(
            model, trainloader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, valloader, criterion, device)
        epoch_time = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["time_per_epoch"].append(epoch_time)

        if verbose and (epoch % 10 == 0 or epoch == epochs):
            print(
                f"    Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                f"Val: {val_loss:.4f}/{val_acc:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )

    if verbose:
        print(f"    {model_name} complete. Final Val Acc: {val_acc:.4f}")
    return history


# ============================================================================
# EGOP basis adapter
# ============================================================================
def compute_V_dict_for_aux(
    og_model: FashionMNISTClassifier,
    trainloader: DataLoader,
    egop_factor: float,
    rsvd_components: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Wrap new repo's compute_V_by_layer to match the old EGOP_basis_by_layer
    semantics: oversampling factor + RSVD with n_components.

    Returns a V_by_layer_dict keyed by submodule name (e.g. "fc1", "fc2"),
    suitable for AuxiliaryReparamFashionMNISTClassifier.
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")
    V_dict = compute_V_by_layer(
        model_OG=og_model,
        data_loader=trainloader,
        criterion=criterion,
        device=device,
        reparam_linear_layers=True,
        use_randomized_svd=True,
        n_components=rsvd_components,
        EGOP_oversampling_factor=egop_factor,
    )
    return V_dict


# ============================================================================
# CSV I/O
# ============================================================================
def save_experiment_results(
    results: List[Dict],
    config: Dict,
    epochs: int,
    save_dir: Path,
) -> Path:
    """Flatten per-trial histories into a long-format CSV and save."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for trial in results:
        seed = trial["seed"]
        rsvd = trial["rsvd_components"]
        for model_type in ("og", "aux_egop"):
            history = trial[model_type]
            for epoch_idx in range(len(history["train_loss"])):
                rows.append(
                    {
                        "seed": seed,
                        "rsvd_components": rsvd,
                        "model_type": model_type,
                        "epoch": epoch_idx,  # 0 = initial state, >=1 = post-training
                        "learning_rate": history["learning_rate"],
                        "weight_decay": history["weight_decay"],
                        "train_loss": history["train_loss"][epoch_idx],
                        "train_acc": history["train_acc"][epoch_idx],
                        "val_loss": history["val_loss"][epoch_idx],
                        "val_acc": history["val_acc"][epoch_idx],
                        "time_per_epoch": history["time_per_epoch"][epoch_idx],
                        **config,
                    }
                )

    df = pd.DataFrame(rows)
    csv_path = save_dir / f"og_vs_aux_rsvd{config['rsvd_components']}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    return csv_path


# ============================================================================
# Plotting (2x2 grid: train/val × loss/acc, median + IQR)
# ============================================================================
PLOT_COLORS = {"og": "blue", "aux_egop": "green"}
PLOT_LABELS = {"og": "Original", "aux_egop": "Auxiliary EGOP"}
PLOT_LINESTYLES = {"og": "-", "aux_egop": "-."}
PLOT_MARKERS = {"og": "o", "aux_egop": "^"}

MARKER_SIZE = 7
MARKER_STRIDE = 10
MARKER_PADDING = 3


def _legend_handles(linewidth: float = 2.0) -> List[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=PLOT_COLORS[m],
            linestyle=PLOT_LINESTYLES[m],
            marker=PLOT_MARKERS[m],
            markersize=MARKER_SIZE,
            linewidth=linewidth,
            label=PLOT_LABELS[m],
        )
        for m in ("og", "aux_egop")
    ]


def plot_training_curves(
    results: List[Dict],
    save_path: Optional[Path] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (16, 10),
):
    """2×2 grid of train/val × loss/acc. Median + IQR shading across trials."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    n_epochs_total = len(results[0]["og"]["train_loss"])  # includes initial state
    epochs_range = np.arange(0, n_epochs_total)  # x axis: 0 = init, 1..N = trained

    panels = [
        ("train_loss", "Training Loss", axes[0, 0], True),
        ("val_loss", "Validation Loss", axes[0, 1], True),
        ("train_acc", "Training Accuracy", axes[1, 0], False),
        ("val_acc", "Validation Accuracy", axes[1, 1], False),
    ]

    for metric_key, metric_title, ax, log_scale in panels:
        for model_idx, model_type in enumerate(("og", "aux_egop")):
            curves = np.array([r[model_type][metric_key] for r in results])

            if len(results) == 1:
                ax.plot(
                    epochs_range,
                    curves[0],
                    color=PLOT_COLORS[model_type],
                    linestyle=PLOT_LINESTYLES[model_type],
                    marker=PLOT_MARKERS[model_type],
                    markevery=(MARKER_PADDING * model_idx, MARKER_STRIDE),
                    markersize=MARKER_SIZE,
                    linewidth=2.0,
                    label=PLOT_LABELS[model_type],
                )
            else:
                median = np.median(curves, axis=0)
                q25 = np.percentile(curves, 25, axis=0)
                q75 = np.percentile(curves, 75, axis=0)
                ax.plot(
                    epochs_range,
                    median,
                    color=PLOT_COLORS[model_type],
                    linestyle=PLOT_LINESTYLES[model_type],
                    marker=PLOT_MARKERS[model_type],
                    markevery=(MARKER_PADDING * model_idx, MARKER_STRIDE),
                    markersize=MARKER_SIZE,
                    linewidth=2.0,
                    label=PLOT_LABELS[model_type],
                )
                ax.fill_between(
                    epochs_range,
                    q25,
                    q75,
                    color=PLOT_COLORS[model_type],
                    alpha=0.2,
                )

        ax.set_title(metric_title, fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric_title, fontsize=12)
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale("log")

    axes[1, 1].legend(handles=_legend_handles(), loc="lower right", fontsize=11)

    rsvd = results[0]["rsvd_components"]
    n_iters = len(results)
    fig.suptitle(
        f"Training Curves: Original vs Auxiliary EGOP\n"
        f"(RSVD components={rsvd}, {n_iters} iteration{'s' if n_iters > 1 else ''})",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()
    return fig


# ============================================================================
# Final summary
# ============================================================================
def print_final_summary(results: List[Dict]) -> None:
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for model_type in ("og", "aux_egop"):
        finals = [r[model_type]["val_acc"][-1] for r in results]
        mean = float(np.mean(finals))
        std = float(np.std(finals))
        print(f"  {PLOT_LABELS[model_type]:18s}: {mean:.4f} ± {std:.4f}")
    print("=" * 70)


# ============================================================================
# Main experiment
# ============================================================================
def run_comparison_experiment(
    rsvd_components: int = 50,
    epochs: int = 30,
    num_iterations: int = 1,
    base_seed: int = 42,
    save_results: bool = True,
    plot_results: bool = True,
    show_plots: bool = True,
    verbose: bool = True,
    output_root: Path = Path("experiments/figures"),
) -> List[Dict]:
    """Run the OG vs auxiliary EGOP comparison."""
    # Hyperparameters
    hp = TUNED_HYPERPARAMS.get(rsvd_components)
    if hp is None:
        print(
            f"Warning: no tuned params for rsvd={rsvd_components}, falling back to rsvd=50"
        )
        hp = TUNED_HYPERPARAMS[50]

    config = DEFAULT_CONFIG.copy()
    config["rsvd_components"] = rsvd_components

    print("=" * 70)
    print("ORIGINAL vs AUXILIARY EGOP COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"Configuration: {config}")
    print(f"Hyperparameters for rsvd={rsvd_components}:")
    for model_type, params in hp.items():
        print(f"  {model_type:15s}: LR={params['lr']:.1e}, WD={params['wd']:.1e}")
    print("=" * 70)

    # Data
    trainloader, valloader, _testloader = fashionMNIST_dataloader(
        batch_size=config["batch_size"],
        num_classes=config["num_classes"],
        num_workers=NUM_WORKERS,
    )

    all_results: List[Dict] = []

    for iteration in range(num_iterations):
        seed = base_seed + iteration
        set_seed(seed)

        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration + 1}/{num_iterations} (seed={seed})")
        print("=" * 60)

        model_kwargs = {
            "pool_factor": config["pool_factor"],
            "hidden_size": config["hidden_size"],
            "num_classes": config["num_classes"],
            "weight_dist": config["weight_dist"],
        }

        # ---- Step 1: compute EGOP basis ----
        print("\n[Step 1] Computing EGOP bases...")
        basis_start = time.time()
        og_for_basis = FashionMNISTClassifier(seed=seed, **model_kwargs).to(DEVICE)
        og_for_basis.reinitialize_seeded(seed=seed)
        V_dict = compute_V_dict_for_aux(
            og_model=og_for_basis,
            trainloader=trainloader,
            egop_factor=config["egop_factor"],
            rsvd_components=rsvd_components,
            device=DEVICE,
        )
        print(f"  EGOP basis computation: {time.time() - basis_start:.2f}s")

        # ---- Step 2: build OG and auxiliary models ----
        print("\n[Step 2] Creating models...")
        og_model = FashionMNISTClassifier(seed=seed, **model_kwargs)
        aux_model = AuxiliaryReparamFashionMNISTClassifier(
            V_by_layer_dict=V_dict,
            seed=seed,
            **model_kwargs,
        )

        # init_equiv: write OG.weight (after seeded reinit) into aux's scratchpad,
        # decompose to weight_d / weight_r so forward W ≈ W_OG at t=0.
        og_model, aux_model = layerwise_reparam_init_equiv(
            EGOP_model=aux_model,
            OG_model=og_model,
            seed=seed,
        )

        # ---- Step 3: train both ----
        print("\n[Step 3] Training models...")
        trial: Dict = {"seed": seed, "rsvd_components": rsvd_components}

        print("\n  >> Original Model")
        trial["og"] = train_model(
            og_model,
            trainloader,
            valloader,
            learning_rate=hp["og"]["lr"],
            weight_decay=hp["og"]["wd"],
            epochs=epochs,
            model_name="Original",
            verbose=verbose,
        )

        print("\n  >> Auxiliary EGOP Model")
        trial["aux_egop"] = train_model(
            aux_model,
            trainloader,
            valloader,
            learning_rate=hp["aux_egop"]["lr"],
            weight_decay=hp["aux_egop"]["wd"],
            epochs=epochs,
            model_name="Auxiliary EGOP",
            verbose=verbose,
        )

        all_results.append(trial)

        print(f"\n  Summary (Iteration {iteration + 1}):")
        print(f"    Original:       Val Acc = {trial['og']['val_acc'][-1]:.4f}")
        print(
            f"    Auxiliary EGOP: Val Acc = {trial['aux_egop']['val_acc'][-1]:.4f}"
        )

    # ---- Save & plot ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_results:
        save_experiment_results(
            all_results,
            config,
            epochs,
            save_dir=output_root.parent / "csv",
        )
    if plot_results:
        output_root.mkdir(parents=True, exist_ok=True)
        plot_path = (
            output_root
            / f"training_curves_rsvd{rsvd_components}_{timestamp}.png"
        )
        plot_training_curves(
            all_results,
            save_path=plot_path,
            show_plot=show_plots,
        )

    print_final_summary(all_results)
    return all_results


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OG vs Auxiliary EGOP comparison on FashionMNIST"
    )
    parser.add_argument(
        "--rsvd_components",
        type=int,
        default=50,
        choices=[50, 100, 200, 400, 800, 1000, 2000, 4000, 7940],
        help="Number of RSVD components for EGOP basis (default: 50)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs per model (default: 30)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1,
        help="Number of trials with different seeds (default: 1)",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--full_experiment",
        action="store_true",
        help="Run full experiment: 10 iterations, 100 epochs",
    )
    parser.add_argument("--no_save", action="store_true", help="Do not save CSV")
    parser.add_argument("--no_plot", action="store_true", help="Do not generate plots")
    parser.add_argument(
        "--no_show", action="store_true", help="Do not display plots (only save)"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    return parser.parse_args()


def main() -> List[Dict]:
    args = parse_args()
    if args.full_experiment:
        args.num_iterations = 10
        args.epochs = 100
        print("Running FULL experiment: 10 iterations, 100 epochs")

    return run_comparison_experiment(
        rsvd_components=args.rsvd_components,
        epochs=args.epochs,
        num_iterations=args.num_iterations,
        base_seed=args.base_seed,
        save_results=not args.no_save,
        plot_results=not args.no_plot,
        show_plots=not args.no_show,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
