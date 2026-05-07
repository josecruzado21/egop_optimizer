"""
Experiment v2: Original FashionMNIST classifier vs Auxiliary EGOP reparam.

Aligned with the new repo's training conventions: each trial's training is
done by `basic_train_loop`, which writes per-epoch metrics to
  logs/{experiment_name}/training.log
After all trials finish, the script parses those logs (no checkpoints, no
modification of basic_train_loop) and produces an aggregated CSV plus a
2x2 comparison plot.

Per trial:
  1. Build OG model (seed) + reinit_seeded
  2. compute_V_by_layer with EGOP_oversampling_factor + RSVD
  3. Build auxiliary reparam model (V_dict, seed)
  4. layerwise_reparam_init_equiv(aux, OG, seed) — guarantees forward W ≈ W_OG at t=0
  5. basic_train_loop on OG and aux separately (different log dirs)

After all trials:
  6. aggregate_trials reads each training.log, builds long-format DataFrame
  7. save aggregated CSV
  8. plot 2x2 grid (train/val × loss/acc), median + IQR shading

Note: init metrics ("Initial Training Loss = ...") logged by basic_train_loop
are NOT aggregated/plotted by design — they exist only in the per-trial log
text files. Curves start from epoch 1.

Usage:
    python experiments/fashionMNIST_auxiliary_compare_v2.py \\
        --rsvd_components 50 --epochs 30 --num_iterations 1
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.FashionMNISTClassfier import (
    AuxiliaryReparamFashionMNISTClassifier,
    FashionMNISTClassifier,
)
from egop_optimizer.dataloaders.fashionMNIST_dataloader import fashionMNIST_dataloader
from egop_optimizer.utils.EGOP_utils import (
    compute_V_by_layer,
    layerwise_reparam_init_equiv,
)
from egop_optimizer.utils.auxiliary_optimizer import create_auxiliary_adamw
from egop_optimizer.utils.experiment_aggregation import (
    aggregate_trials,
    save_aggregated_csv,
)
from egop_optimizer.utils.experiment_plotting import plot_training_comparison
from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()
print(f"Using device: {DEVICE}")

NUM_WORKERS = 0 if sys.platform == "win32" else 2

# ============================================================================
# Tuned hyperparameters (from old repo: reduced_and_auxiliary_example.py)
# ============================================================================
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

# Where logs go (basic_train_loop uses cwd-relative "logs/{experiment_name}")
EXPERIMENT_PREFIX = "auxiliary_compare"
LOG_ROOT = Path("logs")


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


def run_trial(
    seed: int,
    rsvd_components: int,
    epochs: int,
    hp: Dict,
    config: Dict,
    trainloader,
    valloader,
    verbose: bool = True,
) -> None:
    """Run one trial: build models, init_equiv, train OG and aux via basic_train_loop."""
    if verbose:
        print(f"\n{'=' * 60}\nTrial seed={seed}\n{'=' * 60}")

    model_kwargs = {
        "pool_factor": config["pool_factor"],
        "hidden_size": config["hidden_size"],
        "num_classes": config["num_classes"],
        "weight_dist": config["weight_dist"],
    }

    # ---- Step 1: build OG and seed ----
    og_model = FashionMNISTClassifier(seed=seed, **model_kwargs).to(DEVICE)
    og_model.reinitialize_seeded(seed=seed)

    # ---- Step 2: compute V_dict ----
    if verbose:
        print("[Step 1] Computing EGOP basis...")
    basis_start = time.time()
    V_dict = compute_V_by_layer(
        model_OG=og_model,
        data_loader=trainloader,
        criterion=nn.CrossEntropyLoss(reduction="mean"),
        device=DEVICE,
        reparam_linear_layers=True,
        use_randomized_svd=True,
        n_components=rsvd_components,
        EGOP_oversampling_factor=config["egop_factor"],
    )
    if verbose:
        print(f"  EGOP basis: {time.time() - basis_start:.2f}s")

    # ---- Step 3: build aux + init_equiv ----
    if verbose:
        print("[Step 2] Building aux model + init_equiv...")
    aux_model = AuxiliaryReparamFashionMNISTClassifier(
        V_by_layer_dict=V_dict, seed=seed, **model_kwargs
    ).to(DEVICE)
    aux_model.move_bases_to_device(DEVICE)

    og_model, aux_model = layerwise_reparam_init_equiv(
        EGOP_model=aux_model, OG_model=og_model, seed=seed
    )

    # ---- Step 4: train OG ----
    if verbose:
        print("[Step 3] Training OG...")
    og_optimizer = create_auxiliary_adamw(
        og_model, learning_rate=hp["og"]["lr"], weight_decay=hp["og"]["wd"]
    )
    basic_train_loop(
        model=og_model,
        trainloader=trainloader,
        optimizer=og_optimizer,
        loss_method=lambda reduction: nn.CrossEntropyLoss(reduction=reduction),
        epochs=epochs,
        LR_scheduler=None,
        valloader=valloader,
        device=DEVICE,
        experiment_name=f"{EXPERIMENT_PREFIX}/og/trial_{seed}",
        report_validation_metrics=True,
        checkpoint=True,
        initial_metrics=True,
    )

    # ---- Step 5: train auxiliary ----
    if verbose:
        print("[Step 4] Training auxiliary EGOP...")
    aux_optimizer = create_auxiliary_adamw(
        aux_model,
        learning_rate=hp["aux_egop"]["lr"],
        weight_decay=hp["aux_egop"]["wd"],
    )
    basic_train_loop(
        model=aux_model,
        trainloader=trainloader,
        optimizer=aux_optimizer,
        loss_method=lambda reduction: nn.CrossEntropyLoss(reduction=reduction),
        epochs=epochs,
        LR_scheduler=None,
        valloader=valloader,
        device=DEVICE,
        experiment_name=f"{EXPERIMENT_PREFIX}/aux_egop/trial_{seed}",
        report_validation_metrics=True,
        checkpoint=True,
        initial_metrics=True,
    )


def run_comparison_experiment(
    rsvd_components: int = 50,
    epochs: int = 30,
    num_iterations: int = 1,
    base_seed: int = 42,
    egop_factor: Optional[float] = None,
    save_csv: bool = True,
    plot_results: bool = True,
    show_plot: bool = True,
    verbose: bool = True,
    csv_root: Path = Path("experiments/csv"),
    figure_root: Path = Path("experiments/figures"),
) -> None:
    """Run multi-trial OG vs auxiliary EGOP comparison.

    egop_factor (float, optional): Override DEFAULT_CONFIG["egop_factor"]
        (the EGOP oversampling factor controlling k = int(factor * count_params)).
        If None, the default 0.1 from DEFAULT_CONFIG is used.
    """
    hp = TUNED_HYPERPARAMS.get(rsvd_components)
    if hp is None:
        print(
            f"Warning: no tuned params for rsvd={rsvd_components}, falling back to rsvd=50"
        )
        hp = TUNED_HYPERPARAMS[50]

    config = DEFAULT_CONFIG.copy()
    config["rsvd_components"] = rsvd_components
    if egop_factor is not None:
        config["egop_factor"] = egop_factor

    print("=" * 70)
    print("ORIGINAL vs AUXILIARY EGOP COMPARISON (v2 — basic_train_loop)")
    print("=" * 70)
    print(f"Configuration: {config}")
    print(f"Hyperparameters for rsvd={rsvd_components}:")
    for model_type, params in hp.items():
        print(f"  {model_type:15s}: LR={params['lr']:.1e}, WD={params['wd']:.1e}")
    print("=" * 70)

    trainloader, valloader, _testloader = fashionMNIST_dataloader(
        batch_size=config["batch_size"],
        num_classes=config["num_classes"],
        num_workers=NUM_WORKERS,
    )

    trial_seeds: List[int] = []
    for i in range(num_iterations):
        seed = base_seed + i
        set_seed(seed)
        run_trial(
            seed=seed,
            rsvd_components=rsvd_components,
            epochs=epochs,
            hp=hp,
            config=config,
            trainloader=trainloader,
            valloader=valloader,
            verbose=verbose,
        )
        trial_seeds.append(seed)

    # ---- aggregate logs ----
    print("\n" + "=" * 60)
    print("Aggregating trial logs...")
    print("=" * 60)
    df = aggregate_trials(
        log_root=LOG_ROOT,
        model_subdirs={
            "og": f"{EXPERIMENT_PREFIX}/og",
            "aux_egop": f"{EXPERIMENT_PREFIX}/aux_egop",
        },
        trial_seeds=trial_seeds,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_csv:
        save_aggregated_csv(
            df,
            save_path=csv_root / f"og_vs_aux_rsvd{rsvd_components}_{timestamp}.csv",
        )
    if plot_results:
        plot_training_comparison(
            df,
            save_path=figure_root
            / f"training_curves_rsvd{rsvd_components}_{timestamp}.png",
            show_plot=show_plot,
            title_suffix=f"RSVD components={rsvd_components}",
        )

    # ---- final summary ----
    if "val_acc" in df.columns:
        print("\nFinal validation accuracy (last epoch, mean ± std across trials):")
        last_epoch = df["epoch"].max()
        for model_type in ("og", "aux_egop"):
            sub = df[(df["model_type"] == model_type) & (df["epoch"] == last_epoch)]
            mean = sub["val_acc"].mean()
            std = sub["val_acc"].std()
            print(f"  {model_type:10s}: {mean:.4f} ± {std:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OG vs Auxiliary EGOP comparison on FashionMNIST (basic_train_loop based)"
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
        "--base_seed", type=int, default=42, help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--egop_factor",
        type=float,
        default=None,
        help="EGOP oversampling factor; if set, overrides DEFAULT_CONFIG['egop_factor'] (0.1).",
    )
    parser.add_argument(
        "--full_experiment",
        action="store_true",
        help="Run full experiment: 10 iterations, 100 epochs",
    )
    parser.add_argument("--no_save", action="store_true", help="Do not save aggregated CSV")
    parser.add_argument(
        "--no_plot", action="store_true", help="Do not generate comparison plot"
    )
    parser.add_argument(
        "--no_show", action="store_true", help="Save plot but do not display"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_experiment:
        args.num_iterations = 10
        args.epochs = 100
        print("Running FULL experiment: 10 iterations, 100 epochs")

    run_comparison_experiment(
        rsvd_components=args.rsvd_components,
        epochs=args.epochs,
        num_iterations=args.num_iterations,
        base_seed=args.base_seed,
        egop_factor=args.egop_factor,
        save_csv=not args.no_save,
        plot_results=not args.no_plot,
        show_plot=not args.no_show,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
