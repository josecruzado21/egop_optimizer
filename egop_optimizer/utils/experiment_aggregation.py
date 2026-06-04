"""Aggregate per-trial training logs into a single long-format DataFrame.

Used by multi-trial experiment scripts (e.g. auxiliary EGOP comparison) to
collect per-epoch metrics across trials and model types after each trial's
basic_train_loop has written its training.log.

Workflow:
  experiment_root/                 (e.g. "logs")
  ├── auxiliary_compare/
  │   ├── og/
  │   │   ├── trial_42/training.log
  │   │   ├── trial_43/training.log
  │   │   └── ...
  │   └── aux_egop/
  │       ├── trial_42/training.log
  │       └── ...

After all trials done, call:

    df = aggregate_trials(
        log_root=Path("logs"),
        model_subdirs={"og": "auxiliary_compare/og",
                       "aux_egop": "auxiliary_compare/aux_egop"},
        trial_seeds=[42, 43, 44],
    )

    save_aggregated_csv(df, save_path=Path("experiments/csv/og_vs_aux.csv"))
    median_df = compute_median_iqr(df)

The long-format DataFrame `df` has one row per (model_type, seed, epoch) and
is ready for downstream plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from egop_optimizer.utils.log_parser import parse_training_log


def aggregate_trials(
    log_root: Path,
    model_subdirs: Dict[str, str],
    trial_seeds: List[int],
) -> pd.DataFrame:
    """Collect per-epoch metrics from each trial's training.log.

    Args:
        log_root: Directory containing the per-experiment subtree
            (e.g. "logs" — basic_train_loop writes under cwd-relative
            "logs/{experiment_name}/training.log").
        model_subdirs: Mapping model_type -> path-suffix relative to log_root.
            E.g. {"og": "auxiliary_compare/og", "aux_egop": "auxiliary_compare/aux_egop"}.
        trial_seeds: List of integer seeds. For each model_type and seed, the
            function reads `log_root / subdir / f"trial_{seed}" / "training.log"`.

    Returns:
        Long-format DataFrame with columns:
            model_type, seed, epoch, train_loss, train_acc
            [, val_loss, val_acc, train_time, val_time]

        Validation / time columns are present only if at least one log line had them.
        Time values are in seconds (per basic_train_loop's log format).
    """
    log_root = Path(log_root)
    rows: List[Dict] = []

    for model_type, subdir in model_subdirs.items():
        for seed in trial_seeds:
            log_path = log_root / subdir / f"trial_{seed}" / "training.log"
            parsed = parse_training_log(log_path)
            n = len(parsed["epoch"])
            for i in range(n):
                row = {
                    "model_type": model_type,
                    "seed": seed,
                    "epoch": parsed["epoch"][i],
                    "train_loss": parsed["train_loss"][i],
                    "train_acc": parsed["train_acc"][i],
                }
                if "val_loss" in parsed:
                    row["val_loss"] = parsed["val_loss"][i]
                if "val_acc" in parsed:
                    row["val_acc"] = parsed["val_acc"][i]
                if "train_time" in parsed:
                    row["train_time"] = parsed["train_time"][i]
                if "val_time" in parsed:
                    row["val_time"] = parsed["val_time"][i]
                rows.append(row)

    return pd.DataFrame(rows)


def compute_median_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (model_type, epoch); aggregate median / Q25 / Q75 across trials.

    Args:
        df: Long-format DataFrame from aggregate_trials.

    Returns:
        Wide-format DataFrame with columns:
            model_type, epoch, <metric>_median, <metric>_q25, <metric>_q75
        for each metric in {train_loss, train_acc, val_loss?, val_acc?}.
        Index is RangeIndex.
    """
    metric_cols = [
        c
        for c in (
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "train_time",
            "val_time",
        )
        if c in df.columns
    ]

    grouped = df.groupby(["model_type", "epoch"])[metric_cols]

    summary_parts = {}
    for col in metric_cols:
        summary_parts[f"{col}_median"] = grouped[col].median()
        summary_parts[f"{col}_q25"] = grouped[col].quantile(0.25)
        summary_parts[f"{col}_q75"] = grouped[col].quantile(0.75)

    summary = pd.concat(summary_parts, axis=1).reset_index()
    return summary


def save_aggregated_csv(df: pd.DataFrame, save_path: Path) -> None:
    """Save a DataFrame to CSV, creating parent dirs if needed."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Aggregated results saved to: {save_path}")
