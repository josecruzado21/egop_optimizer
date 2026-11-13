"""Parse `training.log` files produced by basic_train_loop.

basic_train_loop writes per-epoch metrics with a fixed format:

    YYYY-MM-DD HH:MM:SS,fff [INFO] Epoch <n>: Training Loss = X, Validation Loss = Y,
        Training Acc. = A, Validation Acc. = B, Training Time = T1s, Validation Time = T2s

Times are in seconds (per basic_train_loop's log format).
(With validation off, the line lacks "Validation Loss" / "Validation Acc.".)

This module supplies a simple regex-based parser that reads a log file and
returns the structured per-epoch metrics, used by the auxiliary experiment
aggregation step. Init metric lines ("Initial Training Loss = ...") are
intentionally skipped — by design we do not aggregate them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

# Match a training-loop epoch line, e.g.:
# "[INFO] Epoch 5: Training Loss = 1.2345, Validation Loss = 1.4567,
#  Training Acc. = 0.5000, Validation Acc. = 0.5500,
#  Training Time = 35.20s, Validation Time = 1.80s"
# Time values are in seconds (the trailing 's' character is consumed but not captured).
# Validation fields (loss / acc / time) are all optional — basic_train_loop omits
# them when valloader is None.
_EPOCH_LINE_PATTERN = re.compile(
    r"\[INFO\]\s+Epoch\s+(?P<epoch>\d+):\s+"
    r"Training\s+Loss\s+=\s+(?P<train_loss>[-+\d.eE]+)"
    r"(?:,\s+Validation\s+Loss\s+=\s+(?P<val_loss>[-+\d.eE]+))?"
    r",?\s+Training\s+Acc\.?\s+=\s+(?P<train_acc>[-+\d.eE]+)"
    r"(?:,\s+Validation\s+Acc\.?\s+=\s+(?P<val_acc>[-+\d.eE]+))?"
    r"(?:,\s+Training\s+Time\s+=\s+(?P<train_time>[-+\d.eE]+)s)?"
    r"(?:,\s+Validation\s+Time\s+=\s+(?P<val_time>[-+\d.eE]+)s)?"
)


def parse_training_log(log_path: Path) -> Dict[str, List[float]]:
    """Parse a training.log file produced by basic_train_loop.

    Args:
        log_path: Path to a training.log file.

    Returns:
        Dict with keys "epoch", "train_loss", "train_acc". If validation metrics
        are present in the log, also includes "val_loss" and "val_acc". If
        Training Time / Validation Time fields are present, also includes
        "train_time" and "val_time" (values in seconds). All lists are aligned
        by index (entry i corresponds to the i-th matched epoch line).
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"training log not found: {log_path}")

    epochs: List[int] = []
    train_losses: List[float] = []
    train_accs: List[float] = []
    val_losses: List[float] = []
    val_accs: List[float] = []
    train_times: List[float] = []
    val_times: List[float] = []
    has_val = False
    has_train_time = False
    has_val_time = False

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = _EPOCH_LINE_PATTERN.search(line)
            if match is None:
                continue
            epochs.append(int(match.group("epoch")))
            train_losses.append(float(match.group("train_loss")))
            train_accs.append(float(match.group("train_acc")))

            v_loss = match.group("val_loss")
            v_acc = match.group("val_acc")
            if v_loss is not None and v_acc is not None:
                has_val = True
                val_losses.append(float(v_loss))
                val_accs.append(float(v_acc))

            t_time = match.group("train_time")
            if t_time is not None:
                has_train_time = True
                train_times.append(float(t_time))

            v_time = match.group("val_time")
            if v_time is not None:
                has_val_time = True
                val_times.append(float(v_time))

    result: Dict[str, List[float]] = {
        "epoch": epochs,
        "train_loss": train_losses,
        "train_acc": train_accs,
    }
    if has_val:
        result["val_loss"] = val_losses
        result["val_acc"] = val_accs
    if has_train_time:
        result["train_time"] = train_times
    if has_val_time:
        result["val_time"] = val_times
    return result
