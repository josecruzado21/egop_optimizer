"""2x2 grid plotting for OG vs auxiliary EGOP training-curve comparison.

Reads the long-format DataFrame produced by `experiment_aggregation.aggregate_trials`
and plots:
  - Top-left   : Training Loss   (log y-axis)
  - Top-right  : Validation Loss (log y-axis)
  - Bottom-left: Training Accuracy
  - Bottom-right: Validation Accuracy

For multi-trial runs, the median curve is drawn with IQR (Q25–Q75) shading.
For single-trial runs, the lone curve is drawn without shading.

Color / linestyle / marker conventions match the old repo's
reduced_and_auxiliary_example.py (og: blue solid 'o'; aux_egop: green dot-dash '^').
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


PLOT_COLORS = {"og": "blue", "aux_egop": "green"}
PLOT_LABELS = {"og": "Original", "aux_egop": "Auxiliary EGOP"}
PLOT_LINESTYLES = {"og": "-", "aux_egop": "-."}
PLOT_MARKERS = {"og": "o", "aux_egop": "^"}

MARKER_SIZE = 7
MARKER_STRIDE = 10
MARKER_PADDING = 3


def _legend_handles(model_types: Tuple[str, ...], linewidth: float) -> List[Line2D]:
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
        for m in model_types
    ]


def plot_training_comparison(
    aggregated_long_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    show_plot: bool = True,
    model_types: Tuple[str, ...] = ("og", "aux_egop"),
    title_suffix: str = "",
    figsize: Tuple[int, int] = (16, 10),
) -> Figure:
    """Plot 2x2 training comparison from long-format aggregated DataFrame.

    Args:
        aggregated_long_df: DataFrame with columns:
            model_type, seed, epoch, train_loss, train_acc[, val_loss, val_acc]
        save_path: If given, save the figure here (parent dirs created).
        show_plot: If True, plt.show(); otherwise close after save.
        model_types: Which model types to plot. Each must appear in the df.
        title_suffix: Extra text appended to suptitle (e.g. RSVD config).
        figsize: Figure size (W, H) in inches.

    Returns:
        The matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    linewidth = 2.0

    has_val = "val_loss" in aggregated_long_df.columns and "val_acc" in aggregated_long_df.columns

    panels = [
        ("train_loss", "Training Loss", axes[0, 0], True),
        ("val_loss", "Validation Loss", axes[0, 1], True),
        ("train_acc", "Training Accuracy", axes[1, 0], False),
        ("val_acc", "Validation Accuracy", axes[1, 1], False),
    ]

    for metric_key, metric_title, ax, log_scale in panels:
        if (metric_key in ("val_loss", "val_acc")) and not has_val:
            ax.text(
                0.5,
                0.5,
                "(no validation metrics)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="gray",
            )
            ax.set_title(metric_title, fontsize=14)
            continue

        for model_idx, model_type in enumerate(model_types):
            sub = aggregated_long_df[aggregated_long_df["model_type"] == model_type]
            if sub.empty:
                continue

            # epochs are integers; pivot to (seed -> epoch) wide
            pivot = sub.pivot(index="seed", columns="epoch", values=metric_key)
            epochs_range = pivot.columns.to_numpy()
            curves = pivot.to_numpy()  # shape (n_trials, n_epochs)

            color = PLOT_COLORS[model_type]
            linestyle = PLOT_LINESTYLES[model_type]
            marker = PLOT_MARKERS[model_type]
            markevery = (MARKER_PADDING * model_idx, MARKER_STRIDE)

            if curves.shape[0] == 1:
                ax.plot(
                    epochs_range,
                    curves[0],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markevery=markevery,
                    markersize=MARKER_SIZE,
                    linewidth=linewidth,
                    label=PLOT_LABELS[model_type],
                )
            else:
                median = np.median(curves, axis=0)
                q25 = np.percentile(curves, 25, axis=0)
                q75 = np.percentile(curves, 75, axis=0)
                ax.plot(
                    epochs_range,
                    median,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markevery=markevery,
                    markersize=MARKER_SIZE,
                    linewidth=linewidth,
                    label=PLOT_LABELS[model_type],
                )
                ax.fill_between(
                    epochs_range,
                    q25,
                    q75,
                    color=color,
                    alpha=0.2,
                )

        ax.set_title(metric_title, fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric_title, fontsize=12)
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale("log")

    # Legend on bottom-right
    axes[1, 1].legend(
        handles=_legend_handles(model_types, linewidth),
        loc="lower right",
        fontsize=11,
    )

    n_trials = aggregated_long_df["seed"].nunique()
    suptitle = (
        f"Training Curves: Original vs Auxiliary EGOP "
        f"({n_trials} iteration{'s' if n_trials > 1 else ''})"
    )
    if title_suffix:
        suptitle += f"\n{title_suffix}"
    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig
