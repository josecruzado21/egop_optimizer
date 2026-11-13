"""Analysis utilities for OG vs Auxiliary EGOP comparison results.

Adapted from cluster23 plot.py for the new repo's CSV format:
  model_type, seed, epoch, train_loss, train_acc, val_loss, val_acc, train_time, val_time

Each CSV contains all trials (rows) for one rsvd_components value, one row per
(model_type, seed, epoch). Use load_results() to combine multiple rsvd CSVs into
a single long DataFrame keyed by rsvd_components.
"""

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MONOTONIC_BLUES = [
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#08519c",
    "#08306b",
    "#020b1a",
]


def setup_fonts():
    """CMU Serif if installed (Windows path baked from Adela's machine);
    falls back to system serif silently otherwise."""
    candidate = Path("C:/Users/hp/AppData/Local/Microsoft/Windows/Fonts/cmunrm.ttf")
    if candidate.exists():
        mpl.font_manager.fontManager.addfont(str(candidate))
        prop = mpl.font_manager.FontProperties(fname=str(candidate))
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = [prop.get_name()]
    else:
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = ["CMU Serif", "Times New Roman", "DejaVu Serif"]
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["axes.unicode_minus"] = False


def _clean_axes(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")


def load_results(csv_dir, rsvd_values=None, pattern="og_vs_aux_rsvd{r}_*.csv"):
    """Load CSVs into a single long DataFrame with `rsvd_components` column.

    For each rsvd value, takes the MOST RECENT matching CSV (by sorted name,
    which works because timestamp suffix is YYYYMMDD_HHMMSS).
    """
    csv_dir = Path(csv_dir)
    if rsvd_values is None:
        all_files = list(csv_dir.glob("og_vs_aux_rsvd*.csv"))
        rsvd_values = sorted(set(
            int(re.search(r"rsvd(\d+)", f.name).group(1)) for f in all_files
        ))
    dfs = []
    for r in rsvd_values:
        matches = sorted(csv_dir.glob(pattern.format(r=r)))
        if not matches:
            print(f"  [skip] no CSV for rsvd={r}")
            continue
        latest = matches[-1]
        df = pd.read_csv(latest)
        df["rsvd_components"] = r
        dfs.append(df)
        print(f"  loaded rsvd={r:>5}  ({latest.name}, {len(df):>5} rows, "
              f"{df['seed'].nunique()} trials)")
    if not dfs:
        raise ValueError(f"No CSVs found in {csv_dir}")
    return pd.concat(dfs, ignore_index=True)


def _convergence_epoch(trial_df, target_frac=0.995):
    """First epoch where val_acc >= target_frac × max(val_acc) for one trial."""
    trial_df = trial_df.sort_values("epoch")
    target = trial_df["val_acc"].max() * target_frac
    above = trial_df[trial_df["val_acc"] >= target]
    return above["epoch"].iloc[0] if len(above) else np.nan


def _mark_median_convergence(ax, median_series, trial_data, color, marker,
                             target_frac=0.995, markersize=18):
    """Plot a marker on `median_series` at the median convergence epoch
    across the trials in `trial_data` (must have 'seed' & 'val_acc' & 'epoch')."""
    conv_epochs = [
        _convergence_epoch(trial_data[trial_data["seed"] == s], target_frac)
        for s in trial_data["seed"].unique()
    ]
    conv_epochs = [e for e in conv_epochs if not np.isnan(e)]
    if not conv_epochs:
        return
    median_epoch = np.median(conv_epochs)
    closest_idx = np.abs(median_series.index - median_epoch).argmin()
    acc_at_marker = median_series.iloc[closest_idx]
    ax.plot(median_epoch, acc_at_marker, marker,
            color=color, markersize=markersize,
            markeredgecolor="white", markeredgewidth=2,
            zorder=10)


def plot_val_acc_curves(df, rsvd_values=None, ax=None, ylim=None, xlim=None,
                        title=None, target_frac=0.995,
                        show_convergence_markers=True):
    """Validation accuracy by epoch — OG (one solid blue curve, aggregated across
    all OG runs) vs Aux (dashed blue gradient per rsvd), median + IQR shading.

    If `show_convergence_markers=True`, also plots a circle/square at the
    median epoch where val_acc first reached `target_frac` × max(val_acc)."""
    if rsvd_values is None:
        rsvd_values = sorted(df["rsvd_components"].unique())
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 8))

    for idx, r in enumerate(rsvd_values):
        aux = df[(df["model_type"] == "aux_egop") & (df["rsvd_components"] == r)]
        if aux.empty:
            continue
        grp = aux.groupby("epoch")["val_acc"]
        med, q25, q75 = grp.median(), grp.quantile(0.25), grp.quantile(0.75)
        color = MONOTONIC_BLUES[idx % len(MONOTONIC_BLUES)]
        ax.plot(med.index, med, label=fr"Aux. EGOP, $r={r}$",
                linewidth=2.5, color=color, linestyle="--")
        ax.fill_between(med.index, q25, q75, alpha=0.25, color=color)
        if show_convergence_markers:
            _mark_median_convergence(ax, med, aux, color, "o", target_frac)

    og = df[df["model_type"] == "og"]
    if not og.empty:
        grp = og.groupby("epoch")["val_acc"]
        med, q25, q75 = grp.median(), grp.quantile(0.25), grp.quantile(0.75)
        ax.plot(med.index, med, label="Original Coordinates",
                linewidth=3, color="blue", linestyle="-")
        ax.fill_between(med.index, q25, q75, alpha=0.25, color="blue")
        if show_convergence_markers:
            og_trials = og.copy()
            og_trials["seed"] = (
                og_trials["rsvd_components"].astype(str)
                + "_" + og_trials["seed"].astype(str)
            )
            _mark_median_convergence(ax, med, og_trials, "blue", "s", target_frac)

    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel("Validation Accuracy", fontsize=20)
    ax.set_title(title or "OG vs Auxiliary EGOP — Validation Accuracy",
                 fontsize=20)
    if ylim:
        ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend(loc="best", fontsize=13)
    ax.tick_params(labelsize=16)
    _clean_axes(ax)
    return ax


def plot_loss_curves(df, rsvd_values=None, loss_col="train_loss",
                     ax=None, log_scale=True, title=None,
                     xlim=None, ylim=None):
    """Loss curves (default log scale), OG vs Aux at each rsvd."""
    if rsvd_values is None:
        rsvd_values = sorted(df["rsvd_components"].unique())
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 8))

    markers = ["^", "o", "s", "D", "v", "P", "*", "X", "p"]
    stride = 10

    for idx, r in enumerate(rsvd_values):
        aux = df[(df["model_type"] == "aux_egop") & (df["rsvd_components"] == r)]
        if aux.empty:
            continue
        grp = aux.groupby("epoch")[loss_col]
        med, q25, q75 = grp.median(), grp.quantile(0.25), grp.quantile(0.75)
        color = MONOTONIC_BLUES[idx % len(MONOTONIC_BLUES)]
        ax.plot(med.index, med, label=fr"Aux. EGOP, $r={r}$",
                linewidth=2, color=color, linestyle="--",
                marker=markers[idx % len(markers)],
                markevery=(7 * idx, stride), markersize=12,
                markerfacecolor=color, markeredgecolor=color, markeredgewidth=0)
        ax.fill_between(med.index, q25, q75, alpha=0.15, color=color)

    og = df[df["model_type"] == "og"]
    if not og.empty:
        grp = og.groupby("epoch")[loss_col]
        med, q25, q75 = grp.median(), grp.quantile(0.25), grp.quantile(0.75)
        ax.plot(med.index, med, label="Original Coordinates",
                linewidth=3, color="blue", linestyle="-")
        ax.fill_between(med.index, q25, q75, alpha=0.2, color="blue")

    if log_scale:
        ax.set_yscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel(loss_col.replace("_", " ").title(), fontsize=20)
    ax.set_title(title or f"OG vs Auxiliary EGOP — {loss_col}", fontsize=20)
    ax.legend(loc="best", fontsize=13)
    ax.tick_params(labelsize=16)
    _clean_axes(ax)
    return ax


def _time_to_target(trial_df, target_frac=0.995):
    """Cumulative train_time to first reach target_frac × max(val_acc) for one trial."""
    trial_df = trial_df.sort_values("epoch")
    conv_epoch = _convergence_epoch(trial_df, target_frac)
    if np.isnan(conv_epoch):
        return np.nan
    return trial_df[trial_df["epoch"] <= conv_epoch]["train_time"].sum()


def plot_convergence_time_box(df, rsvd_values=None, target_frac=0.995,
                              ax=None, title=None):
    """Boxplot of per-trial cumulative train_time to reach target_frac × max
    val_acc. One box per (Aux, rsvd); single OG box aggregating all OG runs."""
    if rsvd_values is None:
        rsvd_values = sorted(df["rsvd_components"].unique())
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))

    times_per_label, labels, colors = [], [], []

    for idx, r in enumerate(rsvd_values):
        aux = df[(df["model_type"] == "aux_egop") & (df["rsvd_components"] == r)]
        ts = [_time_to_target(aux[aux["seed"] == s], target_frac)
              for s in aux["seed"].unique()]
        ts = [t for t in ts if not np.isnan(t)]
        if ts:
            times_per_label.append(ts)
            labels.append(fr"$r={r}$")
            colors.append(MONOTONIC_BLUES[idx % len(MONOTONIC_BLUES)])

    og = df[df["model_type"] == "og"]
    og_ts = []
    for (_, _), grp in og.groupby(["rsvd_components", "seed"]):
        t = _time_to_target(grp, target_frac)
        if not np.isnan(t):
            og_ts.append(t)
    if og_ts:
        times_per_label.append(og_ts)
        labels.append("OG")
        colors.append("blue")

    bp = ax.boxplot(times_per_label, labels=labels, patch_artist=True, widths=0.6,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=8),
                    medianprops=dict(color="black", linewidth=2))

    for i, c in enumerate(colors):
        bp["boxes"][i].set_facecolor(mcolors.to_rgba(c, 0.4))
        bp["boxes"][i].set_edgecolor(c)
        bp["boxes"][i].set_linewidth(2)

    ax.set_xlabel("Method", fontsize=20)
    ax.set_ylabel(f"Cum. train_time to {target_frac*100:.1f}% max val_acc (s)",
                  fontsize=16)
    ax.set_title(title or "Convergence Time Distribution", fontsize=20)
    ax.tick_params(labelsize=14)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    _clean_axes(ax)
    ax.grid(True, axis="y", which="major")
    ax.set_ylim(bottom=0)
    return ax


def summary_table(df, rsvd_values=None):
    """Per-(rsvd, model_type) final-epoch summary: mean ± std of val_acc,
    train_acc, and total train_time across trials."""
    if rsvd_values is None:
        rsvd_values = sorted(df["rsvd_components"].unique())
    rows = []
    for r in rsvd_values:
        sub = df[df["rsvd_components"] == r]
        for mt in ["og", "aux_egop"]:
            mt_data = sub[sub["model_type"] == mt]
            if mt_data.empty:
                continue
            finals = mt_data.loc[mt_data.groupby("seed")["epoch"].idxmax()]
            total_time = mt_data.groupby("seed")["train_time"].sum()
            rows.append({
                "rsvd": r,
                "model": mt,
                "n_trials": finals["seed"].nunique(),
                "val_acc_mean": finals["val_acc"].mean(),
                "val_acc_std": finals["val_acc"].std(),
                "train_acc_mean": finals["train_acc"].mean(),
                "total_time_mean": total_time.mean(),
                "total_time_std": total_time.std(),
            })
    return pd.DataFrame(rows)
