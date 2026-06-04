import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
import os
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib import font_manager
from pathlib import Path

import pdb

# ========================================
# Set up CMU Serif fonts in matplotlib (Windows)
# ========================================

## ========================================
# Set up CMU Serif fonts in matplotlib
# SPECIFIC TO ADELA'S COMPUTER. Sorry everyone else.
# You can delete this block without any impact other than choice of font family for plots.
import matplotlib.font_manager as font_manager

# font_path = "/Users/adeladepavia/Library/Fonts/cmunrm.ttf"
# font_manager.fontManager.addfont(font_path)
# prop = font_manager.FontProperties(fname=font_path)
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = prop.get_name()
# mpl.rcParams["mathtext.fontset"] = "cm"
## ========================================

# 9 entries
# MONOTONIC_BLUES = [
#     "#9ecae1",
#     "#6baed6",
#     "#4292c6",
#     "#2171b5",
#     "#08519c",
#     "#08306b",
#     "#05224d",
#     "#031633",
#     "#020b1a",
# ]

# 7 entries
MONOTONIC_BLUES = [
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#08519c",
    "#08306b",
    "#020b1a",
]


def _set_cmu_serif_windows(font_path=None):
    candidate_paths = []

    if font_path:
        candidate_paths.append(Path(font_path).expanduser())

    candidate_paths += [
        Path(r"C:/Users/hp/AppData/Local/Microsoft/Windows/Fonts/cmunrm.ttf"),
        Path.home() / "Windows/Fonts/cmunrm.ttf",
        Path("cmunrm.ttf"),
    ]

    chosen = next((pp for pp in candidate_paths if pp.exists()), None)

    if chosen is not None:
        print(f"Using CMU Serif font at: {chosen}")
        font_manager.fontManager.addfont(str(chosen))
        prop = font_manager.FontProperties(fname=str(chosen))
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = [prop.get_name()]
    else:

        print("CMU Serif font not found. Using default serif fonts.")
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = ["CMU Serif", "Times New Roman", "DejaVu Serif"]

    mpl.rcParams["mathtext.fontset"] = "cm"


_set_cmu_serif_windows()


def _clean_axes(ax):

    ax.grid(False)
    ax.grid(False, which="both", axis="both")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(top=False, right=False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")


def epoch_accuracy_analysis_time(
    data_dir, grouped_files, rsvd_components, sampling_time_df=None, save_path=None
):
    """
    Left: Validation accuracy curves with convergence markers
    Right: Cumulative time to 99.5% max accuracy (including EGOP basis time)
    """
    n_r = len(rsvd_components)

    og_color = "blue"

    # Re-ordered for best distinction between reparameterized methods
    # pink_purple_colors = [
    #     "#da70d6",
    #     "#8b008b",
    #     "#da70d6",
    #     "#ba55d3",
    #     "#9932cc",
    #     "#1b032b",
    #     "#8b008b",
    #     "#800080",
    #     "#1b032b",
    # ]
    # monotonically increasing
    monotonic_blues = MONOTONIC_BLUES

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), "-", "--", "-."]

    # Load sampling time data if provided
    if sampling_time_df is not None:
        if isinstance(sampling_time_df, str):
            sampling_df = pd.read_csv(sampling_time_df)
        else:
            sampling_df = sampling_time_df
    else:
        sampling_df = None

    # Store per-r data
    r_labels = []
    all_times_by_r = []
    label_colors = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))  # , dpi=150)

    all_og_data = []
    og_times_list = []

    # for idx, (group_files, r_value) in enumerate(zip(grouped_files, rsvd_components)):
    for idx, r_value in enumerate(rsvd_components):
        group_files = grouped_files.get(r_value, [])
        # pdb.set_trace()
        all_trials = []
        for i, fname in enumerate(group_files):
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                df_trial = pd.read_csv(fpath)
                df_trial["trial"] = i
                all_trials.append(df_trial)

        if len(all_trials) == 0:
            print(f"No data found for r={r_value}, skipping...")
            continue

        all_data = pd.concat(all_trials, ignore_index=True)

        # Auxiliary model (RSVD)
        aux_data = all_data[all_data["model_type"] == "aux_egop"]
        if len(aux_data) > 0:
            group_aux = aux_data.groupby("epoch")["val_acc"]
            median_aux = group_aux.median()
            q25_aux = group_aux.quantile(0.25)
            q75_aux = group_aux.quantile(0.75)
            epochs = median_aux.index

            line_color = monotonic_blues[idx % len(monotonic_blues)]

            ax1.plot(
                epochs,
                median_aux,
                label=r"Aux. var. EGOP, $r=$" + f"{r_value}",
                linewidth=2.5,
                color=line_color,
                linestyle="--",  # linestyles[idx % len(linestyles)],
            )
            ax1.fill_between(epochs, q25_aux, q75_aux, alpha=0.25, color=line_color)

            # Compute cumulative time to 99.5% max per trial
            times_for_this_r = []
            convergence_epochs_for_marker = []

            for trial_id in aux_data["trial"].unique():
                trial_data = aux_data[aux_data["trial"] == trial_id].sort_values(
                    "epoch"
                )

                trial_max = trial_data["val_acc"].max()
                trial_target = trial_max * 0.995

                converged = trial_data[trial_data["val_acc"] >= trial_target]

                if len(converged) > 0:
                    convergence_epoch = converged["epoch"].iloc[0]
                    convergence_epochs_for_marker.append(convergence_epoch)

                    time_data = trial_data[trial_data["epoch"] <= convergence_epoch]
                    cumulative_time = time_data["time_per_epoch"].sum()

                    if sampling_df is not None:
                        rsvd_col = f"rsvd{r_value}"
                        if rsvd_col in sampling_df.columns:
                            iteration_idx = int(trial_id)
                            if iteration_idx < len(sampling_df):
                                sampling_time = sampling_df.iloc[iteration_idx][
                                    rsvd_col
                                ]
                                if pd.notna(sampling_time):
                                    cumulative_time += sampling_time

                    times_for_this_r.append(cumulative_time)

            if len(times_for_this_r) > 0:
                r_labels.append(r"$r=$" + f"{r_value}")
                all_times_by_r.append(times_for_this_r)
                label_colors.append(line_color)

                # Mark median convergence epoch on curve
                median_epoch = np.median(convergence_epochs_for_marker)
                closest_epoch_idx = np.abs(median_aux.index - median_epoch).argmin()
                acc_at_marker = median_aux.iloc[closest_epoch_idx]

                ax1.plot(
                    median_epoch,
                    acc_at_marker,
                    "o",
                    color=line_color,
                    markersize=18,
                    markeredgecolor="white",
                    markeredgewidth=2,
                )

        # Collect OG data
        og_data = all_data[all_data["model_type"] == "og"]
        if len(og_data) > 0:
            all_og_data.append(og_data)

            for trial_id in og_data["trial"].unique():
                trial_data = og_data[og_data["trial"] == trial_id].sort_values("epoch")

                trial_max = trial_data["val_acc"].max()
                trial_target = trial_max * 0.995

                converged = trial_data[trial_data["val_acc"] >= trial_target]

                if len(converged) > 0:
                    convergence_epoch = converged["epoch"].iloc[0]
                    time_data = trial_data[trial_data["epoch"] <= convergence_epoch]
                    cumulative_time = time_data["time_per_epoch"].sum()
                    og_times_list.append(cumulative_time)

    og_convergence_epochs = []
    if len(all_og_data) > 0:
        all_og_combined = pd.concat(all_og_data, ignore_index=True)
        group_og = all_og_combined.groupby("epoch")["val_acc"]
        median_og = group_og.median()
        q25_og = group_og.quantile(0.25)
        q75_og = group_og.quantile(0.75)
        epochs = median_og.index

        ax1.plot(
            epochs,
            median_og,
            label="Original Coordinates",
            linewidth=3,
            color=og_color,
            linestyle="-",
        )
        ax1.fill_between(epochs, q25_og, q75_og, alpha=0.25, color=og_color)

        for trial_id in all_og_combined["trial"].unique():
            trial_data = all_og_combined[all_og_combined["trial"] == trial_id]
            trial_grouped = trial_data.groupby("epoch")["val_acc"].mean()

            trial_max = trial_grouped.max()
            trial_target = trial_max * 0.995

            trial_epochs = trial_grouped.index
            trial_epochs_above = trial_epochs[trial_grouped >= trial_target]
            if len(trial_epochs_above) > 0:
                og_convergence_epochs.append(trial_epochs_above[0])

        if len(og_convergence_epochs) > 0:
            median_epoch_og = np.median(og_convergence_epochs)
            closest_epoch_idx = np.abs(median_og.index - median_epoch_og).argmin()
            acc_at_marker = median_og.iloc[closest_epoch_idx]

            ax1.plot(
                median_epoch_og,
                acc_at_marker,
                "s",
                color=og_color,
                markersize=18,
                markeredgecolor="white",
                markeredgewidth=2,
            )

    if len(og_times_list) > 0:
        r_labels.append("Orig. Coors.")
        all_times_by_r.append(og_times_list)
        label_colors.append(og_color)

    ax1.set_xlabel("Epoch", fontsize=26)
    ax1.set_ylabel("Validation Accuracy", fontsize=26)
    ax1.set_title("Validation Accuracy by Epoch", fontsize=26)
    ax1.legend(loc="best", fontsize=26)
    ax1.set_ylim(0.80, 0.89)

    # Boxplot
    if len(all_times_by_r) > 0:
        bp = ax2.boxplot(
            all_times_by_r,
            labels=r_labels,
            patch_artist=True,
            widths=0.6,
            showmeans=True,
            meanprops=dict(
                marker="D",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=8,
            ),
            medianprops=dict(color="black", linewidth=2),
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=1.5),
            capprops=dict(color="black", linewidth=1.5),
        )

        for i, c in enumerate(label_colors):
            edge = mcolors.to_rgba(c, 1.0)
            face = mcolors.to_rgba(c, 0.4)

            bp["boxes"][i].set_facecolor(face)
            bp["boxes"][i].set_edgecolor(edge)
            bp["boxes"][i].set_linewidth(2)

            bp["medians"][i].set_color("black")
            bp["medians"][i].set_linewidth(2.5)

            bp["whiskers"][2 * i].set_color(edge)
            bp["whiskers"][2 * i + 1].set_color(edge)
            bp["whiskers"][2 * i].set_linewidth(2)
            bp["whiskers"][2 * i + 1].set_linewidth(2)

            bp["caps"][2 * i].set_color(edge)
            bp["caps"][2 * i + 1].set_color(edge)
            bp["caps"][2 * i].set_linewidth(2)
            bp["caps"][2 * i + 1].set_linewidth(2)

            if "means" in bp and i < len(bp["means"]):
                bp["means"][i].set_markeredgecolor("black")
                bp["means"][i].set_markerfacecolor("white")
                bp["means"][i].set_markersize(8)

        ax2.set_xlabel("Optimization Method", fontsize=26)
        ax2.set_ylabel("Time to 99.5% Max Validation Accuracy (s)", fontsize=26)
        ax2.set_title(
            "Convergence Time Distribution \n (incl. EGOP Eigenbasis Computation)",
            fontsize=26,
        )
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax1.tick_params(axis="both", which="both", labelsize=26)  # , length=8, width=1)
    ax2.tick_params(axis="both", which="both", labelsize=26)  # , length=8, width=1)
    _clean_axes(ax1)
    _clean_axes(ax2)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, axis="y", which="major")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

    # Print summary
    print("\nSummary: Cumulative time to reach 99.5% of maximum accuracy")
    print("=" * 90)
    print(
        f"{'Model':<10} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<12} {'Max':<12}"
    )
    print("-" * 90)
    for label, times_list in zip(r_labels, all_times_by_r):
        times_array = np.array(times_list)
        print(
            f"{label:<10} {times_array.mean():<12.2f} {np.median(times_array):<12.2f} "
            f"{times_array.std():<12.2f} {times_array.min():<12.2f} {times_array.max():<12.2f}"
        )
    print("=" * 90)
    return


def timing_plot_only(
    data_dir, grouped_files, rsvd_components, sampling_time_df=None, save_path=None
):
    """
    Display cumulative time to 99.5% max accuracy (including EGOP basis time)
    """

    og_color = "blue"

    # monotonically increasing
    monotonic_blues = MONOTONIC_BLUES

    # Load sampling time data if provided
    if sampling_time_df is not None:
        if isinstance(sampling_time_df, str):
            sampling_df = pd.read_csv(sampling_time_df)
        else:
            sampling_df = sampling_time_df
    else:
        sampling_df = None

    # Store per-r data
    r_labels = []
    all_times_by_r = []
    label_colors = []

    fig, ax = plt.subplots(figsize=(11, 10))  # , dpi=150)

    all_og_data = []
    og_times_list = []

    # for idx, (group_files, r_value) in enumerate(zip(grouped_files, rsvd_components)):
    for idx, r_value in enumerate(rsvd_components):
        group_files = grouped_files.get(r_value, [])
        all_trials = []
        for i, fname in enumerate(group_files):
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                df_trial = pd.read_csv(fpath)
                df_trial["trial"] = i
                all_trials.append(df_trial)

        if len(all_trials) == 0:
            print(f"No data found for r={r_value}, skipping...")
            continue

        all_data = pd.concat(all_trials, ignore_index=True)

        # Auxiliary model (RSVD)
        aux_data = all_data[all_data["model_type"] == "aux_egop"]
        if len(aux_data) > 0:
            line_color = monotonic_blues[idx % len(monotonic_blues)]

            # Compute cumulative time to 99.5% max per trial
            times_for_this_r = []
            convergence_epochs_for_marker = []

            for trial_id in aux_data["trial"].unique():
                trial_data = aux_data[aux_data["trial"] == trial_id].sort_values(
                    "epoch"
                )

                trial_max = trial_data["val_acc"].max()
                trial_target = trial_max * 0.995

                converged = trial_data[trial_data["val_acc"] >= trial_target]

                if len(converged) > 0:
                    convergence_epoch = converged["epoch"].iloc[0]
                    convergence_epochs_for_marker.append(convergence_epoch)

                    time_data = trial_data[trial_data["epoch"] <= convergence_epoch]
                    cumulative_time = time_data["time_per_epoch"].sum()

                    if sampling_df is not None:
                        rsvd_col = f"rsvd{r_value}"
                        if rsvd_col in sampling_df.columns:
                            iteration_idx = int(trial_id)
                            if iteration_idx < len(sampling_df):
                                sampling_time = sampling_df.iloc[iteration_idx][
                                    rsvd_col
                                ]
                                if pd.notna(sampling_time):
                                    cumulative_time += sampling_time

                    times_for_this_r.append(cumulative_time)

            if len(times_for_this_r) > 0:
                r_labels.append(r"$r=$" + f"{r_value}")
                all_times_by_r.append(times_for_this_r)
                label_colors.append(line_color)

        # Collect OG data
        og_data = all_data[all_data["model_type"] == "og"]
        if len(og_data) > 0:
            all_og_data.append(og_data)

            for trial_id in og_data["trial"].unique():
                trial_data = og_data[og_data["trial"] == trial_id].sort_values("epoch")

                trial_max = trial_data["val_acc"].max()
                trial_target = trial_max * 0.995

                converged = trial_data[trial_data["val_acc"] >= trial_target]

                if len(converged) > 0:
                    convergence_epoch = converged["epoch"].iloc[0]
                    time_data = trial_data[trial_data["epoch"] <= convergence_epoch]
                    cumulative_time = time_data["time_per_epoch"].sum()
                    og_times_list.append(cumulative_time)

    og_convergence_epochs = []
    if len(all_og_data) > 0:
        all_og_combined = pd.concat(all_og_data, ignore_index=True)
        group_og = all_og_combined.groupby("epoch")["val_acc"]
        median_og = group_og.median()

        for trial_id in all_og_combined["trial"].unique():
            trial_data = all_og_combined[all_og_combined["trial"] == trial_id]
            trial_grouped = trial_data.groupby("epoch")["val_acc"].mean()

            trial_max = trial_grouped.max()
            trial_target = trial_max * 0.995

            trial_epochs = trial_grouped.index
            trial_epochs_above = trial_epochs[trial_grouped >= trial_target]
            if len(trial_epochs_above) > 0:
                og_convergence_epochs.append(trial_epochs_above[0])

        if len(og_convergence_epochs) > 0:
            median_epoch_og = np.median(og_convergence_epochs)
            closest_epoch_idx = np.abs(median_og.index - median_epoch_og).argmin()

    if len(og_times_list) > 0:
        r_labels.append("Orig. Coors.")
        all_times_by_r.append(og_times_list)
        label_colors.append(og_color)

    # Boxplot
    if len(all_times_by_r) > 0:
        bp = ax.boxplot(
            all_times_by_r,
            labels=r_labels,
            patch_artist=True,
            widths=0.6,
            showmeans=False,
            meanprops=dict(
                marker="D",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=8,
            ),
            medianprops=dict(color="black", linewidth=2),
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=1.5),
            capprops=dict(color="black", linewidth=1.5),
        )

        for i, c in enumerate(label_colors):
            edge = mcolors.to_rgba(c, 1.0)
            face = mcolors.to_rgba(c, 0.4)

            bp["boxes"][i].set_facecolor(face)
            bp["boxes"][i].set_edgecolor(edge)
            bp["boxes"][i].set_linewidth(2)

            bp["medians"][i].set_color("black")
            bp["medians"][i].set_linewidth(2.5)

            bp["whiskers"][2 * i].set_color(edge)
            bp["whiskers"][2 * i + 1].set_color(edge)
            bp["whiskers"][2 * i].set_linewidth(2)
            bp["whiskers"][2 * i + 1].set_linewidth(2)

            bp["caps"][2 * i].set_color(edge)
            bp["caps"][2 * i + 1].set_color(edge)
            bp["caps"][2 * i].set_linewidth(2)
            bp["caps"][2 * i + 1].set_linewidth(2)

            if "means" in bp and i < len(bp["means"]):
                bp["means"][i].set_markeredgecolor("black")
                bp["means"][i].set_markerfacecolor("white")
                bp["means"][i].set_markersize(8)

        ax.set_ylabel("Time to 99.5% Max Validation Accuracy (s)", fontsize=30)
        ax.set_title(
            "Convergence Time Distribution \n (incl. EGOP Eigenbasis Computation)",
            fontsize=30,
        )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.tick_params(axis="both", which="both", labelsize=30)  # , length=8, width=1)
    _clean_axes(ax)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", which="major")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

    # Print summary
    print("\nSummary: Cumulative time to reach 99.5% of maximum accuracy")
    print("=" * 90)
    print(
        f"{'Model':<10} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<12} {'Max':<12}"
    )
    print("-" * 90)
    for label, times_list in zip(r_labels, all_times_by_r):
        times_array = np.array(times_list)
        print(
            f"{label:<10} {times_array.mean():<12.2f} {np.median(times_array):<12.2f} "
            f"{times_array.std():<12.2f} {times_array.min():<12.2f} {times_array.max():<12.2f}"
        )
    print("=" * 90)
    return


def get_group(dir, prefix="comparison_results_with_sweep_params_"):
    """
    Returns a dictionary mapping rsvd component values to their corresponding files.
    """
    file_names = os.listdir(dir)
    filtered_files = [f for f in file_names if f.startswith(prefix)]

    all_rsvd = [50, 100, 200, 400, 800, 1000, 2000, 4000, 7940]
    ## new method: use dict comprehension
    grouped_files = {
        rsvd: [f for f in filtered_files if f"rsvd{rsvd}_" in f] for rsvd in all_rsvd
    }

    return grouped_files


def loss_plot(
    data_dir, grouped_files, rsvd_components, loss_type="test_loss", save_path=None
):
    n_r = len(rsvd_components)

    # OG: matplotlib named 'blue', solid line
    og_color = "blue"

    # All reparam methods: same 'blue' color, dashed line, different markers
    # reparam_color = "blue"
    reparam_linestyle = "--"

    # monotonically increasing
    monotonic_blues = MONOTONIC_BLUES
    # Different markers for each r value
    markers = [
        "^",
        "o",
        "s",
        "D",
        "v",
        "P",
        "*",
        "X",
        "p",
    ]  # circle, square, triangle, diamond, etc.
    marker_interval = 5  # show marker every N epochs

    fig = plt.figure(figsize=(11, 8.5))

    all_og_data = []

    stride = 10

    # for idx, (group_files, r_value) in enumerate(zip(grouped_files, rsvd_components)):
    for idx, r_value in enumerate(rsvd_components):
        group_files = grouped_files.get(r_value, [])
        # try:
        reparam_color = monotonic_blues[idx]
        # except:
        #     pdb.set_trace()
        all_trials = []
        for i, fname in enumerate(group_files):
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                df_trial = pd.read_csv(fpath)
                df_trial["trial"] = i
                all_trials.append(df_trial)

        if len(all_trials) == 0:
            print(f"No data found for r={r_value}, skipping...")
            continue

        all_data = pd.concat(all_trials, ignore_index=True)

        aux_data = all_data[all_data["model_type"] == "aux_egop"]
        if len(aux_data) > 0:
            group_aux = aux_data.groupby("epoch")[loss_type]
            median_aux = group_aux.median()
            q25_aux = group_aux.quantile(0.25)
            q75_aux = group_aux.quantile(0.75)
            epochs = median_aux.index

            marker = markers[idx % len(markers)]
            if r_value == 1000:
                marker_size = 18
            else:
                marker_size = 14
            plt.plot(
                epochs,
                median_aux,
                label=r"Aux. var. EGOP, $r=$" + f"{r_value}",
                linewidth=2,
                color=reparam_color,
                linestyle=reparam_linestyle,
                marker=marker,
                # markevery=marker_interval,
                markevery=(7 * idx, stride),
                markersize=marker_size,
                markerfacecolor=reparam_color,
                markeredgecolor=reparam_color,
                markeredgewidth=0,
            )
            plt.fill_between(epochs, q25_aux, q75_aux, alpha=0.15, color=reparam_color)

        og_data = all_data[all_data["model_type"] == "og"]
        if len(og_data) > 0:
            all_og_data.append(og_data)

    # OG: blue solid line
    if len(all_og_data) > 0:
        all_og_combined = pd.concat(all_og_data, ignore_index=True)
        group_og = all_og_combined.groupby("epoch")[loss_type]
        median_og = group_og.median()
        q25_og = group_og.quantile(0.25)
        q75_og = group_og.quantile(0.75)
        epochs = median_og.index

        plt.plot(
            epochs,
            median_og,
            label="Original Coordinates",
            linewidth=3,
            color=og_color,
            linestyle="-",
        )
        plt.fill_between(epochs, q25_og, q75_og, alpha=0.2, color=og_color)

    # Axis labels with bigger fonts
    loss_label = (
        "Training Loss"
        if loss_type == "train_loss"
        else loss_type.replace("_", " ").title()
    )
    plt.xlabel("Epochs", fontsize=32)
    plt.ylabel(loss_label, fontsize=32)
    plt.yscale("log")
    # plt.ylim((0, 1.0))

    # plt.title(...)

    # Legend with bigger font
    plt.legend(loc="upper right", fontsize=30, ncol=1)

    # Bigger tick labels
    ax = plt.gca()
    ax.tick_params(axis="both", which="both", labelsize=30, length=8, width=1)

    _clean_axes(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


# def all_time_v(data_dir, grouped_files, rsvd_components, sampling_time_df=None):
#     """
#     Modified version of all_time that includes EGOP basis computation time (sampling time).

#     Parameters:
#     -----------
#     data_dir : str
#         Directory containing the CSV files
#     grouped_files : list of lists
#         Each inner list contains filenames for a specific r value
#     rsvd_components : list
#         List of r values corresponding to grouped_files
#     sampling_time_df : pd.DataFrame or str, optional
#         DataFrame or path to CSV with EGOP basis computation times.
#         Expected format: columns = ['iteration', 'rsvd50', 'rsvd100', ...]
#         If None, sampling time is not added.
#     """
#     colors = plt.cm.tab10(np.linspace(0, 1, 9))
#     time_data_dir = data_dir

#     # Load sampling time data if provided
#     if sampling_time_df is not None:
#         if isinstance(sampling_time_df, str):
#             sampling_df = pd.read_csv(sampling_time_df)
#         else:
#             sampling_df = sampling_time_df
#     else:
#         sampling_df = None

#     # Store results for each r value (all trials)
#     r_labels = []
#     all_times_by_r = []

#     # Collect OG times directly (not data)
#     og_times_list = []

#     # Process each r value group
#     for idx, (group_files, r_value) in enumerate(zip(grouped_files, rsvd_components)):
#         all_trials = []
#         for i, fname in enumerate(group_files):
#             fpath = os.path.join(time_data_dir, fname)
#             if os.path.exists(fpath):
#                 df_trial = pd.read_csv(fpath)
#                 df_trial['trial'] = i
#                 all_trials.append(df_trial)

#         if len(all_trials) == 0:
#             print(f"No data found for r={r_value}, skipping...")
#             continue

#         all_data = pd.concat(all_trials, ignore_index=True)

#         # Get aux model cumulative time to 99.5% max
#         aux_data = all_data[all_data['model_type'] == 'aux_egop']
#         if len(aux_data) > 0:
#             times_for_this_r = []
#             for trial_id in aux_data['trial'].unique():
#                 trial_data = aux_data[aux_data['trial'] == trial_id].sort_values('epoch')

#                 trial_max = trial_data['val_acc'].max()
#                 trial_target = trial_max * 0.995

#                 converged = trial_data[trial_data['val_acc'] >= trial_target]

#                 if len(converged) > 0:
#                     convergence_epoch = converged['epoch'].iloc[0]
#                     time_data = trial_data[trial_data['epoch'] <= convergence_epoch]
#                     cumulative_time = time_data['time_per_epoch'].sum()

#                     # Add sampling time if available
#                     if sampling_df is not None:
#                         rsvd_col = f'rsvd{r_value}'
#                         if rsvd_col in sampling_df.columns:
#                             # trial_id corresponds to iteration index (0-based)
#                             # sampling_df iteration is 1-based
#                             iteration_idx = int(trial_id)
#                             if iteration_idx < len(sampling_df):
#                                 sampling_time = sampling_df.iloc[iteration_idx][rsvd_col]
#                                 if pd.notna(sampling_time):
#                                     cumulative_time += sampling_time

#                     times_for_this_r.append(cumulative_time)

#             if len(times_for_this_r) > 0:
#                 r_labels.append(f'r={r_value}')
#                 all_times_by_r.append(times_for_this_r)


#         og_data = all_data[all_data['model_type'] == 'og']
#         if len(og_data) > 0:
#             for trial_id in og_data['trial'].unique():
#                 trial_data = og_data[og_data['trial'] == trial_id].sort_values('epoch')

#                 trial_max = trial_data['val_acc'].max()
#                 trial_target = trial_max * 0.995

#                 converged = trial_data[trial_data['val_acc'] >= trial_target]

#                 if len(converged) > 0:
#                     convergence_epoch = converged['epoch'].iloc[0]
#                     time_data = trial_data[trial_data['epoch'] <= convergence_epoch]
#                     cumulative_time = time_data['time_per_epoch'].sum()
#                     og_times_list.append(cumulative_time)


#     if len(og_times_list) > 0:
#         r_labels.append('OG')
#         all_times_by_r.append(og_times_list)

#     # Plot
#     if len(all_times_by_r) == 0:
#         print("No data collected!")
#     else:
#         fig, ax = plt.subplots(figsize=(10, 7))

#         bp = ax.boxplot(all_times_by_r, labels=r_labels, patch_artist=True,
#                         widths=0.6, showmeans=True,
#                         meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='black', markersize=8),
#                         medianprops=dict(color='black', linewidth=2),
#                         boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
#                         whiskerprops=dict(color='black', linewidth=1.5),
#                         capprops=dict(color='black', linewidth=1.5))

#         if 'OG' in r_labels:
#             og_idx = r_labels.index('OG')
#             bp['boxes'][og_idx].set_facecolor('lightcoral')

#         ax.set_xlabel('RSVD Component (r)', fontsize=13)
#         ax.set_ylabel('Cumulative Time to 99.5% Max Accuracy (seconds)', fontsize=13)
#         ax.set_title('Convergence Time Distribution (with EGOP Basis Time): Time to 99.5% of Max vs r', fontsize=15)
#         # ax.grid(True, alpha=0.3, axis='y')

#         _clean_axes(ax)
#         plt.tight_layout()
#         plt.show()

#         # Print summary
#         print("\nSummary: Cumulative time to reach 99.5% of maximum accuracy (including EGOP basis computation)")
#         print("=" * 90)
#         print(f"{'Model':<10} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
#         print("-" * 90)
#         for label, times_list in zip(r_labels, all_times_by_r):
#             times_array = np.array(times_list)
#             print(f"{label:<10} {times_array.mean():<12.2f} {np.median(times_array):<12.2f} "
#                 f"{times_array.std():<12.2f} {times_array.min():<12.2f} {times_array.max():<12.2f}")
#         print("=" * 90)


# def epoch_to_converge(grouped_files, pos, data_dir):

#     group_rsvd = grouped_files[pos]
#     dfs = [pd.read_csv(os.path.join(data_dir, fname)) for fname in group_rsvd]

#     all_trials = []
#     for i, df_trial in enumerate(dfs):
#         df_trial = df_trial.copy()
#         df_trial['trial'] = i
#         all_trials.append(df_trial)
#     all_data = pd.concat(all_trials, ignore_index=True)

#     trial_results = []

#     for model in all_data['model_type'].unique():
#         model_data = all_data[all_data['model_type'] == model]

#         for trial in model_data['trial'].unique():
#             trial_data = model_data[model_data['trial'] == trial].sort_values('epoch')

#             max_val_acc = trial_data['val_acc'].max()
#             threshold = 0.995 * max_val_acc
#             print(f"Model: {model}, Trial: {trial}, Max val_acc: {max_val_acc:.4f}, Threshold: {threshold:.4f}")
#             converged = trial_data[trial_data['val_acc'] >= threshold]

#             if len(converged) > 0:
#                 convergence_epoch = converged['epoch'].iloc[0]

#                 trial_results.append({
#                     'model_type': model,
#                     'convergence_epoch': convergence_epoch
#                 })

#     df_results = pd.DataFrame(trial_results)

#     # Filter for only OG and Auxiliary
#     og_data = df_results[df_results['model_type'].str.contains('og|baseline', case=False, na=False)]
#     aux_data = df_results[df_results['model_type'].str.contains('aux', case=False, na=False)]

#     fig, ax = plt.subplots(figsize=(8, 6))

#     data_to_plot = [og_data['convergence_epoch'].values,
#                     aux_data['convergence_epoch'].values]
#     labels = ['OG', 'Auxiliary']

#     bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
#                     showmeans=True, meanline=True)

#     # Color the boxes
#     bp['boxes'][0].set_facecolor('lightblue')
#     bp['boxes'][1].set_facecolor('lightcoral')

#     ax.set_ylabel('Epoch', fontsize=14)
#     ax.set_title('Epochs to Reach 99.5% Max val_acc', fontsize=16, fontweight='bold')
#     # ax.grid(axis='y', alpha=0.3)
#     _clean_axes(ax)
#     plt.tight_layout()
#     plt.show()

#     # Print statistics
#     print(f"OG: {og_data['convergence_epoch'].mean():.1f} ± {og_data['convergence_epoch'].std():.1f} epochs")
#     print(f"Auxiliary: {aux_data['convergence_epoch'].mean():.1f} ± {aux_data['convergence_epoch'].std():.1f} epochs")
#     return


# def epoch_accuracy_analysis(data_dir, grouped_files, rsvd_components):
#     n_r = len(rsvd_components)
#     r_colors = _blue_gradient(n_r, start=0.35, end=0.90)   # gradient blues for RSVD
#     r_styles = _rsvd_linestyles(n_r)                       # distinct dotted/dashed styles for RSVD
#     og_color = 'blue'                                      # OG uses pure blue

#     # Store per-r convergence epochs (all trials)
#     r_labels = []
#     all_epochs_by_r = []
#     label_colors = []   # keep color aligned with r_labels for the boxplot

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

#     # Collect OG data from all groups to average later
#     all_og_data = []

#     for idx, (group_files, r_value) in enumerate(zip(grouped_files, rsvd_components)):
#         all_trials = []
#         for i, fname in enumerate(group_files):
#             fpath = os.path.join(data_dir, fname)
#             if os.path.exists(fpath):
#                 df_trial = pd.read_csv(fpath)
#                 df_trial['trial'] = i
#                 all_trials.append(df_trial)

#         if len(all_trials) == 0:
#             print(f"No data found for r={r_value}, skipping...")
#             continue

#         all_data = pd.concat(all_trials, ignore_index=True)

#         # Auxiliary model (RSVD)
#         aux_data = all_data[all_data['model_type'] == 'aux_egop']
#         if len(aux_data) > 0:
#             group_aux = aux_data.groupby('epoch')['val_acc']
#             mean_aux = group_aux.mean()
#             error_aux = group_aux.apply(sem)
#             epochs = mean_aux.index

#             # Plot validation accuracy (RSVD: gradient blue + dotted/dashed)
#             ax1.plot(
#                 epochs, mean_aux,
#                 label=f'aux, r={r_value}',
#                 linewidth=2,
#                 color=r_colors[idx],
#                 linestyle=r_styles[idx]
#             )
#             ax1.fill_between(
#                 epochs, mean_aux - error_aux, mean_aux + error_aux,
#                 alpha=0.12,
#                 color=r_colors[idx]
#             )

#             # Compute "epoch to reach 99.5% of max" per trial
#             epochs_for_this_r = []
#             for trial_id in aux_data['trial'].unique():
#                 trial_data = aux_data[aux_data['trial'] == trial_id]
#                 trial_grouped = trial_data.groupby('epoch')['val_acc'].mean()

#                 trial_max = trial_grouped.max()
#                 trial_target = trial_max * 0.995

#                 trial_epochs = trial_grouped.index
#                 trial_epochs_above = trial_epochs[trial_grouped >= trial_target]
#                 if len(trial_epochs_above) > 0:
#                     epochs_for_this_r.append(trial_epochs_above[0])

#             if len(epochs_for_this_r) > 0:
#                 r_labels.append(f'r={r_value}')
#                 all_epochs_by_r.append(epochs_for_this_r)
#                 label_colors.append(r_colors[idx])

#                 # Mark the mean convergence epoch on the curve
#                 mean_epoch = np.mean(epochs_for_this_r)
#                 closest_epoch_idx = np.abs(mean_aux.index - mean_epoch).argmin()
#                 acc_at_marker = mean_aux.iloc[closest_epoch_idx]

#                 ax1.plot(
#                     mean_epoch, acc_at_marker, 'o',
#                     color=r_colors[idx],
#                     markersize=8,
#                     markeredgecolor='black',
#                     markeredgewidth=1.5
#                 )

#         # Collect OG data for averaging
#         og_data = all_data[all_data['model_type'] == 'og']
#         if len(og_data) > 0:
#             all_og_data.append(og_data)

#     # OG: average across all groups (pure blue, solid line)
#     og_epochs_list = []
#     if len(all_og_data) > 0:
#         all_og_combined = pd.concat(all_og_data, ignore_index=True)
#         group_og = all_og_combined.groupby('epoch')['val_acc']
#         mean_og = group_og.mean()
#         error_og = group_og.apply(sem)
#         epochs = mean_og.index

#         ax1.plot(epochs, mean_og, label='OG', linewidth=3, color=og_color, linestyle='-')
#         ax1.fill_between(epochs, mean_og - error_og, mean_og + error_og, alpha=0.12, color=og_color)

#         # Compute "epoch to reach 99.5% of max" per OG trial
#         for trial_id in all_og_combined['trial'].unique():
#             trial_data = all_og_combined[all_og_combined['trial'] == trial_id]
#             trial_grouped = trial_data.groupby('epoch')['val_acc'].mean()

#             trial_max = trial_grouped.max()
#             trial_target = trial_max * 0.995

#             trial_epochs = trial_grouped.index
#             trial_epochs_above = trial_epochs[trial_grouped >= trial_target]
#             if len(trial_epochs_above) > 0:
#                 og_epochs_list.append(trial_epochs_above[0])

#         if len(og_epochs_list) > 0:
#             r_labels.append('OG')
#             all_epochs_by_r.append(og_epochs_list)
#             label_colors.append(og_color)

#             # Mark the mean OG convergence epoch
#             mean_epoch_og = np.mean(og_epochs_list)
#             closest_epoch_idx = np.abs(mean_og.index - mean_epoch_og).argmin()
#             acc_at_marker = mean_og.iloc[closest_epoch_idx]

#             ax1.plot(
#                 mean_epoch_og, acc_at_marker, 's',
#                 color=og_color,
#                 markersize=10,
#                 markeredgecolor='black',
#                 markeredgewidth=2
#             )

#     ax1.set_xlabel('Epoch', fontsize=13)
#     ax1.set_ylabel('Validation Accuracy', fontsize=13)
#     ax1.set_title('Validation Accuracy: Markers show mean epoch to 99.5% of max', fontsize=15)
#     ax1.legend(loc='best', fontsize=9, ncol=2)

#     # Boxplot: color each box to match the corresponding curve color (including OG)
#     if len(all_epochs_by_r) > 0:
#         bp = ax2.boxplot(
#             all_epochs_by_r,
#             labels=r_labels,
#             patch_artist=True,
#             widths=0.6,
#             showmeans=True,
#             meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8),
#             medianprops=dict(color='black', linewidth=2),
#             boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
#             whiskerprops=dict(color='black', linewidth=1.5),
#             capprops=dict(color='black', linewidth=1.5)
#         )

#         for i, c in enumerate(label_colors):
#             edge = mcolors.to_rgba(c, 1.0)
#             face = mcolors.to_rgba(c, 0.20)

#             # Box face/edge
#             bp['boxes'][i].set_facecolor(face)
#             bp['boxes'][i].set_edgecolor(edge)
#             bp['boxes'][i].set_linewidth(1.8)

#             # Median line
#             bp['medians'][i].set_color(edge)
#             bp['medians'][i].set_linewidth(2.2)

#             # Whiskers (two per box)
#             bp['whiskers'][2*i].set_color(edge)
#             bp['whiskers'][2*i + 1].set_color(edge)
#             bp['whiskers'][2*i].set_linewidth(1.6)
#             bp['whiskers'][2*i + 1].set_linewidth(1.6)

#             # Caps (two per box)
#             bp['caps'][2*i].set_color(edge)
#             bp['caps'][2*i + 1].set_color(edge)
#             bp['caps'][2*i].set_linewidth(1.6)
#             bp['caps'][2*i + 1].set_linewidth(1.6)

#             # Mean marker
#             if 'means' in bp and i < len(bp['means']):
#                 bp['means'][i].set_markeredgecolor(edge)
#                 bp['means'][i].set_markerfacecolor('white')
#                 bp['means'][i].set_markersize(8)

#         ax2.set_xlabel('RSVD Component (r)', fontsize=13)
#         ax2.set_ylabel('Epoch to Reach 99.5% of Max Accuracy', fontsize=13)
#         ax2.set_title('Convergence Speed Distribution: Epoch to 99.5% of Max vs r', fontsize=15)

#         # Use consistent ticks for epoch-like values
#         ax2.set_ylim(0, 110)
#         ax2.set_yticks(np.arange(0, 110, 5))
#         plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

#     _clean_axes(ax1)
#     _clean_axes(ax2)
#     plt.tight_layout()
#     plt.show()

#     # Print summary statistics
#     print("\nSummary: Epoch to reach 99.5% of maximum accuracy (across all trials)")
#     print("=" * 90)
#     print(f"{'Model':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
#     print("-" * 90)
#     for label, epochs_list in zip(r_labels, all_epochs_by_r):
#         epochs_array = np.array(epochs_list)
#         print(f"{label:<10} {epochs_array.mean():<10.2f} {np.median(epochs_array):<10.1f} "
#               f"{epochs_array.std():<10.2f} {epochs_array.min():<10} {epochs_array.max():<10}")
#     print("=" * 90)
#     return


# def time(grouped_files, pos,data_dir):

#     group_rsvd = grouped_files[pos]
#     dfs = [pd.read_csv(os.path.join(data_dir, fname)) for fname in group_rsvd]

#     all_trials = []
#     for i, df_trial in enumerate(dfs):
#         df_trial = df_trial.copy()
#         df_trial['trial'] = i
#         all_trials.append(df_trial)
#     all_data = pd.concat(all_trials, ignore_index=True)


#     trial_results = []

#     for model in all_data['model_type'].unique():
#         model_data = all_data[all_data['model_type'] == model]

#         for trial in model_data['trial'].unique():
#             trial_data = model_data[model_data['trial'] == trial].sort_values('epoch')

#             max_val_acc = trial_data['val_acc'].max()
#             threshold = 0.995 * max_val_acc
#             print(f"Model: {model}, Trial: {trial}, Max val_acc: {max_val_acc:.4f}, Threshold: {threshold:.4f}")
#             converged = trial_data[trial_data['val_acc'] >= threshold]

#             if len(converged) > 0:
#                 convergence_epoch = converged['epoch'].iloc[0]
#                 time_data = trial_data[trial_data['epoch'] <= convergence_epoch]
#                 cumulative_time = time_data['time_per_epoch'].sum()

#                 trial_results.append({
#                     'model_type': model,
#                     'cumulative_time_seconds': cumulative_time
#                 })

#     df_results = pd.DataFrame(trial_results)

#     # Filter for only OG and Auxiliary
#     og_data = df_results[df_results['model_type'].str.contains('og|baseline', case=False, na=False)]
#     aux_data = df_results[df_results['model_type'].str.contains('aux', case=False, na=False)]


#     fig, ax = plt.subplots(figsize=(8, 6))

#     data_to_plot = [og_data['cumulative_time_seconds'].values,
#                     aux_data['cumulative_time_seconds'].values]
#     labels = ['OG', 'Auxiliary']

#     bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
#                     showmeans=True, meanline=True)

#     # Color the boxes
#     bp['boxes'][0].set_facecolor('lightblue')
#     bp['boxes'][1].set_facecolor('lightcoral')

#     ax.set_ylabel('Cumulative Time (seconds)', fontsize=14)
#     ax.set_title('Time to Reach 99.5% Max val_acc', fontsize=16, fontweight='bold')
#     # ax.grid(axis='y', alpha=0.3)
#     _clean_axes(ax)
#     plt.tight_layout()

#     plt.show()

#     # Print statistics
#     print(f"OG: {og_data['cumulative_time_seconds'].mean():.1f} ± {og_data['cumulative_time_seconds'].std():.1f} seconds")
#     print(f"Auxiliary: {aux_data['cumulative_time_seconds'].mean():.1f} ± {aux_data['cumulative_time_seconds'].std():.1f} seconds")
#     return


# def time_v(grouped_files, pos, data_dir, V_time):
#     group_rsvd = grouped_files[pos]
#     dfs = [pd.read_csv(os.path.join(data_dir, fname)) for fname in group_rsvd]

#     all_trials = []
#     for i, df_trial in enumerate(dfs):
#         df_trial = df_trial.copy()
#         df_trial['trial'] = i
#         all_trials.append(df_trial)
#     all_data = pd.concat(all_trials, ignore_index=True)

#     trial_results = []

#     for model in all_data['model_type'].unique():
#         model_data = all_data[all_data['model_type'] == model]

#         for trial in model_data['trial'].unique():
#             trial_data = model_data[model_data['trial'] == trial].sort_values('epoch')

#             max_val_acc = trial_data['val_acc'].max()
#             threshold = 0.995 * max_val_acc
#             print(f"Model: {model}, Trial: {trial}, Max val_acc: {max_val_acc:.4f}, Threshold: {threshold:.4f}")
#             converged = trial_data[trial_data['val_acc'] >= threshold]

#             if len(converged) > 0:
#                 convergence_epoch = converged['epoch'].iloc[0]
#                 time_data = trial_data[trial_data['epoch'] <= convergence_epoch]
#                 cumulative_time = time_data['time_per_epoch'].sum()

#                 trial_results.append({
#                     'model_type': model,
#                     'trial': trial,
#                     'cumulative_time_seconds': cumulative_time
#                 })

#     df_results = pd.DataFrame(trial_results)


#     og_data = df_results[df_results['model_type'].str.contains('og|baseline', case=False, na=False)]
#     aux_data = df_results[df_results['model_type'].str.contains('aux', case=False, na=False)].copy()


#     aux_data = aux_data.sort_values("trial").reset_index(drop=True)
#     aux_data["cumulative_time_seconds"] += V_time[:len(aux_data)]


#     df_results.update(aux_data)


#     fig, ax = plt.subplots(figsize=(8, 6))

#     data_to_plot = [
#         og_data['cumulative_time_seconds'].values,
#         aux_data['cumulative_time_seconds'].values
#     ]
#     labels = ['OG', 'Auxiliary']

#     bp = ax.boxplot(
#         data_to_plot, labels=labels, patch_artist=True,
#         showmeans=True, meanline=True
#     )

#     bp['boxes'][0].set_facecolor('lightblue')
#     bp['boxes'][1].set_facecolor('lightcoral')

#     ax.set_ylabel('Cumulative Time (seconds)', fontsize=14)
#     ax.set_title('Time to Reach 99.5% Max val_acc', fontsize=16, fontweight='bold')
#     # ax.grid(axis='y', alpha=0.3)
#     _clean_axes(ax)
#     plt.tight_layout()
#     plt.show()

#     print(f"OG: {og_data['cumulative_time_seconds'].mean():.1f} ± {og_data['cumulative_time_seconds'].std():.1f} seconds")
#     print(f"Auxiliary: {aux_data['cumulative_time_seconds'].mean():.1f} ± {aux_data['cumulative_time_seconds'].std():.1f} seconds")

#     return df_results
