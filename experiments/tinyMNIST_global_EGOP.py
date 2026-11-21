from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

from matplotlib.patches import ConnectionPatch, Rectangle

import matplotlib as mpl

mpl.use("Agg")  # Works without a graphical display.
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import torch
from tqdm.auto import tqdm

import pdb

## ========================================
# Set up CMU Serif fonts in matplotlib
# SPECIFIC TO ADELA'S COMPUTER. Sorry everyone else.
# You can delete this block without any impact other than choice of font family for plots.
font_path = Path("/Users/adeladepavia/Library/Fonts/cmunrm.ttf")
if font_path.is_file():
    # Retain the locally installed CMU Serif font when available.
    font_manager.fontManager.addfont(str(font_path))
    prop = font_manager.FontProperties(fname=str(font_path))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = prop.get_name()
else:
    # Allow plotting on machines without this user-specific font file.
    plt.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "cm"
## ========================================
AXIS_LABEL_FONTSIZE = 30
TICK_FONTSIZE = 30
INSET_FONTSIZE = 25
MARKERSIZE = 8.0
INSET_X_PADDING = 1.0
INSET_Y_PADDING = 0.02

# Resolve paths relative to this file, regardless of the working directory.
EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── User-editable settings ────────────────────────────────────────────────
NUM_SAMPLES = 20000  # Number of independent model initializations / gradient draws
CHUNK_SIZE = 2000  # Number of gradient samples accumulated per matrix multiply
FULL_BATCH_GRADIENTS = True  # True: gradient over entire training split per draw
BATCH_SIZE = (
    300  # Examples per gradient in mini-batch mode; memory chunk in full-batch mode
)
SEED = 0  # RNG used for reinitializations and batch sampling
DATA_SPLIT_SEED = 0  # Separate RNG used to construct the data split
INCLUDE_BIASES = True  # True = all parameters; False = weights only as in EGOP_utils
PLOT_ONLY = False  # True = load a previous estimate and regenerate its plots

INPUT_SIZE = 64
HIDDEN_SIZES = [32]
NUM_CLASSES = 10
TRAIN_VAL_TEST_SPLIT = [None, 0.33, 0.66]

# EVD happens on CPU in float64 for reproducible, numerically accurate spectra.
# Forward/backward gradient evaluations use the device returned by the repo.
EVD_DTYPE = torch.float64
TOP_EIGENVECTORS_TO_PLOT = 60
INSET_NUM_EIGENVALUES = 100  # Leading eigenvalues shown in the linear-scale inset.
INSET_POSITION = (
    0.12,
    0.12,
    0.43,
    0.40,
)  # Bottom-left: (left, bottom, width, height), axes fractions
# ─────────────────────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_layout(model: torch.nn.Module, include_biases: bool) -> list[dict]:
    """Record the exact flatten/concatenation order used for global gradients.

    Each entry describes the inclusive-exclusive row slice of the global vector
    occupied by the named parameter tensor. All trainable parameters are included
    by default. Setting include_biases=False excludes tensors named *.bias.
    """
    layout = []
    offset = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not include_biases and name.split(".")[-1] == "bias":
            continue
        count = parameter.numel()
        layout.append(
            {
                "name": name,
                "start": offset,
                "end": offset + count,
                "shape": list(parameter.shape),
            }
        )
        offset += count
    if not layout:
        raise ValueError("No eligible trainable parameters found.")
    return layout


def infinite_batches(data_loader):
    """Cycle through the data loader without caching all its batches in memory."""
    while True:
        yielded = False
        for batch in data_loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("Data loader produced no batches.")


def estimate_global_egop(
    model: torch.nn.Module,
    data_loader,
    criterion,
    *,
    num_samples: int,
    chunk_size: int = CHUNK_SIZE,
    device: torch.device | str,
    include_biases: bool = True,
    full_batch_gradients: bool = FULL_BATCH_GRADIENTS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """Compute the global EGOP and return (Lambda, V, EGOP, slices).

    For every gradient draw, freshly reinitialize the model. If
    full_batch_gradients=True, use the gradient of the mean loss over the
    entire training split (accumulated over microbatches without updating the
    model). Otherwise, sample one mini-batch gradient per initialization.

    Accumulate global gradient outer products in chunks:

        EGOP_sum += G_chunk @ G_chunk.T.

    Divide by num_samples after all complete and partial chunks have been
    added; this preserves cross-parameter and cross-layer correlations.
    The full EGOP, eigenvalues and eigenvectors are on CPU in EVD_DTYPE
    with eigenpairs sorted by descending eigenvalue. No optimization occurs.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not hasattr(model, "reinitialize"):
        raise TypeError("Expected the OG model to provide its reinitialize() method")

    device = torch.device(device)
    model = model.to(device)
    model.train()
    layout = parameter_layout(model, include_biases)
    d = layout[-1]["end"]

    # Both the accumulator and each chunk live on CPU to keep GPU memory use
    # low and make the full eigen-decomposition independent of NUM_SAMPLES.
    EGOP_sum = torch.zeros((d, d), dtype=EVD_DTYPE, device="cpu")
    G_chunk = torch.empty(
        (d, min(chunk_size, num_samples)), dtype=EVD_DTYPE, device="cpu"
    )
    chunk_count = 0
    batches = None if full_batch_gradients else infinite_batches(data_loader)
    if full_batch_gradients and getattr(data_loader, "drop_last", False):
        raise ValueError("Full-batch gradients require a loader with drop_last=False")

    print(
        f"Sampling {num_samples} global gradients in d={d} dimensions "
        f"(chunk size {chunk_size}; "
        f"{'full training split' if full_batch_gradients else 'one mini-batch per draw'})",
        flush=True,
    )
    for sample_idx in tqdm(range(num_samples), desc="Global EGOP gradients"):
        # Match the existing EGOP utility: fresh weights for every draw.
        model.reinitialize()
        model.zero_grad(set_to_none=True)

        if full_batch_gradients:
            # The criterion uses reduction="mean". Weight each microbatch
            # by its number of examples so a shorter final batch contributes
            # proportionally. Do not clear gradients between microbatches.
            total_examples = 0
            for x, y in data_loader:
                x = x.to(device)
                y = y.to(device)
                batch_examples = x.shape[0]
                (criterion(model(x), y) * batch_examples).backward()
                total_examples += batch_examples
            if total_examples == 0:
                raise ValueError(
                    "Cannot compute full-batch gradient on an empty loader"
                )
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(total_examples)
        else:
            x, y = next(batches)
            x = x.to(device)
            y = y.to(device)
            criterion(model(x), y).backward()

        # Resolve parameters again in case reinitialize() replaces Parameters.
        parameters = dict(model.named_parameters())
        grad_parts = []
        for entry in layout:
            name = entry["name"]
            if name not in parameters:
                raise RuntimeError(f"Model reinitialization removed parameter {name}")
            parameter = parameters[name]
            if tuple(parameter.shape) != tuple(entry["shape"]):
                raise RuntimeError(
                    f"Parameter shape changed after reinitialization: {name}"
                )
            if parameter.grad is None:
                raise RuntimeError(
                    f"Missing gradient for {name} at sample {sample_idx}"
                )
            grad_parts.append(
                parameter.grad.detach().reshape(-1).to(device="cpu", dtype=EVD_DTYPE)
            )

        g = torch.cat(grad_parts)
        if not torch.isfinite(g).all():
            raise RuntimeError(f"Non-finite gradient at sample {sample_idx}")
        G_chunk[:, chunk_count] = g
        chunk_count += 1

        # Accumulate a complete chunk, or the final short chunk. This never
        # mixes samples between parameter coordinates (columns are gradients).
        if chunk_count == G_chunk.shape[1] or sample_idx + 1 == num_samples:
            active_chunk = G_chunk[:, :chunk_count]
            EGOP_sum.addmm_(active_chunk, active_chunk.T)
            chunk_count = 0

    EGOP = EGOP_sum / num_samples
    EGOP = (EGOP + EGOP.T) / 2  # Remove roundoff-induced asymmetry.
    del G_chunk, EGOP_sum

    print(f"Eigendecomposing full {d} x {d} EGOP matrix ...", flush=True)
    # eigh returns eigenvalues ascending; reverse values and columns together.
    Lambda, V = torch.linalg.eigh(EGOP)
    Lambda = Lambda.flip(0).clamp_min(0.0).contiguous()
    V = V.flip(1).contiguous()
    return Lambda, V, EGOP, layout


def eigenvector_diagnostics(V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (beta, normalized IPR), each in [1/d, 1] up to roundoff.

    beta[k] = ||V[:, k]||_1^2 / d measures eigenvector coordinate density.
    ipr_normalized[k] = 1 / (d * sum_j V[j, k]^4) measures participation.

    For repeated (or numerically tied) eigenvalues these are diagnostics of
    the EVD's chosen basis, NOT invariant quantities of an eigenspace.
    """
    d = V.shape[0]
    beta = V.abs().sum(dim=0).square() / d
    ipr_normalized = 1.0 / (d * V.pow(4).sum(dim=0))
    return beta, ipr_normalized


def plot_global_egop_spectrum(result: dict, *, save_stem: Path) -> None:
    """Plot the global spectrum on log scale, with a linear-scale leading inset.

    The inset uses a linear y-axis starting at zero. Since the main y-axis is
    logarithmic, we draw the zoom rectangle/connectors manually rather than
    using mark_inset (which would try to transform y=0 through the log axis).
    """
    Lambda = result["Lambda"].cpu().numpy()
    d = len(Lambda)
    stable_rank = np.sum(np.sqrt(Lambda)) / np.sqrt(Lambda[0])
    print("Stable rank:", stable_rank)

    fig, ax = plt.subplots(figsize=(13, 9), constrained_layout=True)
    ax.set_yscale("log")
    if d and np.isfinite(Lambda[0]) and Lambda[0] > 0:
        normalized = Lambda / Lambda[0]
        k = np.arange(1, d + 1)  # λ_1 is indexed by k = 1.
        positive = np.isfinite(normalized) & (normalized > 0)
        (spectrum_line,) = ax.plot(
            k[positive],
            normalized[positive],
            "-o",
            linewidth=1.3,
            markersize=MARKERSIZE,
        )

        # A linear-scale magnification of the first 100 (or fewer) eigenvalues.
        n_inset = min(INSET_NUM_EIGENVALUES, d)
        if n_inset > 0:
            ax_inset = ax.inset_axes(INSET_POSITION)
            ax_inset.set_facecolor("white")
            ax_inset.plot(
                k[:n_inset],
                normalized[:n_inset],
                "-o",
                color=spectrum_line.get_color(),
                linewidth=1.5,
                markersize=MARKERSIZE,
            )
            ax_inset.set_xlim(
                1 - INSET_X_PADDING,
                max(2, n_inset) + INSET_X_PADDING,
            )
            ax_inset.set_ylim(-INSET_Y_PADDING, 1.05)
            ax_inset.set_yscale("linear")
            ax_inset.set_xlabel("Leading 100 indices", fontsize=INSET_FONTSIZE)
            ax_inset.tick_params(axis="both", which="major", labelsize=INSET_FONTSIZE)
            ax_inset.grid(alpha=0.2)
            ax_inset.set_zorder(ax.get_zorder() + 1)

            # Highlight the region being magnified on the logarithmic plot.
            # Its lower edge must be strictly positive on that axis, even though
            # the inset itself properly extends down to zero on a linear scale.
            leading_positive = normalized[:n_inset]
            leading_positive = leading_positive[
                np.isfinite(leading_positive) & (leading_positive > 0)
            ]
            if leading_positive.size and n_inset >= 10:
                zoom_bottom = max(
                    float(ax.get_ylim()[0]), float(leading_positive.min()) * 0.8
                )
                zoom_top = min(float(ax.get_ylim()[1]), 1.05)
                if zoom_bottom < zoom_top:
                    rectangle = Rectangle(
                        (1, zoom_bottom),
                        n_inset - 1,
                        zoom_top - zoom_bottom,
                        fill=False,
                        edgecolor="0.5",
                        linewidth=1.1,
                        linestyle="--",
                    )
                    ax.add_patch(rectangle)
                    # Dashed gray guide lines connect matching corners of
                    # the zoom box and the inset: bottom-left to bottom-left,
                    # and top-right to top-right.
                    for xy_parent, xy_inset in (
                        ((1, zoom_bottom), (0, 0)),
                        ((n_inset, zoom_top), (1, 1)),
                    ):
                        connector = ConnectionPatch(
                            xyA=xy_parent,
                            coordsA=ax.transData,
                            xyB=xy_inset,
                            coordsB=ax_inset.transAxes,
                            color="0.5",
                            linewidth=0.9,
                            linestyle="--",
                            alpha=0.75,
                            clip_on=False,
                        )
                        ax.add_artist(connector)
    else:
        ax.text(
            0.5,
            0.5,
            "No positive eigenvalues",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_ylim(top=2, bottom=1e-13)
    ax.set_xlabel(r"Eigenvalue index $k$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(
        r"Normalized eigenvalue $\lambda_k/\lambda_1$", fontsize=AXIS_LABEL_FONTSIZE
    )
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    save_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        file_path = save_stem.with_suffix(f".{ext}")
        fig.savefig(file_path, dpi=180)
        print(f"Saved: {file_path}")
    plt.close(fig)


def plot_global_egop_betas(result: dict, *, save_stem: Path) -> None:
    """Plot the betas for the leading 100 eigenvalues."""
    beta = result["beta"].cpu().numpy()
    Lambda = result["Lambda"].cpu().numpy()
    d = len(Lambda)

    fig, ax = plt.subplots(figsize=(13, 9), constrained_layout=True)
    ax.plot(np.arange(1, 101), beta[:100], "-o", markersize=MARKERSIZE)
    ax.axhline(
        y=1 / d,
        linestyle="--",
        color="red",
        label=r"$1/d$",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"Leading 100 Eigenvalue Indices", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Density measure $\|v_k\|^2_1/d$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=TICK_FONTSIZE)

    save_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        file_path = save_stem.with_suffix(f".{ext}")
        fig.savefig(file_path, dpi=180)
        print(f"Saved: {file_path}")
    plt.close(fig)


def plot_global_egop_old(result: dict, *, save_stem: Path) -> None:
    """Optional legacy four-panel diagnostics (not invoked by main)."""
    Lambda = result["Lambda"].cpu().numpy()
    beta = result["beta"].cpu().numpy()
    ipr = result["ipr_normalized"].cpu().numpy()
    V = result["V"]
    layout = result["parameter_slices"]
    num_samples = result["metadata"]["num_samples"]
    batch_size = result["metadata"]["batch_size"]

    # Exclude numerically zero eigenvalues: vectors in the nullspace may be an
    # arbitrary basis, especially when num_samples < parameter dimension.
    threshold = max(float(Lambda.max()) * 1e-12, 0.0)
    nonzero_count = int((Lambda > threshold).sum())
    ranks = np.arange(1, len(Lambda) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    if nonzero_count:
        ax.semilogy(ranks[:nonzero_count], Lambda[:nonzero_count], linewidth=1.5)
    ax.set(
        xlabel="Eigenvalue index (descending)",
        ylabel="Eigenvalue",
        title="Global EGOP eigenvalue spectrum",
    )
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    total = float(Lambda.sum())
    if total > 0:
        ax.plot(ranks, np.cumsum(Lambda) / total, linewidth=1.6)
    ax.set(
        xlabel="Top k eigenvectors",
        ylabel="Fraction of EGOP trace",
        ylim=(-0.02, 1.02),
        title="Cumulative spectral mass",
    )
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    n_plot = min(nonzero_count, max(200, TOP_EIGENVECTORS_TO_PLOT))
    if n_plot:
        ax.plot(ranks[:n_plot], beta[:n_plot], label=r"$\beta_k=\|v_k\|_1^2/d$", lw=1.4)
        ax.plot(
            ranks[:n_plot], ipr[:n_plot], label="Normalized participation ratio", lw=1.4
        )
    ax.set(
        xlabel="Eigenvector index (descending eigenvalue)",
        ylabel="Normalized density",
        title="Eigenvector coordinate density",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    top_k = min(nonzero_count, TOP_EIGENVECTORS_TO_PLOT)
    if top_k:
        energies = np.stack(
            [
                V[entry["start"] : entry["end"], :top_k]
                .square()
                .sum(dim=0)
                .cpu()
                .numpy()
                for entry in layout
            ]
        )
        heatmap = ax.imshow(
            energies,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(
            heatmap, ax=ax, fraction=0.047, pad=0.04, label="Squared eigenvector norm"
        )
        ax.set_yticks(
            np.arange(len(layout)), labels=[entry["name"] for entry in layout]
        )
        ax.set(
            xlabel="Top eigenvector index (zero based)",
            title="Eigenvector energy by parameter tensor",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No nonzero eigenvalues",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()

    d = len(Lambda)
    fig.suptitle(
        f"TinyMNIST global EGOP | d={d}, gradient samples={num_samples}, batch={batch_size}",
        fontsize=14,
    )
    save_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        file_path = save_stem.with_suffix(f".{ext}")
        fig.savefig(file_path, dpi=180)
        print(f"Saved: {file_path}")
    plt.close(fig)


def main() -> None:
    if NUM_SAMPLES < 1:
        raise ValueError("NUM_SAMPLES must be positive")
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be positive")
    if CHUNK_SIZE < 1:
        raise ValueError("CHUNK_SIZE must be positive")

    mode = "all_params" if INCLUDE_BIASES else "weights_only"
    run_name = f"samples_{NUM_SAMPLES}_batch_{BATCH_SIZE}_seed_{SEED}_{mode}"
    if FULL_BATCH_GRADIENTS:
        run_name += (
            "_full_batch"  # Keep separate from previously saved mini-batch EGOPs.
        )
    output_dir = EXPERIMENTS_DIR / "logs" / "tinyMNIST_global_EGOP" / run_name
    matrix_path = output_dir / "global_EGOP.pt"
    spectrum_figure_stem = (
        EXPERIMENTS_DIR
        / "figures"
        / datetime.now().strftime("%b-%d-%Y")
        / f"tinyMNIST_global_EGOP_{run_name}_spectrum"
    )
    betas_figure_stem = (
        EXPERIMENTS_DIR
        / "figures"
        / datetime.now().strftime("%b-%d-%Y")
        / f"tinyMNIST_global_EGOP_{run_name}_betas"
    )

    if PLOT_ONLY:
        if not matrix_path.exists():
            raise FileNotFoundError(f"Saved EGOP not found: {matrix_path}")
        result = torch.load(matrix_path, map_location="cpu", weights_only=True)
    else:
        # Import the repository-specific model and data only when estimating.
        from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader
        from egop_optimizer.models.TinyMNISTClassifier import TinyMNISTClassifier
        from egop_optimizer.utils.device_utils import get_available_device

        seed_everything(DATA_SPLIT_SEED)
        trainloader, _, _ = tinyMNIST_dataloader(
            batch_size=BATCH_SIZE,
            train_val_test_split=TRAIN_VAL_TEST_SPLIT,
        )
        seed_everything(SEED)
        model = TinyMNISTClassifier(
            input_size=INPUT_SIZE,
            hidden_sizes=HIDDEN_SIZES,
            num_classes=NUM_CLASSES,
            seed=SEED,
        )
        device = get_available_device()
        print(f"Computing global EGOP on {device}; dense eigendecomposition on CPU")
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        # Both matrices are accessible directly as Lambda and V at this point.
        Lambda, V, EGOP, parameter_slices = estimate_global_egop(
            model,
            trainloader,
            criterion,
            num_samples=NUM_SAMPLES,
            chunk_size=CHUNK_SIZE,
            device=device,
            include_biases=INCLUDE_BIASES,
            full_batch_gradients=FULL_BATCH_GRADIENTS,
        )
        print("Numerical rank:", torch.linalg.matrix_rank(EGOP))
        beta, ipr_normalized = eigenvector_diagnostics(V)
        result = {
            "Lambda": Lambda,
            "V": V,
            "EGOP": EGOP,
            "beta": beta,
            "ipr_normalized": ipr_normalized,
            "parameter_slices": parameter_slices,
            "metadata": {
                "num_samples": NUM_SAMPLES,
                "chunk_size": CHUNK_SIZE,
                "batch_size": BATCH_SIZE,
                "full_batch_gradients": FULL_BATCH_GRADIENTS,
                "seed": SEED,
                "data_split_seed": DATA_SPLIT_SEED,
                "include_biases": INCLUDE_BIASES,
                "input_size": INPUT_SIZE,
                "hidden_sizes": HIDDEN_SIZES,
                "num_classes": NUM_CLASSES,
                "train_val_test_split": TRAIN_VAL_TEST_SPLIT,
                "sampling": (
                    "fresh random initialization and mean-loss full training-set gradient per draw"
                    if FULL_BATCH_GRADIENTS
                    else "fresh random initialization and mean-loss mini-batch gradient per draw"
                ),
                "parameter_order": "model.named_parameters() traversal, with selected biases",
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(result, matrix_path)
        (output_dir / "configuration.json").write_text(
            json.dumps(result["metadata"], indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved EGOP, Lambda, V, and diagnostics: {matrix_path}")

    Lambda, V = result["Lambda"], result["V"]
    print(f"Lambda.shape = {tuple(Lambda.shape)}, V.shape = {tuple(V.shape)}")
    print("Top 10 eigenvalues:", Lambda[:10].tolist())
    print("Top 10 beta values:", result["beta"][:10].tolist())
    plot_global_egop_spectrum(result, save_stem=spectrum_figure_stem)
    plot_global_egop_betas(result, save_stem=betas_figure_stem)


if __name__ == "__main__":
    main()
