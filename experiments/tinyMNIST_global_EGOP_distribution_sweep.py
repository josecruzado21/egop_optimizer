"""Compare ten global EGOP spectra across parameter sampling distributions.

Place this file and tinyMNIST_global_viz.py inside the repository's
``experiments/`` directory. Run from any working directory with:

    python experiments/tinyMNIST_global_EGOP_distribution_sweep.py

Set PLOT_ONLY=True to regenerate comparison figures without gradient sampling.
Complete distribution runs are automatically skipped unless OVERWRITE=True.

Each global gradient is the *full training-set* mean-loss gradient evaluated at
an independently sampled parameter vector. The same training split is used
for all ten distributions.

Distributions:
  - Default (existing nn.Linear.reset_parameters initialization)
  - N(0, sigma^2 I_d), sigma in (10, 1, 0.1)
  - Uniform ball B_d(0, sigma sqrt(d)), sigma in (10, 1, 0.1)
  - Covariance-matched multivariate Student t, nu=10 and sigma in (10, 1, 0.1)

Two comparison figures (PDF and PNG) are saved in experiments/figures/<date>/:
  - full normalized eigenvalue spectrum, log y-scale, [1e-13, 2]
  - leading 100 normalized eigenvalues, linear y-scale

For disk efficiency, results save eigenvalues and eigenvectors rather than an
additional duplicate d x d EGOP. Set SAVE_FULL_EGOP=True to save both.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch

# Importing the shared module also imports the existing CMU Serif font setup.
import tinyMNIST_global_EGOP as viz

# ── Experiment configuration (edit here; no argparse) ────────────────────
NUM_SAMPLES = 20000
CHUNK_SIZE = 2000
BATCH_SIZE = 300  # Memory microbatch size for full-dataset gradients
SEED = 0
DATA_SPLIT_SEED = 0
PLOT_ONLY = False
OVERWRITE = False
SAVE_FULL_EGOP = False  # Off by default to reduce disk requirements

INPUT_SIZE = 64
HIDDEN_SIZES = [32]
NUM_CLASSES = 10
INCLUDE_BIASES = True
TRAIN_VAL_TEST_SPLIT = [None, 0.33, 0.66]

LEADING_EIGENVALUES = 100
STUDENT_T_DF = 10  # Student-t degrees of freedom; must be an integer > 2
LEGEND_FONTSIZE = viz.INSET_FONTSIZE  # Matches existing 30-point inset labels
# Markers are staggered across the ten spectra, not the data themselves.
# Keep markers sparse on the full spectrum so nearby curves remain legible.
MARKERS_PER_CURVE_FULL = 12
MARKERS_PER_CURVE_LEADING = 8
MARKERSIZE_FULL = 11.0
MARKERSIZE_LEADING = 12.0
# ────────────────────────────────────────────────────────────────────────

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RUN_NAME = (
    f"d{INPUT_SIZE}_{'-'.join(map(str, HIDDEN_SIZES))}_{NUM_CLASSES}"
    f"_samples_{NUM_SAMPLES}_batch_{BATCH_SIZE}"
    f"_seed_{SEED}_split_{DATA_SPLIT_SEED}"
    f"_{'with_biases' if INCLUDE_BIASES else 'weights_only'}_full_batch"
)
LOG_ROOT = (
    EXPERIMENTS_DIR / "logs" / "tinyMNIST_global_EGOP_distribution_sweep" / RUN_NAME
)
FIGURE_ROOT = EXPERIMENTS_DIR / "figures" / datetime.now().strftime("%b-%d-%Y")

# One color per scale, one marker/linestyle per distribution family.
# This makes the legend interpretable even without distinguishing subtle hues.
COLORS_BY_SCALE = {10.0: "tab:blue", 1.0: "tab:orange", 0.1: "tab:green"}
SCENARIOS = (
    {
        "name": "default",
        "family": "default",
        "scale": None,
        "label": "Default initialization",
        "color": "black",
        "marker": "D",
        "linestyle": "-",
    },
    *(
        {
            "name": f"gaussian_std_{scale:g}",
            "family": "gaussian",
            "scale": scale,
            "label": rf"Gaussian ($\sigma={scale:g}$)",
            "color": COLORS_BY_SCALE[scale],
            "marker": "o",
            "linestyle": "-",
        }
        for scale in (10.0, 1.0, 0.1)
    ),
    *(
        {
            "name": f"uniform_ball_scale_{scale:g}",
            "family": "uniform_ball",
            "scale": scale,
            "label": rf"Uniform ball ($R={scale:g}\sqrt{{d}}$)",
            "color": COLORS_BY_SCALE[scale],
            "marker": "s",
            "linestyle": "--",
        }
        for scale in (10.0, 1.0, 0.1)
    ),
    *(
        {
            "name": f"student_t_nu_{STUDENT_T_DF}_std_{scale:g}",
            "family": "student_t",
            "scale": scale,
            "nu": STUDENT_T_DF,
            "label": rf"Student $t$ ($\nu={STUDENT_T_DF}$, $\sigma={scale:g}$)",
            "color": COLORS_BY_SCALE[scale],
            "marker": "^",
            "linestyle": "-.",
        }
        for scale in (10.0, 1.0, 0.1)
    ),
)


def result_path(spec: dict) -> Path:
    """Storage is scoped to experiment configuration and distribution."""
    return LOG_ROOT / spec["name"] / "global_EGOP.pt"


@torch.no_grad()
def sample_uniform_ball(model: torch.nn.Module, *, scale: float) -> None:
    """Draw the full concatenated parameter vector uniformly from B(0, scale*sqrt(d)).

    Uses the same per-device seeded generator as BaseClassifier.reinitialize.
    All trainable parameter tensors, including biases, are sampled JOINTLY.
    Independent per-layer ball samples would not be uniform in the d-ball.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Uniform-ball scale must be positive and finite")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Model has no trainable parameters")
    device, dtype = params[0].device, params[0].dtype
    if any(p.device != device or p.dtype != dtype for p in params):
        raise ValueError(
            "Uniform-ball sampler expects parameters on one device and dtype"
        )

    d = sum(p.numel() for p in params)
    generator = model._generator_for(device)
    z = torch.randn(d, device=device, dtype=dtype, generator=generator)
    u = torch.rand((), device=device, dtype=dtype, generator=generator)
    radius = (scale * math.sqrt(d)) * u.pow(1.0 / d)
    theta = (radius / z.norm()) * z

    offset = 0
    for param in params:
        width = param.numel()
        param.copy_(theta[offset : offset + width].reshape_as(param))
        offset += width


@torch.no_grad()
def sample_student_t(model: torch.nn.Module, *, scale: float, nu: int = 10) -> None:
    """Sample the *joint* parameter vector from isotropic Student t.

    theta = scale * sqrt((nu - 2) / U) * Z,
    Z ~ N(0, I_d), U ~ chi-square(nu), independent.

    With nu > 2 the covariance is scale**2 * I_d, so 'scale' denotes
    marginal standard deviation, as in the Gaussian scenarios. A single U
    is shared across all coordinates to preserve rotational invariance.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Student-t scale must be positive and finite")
    if not isinstance(nu, int) or nu <= 2:
        raise ValueError("Student-t degrees of freedom must be an integer > 2")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Model has no trainable parameters")
    device, dtype = params[0].device, params[0].dtype
    if any(p.device != device or p.dtype != dtype for p in params):
        raise ValueError("Student-t sampler expects parameters on one device and dtype")

    d = sum(p.numel() for p in params)
    generator = model._generator_for(device)
    z = torch.randn(d, device=device, dtype=dtype, generator=generator)
    chi_square = (
        torch.randn(nu, device=device, dtype=dtype, generator=generator).square().sum()
    )
    theta = scale * math.sqrt(nu - 2) * z / chi_square.sqrt()

    offset = 0
    for param in params:
        width = param.numel()
        param.copy_(theta[offset : offset + width].reshape_as(param))
        offset += width


def make_sampler(spec: dict):
    """Return a per-draw parameter sampler for estimate_global_egop."""
    if spec["family"] == "default":
        return lambda model: model.reinitialize()
    if spec["family"] == "gaussian":
        scale = spec["scale"]
        return lambda model: model.reinitialize(sampling_scale=scale)
    if spec["family"] == "uniform_ball":
        return partial(sample_uniform_ball, scale=spec["scale"])
    if spec["family"] == "student_t":
        return partial(sample_student_t, scale=spec["scale"], nu=spec["nu"])
    raise ValueError(f"Unknown distribution family: {spec['family']}")


def expected_metadata(spec: dict, *, d: int) -> dict:
    """Record sampler settings, data split, and global gradient dimension."""
    return {
        "name": spec["name"],
        "parameter_distribution": spec["family"],
        "scale": spec["scale"],
        "gaussian_std": spec["scale"] if spec["family"] == "gaussian" else None,
        "student_t_df": spec.get("nu") if spec["family"] == "student_t" else None,
        "student_t_covariance_std": (
            spec["scale"] if spec["family"] == "student_t" else None
        ),
        "ball_radius": (
            spec["scale"] * math.sqrt(d) if spec["family"] == "uniform_ball" else None
        ),
        "parameter_dimension": d,
        "num_samples": NUM_SAMPLES,
        "chunk_size": CHUNK_SIZE,
        "batch_size": BATCH_SIZE,
        "full_batch_gradients": True,
        "seed": SEED,
        "data_split_seed": DATA_SPLIT_SEED,
        "include_biases": INCLUDE_BIASES,
        "input_size": INPUT_SIZE,
        "hidden_sizes": HIDDEN_SIZES,
        "num_classes": NUM_CLASSES,
        "train_val_test_split": TRAIN_VAL_TEST_SPLIT,
        "sampling": "independent parameter draws, full training-split mean-loss gradient",
        "parameter_order": "model.named_parameters() traversal",
    }


def save_result(spec: dict, result: dict) -> None:
    """Write one distribution result atomically; omit EGOP duplicate by default."""
    path = result_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = {k: v for k, v in result.items() if SAVE_FULL_EGOP or k != "EGOP"}
    partial_path = path.with_suffix(".partial.pt")
    try:
        torch.save(subset, partial_path)
        partial_path.replace(path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise
    (path.parent / "configuration.json").write_text(
        json.dumps(result["metadata"], indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {spec['label']}: {path}", flush=True)


def load_completed_spectra() -> dict[str, np.ndarray]:
    """Load available completed runs; keep only eigenvalues in plotting memory."""
    spectra = {}
    for spec in SCENARIOS:
        path = result_path(spec)
        if not path.is_file():
            continue
        try:
            result = torch.load(path, map_location="cpu", weights_only=True)
            metadata = result.get("metadata", {})
            if (
                metadata.get("name") != spec["name"]
                or metadata.get("num_samples") != NUM_SAMPLES
            ):
                print(f"Ignoring incompatible saved result: {path}")
                continue
            lam = result["Lambda"].detach().cpu().numpy().copy()
            spectra[spec["name"]] = lam
        except Exception as exc:
            print(f"Could not load {path}: {exc}")
    return spectra


def plot_spectra(spectra: dict[str, np.ndarray], *, leading_only: bool) -> None:
    """Create one overlay plot using the exact font sizes from global_viz."""
    if not spectra:
        print("No completed spectra available; skipping comparison figure.")
        return

    # Match the original 13x9 spectrum figure. Adjust the axes explicitly so
    # the outside legend does not squeeze the plotting region.
    fig, ax = viz.plt.subplots(figsize=(13, 9))
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.19, top=0.96)
    for scenario_index, spec in enumerate(SCENARIOS):
        if spec["name"] not in spectra:
            continue
        lam = spectra[spec["name"]]
        if lam.size == 0 or not np.isfinite(lam[0]) or lam[0] <= 0:
            print(
                f"Skipping {spec['name']}: nonpositive or nonfinite leading eigenvalue"
            )
            continue
        normalized = lam / lam[0]
        n = (
            min(LEADING_EIGENVALUES, len(normalized))
            if leading_only
            else len(normalized)
        )
        k = np.arange(1, n + 1)
        y = normalized[:n]
        mask = np.isfinite(y) & ((y >= 0) if leading_only else (y > 0))
        # Draw the lines at their true eigenvalue indices, but stagger marker
        # positions between distributions. Each curve gets a distinct phase in
        # a common stride, avoiding overlaid symbols on near-identical spectra.
        markers_per_curve = (
            MARKERS_PER_CURVE_LEADING if leading_only else MARKERS_PER_CURVE_FULL
        )
        marker_stride = max(len(SCENARIOS), math.ceil(n / markers_per_curve))
        marker_offset = int((scenario_index + 0.5) * marker_stride / len(SCENARIOS))
        markevery = (marker_offset, marker_stride)
        marker_size = MARKERSIZE_LEADING if leading_only else MARKERSIZE_FULL
        ax.plot(
            k[mask],
            y[mask],
            color=spec["color"],
            linestyle=spec["linestyle"],
            marker=spec["marker"],
            markersize=marker_size,
            markevery=markevery,
            linewidth=1.7,
            label=spec["label"],
        )

    if leading_only:
        ax.set_yscale("linear")
        ax.set_xlim(0, LEADING_EIGENVALUES + 1)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Leading 100 indices", fontsize=viz.AXIS_LABEL_FONTSIZE)
        suffix = "leading_100_linear"
    else:
        ax.set_yscale("log")
        ax.set_ylim(1e-13, 2)
        ax.set_xlabel(r"Eigenvalue index $k$", fontsize=viz.AXIS_LABEL_FONTSIZE)
        suffix = "all_log"

    ax.set_ylabel(
        r"Normalized eigenvalue $\lambda_k/\lambda_1$", fontsize=viz.AXIS_LABEL_FONTSIZE
    )
    ax.tick_params(axis="both", which="major", labelsize=viz.TICK_FONTSIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.15)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
    )

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = (
            FIGURE_ROOT
            / f"tinyMNIST_global_EGOP_distribution_sweep_{suffix}.{extension}"
        )
        fig.savefig(path, bbox_inches="tight", dpi=180)
        print(f"Saved figure: {path}")
    viz.plt.close(fig)


def run_sweep() -> None:
    """Estimate each of the ten full-batch EGOPs, or skip completed runs."""
    if PLOT_ONLY:
        print("PLOT_ONLY=True; loading saved results rather than sampling gradients.")
        return

    from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader
    from egop_optimizer.models.TinyMNISTClassifier import TinyMNISTClassifier
    from egop_optimizer.utils.device_utils import get_available_device

    viz.seed_everything(DATA_SPLIT_SEED)
    trainloader, _, _ = tinyMNIST_dataloader(
        batch_size=BATCH_SIZE,
        train_val_test_split=TRAIN_VAL_TEST_SPLIT,
    )
    device = get_available_device()
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    for index, spec in enumerate(SCENARIOS, start=1):
        path = result_path(spec)
        if path.exists() and not OVERWRITE:
            try:
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    saved.get("metadata", {}).get("name") == spec["name"]
                    and saved.get("metadata", {}).get("num_samples") == NUM_SAMPLES
                ):
                    print(
                        f"[{index}/{len(SCENARIOS)}] Skipping completed {spec['label']}: {path}"
                    )
                    del saved
                    continue
                del saved
            except Exception as exc:
                print(f"Saved result unreadable; recomputing {spec['name']}: {exc}")

        viz.seed_everything(SEED)
        model = TinyMNISTClassifier(
            input_size=INPUT_SIZE,
            hidden_sizes=HIDDEN_SIZES,
            num_classes=NUM_CLASSES,
            bias=INCLUDE_BIASES,
            seed=SEED,
            weight_dist="gaussian" if spec["family"] == "gaussian" else "default",
        )
        d = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[{index}/{len(SCENARIOS)}] Estimating {spec['label']}, d={d}, N={NUM_SAMPLES}",
            flush=True,
        )
        Lambda, V, EGOP, parameter_slices = viz.estimate_global_egop(
            model,
            trainloader,
            criterion,
            num_samples=NUM_SAMPLES,
            chunk_size=CHUNK_SIZE,
            device=device,
            include_biases=INCLUDE_BIASES,
            full_batch_gradients=True,
            parameter_sampler=make_sampler(spec),
        )
        beta, ipr_normalized = viz.eigenvector_diagnostics(V)
        result = {
            "Lambda": Lambda,
            "V": V,
            "beta": beta,
            "ipr_normalized": ipr_normalized,
            "parameter_slices": parameter_slices,
            "metadata": expected_metadata(spec, d=d),
        }
        if SAVE_FULL_EGOP:
            result["EGOP"] = EGOP
        save_result(spec, result)
        del Lambda, V, EGOP, beta, ipr_normalized, result, model


def main() -> None:
    if STUDENT_T_DF <= 2 or not isinstance(STUDENT_T_DF, int):
        raise ValueError("STUDENT_T_DF must be an integer > 2")
    if NUM_SAMPLES < 1 or CHUNK_SIZE < 1 or BATCH_SIZE < 1:
        raise ValueError("NUM_SAMPLES, CHUNK_SIZE and BATCH_SIZE must be positive")
    if not INCLUDE_BIASES:
        raise ValueError(
            "Set INCLUDE_BIASES=True for the full d-dimensional distribution comparison"
        )

    try:
        run_sweep()
    finally:
        # Also generate a partial comparison if the long experiment is interrupted.
        spectra = load_completed_spectra()
        print(f"Available completed distributions: {len(spectra)}/{len(SCENARIOS)}")
        plot_spectra(spectra, leading_only=False)
        plot_spectra(spectra, leading_only=True)


if __name__ == "__main__":
    main()
