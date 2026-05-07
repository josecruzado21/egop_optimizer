"""AdamW optimizer factory with selective weight decay for auxiliary EGOP models.

The auxiliary layer (`EGOP_auxiliary_variables_linear_layer`) carries multiple
parameters per layer:
  - weight   : nn.Linear-style scratchpad (requires_grad=False — not in optimizer)
  - weight_d : 1D, length d=out*in
  - weight_r : 1D, length r
  - bias     : 1D, length out

Following the old repo's convention, weight decay is applied to weight_d and
weight_r, but NOT to bias (which would shift outputs unnecessarily). This
module wraps the per-group construction so experiment scripts don't have to
repeat it.

For non-auxiliary models (no weight_d / weight_r in named_parameters),
this falls back to a plain AdamW with uniform weight decay.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable


def _select_named(model: nn.Module, substr: str) -> Iterable[str]:
    return [n for n, p in model.named_parameters() if substr in n and p.requires_grad]


def create_auxiliary_adamw(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build AdamW with auxiliary-aware parameter groups.

    Args:
        model: The model to optimize. If it contains parameters whose names
            include "weight_d" and "weight_r", the optimizer applies selective
            weight decay (decay on weight_d / weight_r, no decay on bias).
        learning_rate: Single LR shared across all parameter groups.
        weight_decay: Weight decay applied to weight_d and weight_r groups.
            (Bias / "other" groups always get weight_decay=0.)

    Returns:
        torch.optim.AdamW configured with the appropriate parameter groups.
    """
    weight_d_names = _select_named(model, "weight_d")
    weight_r_names = _select_named(model, "weight_r")

    # No auxiliary structure: plain AdamW with uniform weight decay.
    if not weight_d_names or not weight_r_names:
        return torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    aux_param_names = set(weight_d_names) | set(weight_r_names)
    aux_param_names.update(n for n, p in model.named_parameters() if "bias" in n)

    weight_d_params = [p for n, p in model.named_parameters() if n in weight_d_names]
    weight_r_params = [p for n, p in model.named_parameters() if n in weight_r_names]
    bias_params = [
        p
        for n, p in model.named_parameters()
        if "bias" in n and p.requires_grad
    ]
    other_params = [
        p
        for n, p in model.named_parameters()
        if n not in aux_param_names and p.requires_grad
    ]

    return torch.optim.AdamW(
        [
            {"params": weight_d_params, "weight_decay": weight_decay},
            {"params": weight_r_params, "weight_decay": weight_decay},
            {"params": bias_params, "weight_decay": 0.0},
            {"params": other_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
