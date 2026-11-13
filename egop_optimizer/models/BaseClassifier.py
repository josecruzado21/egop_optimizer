import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import _calculate_fan_in_and_fan_out
from typing import Optional, Union, Tuple
from contextlib import contextmanager
import math


from egop_optimizer.models.reparam_layers.reparam_layers import (
    EGOP_linear_layer,
    EGOP_auxiliary_variables_linear_layer,
    EGOP_conv2d_layer,
    ResBlock,
)


@contextmanager
def _sync_device_rng(device: torch.device, gen: torch.Generator):
    # Swap the device's global RNG state with `gen`'s state so that init ops
    # which don't accept a generator (e.g. nn.Module.reset_parameters) draw
    # deterministically from our tracked stream. Write the advanced state back
    # into `gen` on exit and restore the device's prior global state so we don't
    # perturb other code sharing the same device RNG.
    if device.type == "cuda":
        saved = torch.cuda.get_rng_state(device)
        torch.cuda.set_rng_state(gen.get_state(), device)
        try:
            yield
        finally:
            gen.set_state(torch.cuda.get_rng_state(device))
            torch.cuda.set_rng_state(saved, device)
    elif device.type == "mps":
        saved = torch.mps.get_rng_state()
        torch.mps.set_rng_state(gen.get_state())
        try:
            yield
        finally:
            gen.set_state(torch.mps.get_rng_state())
            torch.mps.set_rng_state(saved)
    elif device.type == "cpu":
        saved = torch.get_rng_state()
        torch.set_rng_state(gen.get_state())
        try:
            yield
        finally:
            gen.set_state(torch.get_rng_state())
            torch.set_rng_state(saved)
    else:
        raise NotImplementedError(
            f"Seeded reinitialize is only implemented for cpu/cuda/mps, got device {device}."
        )


REPARAM_LAYER_CLASSES = [
    EGOP_linear_layer,
    EGOP_auxiliary_variables_linear_layer,
    EGOP_conv2d_layer,
    ResBlock,
]

"""
TODO with ReparamBaseClassifier:
-
"""

"""
TODO with BaseClassifier:
X Unify logic for initializing random parameters
X Reinitialize and reinitialize_seeded(accepts explicit random seed)

Not in first round:
- Accept "current weights" for periodic reparam logic
"""


class BaseClassifier(nn.Module):
    """
    A base class for classifiers that require standard parameter initialization.

    Args:
        seed (int, optional): Defines the pytorch random seed for parameter initialization

    Returns:
    """

    def __init__(self, seed=0, weight_dist="default", **kwargs):
        super().__init__()
        # weight_dist options: "default", "xavier_normal", "gaussian", "kaiming_normal"
        self.weight_dist = weight_dist
        self._seed = seed
        # Lazy per-device generators so seeded init is deterministic on cpu/mps/cuda.
        # A plain torch.Generator() lives on CPU and can't seed init ops run on
        # device weights, so we key generators by (device_type, device_index).
        self._gens: dict = {}

    def _generator_for(self, device: torch.device) -> torch.Generator:
        key = (device.type, device.index)
        gen = self._gens.get(key)
        if gen is None:
            gen = torch.Generator(device=device)
            if self._seed is not None:
                gen.manual_seed(self._seed)
            self._gens[key] = gen
        return gen

    def reinitialize_seeded(self, seed, **kwargs):
        # Reset all known per-device generators to the same seed and remember it
        # so generators created later (e.g. if a layer moves to a new device)
        # start from the same seed.
        self._seed = seed
        for gen in self._gens.values():
            gen.manual_seed(seed)

        # then initialize params
        self.reinitialize(**kwargs)
        return

    def reinitialize(
        self,
        verbose: bool = False,
        sampling_scale: Optional[float] = None,
        mean: Optional[float] = None,
    ):
        """
        Initializes weights using PyTorch's default reset method or Gaussian IID.

        Args:
            verbose (bool): If True, print layer reset info.
            sampling_scale (Optional[float]): Standard deviation for Gaussian initialization.
                                             Must be None for non-Gaussian initialization.
            mean (Optional[float]): Mean for Gaussian initialization.
                                   Must be None for non-Gaussian initialization.
        """
        if self.weight_dist != "gaussian" and (
            sampling_scale is not None or mean is not None
        ):
            raise Exception(
                "sampling_scale and mean should only be provided for Gaussian initialization"
            )

        # Set default values for Gaussian initialization if not provided
        if self.weight_dist == "gaussian":
            if sampling_scale is None:
                sampling_scale = 1.0
            if mean is None:
                mean = 0.0

        for layer in self.modules():
            # If layer has a weight parameter,
            if hasattr(layer, "weight") and layer.weight is not None:
                if verbose:
                    print(f"Initializing layer with {self.weight_dist} distribution")

                device = layer.weight.device
                gen = self._generator_for(device)

                # 1D weights belong to normalization layers (BatchNorm/LayerNorm
                # affine scales). Kaiming/Xavier need >=2D, and a Gaussian draw
                # would clobber the unit-scale init those layers expect, so defer
                # to the layer's own reset_parameters for any weight_dist.
                if layer.weight.dim() < 2:
                    with _sync_device_rng(device, gen):
                        layer.reset_parameters()
                    continue

                if self.weight_dist == "gaussian":
                    # Gaussian IID initialization
                    with torch.no_grad():
                        layer.weight.normal_(
                            mean=mean, std=sampling_scale, generator=gen
                        )
                elif self.weight_dist == "xavier_normal" and isinstance(
                    layer, torch.nn.modules.linear.Linear
                ):
                    centered_xavier_normal_(
                        layer.weight,
                        mean=torch.zeros_like(layer.weight),
                        generator=gen,
                    )
                elif self.weight_dist == "kaiming_normal":
                    nn.init.kaiming_normal_(
                        layer.weight,
                        mode="fan_in",
                        nonlinearity="relu",
                        generator=gen,
                    )
                    if hasattr(layer, "bias") and layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                else:
                    # reset_parameters() doesn't accept a generator, so temporarily
                    # drive the device's global RNG from our tracked generator.
                    with _sync_device_rng(device, gen):
                        layer.reset_parameters()

                # Give the layer a chance to sync any secondary parameters that
                # depend on self.weight (e.g. EGOP_auxiliary_variables_linear_layer
                # derives weight_d / weight_r from self.weight via V).
                # Most layers don't define this method and silently skip.
                if hasattr(layer, "_post_weight_init_hook"):
                    layer._post_weight_init_hook()


class ReparamBaseClassifier(BaseClassifier):
    def __init__(self, V_by_layer_dict, **kwargs):
        super().__init__(**kwargs)
        self.V_by_layer_dict = V_by_layer_dict

    def move_bases_to_device(self, device):
        for idx in self.V_by_layer_dict.keys():
            self.V_by_layer_dict[idx] = self.V_by_layer_dict[idx].to(device)
        for _, module in self.named_modules():
            if type(module) in (
                EGOP_linear_layer,
                EGOP_auxiliary_variables_linear_layer,
                EGOP_conv2d_layer,
            ):
                module.V = module.V.to(device)
        return

    def update_reparam_weights(self, V_prev_dict, V_new_dict):
        """
        Transitions all EGOP layers from V_prev to V_new basis.

        For each layer in both dicts:
          - Recovers the effective weight in original (un-reparameterized) coordinates
            using V_prev.
          - Projects it into the new reparameterized coordinates using V_new.T.
          - Updates the layer's stored weight and its .V (and .V_inv for linear) in place.

        The noise that diversifies the new V comes from the kaiming perturbations
        applied inside compute_V_by_layer(recalculate_V=True) before this is called.
        """
        with torch.no_grad():
            for name, module in self.named_modules():
                if name not in V_prev_dict or name not in V_new_dict:
                    continue
                V_prev = V_prev_dict[name].to(module.weight.device)
                V_new = V_new_dict[name].to(module.weight.device)
                W = module.weight.data

                if isinstance(module, EGOP_conv2d_layer):
                    # W: [out_ch, in_ch, kH, kW]
                    # Per filter b: W_eff[b] = V_prev @ W_stored[b]
                    #               W_new_stored[b] = V_new.T @ W_eff[b]
                    W_flat = W.view(W.shape[0], -1).unsqueeze(-1)       # [out_ch, D, 1]
                    W_orig = torch.matmul(V_prev, W_flat).view_as(W)    # [out_ch, in_ch, kH, kW]
                    W_new_stored = torch.matmul(
                        V_new.T, W_orig.view(W.shape[0], -1).unsqueeze(-1)
                    ).view_as(W)
                    module.weight.data.copy_(W_new_stored)
                    module.V = V_new

                elif isinstance(module, EGOP_linear_layer):
                    # W: [out_features, in_features], V: [D, D] where D = out*in
                    # W_eff_flat = V_prev @ W_flat
                    # W_new_flat = V_new.T @ W_eff_flat
                    W_flat = W.flatten()
                    W_orig_flat = torch.matmul(V_prev, W_flat)
                    W_new_flat = torch.matmul(V_new.T, W_orig_flat)
                    module.weight.data.copy_(W_new_flat.view_as(W))
                    module.V = V_new
                    module.V_inv = V_new.T

        self.V_by_layer_dict = V_new_dict

    def restore_V_from_dict(self, V_dict):
        """
        Restores .V (and .V_inv) on each EGOP layer from V_dict without touching weights.
        Use after loading a checkpoint so that layer bases match what was saved.
        """
        with torch.no_grad():
            for name, module in self.named_modules():
                if name not in V_dict:
                    continue
                V = V_dict[name].to(module.weight.device)
                if isinstance(module, EGOP_conv2d_layer):
                    module.V = V
                elif isinstance(module, EGOP_linear_layer):
                    module.V = V
                    module.V_inv = V.T
        self.V_by_layer_dict = V_dict


def centered_xavier_normal_(
    tensor: Tensor,
    mean: Tensor,
    gain: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Based on torch.nn.init.xavier_normal_
    https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/init.py#L72
    Added option of a nonzero mean.

    Fill the input `Tensor` with values using a Xavier normal distribution.

    The method is described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010). The resulting tensor
    will have values sampled from :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the tensor mean of the normal distribution. Added to a mean-zero tensor of Xavier sampled entries.
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _centered_no_grad_normal_(tensor, mean, std, generator)


def _centered_no_grad_normal_(tensor, mean, std, generator=None):
    """
    Adds mean to a tensor of iid N(0, std) drawn entries
    """
    with torch.no_grad():
        # first draw mean-zero iid entries
        tensor.normal_(mean=0.0, std=std, generator=generator)
        # then center at desired mean
        tensor = tensor.add(mean)
        return tensor
