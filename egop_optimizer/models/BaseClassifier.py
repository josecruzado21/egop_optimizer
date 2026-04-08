import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import _calculate_fan_in_and_fan_out
from typing import Optional, Union, Tuple
import math


from egop_optimizer.models.reparam_layers.reparam_layers import (
    EGOP_linear_layer,
    EGOP_auxiliary_variables_linear_layer,
    Conv2d_reparam,
    ResBlock,
)

REPARAM_LAYER_CLASSES = [EGOP_linear_layer, EGOP_auxiliary_variables_linear_layer, Conv2d_reparam, ResBlock]

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
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

    def reinitialize_seeded(self, seed, **kwargs):
        # reset random number generator
        self._gen.manual_seed(seed)

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
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                if verbose:
                    print(f"Initializing layer with {self.weight_dist} distribution")

                if self.weight_dist == "gaussian":
                    # Gaussian IID initialization
                    with torch.no_grad():
                        layer.weight.normal_(
                            mean=mean, std=sampling_scale, generator=self._gen
                        )
                elif self.weight_dist == "xavier_normal" and isinstance(
                    layer, torch.nn.modules.linear.Linear
                ):
                    centered_xavier_normal_(
                        layer.weight,
                        mean=torch.zeros_like(layer.weight),
                        generator=self._gen,
                    )
                elif self.weight_dist == "kaiming_normal":
                    with torch.random.fork_rng(enabled=True):
                        torch.set_rng_state(self._gen.get_state())
                        if hasattr(layer, "weight") and layer.weight is not None:
                            nn.init.kaiming_normal_(
                                layer.weight, mode="fan_in", nonlinearity="relu"
                            )
                        if hasattr(layer, "bias") and layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
                        self._gen.set_state(torch.get_rng_state())
                else:
                    layer.reset_parameters()
                    # Adela note: I don't know what the below code is doing? Resetting again and getting the random state?
                    with torch.random.fork_rng(enabled=True):
                        torch.set_rng_state(self._gen.get_state())
                        layer.reset_parameters()
                        self._gen.set_state(torch.get_rng_state())


class ReparamBaseClassifier(BaseClassifier):
    def __init__(self, V_by_layer_dict, **kwargs):
        super().__init__(**kwargs)
        self.V_by_layer_dict = V_by_layer_dict

    def move_bases_to_device(self, device):
        for idx in self.V_by_layer_dict.keys():
            self.V_by_layer_dict[idx] = self.V_by_layer_dict[idx].to(device)
        for _, module in self.named_modules():
            if type(module) in (EGOP_linear_layer, EGOP_auxiliary_variables_linear_layer):
                module.V = module.V.to(device)
        return


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
