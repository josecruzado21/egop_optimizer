import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from egop_optimizer.models.BaseClassifier import BaseClassifier, ReparamBaseClassifier
from egop_optimizer.models.reparam_layers.reparam_layers import (
    EGOP_auxiliary_variables_linear_layer,
)

import pdb

class FashionMNISTClassifier(BaseClassifier):
    """
    A simple feedforward neural network for classifying FashionMNIST images.

    This classifier includes an optional pooling layer before flattening the input,
    followed by two fully connected layers. The first layer applies a ReLU activation,
    and the second outputs class logits.

    Args:
        pool_factor (Optional[int]): If provided, applies MaxPooling with the given factor
            to downsample the input. If `None`, no pooling is applied.
        hidden_size (int): Number of hidden units in the first fully connected layer.
        num_classes (int): Number of output classes. Defaults to 10 for FashionMNIST.
    """

    def __init__(
        self,
        pool_factor: Optional[int] = None,
        hidden_size: int = 128,
        num_classes: int = 10,
        weight_dist: str = "default",  #new test on different distributions
        seed=None,
    ):
        super(FashionMNISTClassifier, self).__init__(seed=seed, weight_dist=weight_dist)
        if pool_factor is None:
            self.optional_pool = nn.Identity()
            fc_input_size = 28**2
        else:
            self.optional_pool = nn.MaxPool2d(pool_factor, pool_factor)
            fc_input_size = int(28 / pool_factor) ** 2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        x = self.optional_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AuxiliaryReparamFashionMNISTClassifier(ReparamBaseClassifier):
    """
    A FashionMNIST classifier that uses EGOP auxiliary variables reparameterization.

    Uses EGOP_auxiliary_variables_linear_layer which maintains the full d-dimensional
    parameter space by introducing auxiliary variables in the orthogonal complement
    to range(V). The forward pass computes:
        W = weight_d + V @ (weight_r - V^T @ weight_d)

    Args:
        V_by_layer_dict: Dictionary mapping layer names (e.g. "fc1", "fc2") to d x r
            eigenbasis matrices.
        pool_factor (Optional[int]): If provided, applies MaxPooling to downsample input.
        hidden_size (int): Number of hidden units in the first fully connected layer.
        num_classes (int): Number of output classes.
    """

    def __init__(
        self,
        V_by_layer_dict,
        pool_factor: Optional[int] = None,
        hidden_size: int = 128,
        num_classes: int = 10,
        seed=None,
    ):
        super().__init__(V_by_layer_dict=V_by_layer_dict, seed=seed)
        if pool_factor is None:
            self.optional_pool = nn.Identity()
            fc_input_size = 28**2
        else:
            self.optional_pool = nn.MaxPool2d(pool_factor, pool_factor)
            fc_input_size = int(28 / pool_factor) ** 2
        self.flatten = nn.Flatten()
        self.fc1 = EGOP_auxiliary_variables_linear_layer(
            V=V_by_layer_dict["fc1"],
            in_features=fc_input_size,
            out_features=hidden_size,
            bias=True,
        )
        self.fc2 = EGOP_auxiliary_variables_linear_layer(
            V=V_by_layer_dict["fc2"],
            in_features=hidden_size,
            out_features=num_classes,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.optional_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x