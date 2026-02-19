import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb

from egop_optimizer.models.BaseClassifier import BaseClassifier


class LinearFeedforward(BaseClassifier):
    """
    Implements a 3-layer linear feedforward network with Xavier initialization.

    Args:
        input_size (int): Input feature dimension.
        hidden_sizes (list): Two-element list specifying hidden layer widths.
        output_size (int): Output feature dimension.
        bias (bool): If True, includes bias terms in linear layers (default: False).

    Returns:
        None: Initializes the network layers and parameter bookkeeping.
    """

    def __init__(self, input_size, hidden_sizes, output_size, bias=False, seed=None):
        """
        Initializes the linear feedforward architecture and parameter count.

        Args:
            input_size (int): Input feature dimension.
            hidden_sizes (list): Two-element list specifying hidden layer widths.
            output_size (int): Output feature dimension.
            bias (bool): If True, includes bias terms in linear layers (default: False).

        Returns:
            None: Constructs the network layers and computes total parameter count.
        """

        # Experiments with linear networks used xavier initialization rather than pytorch's default
        # reset_parameters() for linear layers (uniform(-1/sqrt(in_features), 1/sqrt(in_features))).
        super().__init__(weight_dist="xavier_normal", seed=seed)
        self.fc1 = nn.Linear(input_size, hidden_sizes[0], bias=bias)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=bias)
        self.fc3 = nn.Linear(hidden_sizes[1], output_size, bias=bias)
        self.num_params = get_num_params(
            matrix_dim_list=get_matrix_dim_list(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=hidden_sizes,
            )
        )

    def forward(self, x):
        """
        Performs a forward pass through the linear network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    # A method used for initializing weights equivalently when dealing with
    # globally reparameterized models.
    def _gather_flat_params(self):
        """
        Flattens and concatenates all model parameters into a single 1D tensor.

        Args:
            None

        Returns:
            torch.Tensor: One-dimensional tensor containing all model parameters.
        """
        views = []
        for p in self.parameters():
            if p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        return torch.cat(views, 0)


"""
HELPER FUNCTIONS
"""


def get_matrix_dim_list(input_size, output_size, hidden_sizes):
    """
    Computes the list of weight matrix dimensions for a fully connected feedforward network.

    Args:
        input_size (int): Input feature dimension.
        output_size (int): Output feature dimension.
        hidden_sizes (list): List of hidden layer widths.

    Returns:
        list: List of tuples (in_dim, out_dim) specifying weight matrix dimensions for each layer.
    """
    dim_list = [(input_size, hidden_sizes[0])]
    for idx in range(1, len(hidden_sizes)):
        dim_list.append((hidden_sizes[idx - 1], hidden_sizes[idx]))

    dim_list.append((hidden_sizes[-1], output_size))
    return dim_list


def get_num_params(
    input_size=None, output_size=None, hidden_sizes=None, matrix_dim_list=None
):
    """
    Computes the total number of weight parameters in a linear feedforward network, assuming no biases.

    Args:
        input_size (int): Input feature dimension. Required if matrix_dim_list is None.
        output_size (int): Output feature dimension. Required if matrix_dim_list is None.
        hidden_sizes (list): List of hidden layer widths. Required if matrix_dim_list is None.
        matrix_dim_list (list): Optional list of (in_dim, out_dim) tuples specifying layer dimensions.

    Returns:
        int: Total number of weight parameters, assuming no bias terms.
    """
    if matrix_dim_list is None:
        matrix_dim_list = get_matrix_dim_list(
            input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes
        )
    return np.sum(
        np.array([np.multiply(*dim_tuple).item() for dim_tuple in matrix_dim_list])
    )
