import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from egop_optimizer.models.BaseClassifier import BaseClassifier

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
    ):
        super(FashionMNISTClassifier, self).__init__(weight_dist=weight_dist)
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