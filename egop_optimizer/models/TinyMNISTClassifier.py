import torch
import torch.nn as nn
import torch.nn.functional as F

from egop_optimizer.models.BaseClassifier import BaseClassifier

import pdb


class TinyMNISTClassifier(BaseClassifier):
    def __init__(self, input_size=64, hidden_sizes=[32, 16], num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()

        # Dynamically create fully connected layers
        current_input_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            setattr(self, f"fc{i+1}", nn.Linear(current_input_size, hidden_size))
            current_input_size = hidden_size

        # Final output layer
        setattr(
            self, f"fc{len(hidden_sizes)+1}", nn.Linear(current_input_size, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        for i in range(1, len(self._modules)):
            x = getattr(self, f"fc{i}")(x)
            if i < len(self._modules) - 1:
                x = F.relu(x)
        return x
