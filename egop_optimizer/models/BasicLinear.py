import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

from egop_optimizer.models.BaseClassifier import BaseClassifier


class BasicLinear(BaseClassifier):
    def __init__(self, input_size=10, output_size=10, bias=True):
        super().__init__(weight_dist="xavier_normal")
        self.fc1 = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        return x
