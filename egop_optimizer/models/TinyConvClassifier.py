import torch
import torch.nn as nn
import torch.nn.functional as F

from egop_optimizer.models.BaseClassifier import BaseClassifier, ReparamBaseClassifier
from egop_optimizer.models.reparam_layers.reparam_layers import EGOP_conv2d_layer


class TinyConvClassifier(BaseClassifier):
    """
    Small conv-only feature extractor followed by a linear head.
    Input: (batch, 1, 8, 8). Used for fast unit tests.
    """

    def __init__(self, num_classes=10, seed=None):
        super().__init__(seed=seed)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ReparamTinyConvClassifier(ReparamBaseClassifier):
    """
    Reparameterized version of TinyConvClassifier.
    conv1 and conv2 are replaced with EGOP_conv2d_layer; fc is unchanged.
    """

    def __init__(self, V_by_layer_dict, num_classes=10, seed=None):
        super().__init__(V_by_layer_dict=V_by_layer_dict, seed=seed)
        self.conv1 = EGOP_conv2d_layer(
            1, 4, kernel_size=3, V=V_by_layer_dict["conv1"], padding=1, bias=False
        )
        self.conv2 = EGOP_conv2d_layer(
            4, 8, kernel_size=3, V=V_by_layer_dict["conv2"], padding=1, bias=False
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
