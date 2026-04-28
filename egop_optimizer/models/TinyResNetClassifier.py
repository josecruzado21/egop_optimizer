import torch
import torch.nn as nn
import torch.nn.functional as F

from egop_optimizer.models.BaseClassifier import BaseClassifier, ReparamBaseClassifier
from egop_optimizer.models.reparam_layers.reparam_layers import ResBlock


class TinyResNetClassifier(BaseClassifier):
    """
    Small ResNet-style classifier with one ResBlock followed by a linear head.
    Input: (batch, 1, 8, 8). Used for fast unit tests.
    """

    def __init__(self, num_classes=10, seed=None):
        super().__init__(seed=seed)
        self.resblock = ResBlock(1, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        x = self.resblock(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ReparamTinyResNetClassifier(ReparamBaseClassifier):
    """
    Reparameterized version of TinyResNetClassifier.
    The two conv layers inside ResBlock are replaced with EGOP_conv2d_layer; fc is unchanged.
    """

    def __init__(self, V_by_layer_dict, num_classes=10, seed=None):
        super().__init__(V_by_layer_dict=V_by_layer_dict, seed=seed)
        self.resblock = ResBlock(
            1,
            4,
            V1=V_by_layer_dict["resblock.conv1"],
            V2=V_by_layer_dict["resblock.conv2"],
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        x = self.resblock(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
