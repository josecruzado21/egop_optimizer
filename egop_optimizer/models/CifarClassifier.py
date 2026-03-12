import torch
import torch.nn as nn
from egop_optimizer.models.BaseClassifier import BaseClassifier


class TinyCifarClassifier(BaseClassifier):
    def __init__(self, num_classes=10, seed=None, weight_dist="kaiming_normal"):
        super().__init__(seed=seed, weight_dist=weight_dist)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
