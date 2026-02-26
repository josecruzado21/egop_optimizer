import torch
import torch.nn as nn
from egop_optimizer.models.reparam_layers.reparam_layers import ResBlock
from egop_optimizer.models.BaseClassifier import BaseClassifier

class ImageNet_model_34_layer_residual(BaseClassifier):
    def __init__(self, num_classes=1000, seed=None, weight_dist="kaiming_normal"):
        super().__init__(seed=seed, weight_dist=weight_dist)
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features2 = nn.Sequential(
            ResBlock(64, 64, kernel_size=3, stride=1),
            ResBlock(64, 64, kernel_size=3, stride=1),
            ResBlock(64, 64, kernel_size=3, stride=1),
        )
        self.features3 = nn.Sequential(
            ResBlock(64, 128, kernel_size=3, stride=2),
            ResBlock(128, 128, kernel_size=3, stride=1),
            ResBlock(128, 128, kernel_size=3, stride=1),
            ResBlock(128, 128, kernel_size=3, stride=1),
        )
        self.features4 = nn.Sequential(
            ResBlock(128, 256, kernel_size=3, stride=2),
            ResBlock(256, 256, kernel_size=3, stride=1),
            ResBlock(256, 256, kernel_size=3, stride=1),
            ResBlock(256, 256, kernel_size=3, stride=1),
            ResBlock(256, 256, kernel_size=3, stride=1),
            ResBlock(256, 256, kernel_size=3, stride=1),
        )
        self.features5 = nn.Sequential(
            ResBlock(256, 512, kernel_size=3, stride=2),
            ResBlock(512, 512, kernel_size=3, stride=1),
            ResBlock(512, 512, kernel_size=3, stride=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


