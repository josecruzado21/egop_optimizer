import torch
import torch.nn as nn
from egop_optimizer.models.BaseClassifier import BaseClassifier


class TinyCifarClassifier(BaseClassifier):
    """
    Implements a convolutional classifier for CIFAR-style images.

    The model consists of a small convolutional feature extractor followed by
    a fully connected classifier. The convolutional layers reduce the spatial
    resolution using max pooling before flattening into a multilayer perceptron.

    Args:
        num_classes (int): Number of output classes (default: 10).
        seed (int, optional): Random seed used for parameter initialization.
        weight_dist (str): Initialization scheme passed to BaseClassifier
            (default: "kaiming_normal").

    Returns:
        None: Initializes convolutional feature extractor and classifier layers.
    """

    def __init__(self, num_classes=10, seed=None, weight_dist="kaiming_normal"):
        """
        Initializes the TinyCifar classifier architecture.

        Args:
            num_classes (int): Number of output classes (default: 10).
            seed (int, optional): Random seed used for parameter initialization.
            weight_dist (str): Initialization scheme passed to BaseClassifier
                (default: "kaiming_normal").

        Returns:
            None: Constructs convolutional feature extractor and fully connected
            classifier layers.
        """
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
        """
        Performs a forward pass through the classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W),
                typically (batch_size, 3, 32, 32) for CIFAR images.

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
