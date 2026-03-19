import torch
import unittest

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.LinearFeedforward import (
    LinearFeedforward,
    ReparamLinearFeedforward,
)
from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader
from egop_optimizer.utils.EGOP_utils import compute_V_by_layer
from egop_optimizer.utils.device_utils import get_available_device
from egop_optimizer.dataloaders.linear_networks_dataloader import (
    linear_networks_dataloader,
)

import pdb

DEVICE = get_available_device()

_DEFAULT_NETWORK_PARAMS = {
    "input_size": 64,
    "output_size": 10,
    "hidden_sizes": [10, 5],
}

"""
TODO: Switch over to using linear_networks_dataloader instead of tinyMNIST
"""


def setUpModule():
    """
    Prints the test file name when the test module is executed.

    Args:
        None

    Returns:
        None: Writes the file path to stdout.
    """
    print(f"\nRunning tests in {__file__}")


def get_V_dict_for_TinyMNIST(OG_model, use_randomized_svd):
    """
    Instantiates dataloaders, criterion, and runs compute_V_by_layer
    """
    trainloader, _, _ = tinyMNIST_dataloader(batch_size=128)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    return compute_V_by_layer(
        model_OG=OG_model,
        k=100,
        data_loader=trainloader,
        criterion=criterion,
        use_randomized_svd=use_randomized_svd,
    )


class TestBasicSetup(unittest.TestCase):
    def test_initialization(self):
        """
        Instantiates a reparameterized model to verify successful initialization.

        Args:
            None

        Returns:
            None: Raises an exception if model construction fails.
        """
        OG_model = LinearFeedforward(**_DEFAULT_NETWORK_PARAMS)
        V_dict = get_V_dict_for_TinyMNIST(OG_model=OG_model, use_randomized_svd=False)
        reparam_model = ReparamLinearFeedforward(
            V_by_layer_dict=V_dict, **_DEFAULT_NETWORK_PARAMS
        )

    def test_model_eval(self, batch_size=128):
        """
        Instantiates a reparam model, performs a forward pass on one batch, and verifies execution.

        Args:
            batch_size (int): Batch size used to generate a training batch (default: 128).

        Returns:
            None: Raises an exception if the forward pass fails.
        """
        model = LinearFeedforward(**_DEFAULT_NETWORK_PARAMS)
        model = model.to(DEVICE)
        trainloader, _, _ = tinyMNIST_dataloader(batch_size=128)

        train_iterator = iter(trainloader)
        Xb, yb = next(train_iterator)

        y_hat = model(Xb)
        return


if __name__ == "__main__":
    unittest.main()
