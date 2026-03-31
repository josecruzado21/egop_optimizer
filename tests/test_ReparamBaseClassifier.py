import torch
import unittest

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.LinearFeedforward import (
    LinearFeedforward,
    ReparamLinearFeedforward,
)
from egop_optimizer.models.TinyMNISTClassifier import (
    TinyMNISTClassifier,
    ReparamTinyMNISTClassifier,
)
from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader
from egop_optimizer.utils.EGOP_utils import compute_V_by_layer
from egop_optimizer.utils.device_utils import get_available_device
from egop_optimizer.dataloaders.linear_networks_dataloader import (
    linear_networks_dataloader,
)

import pdb

DEVICE = get_available_device()

_DEFAULT_LINEARFEEDFORWARD_PARAMS = {
    "input_size": 64,
    "output_size": 10,
    "hidden_sizes": [10, 5],
}

"""
TODO:
-test with CNN and ResNet blocks
-sanity check ranks of EGOP matrices
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


def get_V_dict_for_LinearFeedforward(OG_model, use_randomized_svd):
    """
    Instantiates dataloaders, criterion, and runs compute_V_by_layer
    """
    trainloader, _, _ = linear_networks_dataloader(
        batch_size=128,
        input_size=_DEFAULT_LINEARFEEDFORWARD_PARAMS["input_size"],
        output_size=_DEFAULT_LINEARFEEDFORWARD_PARAMS["output_size"],
    )
    criterion = torch.nn.MSELoss(reduction="mean")

    return compute_V_by_layer(
        model_OG=OG_model,
        k=100,
        data_loader=trainloader,
        criterion=criterion,
        use_randomized_svd=use_randomized_svd,
    )


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


class TestLinearFeedforwardReparam(unittest.TestCase):
    def test_initialization(self):
        """
        Instantiates a reparameterized model to verify successful initialization.

        Args:
            None

        Returns:
            None: Raises an exception if model construction fails.
        """
        OG_model = LinearFeedforward(**_DEFAULT_LINEARFEEDFORWARD_PARAMS)
        V_dict = get_V_dict_for_LinearFeedforward(
            OG_model=OG_model, use_randomized_svd=False
        )
        reparam_model = ReparamLinearFeedforward(
            V_by_layer_dict=V_dict, **_DEFAULT_LINEARFEEDFORWARD_PARAMS
        )

    def test_model_eval(self, batch_size=128):
        """
        Instantiates a reparam model, performs a forward pass on one batch, and verifies execution.

        Args:
            batch_size (int): Batch size used to generate a training batch (default: 128).

        Returns:
            None: Raises an exception if the forward pass fails.
        """
        OG_model = LinearFeedforward(**_DEFAULT_LINEARFEEDFORWARD_PARAMS)
        V_dict = get_V_dict_for_LinearFeedforward(
            OG_model=OG_model, use_randomized_svd=False
        )
        reparam_model = ReparamLinearFeedforward(
            V_by_layer_dict=V_dict, **_DEFAULT_LINEARFEEDFORWARD_PARAMS
        )
        trainloader, _, _ = linear_networks_dataloader(
            batch_size=128,
            input_size=_DEFAULT_LINEARFEEDFORWARD_PARAMS["input_size"],
            output_size=_DEFAULT_LINEARFEEDFORWARD_PARAMS["output_size"],
        )

        train_iterator = iter(trainloader)
        Xb, yb = next(train_iterator)

        y_hat = reparam_model(Xb)
        return


class TestTinyMNISTReparam:
    def test_initialization(self):
        OG_model = TinyMNISTClassifier()
        V_dict = get_V_dict_for_TinyMNIST(OG_model=OG_model, use_randomized_svd=False)
        reparam_model = ReparamTinyMNISTClassifier(V_by_layer_dict=V_dict)
        return

    def test_model_eval(self, batch_size=128):
        OG_model = TinyMNISTClassifier()
        V_dict = get_V_dict_for_TinyMNIST(OG_model=OG_model, use_randomized_svd=False)
        reparam_model = ReparamTinyMNISTClassifier(V_by_layer_dict=V_dict)

        trainloader, _, _ = tinyMNIST_dataloader(batch_size=128)
        train_iterator = iter(trainloader)
        Xb, yb = next(train_iterator)

        y_hat = reparam_model(Xb)
        return


if __name__ == "__main__":
    unittest.main()
