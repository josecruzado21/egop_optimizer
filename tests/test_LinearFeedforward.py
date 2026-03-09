import unittest

import pdb

from egop_optimizer.models.LinearFeedforward import LinearFeedforward
from egop_optimizer.dataloaders.linear_networks_dataloader import (
    linear_networks_dataloader,
)
from egop_optimizer.utils.device_utils import get_available_device
DEVICE = get_available_device()

_DEFAULT_NETWORK_PARAMS = {
    "input_size": 10,
    "output_size": 10,
    "hidden_sizes": [50, 30],
}


def setUpModule():
    """
    Prints the test file name when the test module is executed.

    Args:
        None

    Returns:
        None: Writes the file path to stdout.
    """
    print(f"\nRunning tests in {__file__}")


class TestBasicSetup(unittest.TestCase):
    def test_initialization(self):
        """
        Instantiates the LinearFeedforward model to verify successful initialization.

        Args:
            None

        Returns:
            None: Raises an exception if model construction fails.
        """
        model = LinearFeedforward(**_DEFAULT_NETWORK_PARAMS)

    def test_model_eval(self, batch_size=128):
        """
        Instantiates the model, performs a forward pass on one batch, and verifies execution.

        Args:
            batch_size (int): Batch size used to generate a training batch (default: 128).

        Returns:
            None: Raises an exception if the forward pass fails.
        """
        model = LinearFeedforward(**_DEFAULT_NETWORK_PARAMS)
        model = model.to(DEVICE)
        trainloader, _, _ = linear_networks_dataloader(
            batch_size=batch_size,
            input_size=_DEFAULT_NETWORK_PARAMS["input_size"],
            output_size=_DEFAULT_NETWORK_PARAMS["output_size"],
        )

        train_iterator = iter(trainloader)
        Xb, yb = next(train_iterator)

        y_hat = model(Xb)
        return


if __name__ == "__main__":
    unittest.main()
