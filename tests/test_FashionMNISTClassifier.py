import unittest

import pdb

from egop_optimizer.models.FashionMNISTClassfier import FashionMNISTClassifier
from egop_optimizer.dataloaders.fashionMNIST_dataloader import fashionMNIST_dataloader


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
        Instantiates the FashionMNISTClassifier to verify successful initialization.

        Args:
            None

        Returns:
            None: Raises an exception if model construction fails.
        """
        model = FashionMNISTClassifier()

    def test_model_eval(self, batch_size=128):
        """
        Instantiates the model, retrieves one batch from the dataloader, and performs a forward pass.

        Args:
            batch_size (int): Batch size used to sample from the dataloader (default: 128).

        Returns:
            None: Raises an exception if the forward pass fails.
        """
        model = FashionMNISTClassifier()
        trainloader, _, _ = fashionMNIST_dataloader(batch_size=batch_size)

        train_iterator = iter(trainloader)
        Xb, yb = next(train_iterator)

        y_hat = model(Xb)
        return


if __name__ == "__main__":
    unittest.main()