import unittest

import pdb

from egop_optimizer.models.TinyMNISTClassifier import TinyMNISTClassifier
from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader


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
        model = TinyMNISTClassifier()

    def test_model_eval(self, batch_size=128):
        model = TinyMNISTClassifier()
        trainloader, _, _ = tinyMNIST_dataloader(batch_size=batch_size)

        train_iterator = iter(trainloader)
        Xb, yb = next(train_iterator)

        y_hat = model(Xb)
        return


if __name__ == "__main__":
    unittest.main()
