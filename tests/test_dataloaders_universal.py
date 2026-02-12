import unittest

from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader
from egop_optimizer.dataloaders.linear_networks_dataloader import (
    linear_networks_dataloader,
)

import pdb

DATALOADER_METHOD_LIST = [tinyMNIST_dataloader, linear_networks_dataloader]


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
    def test_initialization(self, batch_size=128):
        """
        Instantiates each dataloader to verify successful construction.

        Args:
            batch_size (int): Batch size passed to each dataloader (default: 128).

        Returns:
            None: Raises an exception if any dataloader fails to initialize.
        """
        for dataloader_method in DATALOADER_METHOD_LIST:
            _, _, _ = dataloader_method(batch_size=batch_size)
        return

    def test_batches(self, batch_size=128):
        """
        Instantiates each dataloader, retrieves one batch, and checks basic shape consistency.

        Args:
            batch_size (int): Batch size used when sampling from each dataloader (default: 128).

        Returns:
            None: Asserts that feature and label batch dimensions match the requested batch size.
        """
        for dataloader_method in DATALOADER_METHOD_LIST:
            trainloader, _, _ = dataloader_method(batch_size=batch_size)

            train_iterator = iter(trainloader)
            Xb, yb = next(train_iterator)

            # Check Xb.shape[0]==yb.shape[0] == batch_size
            self.assertEqual(
                Xb.shape[0],
                batch_size,
                f"Batch features first dimension = {Xb.shape[0]}, which is not equal to batch_size = {batch_size}",
            )
            self.assertEqual(
                yb.shape[0],
                batch_size,
                f"Batch labels first dimension = {yb.shape[0]}, which is not equal to batch_size = {batch_size}",
            )
        return


if __name__ == "__main__":
    unittest.main()
