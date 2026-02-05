import unittest

from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader

import pdb

DATALOADER_METHOD_LIST = [tinyMNIST_dataloader]


def setUpModule():
    """
    Prints file name when test script is executed.
    """
    print(f"\nRunning tests in {__file__}")


class TestBasicSetup(unittest.TestCase):
    def test_initialization(self, batch_size=128):
        """
        Instantiates dataloaders
        """
        for dataloader_method in DATALOADER_METHOD_LIST:
            _, _, _ = dataloader_method(batch_size=batch_size)
        return

    def test_batches(self, batch_size=128):
        """
        Instantiates dataloaders, converts to iterator, draws samples, sanity checks shape and datatype
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
