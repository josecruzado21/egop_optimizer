import unittest

import pdb

from egop_optimizer.models.BaseClassifier import BaseClassifier
from egop_optimizer.models.BasicLinear import BasicLinear


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
        model = BasicLinear()


if __name__ == "__main__":
    unittest.main()
