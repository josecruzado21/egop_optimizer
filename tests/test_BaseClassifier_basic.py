import unittest

import pdb

from egop_optimizer.models.BaseClassifier import BaseClassifier
from egop_optimizer.models.BasicLinear import BasicLinear
import torch


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

    def test_reinitialize_same_seed_same_weights(self):
        # Initialize with no seed
        model = BasicLinear()
        model.reinitialize_seeded(seed=42)
        w1 = dict(model.named_parameters())["fc1.weight"].detach().clone()
        model.reinitialize_seeded(seed=42)
        w2 = dict(model.named_parameters())["fc1.weight"].detach().clone()
        self.assertTrue(
            torch.allclose(w1, w2, atol=1e-7),
            "Reinitializing with same seed did not produce identical weights",
        )

    def test_same_seed_gives_same_weights(self):
        # Initialize with no seed
        model1 = BasicLinear()
        model1.reinitialize_seeded(seed=2)
        w1 = dict(model1.named_parameters())["fc1.weight"].detach().clone()
        model2 = BasicLinear()
        model2.reinitialize_seeded(seed=2)
        w2 = dict(model2.named_parameters())["fc1.weight"].detach().clone()

        self.assertTrue(
            torch.allclose(w1, w2, atol=1e-7), "Weights are not identical for same seed"
        )

    def test_reinitialize_seed_different_each_time(self):
        # Initialize with seed
        model = BasicLinear(seed=123)
        model.reinitialize()
        w1 = dict(model.named_parameters())["fc1.weight"].detach().clone()
        model.reinitialize()
        w2 = dict(model.named_parameters())["fc1.weight"].detach().clone()
        self.assertFalse(
            torch.allclose(w1, w2, atol=1e-7),
            "Reinitializing without seed should produce different weights",
        )

    def test_reinitialize_noseed_different_each_time(self):
        # Initialize with no seed
        model = BasicLinear()
        model.reinitialize()
        w1 = dict(model.named_parameters())["fc1.weight"].detach().clone()
        model.reinitialize()
        w2 = dict(model.named_parameters())["fc1.weight"].detach().clone()
        self.assertFalse(
            torch.allclose(w1, w2, atol=1e-7),
            "Reinitializing without seed should produce different weights",
        )

    def test_reinitialize_seeded_different_seeds(self):
        # Initialize with no seed
        model = BasicLinear()
        model.reinitialize_seeded(seed=1)
        w1 = dict(model.named_parameters())["fc1.weight"].detach().clone()
        model.reinitialize_seeded(seed=2)
        w2 = dict(model.named_parameters())["fc1.weight"].detach().clone()
        self.assertFalse(
            torch.allclose(w1, w2, atol=1e-7),
            "Different seeds should produce different weights",
        )

    def test_reinitialize_sequence_matches_for_same_seed(self):
        # Initialize with seed
        model1 = BasicLinear(seed=42)
        model2 = BasicLinear(seed=42)
        # First reinitialization
        model1.reinitialize()
        w1_1 = dict(model1.named_parameters())["fc1.weight"].detach().clone()
        model2.reinitialize()
        w2_1 = dict(model2.named_parameters())["fc1.weight"].detach().clone()
        # Second reinitialization
        model1.reinitialize()
        w1_2 = dict(model1.named_parameters())["fc1.weight"].detach().clone()
        model2.reinitialize()
        w2_2 = dict(model2.named_parameters())["fc1.weight"].detach().clone()
        self.assertTrue(
            torch.allclose(w1_1, w2_1, atol=1e-7),
            "First reinitialization weights do not match",
        )
        self.assertTrue(
            torch.allclose(w1_2, w2_2, atol=1e-7),
            "Second reinitialization weights do not match",
        )

    def test_init_with_seed_vs_no_seed_reinitialize_seeded(self):
        model1 = BasicLinear(seed=123)
        model2 = BasicLinear()  # no seed

        model1.reinitialize_seeded(seed=42)
        w1 = dict(model1.named_parameters())["fc1.weight"].detach().clone()
        model2.reinitialize_seeded(seed=42)
        w2 = dict(model2.named_parameters())["fc1.weight"].detach().clone()

        self.assertTrue(
            torch.allclose(w1, w2, atol=1e-7),
            "Models should match after reinitialize_seeded with same seed",
        )


if __name__ == "__main__":
    unittest.main()
