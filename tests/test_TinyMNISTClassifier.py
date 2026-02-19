import unittest
import torch
import torch.nn as nn

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
        """
        Instantiates the TinyMNISTClassifier to verify successful initialization.

        Args:
            None

        Returns:
            None: Raises an exception if model construction fails.
        """
        model = TinyMNISTClassifier()

    def test_model_eval(self, batch_size=128):
        """
        Instantiates the model, retrieves one batch from the dataloader, and performs a forward pass.

        Args:
            batch_size (int): Batch size used to sample from the dataloader (default: 128).

        Returns:
            None: Raises an exception if the forward pass fails.
        """
        model = TinyMNISTClassifier()
        trainloader, _, _ = tinyMNIST_dataloader(batch_size=batch_size)

        train_iterator = iter(trainloader)
        Xb, yb = next(train_iterator)

        y_hat = model(Xb)
        return

    def test_reinitialize_same_seed_same_weights(self):
        """
        reinitializing the same model twice with the same seed gives identical weights
        """
        # Initialize with no seed
        model = TinyMNISTClassifier()
        model.reinitialize_seeded(seed=42)
        first_init_params = dict_of_parameters(model)
        # Immediately reinitialize
        model.reinitialize_seeded(seed=42)
        second_init_params = dict_of_parameters(model)

        self.assertTrue(
            param_dicts_equal(first_init_params, second_init_params),
            "Reinitializing with same seed did not produce identical weights",
        )

    def test_same_seed_gives_same_weights(self):
        """
        two models reinitialized with the same seed have identical weights
        """
        # Initialize with no seed
        model1 = TinyMNISTClassifier()
        model1.reinitialize_seeded(seed=2)
        model1_params = dict_of_parameters(model1)

        model2 = TinyMNISTClassifier()
        model2.reinitialize_seeded(seed=2)
        model2_params = dict_of_parameters(model2)

        self.assertTrue(
            param_dicts_equal(model1_params, model2_params),
            "Weights are not identical for same seed",
        )

    def test_reinitialize_seed_different_each_time(self):
        """
        repeated reinitialization without reseeding gives different weights
        """
        # Initialize with seed
        model = TinyMNISTClassifier(seed=123)
        model.reinitialize()
        first_init_params = dict_of_parameters(model)
        model.reinitialize()
        second_init_params = dict_of_parameters(model)

        self.assertFalse(
            param_dicts_equal(first_init_params, second_init_params),
            "Reinitializing without seed should produce different weights",
        )

    def test_reinitialize_noseed_different_each_time(self):
        """
        model initialized without a seed produces different weights on consecutive reinitializations
        """
        # Initialize with no seed
        model = TinyMNISTClassifier()
        model.reinitialize()
        first_init_params = dict_of_parameters(model)
        model.reinitialize()
        second_init_params = dict_of_parameters(model)

        self.assertFalse(
            param_dicts_equal(first_init_params, second_init_params),
            "Reinitializing without seed should produce different weights",
        )

    def test_reinitialize_seeded_different_seeds(self):
        """
        reinitializing with different seeds gives different weights
        """
        # Initialize with no seed
        model = TinyMNISTClassifier()
        model.reinitialize_seeded(seed=1)
        first_init_params = dict_of_parameters(model)
        model.reinitialize_seeded(seed=2)
        second_init_params = dict_of_parameters(model)

        self.assertFalse(
            param_dicts_equal(first_init_params, second_init_params),
            "Reinitializing without seed should produce different weights",
        )

    def test_reinitialize_sequence_matches_for_same_seed(self):
        """
        two models with the same seed produce the same sequence of weights for consecutive reinitializations
        """
        # Initialize with seed
        model1 = TinyMNISTClassifier(seed=42)
        model2 = TinyMNISTClassifier(seed=42)

        # First reinitialization
        model1.reinitialize()
        model1_param1 = dict_of_parameters(model1)

        model2.reinitialize()
        model2_param1 = dict_of_parameters(model2)

        # Second reinitialization
        model1.reinitialize()
        model1_param2 = dict_of_parameters(model1)

        model2.reinitialize()
        model2_param2 = dict_of_parameters(model2)

        self.assertTrue(
            param_dicts_equal(model1_param1, model2_param1),
            "First reinitialization weights do not match",
        )

        self.assertTrue(
            param_dicts_equal(model1_param2, model2_param2),
            "Second reinitialization weights do not match",
        )

    def test_init_with_seed_vs_no_seed_reinitialize_seeded(self):
        """
        models initialized with and without a seed produce identical weights after both are reinitialized with the same seed
        """
        model1 = TinyMNISTClassifier(seed=123)
        model2 = TinyMNISTClassifier()  # no seed

        model1.reinitialize_seeded(seed=42)
        model1_params = dict_of_parameters(model1)
        model2.reinitialize_seeded(seed=42)
        model2_params = dict_of_parameters(model2)

        self.assertTrue(
            param_dicts_equal(model1_params, model2_params),
            "Models should match after reinitialize_seeded with same seed",
        )


def dict_of_parameters(model):
    """
    Extracts copies of flattened parameter vectors from selected layers of a model.

    This function iterates over the immediate child modules of ``model`` and,
    for each layer that is an instance of ``nn.Conv2d`` or ``nn.Linear``,
    stores a flattened vector of its parameter values (excluding gradients)
    in a dictionary keyed by the layer index.

    Args:
        model (torch.nn.Module):
            The model whose child layers will be inspected.

    Returns:
        Dict[int, torch.Tensor]:
            A dictionary mapping the integer index of each qualifying layer
            (as given by ``enumerate(model.children())``) to a 1D tensor
            containing a snapshot of that layer’s parameter values.

    Notes:
        - Only immediate children of ``model`` are considered; nested
          submodules are not traversed recursively.
        - Only layers of type ``nn.Conv2d`` and ``nn.Linear`` are included.
        - The returned tensors contain parameter values only (no gradients).
    """
    param_dict = dict()
    for idx, layer in enumerate(model.children()):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            param_dict[idx] = layer_params_as_vector(layer)
    return param_dict


@torch.no_grad()
def layer_params_as_vector(layer: torch.nn.Module, *, device=None, dtype=None):
    """
    Returns a 1D tensor containing a copy of all parameter *values* in `layer`,
    concatenated in a deterministic order (layer.parameters()).
    Gradients are ignored.
    """
    vec = torch.cat([p.detach().reshape(-1) for p in layer.parameters()])
    if device is not None:
        vec = vec.to(device)
    if dtype is not None:
        vec = vec.to(dtype)
    return (
        vec.clone()
    )  # clone() makes it an actual snapshot (won’t change if the model updates later)


def param_dicts_equal(d1, d2, rtol=1e-6, atol=1e-7):
    """
    Returns True if two parameter dictionaries are equivalent.

    Checks:
        - identical key sets
        - approximate tensor equality
    """
    if d1.keys() != d2.keys():
        return False

    for k in d1:
        t1, t2 = d1[k], d2[k]

        if t1.shape != t2.shape:
            return False
        if t1.dtype != t2.dtype:
            return False
        if not torch.allclose(t1, t2, rtol=rtol, atol=atol):
            return False

    return True


if __name__ == "__main__":
    unittest.main()
