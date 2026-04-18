import torch
import torch.nn.init as init
import unittest


from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.TinyMNISTClassifier import (
    TinyMNISTClassifier,
    ReparamTinyMNISTClassifier,
)
from egop_optimizer.models.reparam_layers.reparam_layers import (
    EGOP_linear_layer,
    EGOP_conv2d_layer,
)
from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader
from egop_optimizer.utils.EGOP_utils import (
    compute_V_by_layer,
    layerwise_reparam_init_equiv,
    check_model_structure,
)
from egop_optimizer.utils.device_utils import get_available_device
from egop_optimizer.models.TinyConvClassifier import (
    TinyConvClassifier,
    ReparamTinyConvClassifier,
)

import pdb

DEVICE = get_available_device()
# List of implemented EGOP layer classes
# Needs to be a tuple for isinstance to work
EGOP_LAYER_CLASSES = tuple([EGOP_linear_layer, EGOP_conv2d_layer])

# Set generous absolute tolerance for checks with torch.isclose()
ATOL_THRESHOLD = 1e-5


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


def check_if_leaf_node(module):
    """
    Tests if a module has nested submodules
    """
    return len(list(module.children())) == 0


class SharedModelEquivalenceTester:
    """
    A shared class that implements test_models_are_equivalent.
    This can be used accross multiple unittest.TestCase in this file.
    It's important that the method is named assert_models_are_equivalent and not test_models_are_equivalent,
    because otherwise unittest with accidentally auto-discover this method and that will cause an error.
    """

    def assert_models_are_equivalent(self, model_1, model_2, verbose=True):
        # First check if module_1 and module_2 have compatible sets of named_modules and parameters.
        # If they do not, raises an exception.
        check_model_structure(model_1, model_2)

        # Iterate over named modules
        for name, module in model_1.named_modules():
            # Check if we're at a leaf node. We don't want to run these tests on things with nested submodules.
            if check_if_leaf_node(module):
                module_1 = model_1.get_submodule(name)
                module_2 = model_2.get_submodule(name)

                if isinstance(module_1, EGOP_LAYER_CLASSES) or isinstance(
                    module_2, EGOP_LAYER_CLASSES
                ):
                    if verbose:
                        print(
                            f"Testing for EQUIVALENCE on modules of types: {type(module_1), type(module_2)}"
                        )
                    # Get weights in original coordinates (always compare on CPU)
                    if isinstance(module_1, EGOP_LAYER_CLASSES):
                        module_1_W_OG_coors = module_1.return_weight_copy_in_OG_coors().cpu()
                    else:
                        module_1_W_OG_coors = module_1.weight.detach().clone().cpu()
                    if isinstance(module_2, EGOP_LAYER_CLASSES):
                        module_2_W_OG_coors = module_2.return_weight_copy_in_OG_coors().cpu()
                    else:
                        module_2_W_OG_coors = module_2.weight.detach().clone().cpu()

                    # Check if weights and biases are equal up to floating point error
                    self.assertTrue(
                        torch.allclose(
                            module_1_W_OG_coors,
                            module_2_W_OG_coors,
                            atol=ATOL_THRESHOLD,
                        ),
                        f"Weights not equivalent for modules of name {name} (atol={ATOL_THRESHOLD}).",
                    )
                    bias_1 = module_1.bias
                    bias_2 = module_2.bias
                    if bias_1 is None and bias_2 is None:
                        pass
                    elif bias_1 is not None and bias_2 is not None:
                        self.assertTrue(
                            torch.allclose(
                                bias_1.detach().clone().cpu(),
                                bias_2.detach().clone().cpu(),
                                atol=ATOL_THRESHOLD,
                            ),
                            f"Biases not equivalent for modules of name {name} (atol={ATOL_THRESHOLD}).",
                        )
                    else:
                        self.fail(
                            f"Bias mismatch for module {name}: one is None, the other is not."
                        )

                # If not an EGOP layer, check that modules are identical
                else:
                    if verbose:
                        print(
                            f"Testing for identicality on modules of types: {type(module_1), type(module_2)}"
                        )
                    self.assertTrue(
                        self.assert_modules_identical(module_1, module_2),
                        f"Parameters not identical for modules of name {name} (modules of type {type(module_1)}).",
                    )
            # If we're not at a leaf mode, keep iterating over nested submodules
            else:
                pass

        return True

    def assert_models_NOT_equivalent(self, model_1, model_2, verbose=True):
        try:
            self.assert_models_are_equivalent(model_1, model_2, verbose=verbose)
        except AssertionError:
            return  # expected
        self.fail(
            "Models were expected to NOT be equivalent, but were found to be equivalent."
        )

    def assert_modules_identical(self, module_1, module_2):
        """
        Tests if two modules have IDENTICAL parameters, not equivalent.
        """
        state_dict_1 = module_1.state_dict()
        state_dict_2 = module_2.state_dict()

        # Check that module state dicts have all same keys
        self.assertTrue(
            state_dict_1.keys() == state_dict_2.keys(),
            f"Modules do not have identical keys. module_1 keys = {state_dict_1.keys()}, module_2 keys = {state_dict_2.keys()}",
        )

        for k in state_dict_1:
            self.assertTrue(
                torch.allclose(state_dict_1[k].cpu(), state_dict_2[k].cpu(), atol=ATOL_THRESHOLD),
                f"Modules do not have identical parameter values for parameter {k} (atol = {ATOL_THRESHOLD}).",
            )

        return True


class TestSharedEquivalenceTester(SharedModelEquivalenceTester, unittest.TestCase):
    """
    Sanity checks that the methods in SharedModelEquivalenceTester work as expected
    """

    def test_assert_models_are_equivalent(self):
        OG_model = TinyMNISTClassifier()
        OG_model.reinitialize_seeded(seed=42)

        # If we call the equivalence function on two copies of OG
        # model with the same seed, we should find they are equivalent
        new_OG_model_same_seed = TinyMNISTClassifier()
        new_OG_model_same_seed.reinitialize_seeded(seed=42)
        self.assert_models_are_equivalent(
            OG_model, new_OG_model_same_seed, verbose=False
        )

        # If we call the equivalence function on two copies of OG
        # model DIFFERENT seeds,we should find they are not equivalent
        new_OG_model_diff_seed = TinyMNISTClassifier()
        new_OG_model_diff_seed.reinitialize_seeded(seed=987)
        self.assert_models_NOT_equivalent(
            OG_model, new_OG_model_diff_seed, verbose=False
        )
        return


class TestTinyMNISTReparam(SharedModelEquivalenceTester, unittest.TestCase):
    def test_execution(self):
        OG_model = TinyMNISTClassifier()
        V_dict = get_V_dict_for_TinyMNIST(OG_model=OG_model, use_randomized_svd=False)
        reparam_model = ReparamTinyMNISTClassifier(V_by_layer_dict=V_dict)

        layerwise_reparam_init_equiv(
            EGOP_model=reparam_model, OG_model=OG_model, seed=42
        )
        return

    def test_equivalence(self):
        OG_model = TinyMNISTClassifier()
        V_dict = get_V_dict_for_TinyMNIST(OG_model=OG_model, use_randomized_svd=False)
        reparam_model = ReparamTinyMNISTClassifier(V_by_layer_dict=V_dict)

        # Before initializing equivalently, models should not be equivalent
        self.assert_models_NOT_equivalent(OG_model, reparam_model, verbose=False)
        OG_model, reparam_model = layerwise_reparam_init_equiv(
            EGOP_model=reparam_model, OG_model=OG_model, seed=42
        )
        # After initializing equivalently, models should be equivalent
        self.assert_models_are_equivalent(OG_model, reparam_model, verbose=False)

        # If we create a new OG model with the same seed, it should still be equivalent
        # to reparam model.
        new_OG_model_same_seed = TinyMNISTClassifier()
        new_OG_model_same_seed = new_OG_model_same_seed.to(DEVICE)
        new_OG_model_same_seed.reinitialize_seeded(seed=42)
        self.assert_models_are_equivalent(
            new_OG_model_same_seed, reparam_model, verbose=False
        )

        # If we create a new OG model with a different seed, it should NOT be
        # equivalent to the reparam model
        new_OG_model_diff_seed = TinyMNISTClassifier()
        new_OG_model_diff_seed.reinitialize_seeded(seed=987)
        self.assert_models_NOT_equivalent(
            new_OG_model_diff_seed, reparam_model, verbose=False
        )

        # If we randomly reinitialize the first layer of our reparam model, it should no longer be equivalent
        init.normal_(reparam_model.fc1.weight)
        self.assert_models_NOT_equivalent(OG_model, reparam_model, verbose=False)
        return


def get_V_dict_for_TinyConv(OG_model, use_randomized_svd):
    """
    Builds a synthetic dataloader and runs compute_V_by_layer for conv-only layers.
    """
    x = torch.randn(128, 1, 8, 8)
    y = torch.randint(0, 10, (128,))
    dataset = torch.utils.data.TensorDataset(x, y)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    return compute_V_by_layer(
        model_OG=OG_model,
        k=10,
        data_loader=trainloader,
        criterion=criterion,
        reparam_linear_layers=False,
        use_randomized_svd=use_randomized_svd,
    )

class TestTinyConvReparam(SharedModelEquivalenceTester, unittest.TestCase):
    def test_execution(self):
        OG_model = TinyConvClassifier()
        V_dict = get_V_dict_for_TinyConv(OG_model=OG_model, use_randomized_svd=False)
        reparam_model = ReparamTinyConvClassifier(V_by_layer_dict=V_dict)

        layerwise_reparam_init_equiv(
            EGOP_model=reparam_model, OG_model=OG_model, seed=42
        )
        return

    def test_equivalence(self):
        OG_model = TinyConvClassifier()
        V_dict = get_V_dict_for_TinyConv(OG_model=OG_model, use_randomized_svd=False)
        reparam_model = ReparamTinyConvClassifier(V_by_layer_dict=V_dict)
        OG_model, reparam_model = layerwise_reparam_init_equiv(
            EGOP_model=reparam_model, OG_model=OG_model, seed=42
        )
        self.assert_models_are_equivalent(OG_model, reparam_model, verbose=False)
        return

if __name__ == "__main__":
    unittest.main()
