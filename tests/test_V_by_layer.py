import torch
import unittest

from egop_optimizer.utils.EGOP_utils import compute_V_by_layer

from egop_optimizer.models.TinyMNISTClassifier import TinyMNISTClassifier
from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader
from egop_optimizer.models.CifarClassifier import TinyCifarClassifier
from test_TinyCifarClassifier import load_cifar_debugging_subset

import pdb

"""
TODO'S:
-test CNN
    - only basic computation tested, no shapes
-test CNN when reparameterizing linear layers
-test shapes when using randomized SVD
-test shapes when using n_components
-replicability:
    - do we have explicit seed control/can we get identical results when we re-run compute_V, or not?
-functionality: modification to EGOP_utils.py
    - Add option to pass EGOP_oversampling_factor instead of explicit value for k
    num_EGOP_samples = int(EGOP_oversampling_factor * count_params(model))
"""


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


def get_V_dict_for_Cifar10(
    OG_model, use_randomized_svd=False, reparam_linear_layers=False
):
    """
    Instantiates dataloaders, criterion, and runs compute_V_by_layer
    """
    trainloader, _, _ = load_cifar_debugging_subset()
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    return compute_V_by_layer(
        model_OG=OG_model,
        k=100,
        data_loader=trainloader,
        criterion=criterion,
        use_randomized_svd=use_randomized_svd,
        reparam_linear_layers=reparam_linear_layers,
    )


class TestLinearLayersBasic(unittest.TestCase):
    def test_computation_executes(self):
        """
        Tests that we can run compute_V_by_layer without error
        """
        model = TinyMNISTClassifier(input_size=64, hidden_sizes=[10])
        get_V_dict_for_TinyMNIST(OG_model=model, use_randomized_svd=False)
        get_V_dict_for_TinyMNIST(OG_model=model, use_randomized_svd=True)
        return

    def test_num_dict_elts(self):
        """
        Tests that number of elts in dict matches number of linear layers in model.
        """
        for use_randomized_svd in [True, False]:
            for hidden_sizes in [[10], [10, 5], [10, 5, 2]]:
                model = TinyMNISTClassifier(input_size=64, hidden_sizes=hidden_sizes)
                V_dict = get_V_dict_for_TinyMNIST(
                    OG_model=model, use_randomized_svd=use_randomized_svd
                )
                self.assertEqual(
                    len(hidden_sizes) + 1,
                    len(V_dict),
                    "Number of dictionary elements not equal to number of expected linear layers.",
                )
        return

    def test_dict_elt_dims(self):
        """
        Tests that when randomized_svd = False, V_dict contains full-dim elements
        """
        for hidden_sizes in [[10], [10, 5], [10, 5, 2]]:
            model = TinyMNISTClassifier(input_size=64, hidden_sizes=hidden_sizes)
            V_dict = get_V_dict_for_TinyMNIST(OG_model=model, use_randomized_svd=False)
            for name, module in model.named_modules():
                if hasattr(module, "weight") and (module.weight.grad is not None):
                    self.assertEqual(
                        V_dict[name].shape[0],
                        V_dict[name].shape[1],
                        f"V_dict[{name}] is not square.",
                    )
                    self.assertEqual(
                        V_dict[name].shape[0],
                        module.weight.numel(),
                        f"V_dict[{name}] does not have shape {module.weight.numel()}x{module.weight.numel()}",
                    )
        return


class TestCNNLayersNoLinearReparam(unittest.TestCase):
    def test_computation_executes_no_linear_reparam(self):
        """
        Tests that we can run compute_V_by_layer without error
        """
        model = TinyCifarClassifier()
        get_V_dict_for_Cifar10(
            OG_model=model, use_randomized_svd=False, reparam_linear_layers=False
        )
        return


if __name__ == "__main__":
    unittest.main()
