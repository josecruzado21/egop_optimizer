import torch
import unittest

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.utils.EGOP_utils import compute_V_by_layer
from egop_optimizer.models.TinyMNISTClassifier import TinyMNISTClassifier
from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader

import pdb

"""
TODO'S:
-test CNN
-test shapes when using randomized SVD
"""


def get_V_dict_for_tests(OG_model, use_randomized_svd):
    trainloader, _, _ = tinyMNIST_dataloader(batch_size=128)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    return compute_V_by_layer(
        model_OG=OG_model,
        k=100,
        data_loader=trainloader,
        criterion=criterion,
        use_randomized_svd=use_randomized_svd,
    )


class TestLinearLayersBasic(unittest.TestCase):
    def test_computation_executes(self):
        """
        Tests that we can run compute_V_by_layer without error
        """
        model = TinyMNISTClassifier(input_size=64, hidden_sizes=[10])
        get_V_dict_for_tests(OG_model=model, use_randomized_svd=False)
        get_V_dict_for_tests(OG_model=model, use_randomized_svd=True)
        return

    def test_num_dict_elts(self):
        """
        Tests that number of elts in dict matches number of linear layers in model.
        """
        for use_randomized_svd in [True, False]:
            for hidden_sizes in [[10], [10, 5], [10, 5, 2]]:
                model = TinyMNISTClassifier(input_size=64, hidden_sizes=hidden_sizes)
                V_dict = get_V_dict_for_tests(
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
            V_dict = get_V_dict_for_tests(OG_model=model, use_randomized_svd=False)
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


if __name__ == "__main__":
    unittest.main()
