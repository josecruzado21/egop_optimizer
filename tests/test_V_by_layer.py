import torch
import torch.nn as nn
import unittest

from egop_optimizer.utils.EGOP_utils import compute_V_by_layer, count_params

from egop_optimizer.models.TinyMNISTClassifier import TinyMNISTClassifier
from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader
from egop_optimizer.models.CifarClassifier import TinyCifarClassifier
from test_TinyCifarClassifier import load_cifar_debugging_subset

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


def get_V_dict_for_TinyMNIST(OG_model, use_randomized_svd, n_components=None, k=100, EGOP_oversampling_factor=None):
    """
    Instantiates dataloaders, criterion, and runs compute_V_by_layer
    """
    trainloader, _, _ = tinyMNIST_dataloader(batch_size=128)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    return compute_V_by_layer(
        model_OG=OG_model,
        k=k,
        data_loader=trainloader,
        criterion=criterion,
        use_randomized_svd=use_randomized_svd,
        n_components=n_components,
        EGOP_oversampling_factor=EGOP_oversampling_factor,
    )


def get_V_dict_for_Cifar10(
    OG_model, use_randomized_svd=False, reparam_linear_layers=False, n_components=None
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
        n_components=n_components,
    )



class TestLinearLayersBasic(unittest.TestCase):
    def test_computation_executes(self):
        """
        Tests that we can run compute_V_by_layer without error
        """
        model = TinyMNISTClassifier(input_size=64, hidden_sizes=[10])
        get_V_dict_for_TinyMNIST(OG_model=model, use_randomized_svd=False)
        get_V_dict_for_TinyMNIST(OG_model=model, use_randomized_svd=True)

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



class TestCNNLayersNoLinearReparam(unittest.TestCase):
    def test_computation_executes_no_linear_reparam(self):
        """
        Tests that we can run compute_V_by_layer without error
        """
        model = TinyCifarClassifier()
        get_V_dict_for_Cifar10(
            OG_model=model, use_randomized_svd=False, reparam_linear_layers=False
        )

    def test_num_dict_elts_no_linear_reparam(self):
        """
        Without reparam_linear_layers, only Conv2d layers should appear in V_dict.
        """
        model = TinyCifarClassifier()
        V_dict = get_V_dict_for_Cifar10(
            OG_model=model, use_randomized_svd=False, reparam_linear_layers=False
        )
        num_conv_layers = sum(
            1 for m in model.modules() if isinstance(m, nn.Conv2d)
        )
        self.assertEqual(
            len(V_dict), num_conv_layers,
            f"Expected {num_conv_layers} conv entries, got {len(V_dict)}.",
        )

    def test_shapes_no_linear_reparam(self):
        """
        For Conv2d with full SVD: V shape should be (params_per_kernel, params_per_kernel)
        where params_per_kernel = in_channels * kH * kW.
        """
        model = TinyCifarClassifier()
        V_dict = get_V_dict_for_Cifar10(
            OG_model=model, use_randomized_svd=False, reparam_linear_layers=False
        )
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and name in V_dict:
                params_per_kernel = (
                    module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                )
                self.assertEqual(
                    V_dict[name].shape[0], params_per_kernel,
                    f"V_dict[{name}] dim-0 should be {params_per_kernel}, got {V_dict[name].shape[0]}.",
                )
                self.assertEqual(
                    V_dict[name].shape[0], V_dict[name].shape[1],
                    f"V_dict[{name}] should be square for full SVD.",
                )



class TestCNNLayersWithLinearReparam(unittest.TestCase):
    def test_computation_executes_with_linear_reparam(self):
        """
        Tests that we can run compute_V_by_layer with reparam_linear_layers=True.
        Uses randomized SVD to avoid OOM on large linear layers (e.g. 51200 params).
        """
        model = TinyCifarClassifier()
        get_V_dict_for_Cifar10(
            OG_model=model, use_randomized_svd=True, reparam_linear_layers=True
        )

    def test_num_dict_elts_with_linear_reparam(self):
        """
        With reparam_linear_layers, both Conv2d and Linear layers should appear in V_dict.
        """
        model = TinyCifarClassifier()
        V_dict = get_V_dict_for_Cifar10(
            OG_model=model, use_randomized_svd=True, reparam_linear_layers=True
        )
        num_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        self.assertEqual(
            len(V_dict), num_conv + num_linear,
            f"Expected {num_conv + num_linear} entries (conv+linear), got {len(V_dict)}.",
        )

    def test_shapes_with_linear_reparam(self):
        """
        Conv2d: V dim-0 = params_per_kernel
        Linear: V dim-0 = weight.numel()
        Uses randomized SVD so V is not necessarily square.
        """
        model = TinyCifarClassifier()
        V_dict = get_V_dict_for_Cifar10(
            OG_model=model, use_randomized_svd=True, reparam_linear_layers=True
        )
        for name, module in model.named_modules():
            if name not in V_dict:
                continue
            if isinstance(module, nn.Conv2d):
                expected = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                self.assertEqual(
                    V_dict[name].shape[0], expected,
                    f"V_dict[{name}] Conv2d dim-0 should be {expected}.",
                )
            elif isinstance(module, nn.Linear):
                expected = module.weight.numel()
                self.assertEqual(
                    V_dict[name].shape[0], expected,
                    f"V_dict[{name}] Linear dim-0 should be {expected}.",
                )



class TestRandomizedSVDShapes(unittest.TestCase):
    def test_linear_randomized_svd_shape(self):
        """
        With randomized SVD (no explicit n_components), V columns = min(num_params, num_samples).
        """
        model = TinyMNISTClassifier(input_size=64, hidden_sizes=[10])
        V_dict = get_V_dict_for_TinyMNIST(OG_model=model, use_randomized_svd=True)
        for name, module in model.named_modules():
            if name in V_dict:
                num_params = module.weight.numel()
                # svd_lowrank with q = min(num_params, num_samples)
                expected_cols = min(num_params, V_dict[name].shape[0])
                self.assertEqual(
                    V_dict[name].shape[0], num_params,
                    f"V_dict[{name}] dim-0 should be {num_params}.",
                )
                self.assertLessEqual(
                    V_dict[name].shape[1], num_params,
                    f"V_dict[{name}] dim-1 should be <= {num_params}.",
                )

    def test_cnn_randomized_svd_shape(self):
        """
        Conv2d with randomized SVD: V dim-0 = params_per_kernel.
        """
        model = TinyCifarClassifier()
        V_dict = get_V_dict_for_Cifar10(
            OG_model=model, use_randomized_svd=True, reparam_linear_layers=False
        )
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and name in V_dict:
                params_per_kernel = (
                    module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                )
                self.assertEqual(
                    V_dict[name].shape[0], params_per_kernel,
                    f"V_dict[{name}] dim-0 should be {params_per_kernel}.",
                )
                self.assertLessEqual(
                    V_dict[name].shape[1], params_per_kernel,
                    f"V_dict[{name}] dim-1 should be <= {params_per_kernel}.",
                )


class TestNComponentsShapes(unittest.TestCase):
    def test_linear_n_components(self):
        """
        When n_components is specified with randomized SVD, V should have exactly
        n_components columns.
        """
        n_comp = 5
        model = TinyMNISTClassifier(input_size=64, hidden_sizes=[10])
        V_dict = get_V_dict_for_TinyMNIST(
            OG_model=model, use_randomized_svd=True, n_components=n_comp
        )
        for name, module in model.named_modules():
            if name in V_dict:
                num_params = module.weight.numel()
                self.assertEqual(
                    V_dict[name].shape, (num_params, n_comp),
                    f"V_dict[{name}] should be ({num_params}, {n_comp}) with n_components={n_comp}.",
                )

    def test_cnn_n_components_with_reparam(self):
        
        n_comp = 2
        model = TinyCifarClassifier()
        V_dict = get_V_dict_for_Cifar10(
            OG_model=model, use_randomized_svd=True,
            reparam_linear_layers=True, n_components=n_comp,
        )
        for name, module in model.named_modules():
            if name not in V_dict:
                continue
            if isinstance(module, nn.Linear):
                num_params = module.weight.numel()
                self.assertEqual(
                    V_dict[name].shape, (num_params, n_comp),
                    f"V_dict[{name}] Linear should be ({num_params}, {n_comp}).",
                )
            elif isinstance(module, nn.Conv2d):
                params_per_kernel = (
                    module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                )
                self.assertEqual(
                    V_dict[name].shape[0], params_per_kernel,
                    f"V_dict[{name}] Conv2d dim-0 should be {params_per_kernel}.",
                )



class TestEGOPOversamplingFactor(unittest.TestCase):
    def test_oversampling_factor_produces_valid_V_dict(self):
        """
        Using EGOP_oversampling_factor instead of k should produce the same
        V_dict structure as using k directly.
        """
        model = TinyMNISTClassifier(input_size=64, hidden_sizes=[10])
        V_dict = get_V_dict_for_TinyMNIST(
            OG_model=model, use_randomized_svd=True,
            k=None, EGOP_oversampling_factor=0.1,
        )
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        self.assertEqual(len(V_dict), num_linear)

    def test_oversampling_factor_matches_explicit_k(self):
        """
        V_dict from EGOP_oversampling_factor should have the same keys and shapes
        as V_dict from the equivalent explicit k.
        """
        model = TinyMNISTClassifier(input_size=64, hidden_sizes=[10])
        factor = 0.1
        explicit_k = int(factor * count_params(model))

        V_dict_factor = get_V_dict_for_TinyMNIST(
            OG_model=model, use_randomized_svd=True,
            k=None, EGOP_oversampling_factor=factor,
        )

        model.reinitialize()
        V_dict_explicit = get_V_dict_for_TinyMNIST(
            OG_model=model, use_randomized_svd=True,
            k=explicit_k,
        )

        self.assertEqual(V_dict_factor.keys(), V_dict_explicit.keys())
        for name in V_dict_factor:
            self.assertEqual(
                V_dict_factor[name].shape, V_dict_explicit[name].shape,
                f"V_dict[{name}] shape mismatch: {V_dict_factor[name].shape} vs {V_dict_explicit[name].shape}",
            )

    def test_no_k_no_factor_raises_error(self):
        """
        Passing neither k nor EGOP_oversampling_factor should raise ValueError.
        """
        model = TinyMNISTClassifier(input_size=64, hidden_sizes=[10])
        with self.assertRaises(ValueError):
            get_V_dict_for_TinyMNIST(
                OG_model=model, use_randomized_svd=True,
                k=None,
            )


if __name__ == "__main__":
    unittest.main()
