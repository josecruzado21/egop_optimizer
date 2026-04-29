import torch
import torch.nn.init as init
import unittest

from egop_optimizer.models.FashionMNISTClassfier import (
    FashionMNISTClassifier,
    AuxiliaryReparamFashionMNISTClassifier,
)
from egop_optimizer.models.reparam_layers.reparam_layers import (
    EGOP_linear_layer,
    EGOP_auxiliary_variables_linear_layer,
)
from egop_optimizer.dataloaders.fashionMNIST_dataloader import fashionMNIST_dataloader
from egop_optimizer.utils.EGOP_utils import (
    compute_V_by_layer,
    layerwise_reparam_init_equiv,
    check_model_structure,
)
from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()

_MINI_MODEL_PARAMS = {"pool_factor": 1, "hidden_size": 100, "num_classes": 10}

# Tuple for isinstance checks across all EGOP layer types
EGOP_LAYER_CLASSES = tuple([EGOP_linear_layer, EGOP_auxiliary_variables_linear_layer])

# Absolute tolerance for torch.isclose / torch.allclose checks
ATOL_THRESHOLD = 1e-5


def setUpModule():
    print(f"\nRunning tests in {__file__}")
    print(f"Using device: {DEVICE}")


def get_V_dict_for_FashionMNIST(OG_model, use_randomized_svd=True, n_components=100):
    trainloader, _, _ = fashionMNIST_dataloader(batch_size=128)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    return compute_V_by_layer(
        model_OG=OG_model,
        k=100,
        data_loader=trainloader,
        criterion=criterion,
        use_randomized_svd=use_randomized_svd,
        n_components=n_components,
    )


def check_if_leaf_node(module):
    """Tests if a module has nested submodules."""
    return len(list(module.children())) == 0


class SharedModelEquivalenceTester:
    """
    A shared class that implements test_models_are_equivalent.
    Named assert_models_are_equivalent (not test_*) to avoid unittest auto-discovery.
    """

    def assert_models_are_equivalent(self, model_1, model_2, verbose=True):
        check_model_structure(model_1, model_2)

        for name, module in model_1.named_modules():
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
                    if isinstance(module_1, EGOP_LAYER_CLASSES):
                        module_1_W_OG_coors = module_1.return_weight_copy_in_OG_coors()
                    else:
                        module_1_W_OG_coors = module_1.weight.detach().clone()
                    if isinstance(module_2, EGOP_LAYER_CLASSES):
                        module_2_W_OG_coors = module_2.return_weight_copy_in_OG_coors()
                    else:
                        module_2_W_OG_coors = module_2.weight.detach().clone()

                    self.assertTrue(
                        torch.allclose(
                            module_1_W_OG_coors,
                            module_2_W_OG_coors,
                            atol=ATOL_THRESHOLD,
                        ),
                        f"Weights not equivalent for modules of name {name} (atol={ATOL_THRESHOLD}).",
                    )
                    if module_1.bias is not None and module_2.bias is not None:
                        self.assertTrue(
                            torch.allclose(
                                module_1.bias.detach().clone(),
                                module_2.bias.detach().clone(),
                                atol=ATOL_THRESHOLD,
                            ),
                            f"Biases not equivalent for modules of name {name} (atol={ATOL_THRESHOLD}).",
                        )
                else:
                    if verbose:
                        print(
                            f"Testing for identicality on modules of types: {type(module_1), type(module_2)}"
                        )
                    self.assertTrue(
                        self.assert_modules_identical(module_1, module_2),
                        f"Parameters not identical for modules of name {name} (modules of type {type(module_1)}).",
                    )
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
        state_dict_1 = module_1.state_dict()
        state_dict_2 = module_2.state_dict()

        self.assertTrue(
            state_dict_1.keys() == state_dict_2.keys(),
            f"Modules do not have identical keys. module_1 keys = {state_dict_1.keys()}, module_2 keys = {state_dict_2.keys()}",
        )

        for k in state_dict_1:
            self.assertTrue(
                torch.allclose(state_dict_1[k], state_dict_2[k], atol=ATOL_THRESHOLD),
                f"Modules do not have identical parameter values for parameter {k} (atol = {ATOL_THRESHOLD}).",
            )

        return True


class TestAuxiliaryLayer(unittest.TestCase):
    """Tests for the EGOP_auxiliary_variables_linear_layer itself."""

    def test_layer_creation(self):
        """Verify layer can be created with a valid orthonormal V."""
        d, r = 100, 20
        V, _ = torch.linalg.qr(torch.randn(d, r))
        layer = EGOP_auxiliary_variables_linear_layer(
            V=V, in_features=10, out_features=10, device="cpu"
        )
        self.assertEqual(layer.weight_r.shape[0], r)
        # weight_d is now stored as the inherited nn.Linear weight (shape (out, in))
        self.assertEqual(layer.weight.shape, (10, 10))
        self.assertEqual(layer.weight.numel(), d)

    def test_forward_pass(self):
        """Verify forward pass produces correct output shape."""
        in_f, out_f, r = 20, 10, 50
        d = in_f * out_f
        V, _ = torch.linalg.qr(torch.randn(d, r))
        layer = EGOP_auxiliary_variables_linear_layer(
            V=V, in_features=in_f, out_features=out_f, bias=True, device="cpu"
        )
        # Initialize weights
        torch.nn.init.normal_(layer.weight_r)
        torch.nn.init.normal_(layer.weight)
        torch.nn.init.zeros_(layer.bias)

        x = torch.randn(8, in_f)
        out = layer(x)
        self.assertEqual(out.shape, (8, out_f))

    # def test_decomposition_identity(self):
    #     """Verify that V @ weight_r + (I - VV^T) @ weight_d equals the optimized forward formula."""
    #     in_f, out_f, r = 5, 4, 10
    #     d = in_f * out_f
    #     V, _ = torch.linalg.qr(torch.randn(d, r))
    #     layer = EGOP_auxiliary_variables_linear_layer(
    #         V=V, in_features=in_f, out_features=out_f, device="cpu"
    #     )
    #     torch.nn.init.normal_(layer.weight_r)
    #     torch.nn.init.normal_(layer.weight_d)

    #     # Method 1: explicit 3-matmul formula
    #     r_term = V @ layer.weight_r
    #     d_term = layer.weight_d - V @ (V.T @ layer.weight_d)
    #     W_explicit = r_term + d_term

    #     # Method 2: optimized 2-matmul formula (as in forward)
    #     W_optimized = layer.weight_d + V @ (layer.weight_r - V.T @ layer.weight_d)

    #     self.assertTrue(
    #         torch.allclose(W_explicit, W_optimized, atol=1e-5),
    #         "Decomposition formulas should be mathematically equivalent.",
    #     )

    def test_non_orthonormal_V_rejected(self):
        """Verify that a non-orthonormal V raises an exception."""
        d, r = 100, 20
        V = torch.randn(d, r)  # Not orthonormal
        with self.assertRaises(Exception):
            EGOP_auxiliary_variables_linear_layer(
                V=V, in_features=10, out_features=10, device="cpu"
            )


class TestAuxiliaryFashionMNISTModel(unittest.TestCase):
    """Tests for the AuxiliaryReparamFashionMNISTClassifier."""

    def test_initialization(self):
        """Verify model can be constructed with EGOP bases."""
        OG_model = FashionMNISTClassifier(**_MINI_MODEL_PARAMS)
        V_dict = get_V_dict_for_FashionMNIST(OG_model)
        aux_model = AuxiliaryReparamFashionMNISTClassifier(
            V_by_layer_dict=V_dict, **_MINI_MODEL_PARAMS
        )
        # Verify auxiliary layers were created
        self.assertIsInstance(aux_model.fc1, EGOP_auxiliary_variables_linear_layer)
        self.assertIsInstance(aux_model.fc2, EGOP_auxiliary_variables_linear_layer)

    def test_forward_pass(self):
        """Verify forward pass produces correct output shape."""
        OG_model = FashionMNISTClassifier(**_MINI_MODEL_PARAMS)
        V_dict = get_V_dict_for_FashionMNIST(OG_model)
        aux_model = AuxiliaryReparamFashionMNISTClassifier(
            V_by_layer_dict=V_dict, **_MINI_MODEL_PARAMS
        )
        aux_model.to(DEVICE)
        aux_model.move_bases_to_device(DEVICE)

        trainloader, _, _ = fashionMNIST_dataloader(batch_size=32)
        Xb, yb = next(iter(trainloader))
        Xb = Xb.to(DEVICE)

        out = aux_model(Xb)
        self.assertEqual(out.shape, (32, _MINI_MODEL_PARAMS["num_classes"]))


class TestAuxiliaryInitEquiv(unittest.TestCase):

    def test_init_equiv_weight_reconstruction(self):
        """
        After init_equiv, reconstructing W = weight_d + V @ (weight_r - V^T @ weight_d),
        where weight_d := self.weight.flatten(), should recover the original weight matrix.
        """
        abs_tolerance = 1e-4

        OG_model = FashionMNISTClassifier(**_MINI_MODEL_PARAMS)
        V_dict = get_V_dict_for_FashionMNIST(OG_model)
        aux_model = AuxiliaryReparamFashionMNISTClassifier(
            V_by_layer_dict=V_dict, **_MINI_MODEL_PARAMS
        )
        aux_model.to(DEVICE)
        aux_model.move_bases_to_device(DEVICE)
        OG_model.to(DEVICE)

        OG_model, aux_model = layerwise_reparam_init_equiv(
            EGOP_model=aux_model, OG_model=OG_model, seed=42
        )

        # Check each auxiliary layer's reconstructed weight matches OG
        for name, module in aux_model.named_modules():
            if isinstance(module, EGOP_auxiliary_variables_linear_layer):
                V = module.V
                weight_d = module.weight.flatten()
                W_reconstructed = (
                    weight_d + V @ (module.weight_r - V.T @ weight_d)
                )
                W_reconstructed = W_reconstructed.reshape(
                    module.out_features, module.in_features
                )
                W_original = OG_model.get_submodule(name).weight
                self.assertTrue(
                    torch.allclose(W_reconstructed, W_original, atol=abs_tolerance),
                    f"Reconstructed weight for layer '{name}' does not match original. "
                    f"Max diff: {(W_reconstructed - W_original).abs().max().item():.6f}",
                )

    def test_GD_equivariance(self):
        """
        Tests that one step of SGD on the auxiliary model produces the same
        loss as one step of SGD on the OG model (gradient descent equivariance).
        """
        error_tolerance = 1e-4
        LR = 0.01

        OG_model = FashionMNISTClassifier(**_MINI_MODEL_PARAMS)
        V_dict = get_V_dict_for_FashionMNIST(OG_model)
        aux_model = AuxiliaryReparamFashionMNISTClassifier(
            V_by_layer_dict=V_dict, **_MINI_MODEL_PARAMS
        )
        aux_model.to(DEVICE)
        aux_model.move_bases_to_device(DEVICE)
        OG_model.to(DEVICE)

        OG_model, aux_model = layerwise_reparam_init_equiv(
            EGOP_model=aux_model, OG_model=OG_model, seed=42
        )

        trainloader, _, _ = fashionMNIST_dataloader(batch_size=64)
        Xb, yb = next(iter(trainloader))
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

        loss_fn = torch.nn.CrossEntropyLoss()

        # Train OG model for a few steps
        og_optimizer = torch.optim.SGD(OG_model.parameters(), lr=LR)
        aux_optimizer = torch.optim.SGD(aux_model.parameters(), lr=LR)

        for step in range(5):
            og_optimizer.zero_grad()
            og_loss = loss_fn(OG_model(Xb), yb)
            og_loss.backward()
            og_optimizer.step()

            aux_optimizer.zero_grad()
            aux_loss = loss_fn(aux_model(Xb), yb)
            aux_loss.backward()
            aux_optimizer.step()

            self.assertTrue(
                abs(og_loss.item() - aux_loss.item()) < error_tolerance,
                f"Step {step}: Loss mismatch OG={og_loss.item():.6f} vs Aux={aux_loss.item():.6f}",
            )


class TestAuxiliaryReparam(SharedModelEquivalenceTester, unittest.TestCase):
    """
    Mirrors TestTinyMNISTReparam.test_equivalence from test_init_equiv.py,
    but applied to the auxiliary variables reparameterization.
    """

    def test_execution(self):
        OG_model = FashionMNISTClassifier(**_MINI_MODEL_PARAMS)
        V_dict = get_V_dict_for_FashionMNIST(OG_model)
        reparam_model = AuxiliaryReparamFashionMNISTClassifier(
            V_by_layer_dict=V_dict, **_MINI_MODEL_PARAMS
        )
        OG_model.to(DEVICE)
        reparam_model.to(DEVICE)
        reparam_model.move_bases_to_device(DEVICE)

        layerwise_reparam_init_equiv(
            EGOP_model=reparam_model, OG_model=OG_model, seed=42
        )
        return

    def test_equivalence(self):
        OG_model = FashionMNISTClassifier(**_MINI_MODEL_PARAMS)
        V_dict = get_V_dict_for_FashionMNIST(OG_model)
        reparam_model = AuxiliaryReparamFashionMNISTClassifier(
            V_by_layer_dict=V_dict, **_MINI_MODEL_PARAMS
        )
        OG_model.to(DEVICE)
        reparam_model.to(DEVICE)
        reparam_model.move_bases_to_device(DEVICE)

        # Before initializing equivalently, models should not be equivalent
        self.assert_models_NOT_equivalent(OG_model, reparam_model, verbose=False)
        OG_model, reparam_model = layerwise_reparam_init_equiv(
            EGOP_model=reparam_model, OG_model=OG_model, seed=42
        )
        # After initializing equivalently, models should be equivalent
        self.assert_models_are_equivalent(OG_model, reparam_model, verbose=False)

        # New OG model with the same seed should still be equivalent to reparam model
        new_OG_model_same_seed = FashionMNISTClassifier(**_MINI_MODEL_PARAMS)
        new_OG_model_same_seed.to(DEVICE)
        new_OG_model_same_seed.reinitialize_seeded(seed=42)
        self.assert_models_are_equivalent(
            new_OG_model_same_seed, reparam_model, verbose=False
        )

        # New OG model with a different seed should NOT be equivalent to reparam model
        new_OG_model_diff_seed = FashionMNISTClassifier(**_MINI_MODEL_PARAMS)
        new_OG_model_diff_seed.to(DEVICE)
        new_OG_model_diff_seed.reinitialize_seeded(seed=987)
        self.assert_models_NOT_equivalent(
            new_OG_model_diff_seed, reparam_model, verbose=False
        )

        # If we randomly reinitialize a parameter of the reparam model, it should no longer be equivalent
        init.normal_(reparam_model.fc1.weight_r)
        self.assert_models_NOT_equivalent(OG_model, reparam_model, verbose=False)
        return


if __name__ == "__main__":
    unittest.main()
