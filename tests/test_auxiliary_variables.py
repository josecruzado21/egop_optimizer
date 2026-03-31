import torch
import unittest

from egop_optimizer.models.FashionMNISTClassfier import (
    FashionMNISTClassifier,
    AuxiliaryReparamFashionMNISTClassifier,
)
from egop_optimizer.models.reparam_layers.reparam_layers import (
    EGOP_auxiliary_variables_linear_layer,
)
from egop_optimizer.dataloaders.fashionMNIST_dataloader import fashionMNIST_dataloader
from egop_optimizer.utils.EGOP_utils import (
    compute_V_by_layer,
    layerwise_reparam_init_equiv,
)
from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()

_MINI_MODEL_PARAMS = {"pool_factor": 1, "hidden_size": 100, "num_classes": 10}


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
        self.assertEqual(layer.weight_d.shape[0], d)

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
        torch.nn.init.normal_(layer.weight_d)
        torch.nn.init.zeros_(layer.bias)

        x = torch.randn(8, in_f)
        out = layer(x)
        self.assertEqual(out.shape, (8, out_f))

    def test_decomposition_identity(self):
        """Verify that V @ weight_r + (I - VV^T) @ weight_d equals the optimized forward formula."""
        in_f, out_f, r = 5, 4, 10
        d = in_f * out_f
        V, _ = torch.linalg.qr(torch.randn(d, r))
        layer = EGOP_auxiliary_variables_linear_layer(
            V=V, in_features=in_f, out_features=out_f, device="cpu"
        )
        torch.nn.init.normal_(layer.weight_r)
        torch.nn.init.normal_(layer.weight_d)

        # Method 1: explicit 3-matmul formula
        r_term = V @ layer.weight_r
        d_term = layer.weight_d - V @ (V.T @ layer.weight_d)
        W_explicit = r_term + d_term

        # Method 2: optimized 2-matmul formula (as in forward)
        W_optimized = layer.weight_d + V @ (layer.weight_r - V.T @ layer.weight_d)

        self.assertTrue(
            torch.allclose(W_explicit, W_optimized, atol=1e-5),
            "Decomposition formulas should be mathematically equivalent.",
        )

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
    """Tests for layerwise_reparam_init_equiv with auxiliary variables layers."""

    def test_init_equiv_output_match(self):
        """
        After init_equiv, the auxiliary model should produce the same output
        as the OG model for any input, since auxiliary variables preserve the
        full parameter space.
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

        # Test on a batch of data
        trainloader, _, _ = fashionMNIST_dataloader(batch_size=64)
        Xb, _ = next(iter(trainloader))
        Xb = Xb.to(DEVICE)

        OG_model.eval()
        aux_model.eval()
        with torch.no_grad():
            og_out = OG_model(Xb)
            aux_out = aux_model(Xb)

        self.assertTrue(
            torch.allclose(og_out, aux_out, atol=abs_tolerance),
            f"OG and auxiliary models should produce same output after init_equiv. "
            f"Max diff: {(og_out - aux_out).abs().max().item():.6f}",
        )

    def test_init_equiv_weight_reconstruction(self):
        """
        After init_equiv, reconstructing W = weight_d + V @ (weight_r - V^T @ weight_d)
        should recover the original weight matrix.
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
                W_reconstructed = (
                    module.weight_d + V @ (module.weight_r - V.T @ module.weight_d)
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


if __name__ == "__main__":
    unittest.main()
