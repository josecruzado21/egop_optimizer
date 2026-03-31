import torch
import unittest

from egop_optimizer.models.LinearFeedforward import (
    LinearFeedforward,
    ReparamLinearFeedforward,
)
from egop_optimizer.dataloaders.linear_networks_dataloader import (
    linear_networks_dataloader,
)
from egop_optimizer.utils.EGOP_utils import compute_V_by_layer


"""
Randomness control tests for ReparamBaseClassifier / ReparamLinearFeedforward.

Mirrors test_BaseClassifier_basic.py to verify that the seeding and
reinitialisation contract holds for reparameterised models.
"""


_DEFAULT_PARAMS = {
    "input_size": 64,
    "output_size": 10,
    "hidden_sizes": [10, 5],
}


def setUpModule():
    print(f"\nRunning tests in {__file__}")


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_V_dict(use_randomized_svd=False):
    """Compute an eigenbasis dict for the default LinearFeedforward architecture."""
    og_model = LinearFeedforward(**_DEFAULT_PARAMS)
    trainloader, _, _ = linear_networks_dataloader(
        batch_size=128,
        input_size=_DEFAULT_PARAMS["input_size"],
        output_size=_DEFAULT_PARAMS["output_size"],
    )
    criterion = torch.nn.MSELoss(reduction="mean")
    return compute_V_by_layer(
        model_OG=og_model,
        k=100,
        data_loader=trainloader,
        criterion=criterion,
        use_randomized_svd=use_randomized_svd,
    )


def _make_reparam_model(V_dict, seed=None):
    return ReparamLinearFeedforward(
        V_by_layer_dict=V_dict,
        seed=seed,
        **_DEFAULT_PARAMS,
    )


def _fc1_weight(model):
    return dict(model.named_parameters())["fc1.weight"].detach().clone()




class TestReparamRandomnessControl(unittest.TestCase):
    """
    Mirrors TestBasicSetup from test_BaseClassifier_basic.py for reparam models.

    A shared V_dict is computed once per class to keep the suite fast.
    """

    @classmethod
    def setUpClass(cls):
        cls.V_dict = _make_V_dict()

    # ------------------------------------------------------------------
    # reinitialize_seeded: same seed → same weights
    # ------------------------------------------------------------------

    def test_reinitialize_seeded_same_seed_same_weights(self):
        """
        Calling reinitialize_seeded with the same seed twice on the same model
        should produce identical fc1 weights.
        """
        model = _make_reparam_model(self.V_dict)
        model.reinitialize_seeded(seed=42)
        w1 = _fc1_weight(model)
        model.reinitialize_seeded(seed=42)
        w2 = _fc1_weight(model)
        self.assertTrue(
            torch.allclose(w1, w2, atol=1e-7),
            "reinitialize_seeded with the same seed should produce identical fc1 weights",
        )

    # ------------------------------------------------------------------
    # reinitialize_seeded: same seed, two independent models → same weights
    # ------------------------------------------------------------------

    def test_same_seed_gives_same_weights_two_models(self):
        """
        Two independently created reparam models should produce the same fc1 weights
        after calling reinitialize_seeded with the same seed.
        """
        model1 = _make_reparam_model(self.V_dict)
        model2 = _make_reparam_model(self.V_dict)
        model1.reinitialize_seeded(seed=2)
        model2.reinitialize_seeded(seed=2)
        w1 = _fc1_weight(model1)
        w2 = _fc1_weight(model2)
        self.assertTrue(
            torch.allclose(w1, w2, atol=1e-7),
            "Two models reinitialised with the same seed should have identical weights",
        )

    # ------------------------------------------------------------------
    # reinitialize_seeded: different seeds → different weights
    # ------------------------------------------------------------------

    def test_reinitialize_seeded_different_seeds_different_weights(self):
        """
        reinitialize_seeded with two different seeds should produce distinct fc1 weights.
        """
        model = _make_reparam_model(self.V_dict)
        model.reinitialize_seeded(seed=1)
        w1 = _fc1_weight(model)
        model.reinitialize_seeded(seed=2)
        w2 = _fc1_weight(model)
        self.assertFalse(
            torch.allclose(w1, w2, atol=1e-7),
            "Different seeds should produce different fc1 weights",
        )



    def test_reinitialize_with_init_seed_different_each_call(self):
        """
        After seeding at construction time, successive reinitialize() calls (without an
        explicit seed) should advance the internal generator and produce different weights.
        """
        model = _make_reparam_model(self.V_dict, seed=123)
        model.reinitialize()
        w1 = _fc1_weight(model)
        model.reinitialize()
        w2 = _fc1_weight(model)
        self.assertFalse(
            torch.allclose(w1, w2, atol=1e-7),
            "Successive reinitialize() calls should produce different weights",
        )

    def test_reinitialize_no_seed_different_each_call(self):
        """
        A model without a construction seed should also produce different weights on
        successive reinitialize() calls.
        """
        model = _make_reparam_model(self.V_dict)
        model.reinitialize()
        w1 = _fc1_weight(model)
        model.reinitialize()
        w2 = _fc1_weight(model)
        self.assertFalse(
            torch.allclose(w1, w2, atol=1e-7),
            "Reinitializing without seed should produce different weights each time",
        )

    # ------------------------------------------------------------------
    # Sequence reproducibility: same construction seed → same sequence
    # ------------------------------------------------------------------

    def test_reinitialize_sequence_matches_for_same_seed(self):
        """
        Two models constructed with the same seed should produce the same weight sequence
        across multiple reinitialize() calls.
        """
        model1 = _make_reparam_model(self.V_dict, seed=42)
        model2 = _make_reparam_model(self.V_dict, seed=42)

        model1.reinitialize()
        w1_first = _fc1_weight(model1)
        model2.reinitialize()
        w2_first = _fc1_weight(model2)

        model1.reinitialize()
        w1_second = _fc1_weight(model1)
        model2.reinitialize()
        w2_second = _fc1_weight(model2)

        self.assertTrue(
            torch.allclose(w1_first, w2_first, atol=1e-7),
            "First reinitialise weights should match across models with the same seed",
        )
        self.assertTrue(
            torch.allclose(w1_second, w2_second, atol=1e-7),
            "Second reinitialise weights should match across models with the same seed",
        )

    # ------------------------------------------------------------------
    # Construction seed vs reinitialize_seeded: explicit seed overrides
    # ------------------------------------------------------------------

    def test_init_seed_vs_no_seed_after_reinitialize_seeded(self):
        """
        A model initialised with a construction seed and one without should produce
        identical fc1 weights after both call reinitialize_seeded with the same explicit seed.
        """
        model_with_seed = _make_reparam_model(self.V_dict, seed=123)
        model_no_seed = _make_reparam_model(self.V_dict)

        model_with_seed.reinitialize_seeded(seed=42)
        model_no_seed.reinitialize_seeded(seed=42)

        w1 = _fc1_weight(model_with_seed)
        w2 = _fc1_weight(model_no_seed)

        self.assertTrue(
            torch.allclose(w1, w2, atol=1e-7),
            "reinitialize_seeded(42) should override construction seed and yield identical weights",
        )

    # ------------------------------------------------------------------
    # All layers: verify seed consistency across fc1, fc2, fc3
    # ------------------------------------------------------------------

    def test_all_layers_consistent_under_same_seed(self):
        """
        With the same explicit seed, fc1, fc2, and fc3 weights should all match
        between two independently created reparam models.
        """
        model1 = _make_reparam_model(self.V_dict)
        model2 = _make_reparam_model(self.V_dict)
        model1.reinitialize_seeded(seed=7)
        model2.reinitialize_seeded(seed=7)

        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())

        for layer_name in ("fc1.weight", "fc2.weight", "fc3.weight"):
            w1 = params1[layer_name].detach()
            w2 = params2[layer_name].detach()
            self.assertTrue(
                torch.allclose(w1, w2, atol=1e-7),
                f"{layer_name}: weights differ between two models with the same seed",
            )


if __name__ == "__main__":
    unittest.main()
