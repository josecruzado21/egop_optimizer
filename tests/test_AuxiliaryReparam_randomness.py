import unittest

import torch

from egop_optimizer.models.FashionMNISTClassfier import (
    FashionMNISTClassifier,
    AuxiliaryReparamFashionMNISTClassifier,
)
from egop_optimizer.dataloaders.fashionMNIST_dataloader import fashionMNIST_dataloader
from egop_optimizer.utils.EGOP_utils import compute_V_by_layer


"""
Randomness control tests for AuxiliaryReparamFashionMNISTClassifier.

Mirrors test_BaseClassifier_basic.py but applied to the auxiliary EGOP model.
Checks BOTH the weight_r parameter (V-subspace coefficients) and the inherited
nn.Linear weight (which plays the role of weight_d in the auxiliary decomposition)
of the first auxiliary layer.
"""


_MINI_MODEL_PARAMS = {"pool_factor": 1, "hidden_size": 100, "num_classes": 10}


def setUpModule():
    print(f"\nRunning tests in {__file__}")


def _make_V_dict():
    """Compute an EGOP eigenbasis dict for the default FashionMNIST architecture."""
    og_model = FashionMNISTClassifier(**_MINI_MODEL_PARAMS)
    trainloader, _, _ = fashionMNIST_dataloader(batch_size=128)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    return compute_V_by_layer(
        model_OG=og_model,
        k=100,
        data_loader=trainloader,
        criterion=criterion,
        use_randomized_svd=True,
        n_components=100,
    )


def _make_aux_model(V_dict, seed=None, weight_dist="default"):
    return AuxiliaryReparamFashionMNISTClassifier(
        V_by_layer_dict=V_dict,
        seed=seed,
        weight_dist=weight_dist,
        **_MINI_MODEL_PARAMS,
    )


def _fc1_weight_r(model):
    return dict(model.named_parameters())["fc1.weight_r"].detach().clone()


def _fc1_weight_d(model):
    return dict(model.named_parameters())["fc1.weight_d"].detach().clone()


class TestAuxiliaryRandomnessControl(unittest.TestCase):
    """
    Mirrors TestBasicSetup from test_BaseClassifier_basic.py for auxiliary models.
    A shared V_dict is computed once per class to keep the suite fast.
    """

    @classmethod
    def setUpClass(cls):
        cls.V_dict = _make_V_dict()

    # ------------------------------------------------------------------
    # reinitialize_seeded: same seed -> same weights (same model, two calls)
    # ------------------------------------------------------------------

    def test_reinitialize_seeded_same_seed_same_weights(self):
        """Calling reinitialize_seeded with the same seed twice on the same model
        should produce identical weight_r and weight_d."""
        model = _make_aux_model(self.V_dict)
        model.reinitialize_seeded(seed=42)
        wr1 = _fc1_weight_r(model)
        wd1 = _fc1_weight_d(model)
        model.reinitialize_seeded(seed=42)
        wr2 = _fc1_weight_r(model)
        wd2 = _fc1_weight_d(model)
        self.assertTrue(
            torch.allclose(wr1, wr2, atol=1e-7),
            "reinitialize_seeded with same seed should produce identical fc1.weight_r",
        )
        self.assertTrue(
            torch.allclose(wd1, wd2, atol=1e-7),
            "reinitialize_seeded with same seed should produce identical fc1.weight_d",
        )

    # ------------------------------------------------------------------
    # reinitialize_seeded: same seed, two independent models -> same weights
    # ------------------------------------------------------------------

    def test_same_seed_gives_same_weights_two_models(self):
        """Two independently created auxiliary models should produce the same
        fc1 weight_r and weight_d after reinitialize_seeded with the same seed."""
        model1 = _make_aux_model(self.V_dict)
        model2 = _make_aux_model(self.V_dict)
        model1.reinitialize_seeded(seed=2)
        model2.reinitialize_seeded(seed=2)
        self.assertTrue(
            torch.allclose(_fc1_weight_r(model1), _fc1_weight_r(model2), atol=1e-7),
            "Two models reinitialised with the same seed should have identical fc1.weight_r",
        )
        self.assertTrue(
            torch.allclose(_fc1_weight_d(model1), _fc1_weight_d(model2), atol=1e-7),
            "Two models reinitialised with the same seed should have identical fc1.weight_d",
        )

    # ------------------------------------------------------------------
    # reinitialize_seeded: different seeds -> different weights
    # ------------------------------------------------------------------

    def test_reinitialize_seeded_different_seeds_different_weights(self):
        """reinitialize_seeded with two different seeds should produce distinct
        fc1.weight_r and/or fc1.weight_d."""
        model = _make_aux_model(self.V_dict)
        model.reinitialize_seeded(seed=1)
        wr1 = _fc1_weight_r(model)
        wd1 = _fc1_weight_d(model)
        model.reinitialize_seeded(seed=2)
        wr2 = _fc1_weight_r(model)
        wd2 = _fc1_weight_d(model)
        # At least one of weight_r / weight_d should differ across seeds.
        self.assertFalse(
            torch.allclose(wr1, wr2, atol=1e-7) and torch.allclose(wd1, wd2, atol=1e-7),
            "Different seeds should produce different fc1.weight_r or fc1.weight_d",
        )

    # ------------------------------------------------------------------
    # reinitialize() advances generator: successive calls should give different weights
    # ------------------------------------------------------------------

    def test_reinitialize_with_init_seed_different_each_call(self):
        """After seeding at construction time, successive reinitialize() calls
        (without an explicit seed) should advance the internal generator and
        produce different weights."""
        model = _make_aux_model(self.V_dict, seed=123)
        model.reinitialize()
        wr1 = _fc1_weight_r(model)
        wd1 = _fc1_weight_d(model)
        model.reinitialize()
        wr2 = _fc1_weight_r(model)
        wd2 = _fc1_weight_d(model)
        self.assertFalse(
            torch.allclose(wr1, wr2, atol=1e-7) and torch.allclose(wd1, wd2, atol=1e-7),
            "Successive reinitialize() calls should produce different weights",
        )

    def test_reinitialize_no_seed_different_each_call(self):
        """A model without a construction seed should also produce different
        weights on successive reinitialize() calls."""
        model = _make_aux_model(self.V_dict)
        model.reinitialize()
        wr1 = _fc1_weight_r(model)
        wd1 = _fc1_weight_d(model)
        model.reinitialize()
        wr2 = _fc1_weight_r(model)
        wd2 = _fc1_weight_d(model)
        self.assertFalse(
            torch.allclose(wr1, wr2, atol=1e-7) and torch.allclose(wd1, wd2, atol=1e-7),
            "Reinitializing without seed should produce different weights each time",
        )

    # ------------------------------------------------------------------
    # Sequence reproducibility: same construction seed -> same sequence
    # ------------------------------------------------------------------

    def test_reinitialize_sequence_matches_for_same_seed(self):
        """Two models constructed with the same seed should produce the same
        weight sequence across multiple reinitialize() calls."""
        model1 = _make_aux_model(self.V_dict, seed=42)
        model2 = _make_aux_model(self.V_dict, seed=42)

        model1.reinitialize()
        wr1_first = _fc1_weight_r(model1)
        wd1_first = _fc1_weight_d(model1)
        model2.reinitialize()
        wr2_first = _fc1_weight_r(model2)
        wd2_first = _fc1_weight_d(model2)

        model1.reinitialize()
        wr1_second = _fc1_weight_r(model1)
        wd1_second = _fc1_weight_d(model1)
        model2.reinitialize()
        wr2_second = _fc1_weight_r(model2)
        wd2_second = _fc1_weight_d(model2)

        self.assertTrue(
            torch.allclose(wr1_first, wr2_first, atol=1e-7)
            and torch.allclose(wd1_first, wd2_first, atol=1e-7),
            "First reinitialise weights should match across models with the same seed",
        )
        self.assertTrue(
            torch.allclose(wr1_second, wr2_second, atol=1e-7)
            and torch.allclose(wd1_second, wd2_second, atol=1e-7),
            "Second reinitialise weights should match across models with the same seed",
        )

    # ------------------------------------------------------------------
    # Construction seed vs reinitialize_seeded: explicit seed overrides
    # ------------------------------------------------------------------

    def test_init_seed_vs_no_seed_after_reinitialize_seeded(self):
        """A model with a construction seed and one without should produce
        identical weights after both call reinitialize_seeded with the same
        explicit seed."""
        model_with_seed = _make_aux_model(self.V_dict, seed=123)
        model_no_seed = _make_aux_model(self.V_dict)

        model_with_seed.reinitialize_seeded(seed=42)
        model_no_seed.reinitialize_seeded(seed=42)

        self.assertTrue(
            torch.allclose(
                _fc1_weight_r(model_with_seed),
                _fc1_weight_r(model_no_seed),
                atol=1e-7,
            ),
            "reinitialize_seeded(42) should override construction seed (weight_r)",
        )
        self.assertTrue(
            torch.allclose(
                _fc1_weight_d(model_with_seed),
                _fc1_weight_d(model_no_seed),
                atol=1e-7,
            ),
            "reinitialize_seeded(42) should override construction seed (weight_d)",
        )

    # ------------------------------------------------------------------
    # All auxiliary layers: verify seed consistency across fc1 and fc2
    # ------------------------------------------------------------------

    def test_all_layers_consistent_under_same_seed(self):
        """With the same explicit seed, fc1 and fc2 weight_r/weight_d should
        all match between two independently created auxiliary models."""
        model1 = _make_aux_model(self.V_dict)
        model2 = _make_aux_model(self.V_dict)
        model1.reinitialize_seeded(seed=7)
        model2.reinitialize_seeded(seed=7)

        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())

        for name in (
            "fc1.weight_r",
            "fc1.weight_d",
            "fc2.weight_r",
            "fc2.weight_d",
        ):
            self.assertTrue(
                torch.allclose(params1[name].detach(), params2[name].detach(), atol=1e-7),
                f"{name}: weights differ between two models with the same seed",
            )

    # ------------------------------------------------------------------
    # Diagnostic: reinitialize_seeded should actually CHANGE the parameters
    # ------------------------------------------------------------------

    def test_reinitialize_seeded_actually_changes_parameters(self):
        """Sanity check: after a fresh model is created and reinitialize_seeded
        is called, the parameters should be DIFFERENT from their pre-reinit
        values. If this fails, reinitialize_seeded is silently doing nothing."""
        model = _make_aux_model(self.V_dict)
        wr_before = _fc1_weight_r(model)
        wd_before = _fc1_weight_d(model)
        model.reinitialize_seeded(seed=42)
        wr_after = _fc1_weight_r(model)
        wd_after = _fc1_weight_d(model)
        self.assertFalse(
            torch.allclose(wr_before, wr_after, atol=1e-7),
            "reinitialize_seeded should change fc1.weight_r (currently a no-op for auxiliary layers)",
        )
        self.assertFalse(
            torch.allclose(wd_before, wd_after, atol=1e-7),
            "reinitialize_seeded should change fc1.weight_d (currently a no-op for auxiliary layers)",
        )


class TestAuxiliaryWeightDistRandomness(unittest.TestCase):
    """
    Diagnostic suite that runs the core randomness checks for each non-default
    weight_dist value ("gaussian", "kaiming_normal", "xavier_normal") to expose
    how reinitialize_seeded behaves on auxiliary layers under each distribution.

    Expected results given the current implementation (Method A, default branch
    only invokes our reset_parameters):
      - "default": all checks pass — weight + weight_r both reset coherently.
      - "gaussian" / "kaiming_normal": self.weight is overwritten directly,
        but weight_r is NOT touched and falls out of sync with self.weight.
        Some checks may pass (weight changes), others may fail (weight_r stale).
      - "xavier_normal": auxiliary layer is not nn.Linear, so the branch's
        isinstance gate is skipped — silent no-op for both weight and weight_r.
    """

    @classmethod
    def setUpClass(cls):
        cls.V_dict = _make_V_dict()

    def _run_diagnostics(self, weight_dist):
        """Run a battery of checks and return a dict of pass/fail results."""
        results = {}

        # 1. Same seed twice on same model -> identical weight + weight_r
        model = _make_aux_model(self.V_dict, weight_dist=weight_dist)
        model.reinitialize_seeded(seed=42)
        wr1 = _fc1_weight_r(model)
        wd1 = _fc1_weight_d(model)
        model.reinitialize_seeded(seed=42)
        wr2 = _fc1_weight_r(model)
        wd2 = _fc1_weight_d(model)
        results["same_seed_weight_d_consistent"] = torch.allclose(wd1, wd2, atol=1e-7)
        results["same_seed_weight_r_consistent"] = torch.allclose(wr1, wr2, atol=1e-7)

        # 2. Two independent models same seed -> identical
        m1 = _make_aux_model(self.V_dict, weight_dist=weight_dist)
        m2 = _make_aux_model(self.V_dict, weight_dist=weight_dist)
        m1.reinitialize_seeded(seed=2)
        m2.reinitialize_seeded(seed=2)
        results["two_models_same_seed_weight_d"] = torch.allclose(
            _fc1_weight_d(m1), _fc1_weight_d(m2), atol=1e-7
        )
        results["two_models_same_seed_weight_r"] = torch.allclose(
            _fc1_weight_r(m1), _fc1_weight_r(m2), atol=1e-7
        )

        # 3. Different seeds -> different
        model = _make_aux_model(self.V_dict, weight_dist=weight_dist)
        model.reinitialize_seeded(seed=1)
        wr_a = _fc1_weight_r(model)
        wd_a = _fc1_weight_d(model)
        model.reinitialize_seeded(seed=2)
        wr_b = _fc1_weight_r(model)
        wd_b = _fc1_weight_d(model)
        results["diff_seed_weight_d_changes"] = not torch.allclose(wd_a, wd_b, atol=1e-7)
        results["diff_seed_weight_r_changes"] = not torch.allclose(wr_a, wr_b, atol=1e-7)

        # 4. Reinit actually changes parameters from construction
        model = _make_aux_model(self.V_dict, weight_dist=weight_dist)
        wr_before = _fc1_weight_r(model)
        wd_before = _fc1_weight_d(model)
        model.reinitialize_seeded(seed=42)
        wr_after = _fc1_weight_r(model)
        wd_after = _fc1_weight_d(model)
        results["reinit_changes_weight_d"] = not torch.allclose(
            wd_before, wd_after, atol=1e-7
        )
        results["reinit_changes_weight_r"] = not torch.allclose(
            wr_before, wr_after, atol=1e-7
        )

        # 5. Coherence: after reinit, the layer should still satisfy the
        # invariant that forward W == self.weight (the init scratchpad).
        # This means: weight_r = V^T @ self.weight.flatten() and
        #             weight_d = (I - VV^T) @ self.weight.flatten().
        model = _make_aux_model(self.V_dict, weight_dist=weight_dist)
        model.reinitialize_seeded(seed=42)
        with torch.no_grad():
            W_flat = model.fc1.weight.flatten()
            expected_wr = model.fc1.V.T @ W_flat
            expected_wd = W_flat - model.fc1.V @ expected_wr
        actual_wr = model.fc1.weight_r
        actual_wd = model.fc1.weight_d
        results["weight_r_equals_VT_weight"] = torch.allclose(
            expected_wr, actual_wr, atol=1e-5
        ) and torch.allclose(expected_wd, actual_wd, atol=1e-5)

        return results

    def _print_diagnostics(self, weight_dist, results):
        print(f"\n  --- weight_dist={weight_dist!r} ---")
        for k, v in results.items():
            mark = "PASS" if v else "FAIL"
            print(f"    [{mark}] {k}")

    def test_default(self):
        results = self._run_diagnostics("default")
        self._print_diagnostics("default", results)

    def test_gaussian(self):
        results = self._run_diagnostics("gaussian")
        self._print_diagnostics("gaussian", results)

    def test_kaiming_normal(self):
        results = self._run_diagnostics("kaiming_normal")
        self._print_diagnostics("kaiming_normal", results)

    def test_xavier_normal(self):
        results = self._run_diagnostics("xavier_normal")
        self._print_diagnostics("xavier_normal", results)


if __name__ == "__main__":
    unittest.main()
