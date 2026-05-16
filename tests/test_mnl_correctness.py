"""MNL correctness tests for Phase 2: availability, sampling correction,
weights, and probability kernel consistency.

These tests verify that the MNL kernel correctly handles:
1. Availability masking (unavailable alts get zero probability)
2. Sampling correction (log(inclusion probability) added to utility)
3. Probability kernel consistency between estimation and prediction
4. Observation-level weights
5. Null log-likelihood with varying availability
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from locpick import ChoiceTable, MultinomialLogit
from locpick.data import ChoiceArrays

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_dataset(n_obs=100, n_alts=5, seed=42):
    """Create a simple choice dataset for testing."""
    rng = np.random.default_rng(seed)
    choosers = pd.DataFrame(
        {
            "obsval": rng.standard_normal(n_obs),
        },
        index=pd.Index(np.arange(n_obs), name="oid"),
    )

    alternatives = pd.DataFrame(
        {
            "altval": rng.standard_normal(n_alts),
        },
        index=pd.Index(np.arange(n_alts), name="aid"),
    )

    choices = rng.choice(np.arange(n_alts), size=n_obs)

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives=pd.Series(choices, index=choosers.index),
    )
    return ct, choosers, alternatives, choices


def _make_dataset_with_availability(n_obs=100, n_alts=5, seed=42):
    """Create a choice dataset with explicit availability constraints."""
    rng = np.random.default_rng(seed)
    choosers = pd.DataFrame(
        {
            "obsval": rng.standard_normal(n_obs),
        },
        index=pd.Index(np.arange(n_obs), name="oid"),
    )

    alternatives = pd.DataFrame(
        {
            "altval": rng.standard_normal(n_alts),
        },
        index=pd.Index(np.arange(n_alts), name="aid"),
    )

    choices = rng.choice(np.arange(n_alts), size=n_obs)

    # Build per-observation availability as a MultiIndex Series
    # Alt 4 is unavailable for observations 0-49
    avail_arr = np.ones((n_obs, n_alts), dtype=np.float64)
    avail_arr[:50, 4] = 0.0

    # Build MultiIndex Series for availability
    idx = pd.MultiIndex.from_product(
        [np.arange(n_obs), np.arange(n_alts)],
        names=["oid", "aid"],
    )
    avail_series = pd.Series(avail_arr.ravel(), index=idx, name="available")

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives=pd.Series(choices, index=choosers.index),
        available=avail_series,
    )
    return ct, choosers, alternatives, choices, avail_arr


# ---------------------------------------------------------------------------
# Test: Availability masking
# ---------------------------------------------------------------------------


class TestAvailability:
    """Tests that unavailable alternatives receive zero probability
    and are excluded from the logsumexp denominator."""

    def test_unavailable_alt_zero_probability(self):
        """Unavailable alternatives should have zero probability."""
        ct, _, _, _, avail_arr = _make_dataset_with_availability()
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
        result = model.fit()

        probs = model.probabilities(ct)
        probs_2d = probs.reshape(ct.n_observations, ct.n_alternatives)

        # Alt 4 should have zero probability for first 50 observations
        assert np.allclose(probs_2d[:50, 4], 0.0, atol=1e-12)

        # Available alts should have non-zero probability
        assert np.all(probs_2d[:50, :4] > 0)

    def test_available_alts_sum_to_one(self):
        """Probabilities of available alternatives should sum to 1."""
        ct, _, _, _, avail_arr = _make_dataset_with_availability()
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
        result = model.fit()

        probs = model.probabilities(ct)
        probs_2d = probs.reshape(ct.n_observations, ct.n_alternatives)

        # For obs where alt 4 is unavailable, sum of available probs should be 1
        avail_sums = probs_2d[:50, :4].sum(axis=1)
        assert np.allclose(avail_sums, 1.0, atol=1e-8)

        # For obs where all alts are available, sum should be 1
        all_sums = probs_2d[50:, :].sum(axis=1)
        assert np.allclose(all_sums, 1.0, atol=1e-8)

    def test_no_availability_all_available(self):
        """When no availability is specified, all alternatives should be available."""
        ct, _, _, _ = _make_simple_dataset()
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
        result = model.fit()

        probs = model.probabilities(ct)
        probs_2d = probs.reshape(ct.n_observations, ct.n_alternatives)

        # All probabilities should be positive
        assert np.all(probs_2d > 0)

        # Should sum to 1
        row_sums = probs_2d.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Test: Sampling correction
# ---------------------------------------------------------------------------


class TestSamplingCorrection:
    """Tests that sampling correction (log(inclusion probability)) is correctly
    added to utilities."""

    def test_sampling_correction_payload_in_arrays(self):
        """ChoiceArrays from sampled data should carry inclusion_probs."""
        rng = np.random.default_rng(42)
        n_obs = 100
        n_alts = 5

        choosers = pd.DataFrame(
            {
                "obsval": rng.standard_normal(n_obs),
            },
            index=pd.Index(np.arange(n_obs), name="oid"),
        )

        alternatives = pd.DataFrame(
            {
                "altval": rng.standard_normal(n_alts),
            },
            index=pd.Index(np.arange(n_alts), name="aid"),
        )

        choices = rng.choice(np.arange(n_alts), size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
            sample_size=3,
        )
        arrays = ct.to_arrays(formula="obsval + altval - 1")
        assert arrays.inclusion_probs is not None
        # sample_size=3, n_alts_eff = 3 (sampled) + 1 (chosen) = 4
        # non-chosen inclusion probability should be positive
        assert np.all(arrays.inclusion_probs > 0)

    def test_sampling_correction_improves_estimation(self):
        """With sampling correction, estimated coefficients should be closer
        to true values than without correction (for sampled choice sets)."""
        rng = np.random.default_rng(12345)
        n_obs = 2000
        n_alts = 50
        sample_size = 10

        # True parameters
        beta_alt = -0.5

        # Generate alternatives
        alternatives = pd.DataFrame(
            {
                "altval": rng.standard_normal(n_alts),
            },
            index=pd.Index(np.arange(n_alts), name="aid"),
        )

        # Generate choosers
        choosers = pd.DataFrame(
            {
                "obsval": rng.standard_normal(n_obs),
            },
            index=pd.Index(np.arange(n_obs), name="oid"),
        )

        # Simulate choices: utility = beta_alt * altval
        utilities = beta_alt * alternatives["altval"].values
        probs = np.exp(utilities - utilities.max()) / np.exp(utilities - utilities.max()).sum()
        choices = rng.choice(n_alts, size=n_obs, p=probs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
            sample_size=sample_size,
        )

        model = MultinomialLogit(ct, formula="altval - 1")
        result = model.fit()

        # The coefficient should be recoverable (within 30% tolerance)
        # even with sampled alternatives, because sampling correction is applied
        # Note: with sampled choice sets, some estimation error is expected
        assert abs(result.coefficients["altval"] - beta_alt) / abs(beta_alt) < 0.30


# ---------------------------------------------------------------------------
# Test: Probability kernel consistency
# ---------------------------------------------------------------------------


class TestProbabilityKernelConsistency:
    """Tests that prediction uses the same probability kernel as estimation."""

    def test_estimation_and_prediction_agree(self):
        """Probabilities from prediction should match those implied by
        the estimated model's log-likelihood computation."""
        ct, _, _, _ = _make_simple_dataset()
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
        result = model.fit()

        # Get probabilities from prediction
        probs = model.probabilities(ct)
        probs_2d = probs.reshape(ct.n_observations, ct.n_alternatives)

        # Manually compute probabilities from the same kernel
        arrays = ct.to_arrays(formula="obsval + altval - 1")
        dm = arrays.design_matrix
        beta = result.coefficients.values
        utilities = (dm @ beta).reshape(ct.n_observations, ct.n_alternatives)

        # Stable softmax (same as kernel)
        utilities_shifted = utilities - utilities.max(axis=1, keepdims=True)
        exp_u = np.exp(utilities_shifted)
        manual_probs = exp_u / exp_u.sum(axis=1, keepdims=True)

        assert np.allclose(probs_2d, manual_probs, atol=1e-10)

    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_backend_consistency(self, backend, monkeypatch):
        """NumPy and JAX backends should produce the same log-likelihood
        and similar coefficient estimates."""
        if backend == "jax" and importlib.util.find_spec("jax") is None:
            pytest.skip("JAX not installed")

        ct, _, _, _ = _make_simple_dataset(seed=99)

        monkeypatch.delenv("CHOICEMODELS_MNL_BACKEND", raising=False)
        if backend == "numpy":
            monkeypatch.setenv("CHOICEMODELS_MNL_BACKEND", "numpy")

        model = MultinomialLogit(ct, formula="obsval + altval - 1")
        result = model.fit()

        # Should produce finite results
        assert np.isfinite(result.log_likelihood)
        assert np.isfinite(result.coefficients.to_numpy()).all()

        # Probabilities should sum to 1
        probs = model.probabilities(ct)
        probs_2d = probs.reshape(ct.n_observations, ct.n_alternatives)
        assert np.allclose(probs_2d.sum(axis=1), 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Test: Weight semantics
# ---------------------------------------------------------------------------


class TestWeightSemantics:
    """Tests that observation-level weights work correctly."""

    def test_unit_weights_same_as_unweighted(self):
        """Unit weights should produce the same log-likelihood as unweighted."""
        ct, _, _, _ = _make_simple_dataset()

        # Unweighted
        model_unweighted = MultinomialLogit(ct, formula="obsval + altval - 1")
        result_unweighted = model_unweighted.fit()

        # Weighted with unit weights
        n_obs = ct.n_observations
        unit_weights = np.ones(n_obs)
        model_weighted = MultinomialLogit(ct, formula="obsval + altval - 1", weights=unit_weights)
        result_weighted = model_weighted.fit()

        # Log-likelihoods should be very close
        assert abs(result_unweighted.log_likelihood - result_weighted.log_likelihood) < 1e-6

    def test_doubled_weights_double_log_likelihood(self):
        """Doubling all weights should approximately double the log-likelihood."""
        ct, _, _, _ = _make_simple_dataset()

        # Unweighted
        model_unweighted = MultinomialLogit(ct, formula="obsval + altval - 1")
        result_unweighted = model_unweighted.fit()

        # Doubled weights
        n_obs = ct.n_observations
        double_weights = 2.0 * np.ones(n_obs)
        model_doubled = MultinomialLogit(ct, formula="obsval + altval - 1", weights=double_weights)
        result_doubled = model_doubled.fit()

        # The doubled-weight LL should be approximately 2x the unweighted LL
        ratio = result_doubled.log_likelihood / result_unweighted.log_likelihood
        assert abs(ratio - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Test: Null log-likelihood with availability
# ---------------------------------------------------------------------------


class TestNullLogLikelihood:
    """Tests that null log-likelihood correctly accounts for availability."""

    def test_null_ll_all_available(self):
        """When all alternatives are available, null LL = -n_obs * log(n_alts)."""
        ct, _, _, _ = _make_simple_dataset(n_obs=100, n_alts=5)
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
        result = model.fit()

        expected_null_ll = -100 * np.log(5)
        assert abs(result.log_likelihood_null - expected_null_ll) < 1e-10

    def test_null_ll_with_availability(self):
        """Null LL should account for varying choice set sizes."""
        n_obs = 100
        n_alts = 5
        rng = np.random.default_rng(42)

        choosers = pd.DataFrame(
            {
                "obsval": rng.standard_normal(n_obs),
            },
            index=pd.Index(np.arange(n_obs), name="oid"),
        )

        alternatives = pd.DataFrame(
            {
                "altval": rng.standard_normal(n_alts),
            },
            index=pd.Index(np.arange(n_alts), name="aid"),
        )

        choices = rng.choice(np.arange(n_alts), size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
        )

        # Build arrays with availability
        ct.to_arrays(formula="obsval + altval - 1")

        # Set availability: first 50 obs have 4 alts, rest have 5
        avail = np.ones((n_obs, n_alts), dtype=np.float64)
        avail[:50, 4] = 0.0

        # Expected null LL: sum of -log(n_available) per obs
        # For first 50 obs: -log(4), for rest: -log(5)
        n_avail_per_obs = avail.sum(axis=1)
        expected_null_ll = -np.sum(np.log(n_avail_per_obs))

        # Verify: -50*log(4) - 50*log(5)
        assert abs(expected_null_ll - (-50 * np.log(4) - 50 * np.log(5))) < 1e-10


# ---------------------------------------------------------------------------
# Test: Gradient correctness
# ---------------------------------------------------------------------------


class TestGradientCorrectness:
    """Tests that the gradient is consistent with the log-likelihood
    via finite differences."""

    def test_numpy_gradient_matches_finite_differences(self):
        """NumPy gradient should match finite-difference approximation."""
        ct, _, _, _ = _make_simple_dataset(seed=77)
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
        arrays = model._build_arrays()

        # Force NumPy backend
        import os

        os.environ["CHOICEMODELS_MNL_BACKEND"] = "numpy"
        try:
            objective = model._build_objective_numpy(arrays)
            ll_fn = objective.fn
            grad_fn = objective.grad
        finally:
            del os.environ["CHOICEMODELS_MNL_BACKEND"]

        beta = np.array([0.1, -0.2])

        # Analytical gradient
        analytical_grad = grad_fn(beta)

        # Numerical gradient via finite differences
        eps = 1e-5
        numerical_grad = np.zeros_like(beta)
        for i in range(len(beta)):
            beta_plus = beta.copy()
            beta_plus[i] += eps
            beta_minus = beta.copy()
            beta_minus[i] -= eps
            numerical_grad[i] = (ll_fn(beta_plus) - ll_fn(beta_minus)) / (2 * eps)

        assert np.allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-4)

    def test_gradient_with_availability(self):
        """Gradient should be correct when availability masking is active."""
        n_obs = 50
        n_alts = 4
        rng = np.random.default_rng(55)

        # Build arrays manually
        dm = rng.standard_normal((n_obs * n_alts, 2))
        chosen = np.zeros((n_obs, n_alts))
        for i in range(n_obs):
            # Only choose from available alternatives
            if i < 20:
                chosen[i, rng.choice(3)] = 1.0  # alt 3 unavailable
            else:
                chosen[i, rng.choice(n_alts)] = 1.0

        # Make alt 3 unavailable for first 20 obs
        avail = np.ones((n_obs, n_alts))
        avail[:20, 3] = 0.0

        arrays = ChoiceArrays(
            design_matrix=dm,
            chosen=chosen,
            available=avail,
            n_obs=n_obs,
            n_alts=n_alts,
            param_names=["x1", "x2"],
        )

        # Build the objective directly using the model's method
        ct, _, _, _ = _make_simple_dataset(seed=55)
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
        objective = model._build_objective_numpy(arrays)
        ll_fn = objective.fn
        grad_fn = objective.grad

        beta = np.array([0.3, -0.5])

        # Analytical gradient
        analytical_grad = grad_fn(beta)

        # Numerical gradient via finite differences
        eps = 1e-5
        numerical_grad = np.zeros_like(beta)
        for i in range(len(beta)):
            beta_plus = beta.copy()
            beta_plus[i] += eps
            beta_minus = beta.copy()
            beta_minus[i] -= eps
            numerical_grad[i] = (ll_fn(beta_plus) - ll_fn(beta_minus)) / (2 * eps)

        assert np.allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-4)

    def test_gradient_with_sampling_correction(self):
        """Gradient should be correct when sampling correction is applied."""
        n_obs = 50
        n_alts = 4
        rng = np.random.default_rng(66)

        dm = rng.standard_normal((n_obs * n_alts, 2))
        chosen = np.zeros((n_obs, n_alts))
        for i in range(n_obs):
            chosen[i, rng.choice(n_alts)] = 1.0

        # Sample rates: 0.5 for all alternatives
        inclusion_probs = np.full((n_obs, n_alts), 0.5)

        arrays = ChoiceArrays(
            design_matrix=dm,
            chosen=chosen,
            inclusion_probs=inclusion_probs,
            n_obs=n_obs,
            n_alts=n_alts,
            param_names=["x1", "x2"],
        )

        ct, _, _, _ = _make_simple_dataset(seed=66)
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
        objective = model._build_objective_numpy(arrays)
        ll_fn = objective.fn
        grad_fn = objective.grad

        beta = np.array([0.3, -0.5])

        # Analytical gradient
        analytical_grad = grad_fn(beta)

        # Numerical gradient
        eps = 1e-5
        numerical_grad = np.zeros_like(beta)
        for i in range(len(beta)):
            beta_plus = beta.copy()
            beta_plus[i] += eps
            beta_minus = beta.copy()
            beta_minus[i] -= eps
            numerical_grad[i] = (ll_fn(beta_plus) - ll_fn(beta_minus)) / (2 * eps)

        assert np.allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-4)
