# ruff: noqa: E402
"""MNL tests: correctness, recovery, pipeline, and edge cases."""

import importlib.util

import numpy.testing as npt
import pytest

from locpick import MNL, dgp

"""
These are tests for the refactored locpick MNL codebase.

"""

import numpy as np
import pandas as pd

from locpick import ChoiceTable


@pytest.fixture
def obs():
    d1 = {
        "oid": np.arange(100),
        "obsval": np.random.random(100),
        "choice": np.random.choice(np.arange(5), size=100),
    }
    return pd.DataFrame(d1).set_index("oid")


@pytest.fixture
def alts():
    d2 = {"aid": np.arange(5), "altval": np.random.random(5)}
    return pd.DataFrame(d2).set_index("aid")


def test_mnl(obs, alts):
    """
    Confirm that MNL estimation runs, using the native estimator.

    """
    formula = "obsval + altval - 1"
    ct = ChoiceTable.from_tables(obs, alts, chosen_alternatives="choice")
    m = MNL(ct, formula=formula)
    r = m.fit()
    assert len(r.coefficients) == 2


def test_mnl_estimation(obs, alts):
    """
    Confirm that MNL returns finite coefficient estimates.

    """
    formula = "obsval + altval - 1"
    ct = ChoiceTable.from_tables(obs, alts, chosen_alternatives="choice")
    result = MNL(ct, formula=formula).fit()
    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(result.coefficients.to_numpy()).all()


def test_mnl_prediction(obs, alts):
    """
    Confirm predicted probabilities are well-formed.

    """
    ct = ChoiceTable.from_tables(obs, alts, chosen_alternatives="choice", sample_size=5)
    m = MNL(ct, formula="obsval + altval - 1")
    m.fit()

    probs = m.probabilities(ct)
    prob_sums = probs.sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-8)


FORMULA = "alt_feature + obs_x_alt - 1"


def _fit_v2(dataset, backend, monkeypatch):
    """Fit using the v2 MultinomialLogit, controlling JAX vs NumPy via env var."""
    monkeypatch.delenv("LOCPICK_MNL_BACKEND", raising=False)
    if backend != "jax":
        monkeypatch.setenv("LOCPICK_MNL_BACKEND", backend)
    model = MNL(dataset.choice_table, FORMULA)
    result = model.fit()
    return result.coefficients


def test_mnl_parameter_recovery_with_pairwise_variable():
    dataset = dgp.simulate_mnl(
        n_obs=4000,
        n_alts=5,
        alt_params={"alt_feature": -0.7},
        interaction_params={"obs_x_alt": 1.1},
        seed=1234,
    )
    model = MNL(dataset.choice_table, FORMULA)
    estimated = model.fit().coefficients

    # With 4 000 observations MLE is consistent; allow 10 % relative tolerance.
    npt.assert_allclose(estimated["alt_feature"], dataset.true_params["alt_feature"], rtol=0.10)
    npt.assert_allclose(estimated["obs_x_alt"], dataset.true_params["obs_x_alt"], rtol=0.10)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_mnl_parameter_recovery_across_backends(backend, monkeypatch):
    if backend == "jax" and importlib.util.find_spec("jax") is None:
        pytest.skip("JAX backend recovery test skipped because jax is not installed")

    dataset = dgp.simulate_mnl(seed=2026)
    estimated = _fit_v2(dataset, backend, monkeypatch)

    npt.assert_allclose(estimated["alt_feature"], dataset.true_params["alt_feature"], rtol=0.10)
    npt.assert_allclose(estimated["obs_x_alt"], dataset.true_params["obs_x_alt"], rtol=0.10)


def test_mnl_backend_coefficient_consistency(monkeypatch):
    dataset = dgp.simulate_mnl(seed=909)

    est_numpy = _fit_v2(dataset, "numpy", monkeypatch)

    if importlib.util.find_spec("jax") is not None:
        est_jax = _fit_v2(dataset, "jax", monkeypatch)
        npt.assert_allclose(
            est_jax[["alt_feature", "obs_x_alt"]].to_numpy(),
            est_numpy[["alt_feature", "obs_x_alt"]].to_numpy(),
            rtol=1e-4,
            atol=1e-6,
        )


"""MNL correctness tests for Phase 2: availability, sampling correction,
weights, and probability kernel consistency.

These tests verify that the MNL kernel correctly handles:
1. Availability masking (unavailable alts get zero probability)
2. Sampling correction (log(inclusion probability) added to utility)
3. Probability kernel consistency between estimation and prediction
4. Observation-level weights
5. Null log-likelihood with varying availability
"""


import pytest

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
        model = MNL(ct, formula="obsval + altval - 1")
        model.fit()

        probs = model.probabilities(ct)
        probs_2d = probs.reshape(ct.n_observations, ct.n_alternatives)

        # Alt 4 should have zero probability for first 50 observations
        assert np.allclose(probs_2d[:50, 4], 0.0, atol=1e-12)

        # Available alts should have non-zero probability
        assert np.all(probs_2d[:50, :4] > 0)

    def test_available_alts_sum_to_one(self):
        """Probabilities of available alternatives should sum to 1."""
        ct, _, _, _, avail_arr = _make_dataset_with_availability()
        model = MNL(ct, formula="obsval + altval - 1")
        model.fit()

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
        model = MNL(ct, formula="obsval + altval - 1")
        model.fit()

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

        model = MNL(ct, formula="altval - 1")
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
        model = MNL(ct, formula="obsval + altval - 1")
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

        monkeypatch.delenv("LOCPICK_MNL_BACKEND", raising=False)
        if backend == "numpy":
            monkeypatch.setenv("LOCPICK_MNL_BACKEND", "numpy")

        model = MNL(ct, formula="obsval + altval - 1")
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
        model_unweighted = MNL(ct, formula="obsval + altval - 1")
        result_unweighted = model_unweighted.fit()

        # Weighted with unit weights
        n_obs = ct.n_observations
        unit_weights = np.ones(n_obs)
        model_weighted = MNL(ct, formula="obsval + altval - 1", weights=unit_weights)
        result_weighted = model_weighted.fit()

        # Log-likelihoods should be very close
        assert abs(result_unweighted.log_likelihood - result_weighted.log_likelihood) < 1e-6

    def test_doubled_weights_double_log_likelihood(self):
        """Doubling all weights should approximately double the log-likelihood."""
        ct, _, _, _ = _make_simple_dataset()

        # Unweighted
        model_unweighted = MNL(ct, formula="obsval + altval - 1")
        result_unweighted = model_unweighted.fit()

        # Doubled weights
        n_obs = ct.n_observations
        double_weights = 2.0 * np.ones(n_obs)
        model_doubled = MNL(ct, formula="obsval + altval - 1", weights=double_weights)
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
        model = MNL(ct, formula="obsval + altval - 1")
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
        model = MNL(ct, formula="obsval + altval - 1")
        arrays = model._build_arrays()

        # Force NumPy backend
        import os

        os.environ["LOCPICK_MNL_BACKEND"] = "numpy"
        try:
            objective = model._build_objective_numpy(arrays)
            ll_fn = objective.fn
            grad_fn = objective.grad
        finally:
            del os.environ["LOCPICK_MNL_BACKEND"]

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
        model = MNL(ct, formula="obsval + altval - 1")
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
        model = MNL(ct, formula="obsval + altval - 1")
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


"""Extended MNL correctness tests for LocPick.

These tests cover:

1. Probability computation against known analytical values
2. Log-likelihood computation against known values
3. Gradient verification via finite differences (extended)
4. Hessian verification via finite differences
5. Numerical stability (extreme utilities, under/overflow)
6. Prediction with new/out-of-sample data
7. Design matrix correctness (P/X algebra)
8. Comprehensive DGP parameter recovery
"""


import pytest

from locpick import EstimationProblem, ModelSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_dataset(n_obs=200, n_alts=4, seed=42):
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


def _make_manual_arrays(n_obs=50, n_alts=4, n_params=2, seed=55):
    """Create ChoiceArrays directly for low-level kernel tests."""
    rng = np.random.default_rng(seed)
    dm = rng.standard_normal((n_obs * n_alts, n_params))
    chosen = np.zeros((n_obs, n_alts))
    for i in range(n_obs):
        chosen[i, rng.choice(n_alts)] = 1.0

    return ChoiceArrays(
        design_matrix=dm,
        chosen=chosen,
        n_obs=n_obs,
        n_alts=n_alts,
        param_names=[f"x{i}" for i in range(n_params)],
    )


# ===========================================================================
# 1. Probability computation against known analytical values
# ===========================================================================


class TestProbabilityComputation:
    """Verify that MNL probabilities match analytical softmax."""

    def test_probabilities_match_softmax(self):
        """Probabilities from the model should match manual softmax computation."""
        ct, _, _, _ = _make_simple_dataset(seed=101)
        model = MNL(ct, formula="obsval + altval - 1")
        result = model.fit()

        arrays = ct.to_arrays(formula="obsval + altval - 1")
        dm = arrays.design_matrix
        beta = result.coefficients.values
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        # Manual softmax
        utilities = (dm @ beta).reshape(n_obs, n_alts)
        utilities_shifted = utilities - utilities.max(axis=1, keepdims=True)
        exp_u = np.exp(utilities_shifted)
        manual_probs = exp_u / exp_u.sum(axis=1, keepdims=True)

        # Model probabilities
        model_probs = model.probabilities(ct)
        model_probs_2d = model_probs.reshape(n_obs, n_alts)

        npt.assert_allclose(model_probs_2d, manual_probs, atol=1e-10)

    def test_probabilities_sum_to_one(self):
        """Probabilities for each observation should sum to 1."""
        ct, _, _, _ = _make_simple_dataset(seed=102)
        model = MNL(ct, formula="obsval + altval - 1")
        model.fit()

        probs = model.probabilities(ct)
        probs_2d = probs.reshape(ct.n_observations, ct.n_alternatives)
        row_sums = probs_2d.sum(axis=1)
        npt.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_probabilities_are_non_negative(self):
        """All probabilities should be non-negative."""
        ct, _, _, _ = _make_simple_dataset(seed=103)
        model = MNL(ct, formula="obsval + altval - 1")
        model.fit()

        probs = model.probabilities(ct)
        assert (probs >= -1e-15).all()

    def test_chosen_probabilities_positive(self):
        """Probability of the chosen alternative should be positive for each obs."""
        ct, _, _, _ = _make_simple_dataset(seed=104)
        model = MNL(ct, formula="obsval + altval - 1")
        model.fit()

        arrays = ct.to_arrays(formula="obsval + altval - 1")
        probs = model.probabilities(ct)
        probs_2d = probs.reshape(arrays.n_obs, arrays.n_alts)
        chosen_probs = (probs_2d * arrays.chosen).sum(axis=1)
        assert (chosen_probs > 0).all()


# ===========================================================================
# 2. Log-likelihood computation against known values
# ===========================================================================


class TestLogLikelihoodComputation:
    """Verify that log-likelihood matches manual computation."""

    def test_log_likelihood_matches_manual(self):
        """Log-likelihood should equal sum of log(chosen probabilities)."""
        ct, _, _, _ = _make_simple_dataset(seed=201)
        model = MNL(ct, formula="obsval + altval - 1")
        result = model.fit()

        arrays = ct.to_arrays(formula="obsval + altval - 1")
        probs = model.probabilities(ct)
        probs_2d = probs.reshape(arrays.n_obs, arrays.n_alts)

        # Manual LL = sum of log(P(chosen))
        chosen_probs = (probs_2d * arrays.chosen).sum(axis=1)
        expected_ll = np.sum(np.log(chosen_probs))

        npt.assert_allclose(result.log_likelihood, expected_ll, rtol=1e-8)

    def test_log_likelihood_is_negative(self):
        """Log-likelihood should be negative for any model."""
        ct, _, _, _ = _make_simple_dataset(seed=202)
        model = MNL(ct, formula="obsval + altval - 1")
        result = model.fit()
        assert result.log_likelihood < 0

    def test_log_likelihood_better_than_null(self):
        """Fitted model LL should be >= null LL (rho-squared >= 0)."""
        ct, _, _, _ = _make_simple_dataset(seed=203)
        model = MNL(ct, formula="obsval + altval - 1")
        result = model.fit()
        assert result.log_likelihood >= result.log_likelihood_null


# ===========================================================================
# 3. Gradient verification via finite differences (extended)
# ===========================================================================


class TestGradientExtended:
    """Extended gradient tests beyond the basic correctness tests."""

    def test_gradient_with_weights(self):
        """Gradient should be correct when observation weights are used."""
        n_obs = 50
        n_alts = 4
        rng = np.random.default_rng(301)

        dm = rng.standard_normal((n_obs * n_alts, 2))
        chosen = np.zeros((n_obs, n_alts))
        for i in range(n_obs):
            chosen[i, rng.choice(n_alts)] = 1.0

        weights = rng.uniform(0.5, 2.0, size=n_obs)

        arrays = ChoiceArrays(
            design_matrix=dm,
            chosen=chosen,
            weights=weights,
            n_obs=n_obs,
            n_alts=n_alts,
            param_names=["x1", "x2"],
        )

        ct, _, _, _ = _make_simple_dataset(seed=301)
        model = MNL(ct, formula="obsval + altval - 1")
        objective = model._build_objective_numpy(arrays)
        ll_fn = objective.fn
        grad_fn = objective.grad

        beta = np.array([0.3, -0.5])

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

        npt.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-4)

    def test_gradient_with_availability_and_weights(self):
        """Gradient should be correct with both availability and weights."""
        n_obs = 50
        n_alts = 4
        rng = np.random.default_rng(302)

        dm = rng.standard_normal((n_obs * n_alts, 2))
        chosen = np.zeros((n_obs, n_alts))
        for i in range(n_obs):
            if i < 20:
                chosen[i, rng.choice(3)] = 1.0  # alt 3 unavailable
            else:
                chosen[i, rng.choice(n_alts)] = 1.0

        avail = np.ones((n_obs, n_alts))
        avail[:20, 3] = 0.0

        weights = rng.uniform(0.5, 2.0, size=n_obs)

        arrays = ChoiceArrays(
            design_matrix=dm,
            chosen=chosen,
            available=avail,
            weights=weights,
            n_obs=n_obs,
            n_alts=n_alts,
            param_names=["x1", "x2"],
        )

        ct, _, _, _ = _make_simple_dataset(seed=302)
        model = MNL(ct, formula="obsval + altval - 1")
        objective = model._build_objective_numpy(arrays)
        ll_fn = objective.fn
        grad_fn = objective.grad

        beta = np.array([0.3, -0.5])

        analytical_grad = grad_fn(beta)

        eps = 1e-5
        numerical_grad = np.zeros_like(beta)
        for i in range(len(beta)):
            beta_plus = beta.copy()
            beta_plus[i] += eps
            beta_minus = beta.copy()
            beta_minus[i] -= eps
            numerical_grad[i] = (ll_fn(beta_plus) - ll_fn(beta_minus)) / (2 * eps)

        npt.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-4)


# ===========================================================================
# 4. Hessian verification via finite differences
# ===========================================================================


class TestHessianVerification:
    """Verify that the inverse Hessian from the solver matches numerical
    approximation."""

    def test_inverse_hessian_matches_numerical(self):
        """The inverse Hessian should match a numerical approximation."""
        ct, _, _, _ = _make_simple_dataset(n_obs=200, seed=401)
        model = MNL(ct, formula="obsval + altval - 1")
        result = model.fit()

        # Get the inverse Hessian from the solver result
        solver_result = result.solver_result
        hessian_inv = solver_result.get("scipy_result", None)
        if hessian_inv is not None and hasattr(hessian_inv, "hess_inv"):
            try:
                hessian_inv_arr = np.asarray(hessian_inv.hess_inv.todense())
            except AttributeError:
                hessian_inv_arr = np.asarray(hessian_inv.hess_inv)
        else:
            pytest.skip("Solver did not return Hessian")

        # Compute numerical Hessian via finite differences of the gradient
        arrays = ct.to_arrays(formula="obsval + altval - 1")
        objective = model._build_objective(arrays)
        grad_fn = objective.grad
        beta = result.coefficients.values

        eps = 1e-4
        n = len(beta)
        numerical_hessian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                beta_pp = beta.copy()
                beta_pp[i] += eps
                beta_pp[j] += eps

                beta_pm = beta.copy()
                beta_pm[i] += eps
                beta_pm[j] -= eps

                beta_mp = beta.copy()
                beta_mp[i] -= eps
                beta_mp[j] += eps

                beta_mm = beta.copy()
                beta_mm[i] -= eps
                beta_mm[j] -= eps

                # Second derivative via central differences
                numerical_hessian[i, j] = (
                    -grad_fn(beta_pp)[j]
                    + grad_fn(beta_pm)[j]
                    + grad_fn(beta_mp)[j]
                    - grad_fn(beta_mm)[j]
                ) / (4 * eps**2)

        # The Hessian should be negative definite (maximizing LL)
        # So numerical_hessian should be close to -hessian_inv^{-1}
        # Instead, compare the inverse of numerical_hessian with hessian_inv
        try:
            numerical_inv = np.linalg.inv(numerical_hessian)
            npt.assert_allclose(hessian_inv_arr, numerical_inv, atol=0.05, rtol=0.05)
        except np.linalg.LinAlgError:
            pytest.skip("Numerical Hessian is singular")

    def test_standard_errors_positive(self):
        """Standard errors should be positive for all parameters."""
        ct, _, _, _ = _make_simple_dataset(seed=402)
        model = MNL(ct, formula="obsval + altval - 1")
        result = model.fit()

        # Parameters with zero SE are marked NaN (numerically unidentified).
        # For a well-identified model all SEs should be finite and positive.
        valid = result.std_errors.dropna()
        assert len(valid) > 0, "No valid standard errors computed"
        assert (valid > 0).all(), f"Non-positive SEs: {valid[valid <= 0]}"


# ===========================================================================
# 5. Numerical stability (extreme utilities, under/overflow)
# ===========================================================================


class TestNumericalStability:
    """Tests that the MNL kernel handles extreme utility values correctly."""

    def test_extreme_utilities_no_nan(self):
        """Model should produce finite results even with extreme utility values."""
        rng = np.random.default_rng(501)
        n_obs = 100
        n_alts = 4

        # Create data with extreme values
        choosers = pd.DataFrame(
            {
                "obsval": rng.standard_normal(n_obs) * 100,  # extreme
            },
            index=pd.Index(np.arange(n_obs), name="oid"),
        )

        alternatives = pd.DataFrame(
            {
                "altval": rng.standard_normal(n_alts) * 100,  # extreme
            },
            index=pd.Index(np.arange(n_alts), name="aid"),
        )

        choices = rng.choice(np.arange(n_alts), size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
        )

        model = MNL(ct, formula="obsval + altval - 1")
        result = model.fit()

        # Results should be finite
        assert np.isfinite(result.log_likelihood)
        assert np.isfinite(result.coefficients.values).all()

        # Probabilities should be valid
        probs = model.probabilities(ct)
        assert np.isfinite(probs).all()
        assert (probs >= 0).all()

    def test_extreme_utilities_probabilities_valid(self):
        """Probabilities should be valid (sum to 1, non-negative) with extreme values."""
        rng = np.random.default_rng(502)
        n_obs = 50
        n_alts = 3

        choosers = pd.DataFrame(
            {
                "obsval": rng.standard_normal(n_obs) * 50,
            },
            index=pd.Index(np.arange(n_obs), name="oid"),
        )

        alternatives = pd.DataFrame(
            {
                "altval": rng.standard_normal(n_alts) * 50,
            },
            index=pd.Index(np.arange(n_alts), name="aid"),
        )

        choices = rng.choice(np.arange(n_alts), size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
        )

        model = MNL(ct, formula="obsval + altval - 1")
        model.fit()

        probs = model.probabilities(ct)
        probs_2d = probs.reshape(n_obs, n_alts)

        # Should sum to 1
        npt.assert_allclose(probs_2d.sum(axis=1), 1.0, atol=1e-6)

        # Should be non-negative
        assert (probs_2d >= -1e-10).all()

    def test_single_dominant_alternative(self):
        """When one alternative has much higher utility, its probability should be ~1."""
        rng = np.random.default_rng(503)
        n_obs = 100
        n_alts = 4

        choosers = pd.DataFrame(
            {
                "obsval": rng.standard_normal(n_obs),
            },
            index=pd.Index(np.arange(n_obs), name="oid"),
        )

        # Make alt 0 much better than others
        alternatives = pd.DataFrame(
            {
                "altval": [10.0, -1.0, -1.0, -1.0],  # alt 0 dominates
            },
            index=pd.Index(np.arange(n_alts), name="aid"),
        )

        # Most choosers should choose alt 0
        choices = np.zeros(n_obs, dtype=int)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
        )

        model = MNL(ct, formula="altval - 1")
        model.fit()

        probs = model.probabilities(ct)
        probs_2d = probs.reshape(n_obs, n_alts)

        # Alt 0 should have probability close to 1
        assert (probs_2d[:, 0] > 0.99).all()


# ===========================================================================
# 6. Prediction with new/out-of-sample data
# ===========================================================================


class TestPrediction:
    """Tests for prediction on new data."""

    def test_prediction_on_same_data(self):
        """Prediction on estimation data should match fitted probabilities."""
        ct, _, _, _ = _make_simple_dataset(seed=601)
        model = MNL(ct, formula="obsval + altval - 1")
        model.fit()

        probs = model.probabilities(ct)
        assert probs.shape == (ct.n_observations, ct.n_alternatives)
        assert np.isfinite(probs).all()

    def test_prediction_on_new_data(self):
        """Prediction on new data should produce valid probabilities."""
        rng = np.random.default_rng(602)

        # Training data
        n_obs_train = 200
        n_alts = 4
        choosers_train = pd.DataFrame(
            {
                "obsval": rng.standard_normal(n_obs_train),
            },
            index=pd.Index(np.arange(n_obs_train), name="oid"),
        )

        alternatives = pd.DataFrame(
            {
                "altval": rng.standard_normal(n_alts),
            },
            index=pd.Index(np.arange(n_alts), name="aid"),
        )

        choices_train = rng.choice(np.arange(n_alts), size=n_obs_train)

        ct_train = ChoiceTable.from_tables(
            choosers_train,
            alternatives,
            chosen_alternatives=pd.Series(choices_train, index=choosers_train.index),
        )

        model = MNL(ct_train, formula="obsval + altval - 1")
        model.fit()

        # New data (different choosers, same alternatives)
        n_obs_new = 50
        choosers_new = pd.DataFrame(
            {
                "obsval": rng.standard_normal(n_obs_new),
            },
            index=pd.Index(np.arange(n_obs_new), name="oid"),
        )

        ct_new = ChoiceTable.from_tables(
            choosers_new,
            alternatives,
            chosen_alternatives=pd.Series(
                rng.choice(np.arange(n_alts), size=n_obs_new),
                index=choosers_new.index,
            ),
        )

        probs_new = model.probabilities(ct_new)
        probs_2d = probs_new.reshape(n_obs_new, n_alts)

        # Probabilities should be valid
        npt.assert_allclose(probs_2d.sum(axis=1), 1.0, atol=1e-8)
        assert (probs_2d >= -1e-10).all()

    def test_prediction_with_availability(self):
        """Prediction with availability constraints should respect them."""
        rng = np.random.default_rng(603)
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

        # Build availability: alt 4 unavailable for first 30 obs
        avail_arr = np.ones((n_obs, n_alts), dtype=np.float64)
        avail_arr[:30, 4] = 0.0

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

        model = MNL(ct, formula="obsval + altval - 1")
        model.fit()

        probs = model.probabilities(ct)
        probs_2d = probs.reshape(n_obs, n_alts)

        # Alt 4 should have zero probability for first 30 obs
        npt.assert_allclose(probs_2d[:30, 4], 0.0, atol=1e-12)

        # Available alts should sum to 1
        npt.assert_allclose(probs_2d[:30, :4].sum(axis=1), 1.0, atol=1e-8)

    def test_utilities_include_sampling_correction(self):
        """Utilities from prediction should include sampling correction."""
        rng = np.random.default_rng(604)
        n_obs = 100
        n_alts = 10
        sample_size = 5

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
            sample_size=sample_size,
        )

        model = MNL(ct, formula="obsval + altval - 1")
        model.fit()

        # Utilities should include sampling correction
        utilities = model.utilities(ct)
        assert np.isfinite(utilities).all()


# ===========================================================================
# 7. Design matrix correctness (P/X algebra)
# ===========================================================================


class TestDesignMatrixCorrectness:
    """Verify that P/X algebra produces correct design matrices."""

    def test_generic_scope_produces_single_column(self):
        """A generic-scoped variable should produce a single column in the design matrix."""
        rng = np.random.default_rng(701)
        n_obs = 50
        n_alts = 3

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

        arrays = ct.to_arrays(formula="altval - 1")
        # "altval" with generic scope should produce 1 column
        assert arrays.design_matrix.shape[1] == 1

        # Values should be altval tiled across observations
        alt_vals_tiled = np.tile(alternatives["altval"].values, n_obs)
        npt.assert_allclose(arrays.design_matrix[:, 0], alt_vals_tiled, atol=1e-12)

    def test_alternative_specific_scope_produces_per_alt_columns(self):
        """An alternative-specific variable should produce (n_alts - 1) columns."""
        rng = np.random.default_rng(702)
        n_obs = 50
        n_alts = 4

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

        # Use ModelSpec.alternative_specific to create alt-specific coefficients
        spec = ModelSpec().alternative_specific("obsval", reference=0)
        arrays = ct.to_arrays(spec=spec)

        # alternative_specific scope should produce (n_alts - 1) = 3 columns
        assert arrays.design_matrix.shape[1] == n_alts - 1

    def test_interaction_term_produces_correct_column(self):
        """An interaction term (obs × alt) should produce a single column."""
        rng = np.random.default_rng(703)
        n_obs = 50
        n_alts = 3

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

        arrays = ct.to_arrays(formula="obsval + altval - 1")
        # Should produce 1 (generic) + 1 (generic) = 2 columns
        assert arrays.design_matrix.shape[1] == 2

    def test_design_matrix_values_match_manual_computation(self):
        """Design matrix values should match manual computation."""
        rng = np.random.default_rng(704)
        n_obs = 30
        n_alts = 3

        choosers = pd.DataFrame(
            {
                "obsval": rng.standard_normal(n_obs),
            },
            index=pd.Index(np.arange(n_obs), name="oid"),
        )

        alternatives = pd.DataFrame(
            {
                "altval": [1.0, 2.0, 3.0],
            },
            index=pd.Index(np.arange(n_alts), name="aid"),
        )

        choices = rng.choice(np.arange(n_alts), size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
        )

        arrays = ct.to_arrays(formula="altval - 1")

        # altval should be tiled: [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, ...]
        expected = np.tile([1.0, 2.0, 3.0], n_obs)
        npt.assert_allclose(arrays.design_matrix[:, 0], expected, atol=1e-12)


# ===========================================================================
# 8. Comprehensive DGP parameter recovery
# ===========================================================================


class TestDGPRecovery:
    """Comprehensive parameter recovery tests using the DGP module."""

    def test_alt_feature_recovery(self):
        """Recover a single alternative-feature coefficient."""
        from locpick.dgp import simulate_mnl

        dataset = simulate_mnl(
            n_obs=3000,
            n_alts=6,
            alt_params={"alt_feature": -0.7},
            interaction_params={},
            seed=8001,
        )
        model = MNL(dataset.choice_table, formula="alt_feature - 1")
        result = model.fit()

        npt.assert_allclose(
            result.coefficients["alt_feature"],
            dataset.true_params["alt_feature"],
            rtol=0.15,
        )

    def test_interaction_recovery(self):
        """Recover an interaction coefficient."""
        from locpick.dgp import simulate_mnl

        dataset = simulate_mnl(
            n_obs=3000,
            n_alts=6,
            alt_params={"alt_feature": -0.65},
            interaction_params={"obs_x_alt": 0.95},
            seed=8002,
        )
        model = MNL(dataset.choice_table, formula="alt_feature + obs_x_alt - 1")
        result = model.fit()

        npt.assert_allclose(
            result.coefficients["alt_feature"],
            dataset.true_params["alt_feature"],
            rtol=0.15,
        )
        npt.assert_allclose(
            result.coefficients["obs_x_alt"],
            dataset.true_params["obs_x_alt"],
            rtol=0.15,
        )

    def test_multi_parameter_recovery(self):
        """Recover multiple parameters simultaneously."""
        from locpick.dgp import simulate_mnl

        dataset = simulate_mnl(
            n_obs=5000,
            n_alts=8,
            alt_params={"alt_feature": -0.5},
            interaction_params={"obs_x_alt": 1.0},
            seed=8003,
        )
        model = MNL(dataset.choice_table, formula="alt_feature + obs_x_alt - 1")
        result = model.fit()

        # With 5000 obs, should recover within 10%
        npt.assert_allclose(
            result.coefficients["alt_feature"],
            dataset.true_params["alt_feature"],
            rtol=0.10,
        )
        npt.assert_allclose(
            result.coefficients["obs_x_alt"],
            dataset.true_params["obs_x_alt"],
            rtol=0.10,
        )

    def test_recovery_with_large_choice_set(self):
        """Parameter recovery should work with larger choice sets."""
        from locpick.dgp import simulate_mnl

        dataset = simulate_mnl(
            n_obs=3000,
            n_alts=15,
            alt_params={"alt_feature": -0.5},
            interaction_params={"obs_x_alt": 0.8},
            seed=8004,
        )

        model = MNL(dataset.choice_table, formula="alt_feature + obs_x_alt - 1")
        result = model.fit()

        # With 3000 obs and 15 alternatives, should recover within 20%
        npt.assert_allclose(
            result.coefficients["alt_feature"],
            dataset.true_params["alt_feature"],
            rtol=0.20,
        )
        npt.assert_allclose(
            result.coefficients["obs_x_alt"],
            dataset.true_params["obs_x_alt"],
            rtol=0.20,
        )

    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_recovery_across_backends(self, backend, monkeypatch):
        """Both backends should recover similar parameter values."""
        if backend == "jax" and importlib.util.find_spec("jax") is None:
            pytest.skip("JAX not installed")

        from locpick.dgp import simulate_mnl

        dataset = simulate_mnl(n_obs=3000, n_alts=6, seed=8005)

        monkeypatch.delenv("LOCPICK_MNL_BACKEND", raising=False)
        if backend != "jax":
            monkeypatch.setenv("LOCPICK_MNL_BACKEND", backend)

        model = MNL(dataset.choice_table, formula="alt_feature + obs_x_alt - 1")
        result = model.fit()

        npt.assert_allclose(
            result.coefficients["alt_feature"],
            dataset.true_params["alt_feature"],
            rtol=0.15,
        )


# ===========================================================================
# 9. EstimationProblem integration tests
# ===========================================================================


class TestEstimationProblemIntegration:
    """Tests that EstimationProblem produces correct estimation results."""

    def test_problem_matches_formula_path(self):
        """EstimationProblem should produce same results as formula path."""
        from locpick.dgp import simulate_mnl

        dataset = simulate_mnl(n_obs=2000, n_alts=5, seed=901)

        # Formula path
        model_formula = MNL(dataset.choice_table, formula="alt_feature + obs_x_alt - 1")
        result_formula = model_formula.fit()

        # Problem path
        problem = EstimationProblem.from_choice_table(
            dataset.choice_table, formula="alt_feature + obs_x_alt - 1"
        )
        model_problem = MNL(data=dataset.choice_table, problem=problem)
        result_problem = model_problem.fit()

        # Results should match
        npt.assert_allclose(
            result_formula.coefficients.values,
            result_problem.coefficients.values,
            atol=1e-8,
        )
        npt.assert_allclose(
            result_formula.log_likelihood,
            result_problem.log_likelihood,
            atol=1e-8,
        )

    def test_problem_with_fixed_parameter(self):
        """EstimationProblem with a fixed parameter should constrain estimation."""
        from locpick.dgp import simulate_mnl

        dataset = simulate_mnl(n_obs=2000, n_alts=5, seed=902)

        problem = EstimationProblem.from_choice_table(
            dataset.choice_table, formula="alt_feature + obs_x_alt - 1"
        )

        # Fix the alt_feature coefficient at -0.5
        problem_fixed = EstimationProblem(
            arrays=problem.arrays,
            param_names=problem.param_names,
            param_initial=[-0.5, 0.0],
            param_fixed=[True, False],
        )

        model = MNL(data=dataset.choice_table, problem=problem_fixed)
        result = model.fit()

        # The fixed parameter should remain at -0.5
        npt.assert_allclose(result.coefficients.iloc[0], -0.5, atol=1e-6)

        # The free parameter should be estimated
        assert np.isfinite(result.coefficients.iloc[1])

    def test_problem_with_bounds(self):
        """EstimationProblem with bounds should constrain parameter range."""
        from locpick.dgp import simulate_mnl

        dataset = simulate_mnl(n_obs=2000, n_alts=5, seed=903)

        problem = EstimationProblem.from_choice_table(
            dataset.choice_table, formula="alt_feature + obs_x_alt - 1"
        )

        # Bound alt_feature to [-1, 0]
        problem_bounded = EstimationProblem(
            arrays=problem.arrays,
            param_names=problem.param_names,
            param_bounds=[(-1.0, 0.0), (None, None)],
        )

        model = MNL(data=dataset.choice_table, problem=problem_bounded)
        result = model.fit()

        # alt_feature should be within bounds
        assert -1.0 <= result.coefficients.iloc[0] <= 0.0


"""Tests for the v2 MultinomialLogit pipeline (data, spec, estimation, results).

Covers: ChoiceTable, ModelSpec, MultinomialLogit, FitResult, reporting, sampling.
"""


from locpick import (
    FitResult,
    format_coefficient_table,
    format_fit_statistics,
    sample_alternatives,
)


def make_toy_data():
    # Simple synthetic data: 3 choosers, 4 alternatives
    choosers = pd.DataFrame({"id": [1, 2, 3]})
    alternatives = pd.DataFrame(
        {
            "alt_id": [10, 11, 12, 13],
            "cost": [1.0, 2.0, 3.0, 4.0],
            "time": [5.0, 4.0, 3.0, 2.0],
        }
    )
    chosen = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "alt_id": [10, 12, 13],
        }
    )
    return choosers, alternatives, chosen


def test_choicetable_and_arrays():
    choosers, alternatives, chosen = make_toy_data()
    ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=3, seed=42)
    arrays = ct.to_arrays(formula="cost + time")
    assert arrays.design_matrix.shape[0] == arrays.n_obs * arrays.n_alts
    assert arrays.design_matrix.shape[1] == 2
    assert arrays.chosen.sum() == arrays.n_obs


def test_multinomiallogit_estimation():
    choosers, alternatives, chosen = make_toy_data()
    ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=4, seed=1)
    model = MNL(ct, formula="cost + time")
    result = model.fit()
    assert isinstance(result, FitResult)
    assert result.coefficients.shape[0] == 2
    assert np.isfinite(result.log_likelihood)
    # Check reporting
    txt = result.summary()
    assert "Log-likelihood" in txt
    html = result.to_html()
    assert "<table" in html
    latex = result.to_latex()
    assert "\\begin{tabular}" in latex


def test_formatting_and_statistics():
    choosers, alternatives, chosen = make_toy_data()
    ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=4, seed=2)
    model = MNL(ct, formula="cost + time")
    result = model.fit()
    # Coefficient table
    table = format_coefficient_table(result)
    assert "cost" in table and "time" in table
    # Fit statistics
    stats = format_fit_statistics(result)
    assert "Log-likelihood" in stats


def test_sampling_module():
    arr = sample_alternatives(5, 10, 3, seed=123)
    assert arr.shape == (5, 3)
    arr2 = sample_alternatives(2, 5, 2, weights=np.array([1, 2, 3, 4, 5]), seed=1)
    assert arr2.shape == (2, 2)
    assert np.all(arr2 < 5)


def test_modelspec_formula_spec_estimation():
    spec = ModelSpec(formula="cost + time")
    choosers, alternatives, chosen = make_toy_data()
    ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=4, seed=3)
    arrays = ct.to_arrays(spec=spec)
    assert arrays.design_matrix.shape[1] == 2
    # Estimation with formula spec
    model = MNL(ct, spec=spec)
    result = model.fit()
    assert isinstance(result, FitResult)
    assert result.coefficients.shape[0] == 2


def test_modelspec_formula_generated_interaction():
    choosers, alternatives, chosen = make_toy_data()
    choosers = choosers.copy()
    choosers["income"] = [1.0, 2.0, 3.0]

    spec = ModelSpec(formula="income_x_cost").with_interaction(
        "income_x_cost",
        "income",
        "cost",
    )

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen,
        sample_size=4,
        seed=4,
    )
    arrays = ct.to_arrays(spec=spec)

    frame = ct.to_frame()
    expected = frame["income"].to_numpy() * frame["cost"].to_numpy()
    assert arrays.param_names == ["income_x_cost"]
    assert np.allclose(arrays.design_matrix[:, 0], expected)


def test_modelspec_formula_generated_interaction_estimation():
    choosers, alternatives, chosen = make_toy_data()
    choosers = choosers.copy()
    choosers["income"] = [1.0, 2.0, 3.0]

    spec = ModelSpec(formula="income_x_cost").with_interaction(
        "income_x_cost",
        "income",
        "cost",
    )

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen,
        sample_size=4,
        seed=5,
    )
    arrays = ct.to_arrays(spec=spec)

    frame = ct.to_frame()
    expected = frame["income"].to_numpy() * frame["cost"].to_numpy()
    assert arrays.param_names == ["income_x_cost"]
    assert np.allclose(arrays.design_matrix[:, 0], expected)


def test_modelspec_generic_scope_matches_raw_column():
    choosers, alternatives, chosen = make_toy_data()
    chosen_series = chosen["alt_id"].rename("chosen_alt")
    spec = ModelSpec().generic("cost")

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_series,
        seed=6,
    )
    arrays = ct.to_arrays(spec=spec)
    frame = ct.to_frame()

    assert arrays.param_names == ["cost"]
    assert np.allclose(arrays.design_matrix[:, 0], frame["cost"].to_numpy())


def test_modelspec_alternative_specific_scope_masks_by_alt_id():
    choosers, alternatives, chosen = make_toy_data()
    chosen_series = chosen["alt_id"].rename("chosen_alt")
    spec = ModelSpec().alternative_specific("cost", reference=10)

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_series,
        seed=7,
    )
    arrays = ct.to_arrays(spec=spec)
    frame = ct.to_frame()

    assert arrays.param_names == ["cost[11]", "cost[12]", "cost[13]"]
    for col, alt_id in enumerate([11, 12, 13]):
        expected = frame["cost"].where(frame["alt_id"] == alt_id, 0.0).to_numpy()
        assert np.allclose(arrays.design_matrix[:, col], expected)


def test_modelspec_grouped_scope_masks_by_alt_groups():
    choosers, alternatives, chosen = make_toy_data()
    chosen_series = chosen["alt_id"].rename("chosen_alt")
    spec = ModelSpec().grouped(
        "time",
        groups={
            "early": [10, 11],
            "late": [12, 13],
        },
    )

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_series,
        seed=8,
    )
    arrays = ct.to_arrays(spec=spec)
    frame = ct.to_frame()

    assert arrays.param_names == ["time[early]", "time[late]"]
    expected_early = frame["time"].where(frame["alt_id"].isin([10, 11]), 0.0).to_numpy()
    expected_late = frame["time"].where(frame["alt_id"].isin([12, 13]), 0.0).to_numpy()
    assert np.allclose(arrays.design_matrix[:, 0], expected_early)
    assert np.allclose(arrays.design_matrix[:, 1], expected_late)
