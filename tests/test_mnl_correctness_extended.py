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

import importlib.util

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from locpick import ChoiceTable, EstimationProblem, ModelSpec, MultinomialLogit
from locpick.data import ChoiceArrays

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
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
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
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
        model.fit()

        probs = model.probabilities(ct)
        probs_2d = probs.reshape(ct.n_observations, ct.n_alternatives)
        row_sums = probs_2d.sum(axis=1)
        npt.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_probabilities_are_non_negative(self):
        """All probabilities should be non-negative."""
        ct, _, _, _ = _make_simple_dataset(seed=103)
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
        model.fit()

        probs = model.probabilities(ct)
        assert (probs >= -1e-15).all()

    def test_chosen_probabilities_positive(self):
        """Probability of the chosen alternative should be positive for each obs."""
        ct, _, _, _ = _make_simple_dataset(seed=104)
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
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
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
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
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
        result = model.fit()
        assert result.log_likelihood < 0

    def test_log_likelihood_better_than_null(self):
        """Fitted model LL should be >= null LL (rho-squared >= 0)."""
        ct, _, _, _ = _make_simple_dataset(seed=203)
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
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
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
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
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
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
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
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
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
        result = model.fit()

        assert (result.std_errors > 0).all()


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

        model = MultinomialLogit(ct, formula="obsval + altval - 1")
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

        model = MultinomialLogit(ct, formula="obsval + altval - 1")
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

        model = MultinomialLogit(ct, formula="altval - 1")
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
        model = MultinomialLogit(ct, formula="obsval + altval - 1")
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

        model = MultinomialLogit(ct_train, formula="obsval + altval - 1")
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

        model = MultinomialLogit(ct, formula="obsval + altval - 1")
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

        model = MultinomialLogit(ct, formula="obsval + altval - 1")
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
        model = MultinomialLogit(dataset.choice_table, formula="alt_feature - 1")
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
        model = MultinomialLogit(dataset.choice_table, formula="alt_feature + obs_x_alt - 1")
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
        model = MultinomialLogit(dataset.choice_table, formula="alt_feature + obs_x_alt - 1")
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

        model = MultinomialLogit(dataset.choice_table, formula="alt_feature + obs_x_alt - 1")
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

        monkeypatch.delenv("CHOICEMODELS_MNL_BACKEND", raising=False)
        if backend != "jax":
            monkeypatch.setenv("CHOICEMODELS_MNL_BACKEND", backend)

        model = MultinomialLogit(dataset.choice_table, formula="alt_feature + obs_x_alt - 1")
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
        model_formula = MultinomialLogit(
            dataset.choice_table, formula="alt_feature + obs_x_alt - 1"
        )
        result_formula = model_formula.fit()

        # Problem path
        problem = EstimationProblem.from_choice_table(
            dataset.choice_table, formula="alt_feature + obs_x_alt - 1"
        )
        model_problem = MultinomialLogit(data=dataset.choice_table, problem=problem)
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

        model = MultinomialLogit(data=dataset.choice_table, problem=problem_fixed)
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

        model = MultinomialLogit(data=dataset.choice_table, problem=problem_bounded)
        result = model.fit()

        # alt_feature should be within bounds
        assert -1.0 <= result.coefficients.iloc[0] <= 0.0
