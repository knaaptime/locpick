"""Tests for inference: sandwich and cluster-robust covariance estimators.

Covers: observation-level scores, sandwich (robust) covariance,
cluster-robust covariance, and standard error convenience methods.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd

from locpick import ChoiceModel, ChoiceTable

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_simple_data(n_obs=500, n_alts=5, seed=42):
    """Create simple synthetic data for testing."""
    rng = np.random.default_rng(seed)

    choosers = pd.DataFrame(
        {
            "income": rng.standard_normal(n_obs),
        },
        index=pd.Index(np.arange(n_obs), name="oid"),
    )

    alternatives = pd.DataFrame(
        {
            "cost": rng.uniform(1, 10, n_alts),
            "time": rng.uniform(5, 30, n_alts),
        },
        index=pd.Index(np.arange(n_alts), name="aid"),
    )

    # True parameters
    beta_cost = -0.5
    beta_time = -0.1

    # Simulate choices
    utilities = beta_cost * alternatives["cost"].values + beta_time * alternatives["time"].values
    probs = np.exp(utilities - utilities.max())
    probs /= probs.sum()
    choices = rng.choice(n_alts, size=n_obs, p=probs)

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives=pd.Series(choices, index=choosers.index),
    )

    return ct, beta_cost, beta_time


# ---------------------------------------------------------------------------
# Test: Observation-level scores
# ---------------------------------------------------------------------------


class TestObservationScores:
    """Tests for model-level observation score computation."""

    def test_scores_shape(self):
        """Observation scores should have shape (n_obs, n_params)."""
        ct, _, _ = _make_simple_data()
        model = ChoiceModel(ct, formula="cost + time - 1")
        model.fit()

        arrays = ct.to_arrays(formula="cost + time - 1")
        scores = model._observation_scores(arrays)

        assert scores.shape == (ct.n_observations, 2)  # 2 parameters

    def test_scores_sum_to_gradient(self):
        """Sum of observation scores should equal the full gradient."""
        ct, _, _ = _make_simple_data()
        model = ChoiceModel(ct, formula="cost + time - 1")
        model.fit()

        arrays = ct.to_arrays(formula="cost + time - 1")
        scores = model._observation_scores(arrays)

        # At the MLE, the gradient should be approximately zero
        total_gradient = scores.sum(axis=0)
        npt.assert_allclose(total_gradient, 0.0, atol=0.1)

    def test_scores_are_finite(self):
        """All observation scores should be finite."""
        ct, _, _ = _make_simple_data()
        model = ChoiceModel(ct, formula="cost + time - 1")
        model.fit()

        arrays = ct.to_arrays(formula="cost + time - 1")
        scores = model._observation_scores(arrays)

        assert np.all(np.isfinite(scores))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Test: Sandwich (robust) covariance
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: Sandwich (robust) covariance
# ---------------------------------------------------------------------------


class TestRobustCovariance:
    """Tests for the sandwich (Huber-White) robust covariance estimator."""

    def test_robust_covariance_shape(self):
        """Robust covariance should be (n_params, n_params)."""
        ct, _, _ = _make_simple_data()
        model = ChoiceModel(ct, formula="cost + time - 1")
        model.fit()

        cov = model.covariance_robust(ct)

        assert cov.shape == (2, 2)

    def test_robust_covariance_positive_diagonal(self):
        """Robust covariance diagonal should be positive (variances)."""
        ct, _, _ = _make_simple_data()
        model = ChoiceModel(ct, formula="cost + time - 1")
        model.fit()

        cov = model.covariance_robust(ct)

        assert np.all(np.diag(cov) > 0)

    def test_robust_covariance_symmetric(self):
        """Robust covariance should be approximately symmetric."""
        ct, _, _ = _make_simple_data()
        model = ChoiceModel(ct, formula="cost + time - 1")
        model.fit()

        cov = model.covariance_robust(ct)

        npt.assert_allclose(cov, cov.T, atol=1e-8)

    def test_robust_standard_errors(self):
        """Robust standard errors should be positive and finite."""
        ct, _, _ = _make_simple_data()
        model = ChoiceModel(ct, formula="cost + time - 1")
        model.fit()

        se = model.std_errors_robust(ct)

        assert isinstance(se, pd.Series)
        assert len(se) == 2
        assert np.all(se > 0)
        assert np.all(np.isfinite(se))


# ---------------------------------------------------------------------------
# Test: Cluster-robust covariance
# ---------------------------------------------------------------------------


class TestClusteredCovariance:
    """Tests for the cluster-robust covariance estimator."""

    def test_clustered_covariance_shape(self):
        """Cluster-robust covariance should be (n_params, n_params)."""
        ct, _, _ = _make_simple_data()
        model = ChoiceModel(ct, formula="cost + time - 1")
        model.fit()

        # Create cluster groups
        groups = np.repeat([0, 1, 2, 3, 4], ct.n_observations // 5)

        cov = model.covariance_clustered(ct, groups)

        assert cov.shape == (2, 2)

    def test_clustered_covariance_positive_diagonal(self):
        """Cluster-robust covariance diagonal should be positive."""
        ct, _, _ = _make_simple_data()
        model = ChoiceModel(ct, formula="cost + time - 1")
        model.fit()

        groups = np.repeat([0, 1, 2, 3, 4], ct.n_observations // 5)

        cov = model.covariance_clustered(ct, groups)

        assert np.all(np.diag(cov) > 0)

    def test_clustered_standard_errors(self):
        """Cluster-robust standard errors should be positive and finite."""
        ct, _, _ = _make_simple_data()
        model = ChoiceModel(ct, formula="cost + time - 1")
        model.fit()

        groups = np.repeat([0, 1, 2, 3, 4], ct.n_observations // 5)

        se = model.std_errors_clustered(ct, groups)

        assert isinstance(se, pd.Series)
        assert len(se) == 2
        assert np.all(se > 0)
        assert np.all(np.isfinite(se))

    def test_clustered_larger_than_default(self):
        """Cluster-robust SEs should typically be >= default SEs
        (due to within-cluster correlation)."""
        ct, _, _ = _make_simple_data()
        model = ChoiceModel(ct, formula="cost + time - 1")
        result = model.fit()

        # With many small clusters, clustered SEs should be similar to default
        groups = np.arange(ct.n_observations)  # Each obs is its own cluster

        se_clustered = model.std_errors_clustered(ct, groups)
        se_default = result.std_errors

        # With one obs per cluster, clustered SEs should be close to default
        npt.assert_allclose(se_clustered.values, se_default.values, rtol=0.5)


# ---------------------------------------------------------------------------
# Test: Covariance comparison
# ---------------------------------------------------------------------------


class TestCovarianceComparison:
    """Tests comparing different covariance estimators."""

    def test_robust_vs_default_se_order(self):
        """Robust SEs should be in the same order of magnitude as default SEs."""
        ct, _, _ = _make_simple_data()
        model = ChoiceModel(ct, formula="cost + time - 1")
        result = model.fit()

        se_default = result.std_errors.values
        se_robust = model.std_errors_robust(ct).values

        # Robust SEs should be within 5x of default SEs
        ratio = se_robust / se_default
        assert np.all(ratio > 0.2) and np.all(ratio < 5.0)


# ---------------------------------------------------------------------------
# Test: parameterisation of the sandwich for transformed models
# ---------------------------------------------------------------------------


def _numeric_robust_cov(model, theta, chosen):
    """Sandwich covariance built entirely from display-scale finite differences.

    Independent of the package's score/Hessian machinery, so it pins down the
    coordinate system the reported robust covariance is supposed to be in.
    """
    eps = 1e-5
    n_params = len(theta)

    def obs_ll(params):
        probs = model.probabilities(beta=params)
        return np.log(np.maximum((probs * chosen).sum(axis=1), 1e-300))

    def total_score(params):
        out = np.zeros(n_params)
        for i in range(n_params):
            hi, lo = params.copy(), params.copy()
            hi[i] += eps
            lo[i] -= eps
            out[i] = (obs_ll(hi).sum() - obs_ll(lo).sum()) / (2 * eps)
        return out

    scores = np.zeros((len(chosen), n_params))
    for i in range(n_params):
        hi, lo = theta.copy(), theta.copy()
        hi[i] += eps
        lo[i] -= eps
        scores[:, i] = (obs_ll(hi) - obs_ll(lo)) / (2 * eps)

    hess = np.zeros((n_params, n_params))
    for i in range(n_params):
        hi, lo = theta.copy(), theta.copy()
        hi[i] += eps
        lo[i] -= eps
        hess[:, i] = (total_score(hi) - total_score(lo)) / (2 * eps)
    hess = 0.5 * (hess + hess.T)

    hess_inv = np.linalg.inv(-hess)
    return hess_inv @ (scores.T @ scores) @ hess_inv


class TestTransformedParameterScale:
    """Robust SEs must be reported on the display scale.

    Nested/mixed/spatial models are estimated in an unconstrained space
    (lambda = sigmoid(raw), rho = tanh(raw), spread = |raw|).  Scores and
    Hessians therefore have to be evaluated at the raw solution and mapped
    back through the delta method; evaluating them at the display-scale
    coefficients silently mixes coordinate systems.
    """

    def _fit_nested(self):
        from locpick.dgp import simulate_nested_logit

        # Interaction terms make attributes vary across choosers; without
        # them the betas and nest lambdas are not separately identified.
        dataset = simulate_nested_logit(
            n_obs=3000,
            n_alts=4,
            seed=2026,
            interaction_params={"income_x_cost": 0.8, "income_x_time": 0.8},
        )
        ct = dataset.choice_table
        model = ChoiceModel(
            ct,
            formula="cost + time + income_x_cost + income_x_time - 1",
            nests=dataset.nests,
        )
        result = model.fit()
        return ct, model, result

    def test_nested_robust_se_matches_numeric_sandwich(self):
        ct, model, result = self._fit_nested()

        arrays = model._arrays
        chosen = np.asarray(arrays.chosen, dtype=float).reshape(arrays.n_obs, arrays.n_alts)
        cov_numeric = _numeric_robust_cov(model, result.coefficients.values.copy(), chosen)
        se_numeric = np.sqrt(np.maximum(np.diag(cov_numeric), 0))

        se_pkg = model.std_errors_robust().values

        npt.assert_allclose(se_pkg, se_numeric, rtol=0.05)

    def test_nested_clustered_se_finite_and_display_scale(self):
        ct, model, result = self._fit_nested()

        groups = np.arange(ct.n_observations) % 40
        se_clustered = model.std_errors_clustered(groups=groups)

        assert np.all(np.isfinite(se_clustered.values))
        assert list(se_clustered.index) == list(result.coefficients.index)

    def test_covariance_is_display_scale_and_consistent_with_std_errors(self):
        """FitResult.covariance() shares coordinates with the reported SEs."""
        _, _, result = self._fit_nested()

        cov = result.covariance()
        npt.assert_allclose(
            np.sqrt(np.maximum(np.diag(cov), 0)),
            result.std_errors.values,
            rtol=1e-6,
        )
