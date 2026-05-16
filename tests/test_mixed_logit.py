"""Tests for the mixed logit model.

Covers: ParamDistribution, draw generation, probability computation,
log-likelihood, parameter recovery, and MixedLogit model class.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from locpick import ChoiceTable
from locpick.models.mixed import (
    MixedLogit,
    ParamDistribution,
    _halton_sequence,
    _mixed_logit_ll_numpy,
    _mixed_logit_probs_numpy,
    generate_halton_draws,
    generate_random_draws,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_mixed_data(n_obs=300, n_alts=4, seed=42):
    """Create synthetic data for mixed logit estimation."""
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

    # Simulate choices using MNL (no random coefficients in DGP)
    beta_cost = -0.5
    beta_time = -0.1
    choices = np.zeros(n_obs, dtype=int)
    for i in range(n_obs):
        utilities = (
            beta_cost * alternatives["cost"].values + beta_time * alternatives["time"].values
        )
        utilities += rng.gumbel(size=n_alts)
        choices[i] = np.argmax(utilities)

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives=pd.Series(choices, index=choosers.index),
    )

    return ct


# ---------------------------------------------------------------------------
# Test: ParamDistribution
# ---------------------------------------------------------------------------


class TestParamDistribution:
    """Tests for ParamDistribution dataclass."""

    def test_basic_construction(self):
        pd_ = ParamDistribution("normal", "commute_time")
        assert pd_.distribution == "normal"
        assert pd_.param == "commute_time"
        assert pd_.param_name == "commute_time"
        assert pd_.n_params == 2

    def test_paramref_support(self):
        from locpick.spec import ParamRef

        pd_ = ParamDistribution("normal", ParamRef("beta_time"))
        assert pd_.param_name == "beta_time"

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            ParamDistribution("gamma", "x")

    def test_all_distributions(self):
        for dist in ["normal", "lognormal", "triangular", "uniform"]:
            pd_ = ParamDistribution(dist, "x")
            assert pd_.distribution == dist
            assert pd_.n_params == 2


# ---------------------------------------------------------------------------
# Test: Draw generation
# ---------------------------------------------------------------------------


class TestDrawGeneration:
    """Tests for Halton and random draw generation."""

    def test_halton_sequence_basic(self):
        """Halton sequence with base 2 should produce values in [0, 1)."""
        draws = _halton_sequence(10, base=2)
        assert len(draws) == 10
        assert (draws >= 0).all()
        assert (draws < 1).all()

    def test_halton_sequence_known_values(self):
        """First few Halton values for base 2 are known."""
        draws = _halton_sequence(4, base=2)
        # Halton base 2: 1/2, 1/4, 3/4, 1/8
        npt.assert_allclose(draws[0], 0.5, atol=1e-10)
        npt.assert_allclose(draws[1], 0.25, atol=1e-10)
        npt.assert_allclose(draws[2], 0.75, atol=1e-10)

    def test_halton_draws_shape(self):
        """Halton draws should have shape (n_obs, n_draws, n_random)."""
        draws = generate_halton_draws(50, 100, 3, seed=42)
        assert draws.shape == (50, 100, 3)

    def test_random_draws_shape(self):
        """Random draws should have shape (n_obs, n_draws, n_random)."""
        draws = generate_random_draws(50, 100, 3, seed=42)
        assert draws.shape == (50, 100, 3)

    def test_random_draws_standard_normal(self):
        """Random draws should be approximately standard normal."""
        draws = generate_random_draws(10000, 1, 1, seed=42)
        values = draws[:, 0, 0]
        npt.assert_allclose(np.mean(values), 0.0, atol=0.05)
        npt.assert_allclose(np.std(values), 1.0, atol=0.05)

    def test_halton_draws_reproducible(self):
        """Same seed should produce same draws."""
        d1 = generate_halton_draws(50, 100, 2, seed=42)
        d2 = generate_halton_draws(50, 100, 2, seed=42)
        npt.assert_array_equal(d1, d2)

    def test_too_many_random_params_raises(self):
        """Should raise if more random params than available primes."""
        with pytest.raises(ValueError, match="Maximum"):
            generate_halton_draws(50, 100, 20, seed=42)


# ---------------------------------------------------------------------------
# Test: Probability computation
# ---------------------------------------------------------------------------


class TestMixedLogitProbabilities:
    """Tests for mixed logit probability computation."""

    def test_probabilities_sum_to_one(self):
        """Probabilities should sum to 1 for each observation."""
        rng = np.random.default_rng(123)
        n_obs = 50
        n_alts = 4
        k_fixed = 1
        k_random = 1
        n_draws = 50

        beta_fixed = np.array([0.5])
        beta_random_means = np.array([-0.3])
        beta_random_spreads = np.array([0.1])
        random_distributions = ["normal"]

        dm = rng.standard_normal((n_obs * n_alts, k_fixed + k_random))
        draws = generate_random_draws(n_obs, n_draws, k_random, seed=42)
        random_col_indices = [1]  # second column is random

        probs = _mixed_logit_probs_numpy(
            beta_fixed,
            beta_random_means,
            beta_random_spreads,
            random_distributions,
            draws,
            dm,
            random_col_indices,
            n_obs,
            n_alts,
        )

        row_sums = probs.sum(axis=1)
        npt.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_probabilities_are_non_negative(self):
        """All probabilities should be non-negative."""
        rng = np.random.default_rng(456)
        n_obs = 50
        n_alts = 4
        k_fixed = 1
        k_random = 1
        n_draws = 50

        beta_fixed = np.array([0.5])
        beta_random_means = np.array([-0.3])
        beta_random_spreads = np.array([0.1])
        random_distributions = ["normal"]

        dm = rng.standard_normal((n_obs * n_alts, k_fixed + k_random))
        draws = generate_random_draws(n_obs, n_draws, k_random, seed=42)
        random_col_indices = [1]

        probs = _mixed_logit_probs_numpy(
            beta_fixed,
            beta_random_means,
            beta_random_spreads,
            random_distributions,
            draws,
            dm,
            random_col_indices,
            n_obs,
            n_alts,
        )

        assert (probs >= -1e-10).all()

    def test_zero_spread_reduces_to_mnl(self):
        """When all spreads are zero, mixed logit should reduce to MNL."""
        rng = np.random.default_rng(789)
        n_obs = 100
        n_alts = 4
        k_fixed = 1
        k_random = 1
        n_draws = 50

        beta_fixed = np.array([0.5])
        beta_random_means = np.array([-0.3])
        beta_random_spreads = np.array([0.0])  # zero spread -> MNL
        random_distributions = ["normal"]

        dm = rng.standard_normal((n_obs * n_alts, k_fixed + k_random))
        draws = generate_random_draws(n_obs, n_draws, k_random, seed=42)
        random_col_indices = [1]

        # Mixed logit probabilities
        ml_probs = _mixed_logit_probs_numpy(
            beta_fixed,
            beta_random_means,
            beta_random_spreads,
            random_distributions,
            draws,
            dm,
            random_col_indices,
            n_obs,
            n_alts,
        )

        # MNL probabilities (manual softmax)
        beta_full = np.concatenate([beta_fixed, beta_random_means])
        utilities = (dm @ beta_full).reshape(n_obs, n_alts)
        utilities_shifted = utilities - utilities.max(axis=1, keepdims=True)
        exp_u = np.exp(utilities_shifted)
        mnl_probs = exp_u / exp_u.sum(axis=1, keepdims=True)

        npt.assert_allclose(ml_probs, mnl_probs, atol=1e-4)

    def test_lognormal_distribution_positive_coefficients(self):
        """Lognormal distribution should produce positive coefficients."""
        rng = np.random.default_rng(101)
        n_obs = 50
        n_alts = 3
        k_random = 1
        n_draws = 50

        beta_fixed = np.array([])
        beta_random_means = np.array([0.5])  # log-normal mean
        beta_random_spreads = np.array([0.2])  # log-normal sigma
        random_distributions = ["lognormal"]

        dm = rng.standard_normal((n_obs * n_alts, k_random))
        draws = generate_random_draws(n_obs, n_draws, k_random, seed=42)
        random_col_indices = [0]

        probs = _mixed_logit_probs_numpy(
            beta_fixed,
            beta_random_means,
            beta_random_spreads,
            random_distributions,
            draws,
            dm,
            random_col_indices,
            n_obs,
            n_alts,
        )

        # Probabilities should sum to 1
        npt.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: Log-likelihood
# ---------------------------------------------------------------------------


class TestMixedLogitLikelihood:
    """Tests for mixed logit log-likelihood computation."""

    def test_log_likelihood_is_negative(self):
        """Log-likelihood should be negative."""
        rng = np.random.default_rng(202)
        n_obs = 50
        n_alts = 4
        k_fixed = 1
        k_random = 1
        n_draws = 50

        beta_fixed = np.array([0.5])
        beta_random_means = np.array([-0.3])
        beta_random_spreads = np.array([0.1])
        random_distributions = ["normal"]

        dm = rng.standard_normal((n_obs * n_alts, k_fixed + k_random))
        chosen = np.zeros((n_obs, n_alts))
        for i in range(n_obs):
            chosen[i, rng.choice(n_alts)] = 1.0

        draws = generate_random_draws(n_obs, n_draws, k_random, seed=42)
        random_col_indices = [1]

        ll = _mixed_logit_ll_numpy(
            beta_fixed,
            beta_random_means,
            beta_random_spreads,
            random_distributions,
            draws,
            dm,
            chosen,
            random_col_indices,
            n_obs,
            n_alts,
        )

        assert ll < 0

    def test_log_likelihood_matches_manual(self):
        """Log-likelihood should match manual computation from probabilities."""
        rng = np.random.default_rng(303)
        n_obs = 50
        n_alts = 4
        k_fixed = 1
        k_random = 1
        n_draws = 50

        beta_fixed = np.array([0.5])
        beta_random_means = np.array([-0.3])
        beta_random_spreads = np.array([0.1])
        random_distributions = ["normal"]

        dm = rng.standard_normal((n_obs * n_alts, k_fixed + k_random))
        chosen = np.zeros((n_obs, n_alts))
        for i in range(n_obs):
            chosen[i, rng.choice(n_alts)] = 1.0

        draws = generate_random_draws(n_obs, n_draws, k_random, seed=42)
        random_col_indices = [1]

        # Compute LL from probabilities
        probs = _mixed_logit_probs_numpy(
            beta_fixed,
            beta_random_means,
            beta_random_spreads,
            random_distributions,
            draws,
            dm,
            random_col_indices,
            n_obs,
            n_alts,
        )
        chosen_probs = (probs * chosen).sum(axis=1)
        chosen_probs = np.maximum(chosen_probs, 1e-30)
        expected_ll = np.sum(np.log(chosen_probs))

        # Compute LL directly
        ll = _mixed_logit_ll_numpy(
            beta_fixed,
            beta_random_means,
            beta_random_spreads,
            random_distributions,
            draws,
            dm,
            chosen,
            random_col_indices,
            n_obs,
            n_alts,
        )

        npt.assert_allclose(ll, expected_ll, rtol=1e-4)


# ---------------------------------------------------------------------------
# Test: MixedLogit model class
# ---------------------------------------------------------------------------


class TestMixedLogitModel:
    """Tests for the MixedLogit model class."""

    def test_mixed_logit_estimation(self):
        """MixedLogit should estimate and return a FitResult."""
        ct = make_mixed_data(n_obs=200, n_alts=4)

        model = MixedLogit(
            ct,
            formula="cost + time - 1",
            random_params={"time": ParamDistribution("normal", "time")},
            n_draws=50,
            draw_type="random",
            seed=42,
        )
        result = model.fit()

        assert result.coefficients is not None
        assert np.isfinite(result.log_likelihood)
        # Should have: cost (fixed) + mean_time + sd_time = 3 params
        assert result.n_parameters == 3

    def test_mixed_logit_has_random_params(self):
        """FitResult should include mean and sd of random parameters."""
        ct = make_mixed_data(n_obs=200, n_alts=4)

        model = MixedLogit(
            ct,
            formula="cost + time - 1",
            random_params={"time": ParamDistribution("normal", "time")},
            n_draws=50,
            draw_type="random",
            seed=42,
        )
        result = model.fit()

        # Should have mean_time and sd_time parameters
        assert "mean_time" in result.coefficients.index
        assert "sd_time" in result.coefficients.index

    def test_mixed_logit_requires_random_params(self):
        """MixedLogit should raise ValueError without random_params."""
        rng = np.random.default_rng(44)
        n_obs = 50
        n_alts = 3

        choosers = pd.DataFrame(
            {
                "x": rng.standard_normal(n_obs),
            },
            index=pd.Index(np.arange(n_obs), name="oid"),
        )

        alternatives = pd.DataFrame(
            {
                "y": rng.standard_normal(n_alts),
            },
            index=pd.Index(np.arange(n_alts), name="aid"),
        )

        choices = rng.choice(np.arange(n_alts), size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
        )

        with pytest.raises(ValueError, match="at least one random parameter"):
            MixedLogit(ct, formula="y - 1", random_params={})

    def test_mixed_logit_fit_alias(self):
        """fit() should be the primary estimation API."""
        ct = make_mixed_data(n_obs=100, n_alts=4)

        model = MixedLogit(
            ct,
            formula="cost + time - 1",
            random_params={"time": ParamDistribution("normal", "time")},
            n_draws=30,
            draw_type="random",
            seed=42,
        )
        result = model.fit()

        assert np.isfinite(result.log_likelihood)

    def test_mixed_logit_halton_draws(self):
        """MixedLogit should work with Halton draws."""
        ct = make_mixed_data(n_obs=100, n_alts=4)

        model = MixedLogit(
            ct,
            formula="cost + time - 1",
            random_params={"time": ParamDistribution("normal", "time")},
            n_draws=50,
            draw_type="halton",
            seed=42,
        )
        result = model.fit()

        assert np.isfinite(result.log_likelihood)

    def test_mixed_logit_multiple_random_params(self):
        """MixedLogit should handle multiple random parameters."""
        ct = make_mixed_data(n_obs=200, n_alts=4)

        model = MixedLogit(
            ct,
            formula="cost + time - 1",
            random_params={
                "cost": ParamDistribution("normal", "cost"),
                "time": ParamDistribution("normal", "time"),
            },
            n_draws=50,
            draw_type="random",
            seed=42,
        )
        result = model.fit()

        # Should have: mean_cost, sd_cost, mean_time, sd_time = 4 random params
        # No fixed params since both are random
        assert "mean_cost" in result.coefficients.index
        assert "sd_cost" in result.coefficients.index
        assert "mean_time" in result.coefficients.index
        assert "sd_time" in result.coefficients.index
        assert result.n_parameters == 4

    def test_mixed_logit_lognormal(self):
        """MixedLogit should work with lognormal distribution."""
        ct = make_mixed_data(n_obs=200, n_alts=4)

        model = MixedLogit(
            ct,
            formula="cost + time - 1",
            random_params={"time": ParamDistribution("lognormal", "time")},
            n_draws=50,
            draw_type="random",
            seed=42,
        )
        result = model.fit()

        assert np.isfinite(result.log_likelihood)
        assert "mean_time" in result.coefficients.index
        assert "sd_time" in result.coefficients.index
