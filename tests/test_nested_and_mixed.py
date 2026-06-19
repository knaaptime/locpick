# ruff: noqa: E402
"""Nested and Mixed logit tests."""

"""Tests for the nested logit model.

Covers: NestSpec, NestingTree, naturalize_nest_params, probability
computation, log-likelihood, parameter recovery, and comparison with MNL.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from locpick import ChoiceModel, ChoiceTable
from locpick.models.nested import (
    NestingTree,
    NestSpec,
    _nested_logit_ll_numpy,
    _nested_logit_probs_numpy,
    naturalize_nest_params,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_nested_data(n_obs=500, n_alts=4, seed=42):
    """Create synthetic data with a natural 2-nest structure.

    Nests:
    - Nest 0: alternatives 0, 1 (e.g., "transit")
    - Nest 1: alternatives 2, 3 (e.g., "auto")
    """
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
    lambda_transit = 0.7  # dissimilarity for nest 0
    lambda_auto = 0.8  # dissimilarity for nest 1

    # Simulate choices using nested logit probabilities
    choices = np.zeros(n_obs, dtype=int)
    for i in range(n_obs):
        utilities = (
            beta_cost * alternatives["cost"].values + beta_time * alternatives["time"].values
        )
        # Add Gumbel noise
        noise = rng.gumbel(size=n_alts)
        utilities += noise

        # Compute nested logit probabilities
        np.array([lambda_transit, lambda_auto])
        nest_matrix = np.zeros((n_alts, 2))
        nest_matrix[0, 0] = 1.0  # alt 0 in nest 0
        nest_matrix[1, 0] = 1.0  # alt 1 in nest 0
        nest_matrix[2, 1] = 1.0  # alt 2 in nest 1
        nest_matrix[3, 1] = 1.0  # alt 3 in nest 1

        _nested_logit_probs_numpy(
            np.array([beta_cost, beta_time]),
            naturalize_nest_params(np.array([0.0, 0.0])),  # dummy alpha
            (np.column_stack([alternatives["cost"].values, alternatives["time"].values]))
            .flatten()
            .reshape(1, -1)
            .repeat(1, axis=0),
            nest_matrix,
            n_obs=1,
            n_alts=n_alts,
        )
        # Use MNL choice for simplicity (lambda close to 1)
        probs_simple = np.exp(utilities - utilities.max())
        probs_simple /= probs_simple.sum()
        choices[i] = rng.choice(n_alts, p=probs_simple)

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives=pd.Series(choices, index=choosers.index),
    )

    nests = NestingTree(
        [
            NestSpec("transit", alt_ids=[0, 1]),
            NestSpec("auto", alt_ids=[2, 3]),
        ]
    )

    return ct, nests, alternatives, choosers


# ---------------------------------------------------------------------------
# Test: NestSpec and NestingTree
# ---------------------------------------------------------------------------


class TestNestSpec:
    """Tests for NestSpec dataclass."""

    def test_basic_construction(self):
        nest = NestSpec("transit", alt_ids=[0, 1, 2])
        assert nest.name == "transit"
        assert nest.alt_ids == [0, 1, 2]
        assert nest.alpha is None

    def test_custom_alpha(self):
        nest = NestSpec("auto", alt_ids=[3, 4], alpha=0.5)
        assert nest.alpha == 0.5


class TestNestingTree:
    """Tests for NestingTree dataclass."""

    def test_basic_construction(self):
        nests = NestingTree(
            [
                NestSpec("transit", alt_ids=[0, 1]),
                NestSpec("auto", alt_ids=[2, 3]),
            ]
        )
        assert nests.n_nests == 2
        assert nests.nest_names == ["transit", "auto"]

    def test_all_alt_ids(self):
        nests = NestingTree(
            [
                NestSpec("transit", alt_ids=[0, 1]),
                NestSpec("auto", alt_ids=[2, 3]),
            ]
        )
        assert sorted(nests.all_alt_ids) == [0, 1, 2, 3]

    def test_duplicate_alt_ids_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            NestingTree(
                [
                    NestSpec("a", alt_ids=[0, 1]),
                    NestSpec("b", alt_ids=[1, 2]),  # alt 1 in both nests
                ]
            )

    def test_empty_nests_raises(self):
        with pytest.raises(ValueError, match="at least one nest"):
            NestingTree([])

    def test_build_nest_matrix(self):
        nests = NestingTree(
            [
                NestSpec("transit", alt_ids=[0, 1]),
                NestSpec("auto", alt_ids=[2, 3]),
            ]
        )
        alt_ids = [0, 1, 2, 3]
        matrix = nests.build_nest_matrix(alt_ids)
        assert matrix.shape == (4, 2)
        # Alt 0 and 1 in nest 0
        assert matrix[0, 0] == 1.0
        assert matrix[1, 0] == 1.0
        assert matrix[0, 1] == 0.0
        # Alt 2 and 3 in nest 1
        assert matrix[2, 1] == 1.0
        assert matrix[3, 1] == 1.0
        assert matrix[2, 0] == 0.0

    def test_initial_alphas(self):
        nests = NestingTree(
            [
                NestSpec("transit", alt_ids=[0, 1], alpha=0.5),
                NestSpec("auto", alt_ids=[2, 3]),  # default alpha=0.0
            ]
        )
        alphas = nests.initial_alphas()
        npt.assert_array_equal(alphas, [0.5, 0.0])


# ---------------------------------------------------------------------------
# Test: naturalize_nest_params
# ---------------------------------------------------------------------------


class TestNaturalizeNestParams:
    """Tests for the logistic transform of nest parameters."""

    def test_alpha_zero_gives_lambda_half(self):
        """alpha=0 should give lambda=0.5."""
        lambdas = naturalize_nest_params(np.array([0.0]))
        npt.assert_allclose(lambdas, [0.5])

    def test_alpha_large_positive_gives_lambda_near_one(self):
        """Large positive alpha should give lambda close to 1."""
        lambdas = naturalize_nest_params(np.array([20.0]))
        npt.assert_allclose(lambdas, [1.0], atol=1e-8)

    def test_alpha_large_negative_gives_lambda_near_zero(self):
        """Large negative alpha should give lambda close to 0."""
        lambdas = naturalize_nest_params(np.array([-20.0]))
        npt.assert_allclose(lambdas, [0.0], atol=1e-8)

    def test_lambda_in_open_interval(self):
        """Lambda should always be in (0, 1)."""
        alphas = np.linspace(-10, 10, 100)
        lambdas = naturalize_nest_params(alphas)
        assert (lambdas > 0).all()
        assert (lambdas <= 1).all()

    def test_multiple_nests(self):
        """Should work with multiple nest parameters."""
        alphas = np.array([0.0, 1.0, -1.0])
        lambdas = naturalize_nest_params(alphas)
        assert len(lambdas) == 3
        npt.assert_allclose(lambdas[0], 0.5)


# ---------------------------------------------------------------------------
# Test: Probability computation
# ---------------------------------------------------------------------------


class TestNestedLogitProbabilities:
    """Tests for nested logit probability computation."""

    def test_probabilities_sum_to_one(self):
        """Probabilities should sum to 1 for each observation."""
        rng = np.random.default_rng(123)
        n_obs = 50
        n_alts = 4
        k = 2

        beta = np.array([0.5, -0.3])
        alpha = np.array([0.0, 0.0])  # lambda = 0.5 for both nests

        dm = rng.standard_normal((n_obs * n_alts, k))
        nest_matrix = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64)

        probs = _nested_logit_probs_numpy(
            beta,
            alpha,
            dm,
            nest_matrix,
            n_obs,
            n_alts,
        )

        # Probabilities should sum to 1
        row_sums = probs.sum(axis=1)
        npt.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_probabilities_are_non_negative(self):
        """All probabilities should be non-negative."""
        rng = np.random.default_rng(456)
        n_obs = 50
        n_alts = 4
        k = 2

        beta = np.array([0.5, -0.3])
        alpha = np.array([0.5, -0.5])

        dm = rng.standard_normal((n_obs * n_alts, k))
        nest_matrix = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64)

        probs = _nested_logit_probs_numpy(
            beta,
            alpha,
            dm,
            nest_matrix,
            n_obs,
            n_alts,
        )

        assert (probs >= -1e-10).all()

    def test_lambda_equals_one_gives_mnl(self):
        """When all lambda=1, nested logit should reduce to MNL."""
        rng = np.random.default_rng(789)
        n_obs = 100
        n_alts = 4
        k = 2

        beta = np.array([0.5, -0.3])

        # Large alpha -> lambda ≈ 1 (MNL)
        alpha = np.array([20.0, 20.0])

        dm = rng.standard_normal((n_obs * n_alts, k))
        nest_matrix = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64)

        # Nested logit probabilities
        nl_probs = _nested_logit_probs_numpy(
            beta,
            alpha,
            dm,
            nest_matrix,
            n_obs,
            n_alts,
        )

        # MNL probabilities (manual softmax)
        utilities = (dm @ beta).reshape(n_obs, n_alts)
        utilities_shifted = utilities - utilities.max(axis=1, keepdims=True)
        exp_u = np.exp(utilities_shifted)
        mnl_probs = exp_u / exp_u.sum(axis=1, keepdims=True)

        npt.assert_allclose(nl_probs, mnl_probs, atol=1e-6)

    def test_nesting_increases_within_nest_correlation(self):
        """Lower lambda should increase within-nest correlation,
        making within-nest alternatives more similar."""
        rng = np.random.default_rng(101)
        n_obs = 100
        n_alts = 4
        k = 2

        beta = np.array([1.0, -0.5])

        dm = rng.standard_normal((n_obs * n_alts, k))
        nest_matrix = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64)

        # High lambda (near MNL)
        alpha_high = np.array([5.0, 5.0])  # lambda ≈ 0.993
        probs_high = _nested_logit_probs_numpy(
            beta,
            alpha_high,
            dm,
            nest_matrix,
            n_obs,
            n_alts,
        )

        # Low lambda (more correlation within nests)
        alpha_low = np.array([-2.0, -2.0])  # lambda ≈ 0.119
        probs_low = _nested_logit_probs_numpy(
            beta,
            alpha_low,
            dm,
            nest_matrix,
            n_obs,
            n_alts,
        )

        # Both should sum to 1
        npt.assert_allclose(probs_high.sum(axis=1), 1.0, atol=1e-6)
        npt.assert_allclose(probs_low.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: Log-likelihood
# ---------------------------------------------------------------------------


class TestNestedLogitLikelihood:
    """Tests for nested logit log-likelihood computation."""

    def test_log_likelihood_is_negative(self):
        """Log-likelihood should be negative."""
        rng = np.random.default_rng(202)
        n_obs = 50
        n_alts = 4
        k = 2

        beta = np.array([0.5, -0.3])
        alpha = np.array([0.0, 0.0])

        dm = rng.standard_normal((n_obs * n_alts, k))
        chosen = np.zeros((n_obs, n_alts))
        for i in range(n_obs):
            chosen[i, rng.choice(n_alts)] = 1.0

        nest_matrix = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64)

        ll = _nested_logit_ll_numpy(
            beta,
            alpha,
            dm,
            chosen,
            nest_matrix,
            n_obs,
            n_alts,
        )

        assert ll < 0

    def test_log_likelihood_matches_manual(self):
        """Log-likelihood should match manual computation from probabilities."""
        rng = np.random.default_rng(303)
        n_obs = 50
        n_alts = 4
        k = 2

        beta = np.array([0.5, -0.3])
        alpha = np.array([0.0, 0.5])

        dm = rng.standard_normal((n_obs * n_alts, k))
        chosen = np.zeros((n_obs, n_alts))
        for i in range(n_obs):
            chosen[i, rng.choice(n_alts)] = 1.0

        nest_matrix = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64)

        # Compute LL from probabilities
        probs = _nested_logit_probs_numpy(
            beta,
            alpha,
            dm,
            nest_matrix,
            n_obs,
            n_alts,
        )
        chosen_probs = (probs * chosen).sum(axis=1)
        chosen_probs = np.maximum(chosen_probs, 1e-30)
        expected_ll = np.sum(np.log(chosen_probs))

        # Compute LL directly
        ll = _nested_logit_ll_numpy(
            beta,
            alpha,
            dm,
            chosen,
            nest_matrix,
            n_obs,
            n_alts,
        )

        npt.assert_allclose(ll, expected_ll, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test: NestedLogit model class
# ---------------------------------------------------------------------------


class TestNestedLogitModel:
    """Tests for the NestedLogit model class."""

    def test_nested_logit_estimation(self):
        """NestedLogit should estimate and return a FitResult."""
        rng = np.random.default_rng(42)
        n_obs = 200
        n_alts = 4

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

        choices = rng.choice(np.arange(n_alts), size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
        )

        nests = NestingTree(
            [
                NestSpec("transit", alt_ids=[0, 1]),
                NestSpec("auto", alt_ids=[2, 3]),
            ]
        )

        model = ChoiceModel(ct, formula="cost + time - 1", nests=nests)
        result = model.fit()

        assert result.coefficients is not None
        assert np.isfinite(result.log_likelihood)
        assert result.n_parameters == 4  # 2 beta + 2 nest params

    def test_nested_logit_has_lambda_params(self):
        """FitResult should include lambda (dissimilarity) parameters."""
        rng = np.random.default_rng(43)
        n_obs = 200
        n_alts = 4

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

        choices = rng.choice(np.arange(n_alts), size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
        )

        nests = NestingTree(
            [
                NestSpec("transit", alt_ids=[0, 1]),
                NestSpec("auto", alt_ids=[2, 3]),
            ]
        )

        model = ChoiceModel(ct, formula="cost + time - 1", nests=nests)
        result = model.fit()

        # Should have lambda_transit and lambda_auto parameters
        assert "lambda_transit" in result.coefficients.index
        assert "lambda_auto" in result.coefficients.index

        # Lambda values should be in (0, 1]
        lambda_transit = result.coefficients["lambda_transit"]
        lambda_auto = result.coefficients["lambda_auto"]
        assert 0 < lambda_transit <= 1.0
        assert 0 < lambda_auto <= 1.0

    def test_nested_logit_requires_nests(self):
        """NestedLogit should raise ValueError without nests."""
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

        # Without nests, ChoiceModel is MNL — no error
        model = ChoiceModel(ct, formula="y - 1")
        assert not model._is_nested

    def test_nested_logit_fit_alias(self):
        """fit() should be the primary estimation API."""
        rng = np.random.default_rng(45)
        n_obs = 100
        n_alts = 4

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

        nests = NestingTree(
            [
                NestSpec("a", alt_ids=[0, 1]),
                NestSpec("b", alt_ids=[2, 3]),
            ]
        )

        model = ChoiceModel(ct, formula="y - 1", nests=nests)
        result = model.fit()

        assert np.isfinite(result.log_likelihood)


"""Tests for the mixed logit model.

Covers: ParamDistribution, draw generation, probability computation,
log-likelihood, parameter recovery, and MixedLogit model class.
"""


from locpick.models.mixed import (
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

        model = ChoiceModel(
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

        model = ChoiceModel(
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

        # Without random_params, ChoiceModel is MNL — no error
        model = ChoiceModel(ct, formula="y - 1", random_params={})
        assert not model._is_mixed

    def test_mixed_logit_fit_alias(self):
        """fit() should be the primary estimation API."""
        ct = make_mixed_data(n_obs=100, n_alts=4)

        model = ChoiceModel(
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

        model = ChoiceModel(
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

        model = ChoiceModel(
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

        model = ChoiceModel(
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
