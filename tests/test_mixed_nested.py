"""Tests for the Mixed Nested Logit model."""

import numpy as np
import pytest

from locpick import ChoiceModel, ChoiceTable
from locpick.dgp import simulate_mixed_nested_logit
from locpick.models.mixed import ParamDistribution

from locpick.models.nested import NestingTree, NestSpec

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_mixed_nested_data(n_obs=500, n_alts=4, seed=42):
    """Create synthetic data with nests and random coefficients.

    Nests:
    - Nest 0: alternatives 0, 1 (e.g., "transit")
    - Nest 1: alternatives 2, 3 (e.g., "auto")
    """
    rng = np.random.default_rng(seed)

    import pandas as pd

    choosers = pd.DataFrame(
        {"income": rng.standard_normal(n_obs)},
        index=pd.Index(np.arange(n_obs), name="oid"),
    )

    alternatives = pd.DataFrame(
        {
            "cost": rng.uniform(1, 10, n_alts),
            "time": rng.uniform(5, 30, n_alts),
        },
        index=pd.Index(np.arange(n_alts), name="aid"),
    )

    # Simple random choices for now
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

    return ct, nests


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMixedNestedMNL:
    """Tests for the MixedNestedMNL model class."""

    def test_without_nests_is_not_nested(self):
        """ChoiceModel without nests should not raise — it's just mixed."""
        ct, nests = make_mixed_nested_data()

        model = ChoiceModel(
            ct,
            formula="cost + time - 1",
            random_params={"time": ParamDistribution("normal", "time")},
            n_draws=50,
        )
        assert not model._is_nested

    def test_without_random_params_is_not_mixed(self):
        """ChoiceModel without random_params should not raise — it's just nested."""
        ct, nests = make_mixed_nested_data()

        model = ChoiceModel(
            ct,
            formula="cost + time - 1",
            nests=nests,
            n_draws=50,
        )
        assert not model._is_mixed

    def test_mixed_nested_logit_estimation(self):
        """MixedNestedMNL should estimate and return a FitResult."""
        ct, nests = make_mixed_nested_data(n_obs=200)

        model = ChoiceModel(
            ct,
            formula="cost + time - 1",
            nests=nests,
            random_params={"time": ParamDistribution("normal", "time")},
            n_draws=50,
            draw_type="halton",
            seed=42,
        )
        result = model.fit()

        assert result.coefficients is not None
        assert np.isfinite(result.log_likelihood)
        # Should have: cost (fixed), lambda_transit, lambda_auto, mean_time, sd_time
        assert "lambda_transit" in result.coefficients.index
        assert "lambda_auto" in result.coefficients.index
        assert "mean_time" in result.coefficients.index
        assert "sd_time" in result.coefficients.index

    def test_mixed_nested_logit_multiple_random_params(self):
        """MixedNestedMNL should handle multiple random parameters."""
        ct, nests = make_mixed_nested_data(n_obs=200)

        model = ChoiceModel(
            ct,
            formula="cost + time - 1",
            nests=nests,
            random_params={
                "cost": ParamDistribution("normal", "cost"),
                "time": ParamDistribution("normal", "time"),
            },
            n_draws=50,
            draw_type="random",
            seed=42,
        )
        result = model.fit()

        # No fixed params since both are random
        assert "mean_cost" in result.coefficients.index
        assert "sd_cost" in result.coefficients.index
        assert "mean_time" in result.coefficients.index
        assert "sd_time" in result.coefficients.index
        assert "lambda_transit" in result.coefficients.index
        assert "lambda_auto" in result.coefficients.index

    def test_mixed_nested_logit_with_fixed_params(self):
        """MixedNestedMNL should handle mix of fixed and random parameters."""
        ct, nests = make_mixed_nested_data(n_obs=200)

        model = ChoiceModel(
            ct,
            formula="cost + time - 1",
            nests=nests,
            random_params={"time": ParamDistribution("normal", "time")},
            n_draws=50,
            draw_type="halton",
            seed=42,
        )
        result = model.fit()

        # cost is fixed, time is random
        assert "cost" in result.coefficients.index
        assert "mean_time" in result.coefficients.index
        assert "sd_time" in result.coefficients.index

    def test_mixed_nested_logit_lognormal(self):
        """MixedNestedMNL should work with lognormal distribution."""
        ct, nests = make_mixed_nested_data(n_obs=200)

        model = ChoiceModel(
            ct,
            formula="cost + time - 1",
            nests=nests,
            random_params={"time": ParamDistribution("lognormal", "time")},
            n_draws=50,
            draw_type="random",
            seed=42,
        )
        result = model.fit()

        assert np.isfinite(result.log_likelihood)
        assert "mean_time" in result.coefficients.index
        assert "sd_time" in result.coefficients.index


class TestMixedNestedDGP:
    """Tests for the mixed nested logit DGP."""

    def test_simulate_mixed_nested_logit(self):
        """simulate_mixed_nested_logit should return a valid dataset."""
        dataset = simulate_mixed_nested_logit(n_obs=500, n_alts=4, seed=42)

        assert dataset.n_obs == 500
        assert dataset.n_alts == 4
        assert "choice" in dataset.choosers.columns
        assert dataset.nests is not None
        assert len(dataset.true_lambdas) == 2
        assert len(dataset.true_random_means) == 1
        assert len(dataset.true_random_spreads) == 1

    def test_simulate_mixed_nested_logit_custom_params(self):
        """simulate_mixed_nested_logit should accept custom parameters."""
        dataset = simulate_mixed_nested_logit(
            n_obs=1000,
            n_alts=6,
            alt_params={"cost": -0.3, "time": -0.05},
            nest_lambdas={"group_a": 0.6, "group_b": 0.75, "group_c": 0.9},
            random_params={"time": ("normal", -0.2, 0.3)},
            seed=123,
        )

        assert dataset.n_obs == 1000
        assert dataset.n_alts == 6
        assert len(dataset.true_lambdas) == 3
