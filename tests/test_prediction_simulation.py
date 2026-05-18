"""Tests for prediction and simulation functionality.

Covers: model.simulate(), prediction on new data,
prediction with sampled choice sets, and reproducibility.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd

from locpick import ChoiceTable, MNL, NestedMNL, NestSpec
from locpick.models.nested import NestingTree

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_simple_data(n_obs=200, n_alts=5, seed=42):
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


def _make_sampled_data(n_obs=200, n_alts=20, sample_size=5, seed=42):
    """Create synthetic data with sampled choice sets."""
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

    beta_cost = -0.5
    beta_time = -0.1

    utilities = beta_cost * alternatives["cost"].values + beta_time * alternatives["time"].values
    probs = np.exp(utilities - utilities.max())
    probs /= probs.sum()
    choices = rng.choice(n_alts, size=n_obs, p=probs)

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives=pd.Series(choices, index=choosers.index),
        sample_size=sample_size,
        seed=seed,
    )

    return ct, beta_cost, beta_time


# ---------------------------------------------------------------------------
# Test: FitResult.simulate()
# ---------------------------------------------------------------------------


class TestSimulate:
    """Tests for the simulate() method on FitResult."""

    def test_simulate_basic(self):
        """simulate() should return a DataFrame with expected columns."""
        ct, _, _ = _make_simple_data()
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        simulated = model.simulate(ct, n_draws=1, seed=42)

        assert isinstance(simulated, pd.DataFrame)
        assert "draw" in simulated.columns
        assert "probability" in simulated.columns
        assert len(simulated) == ct.n_observations

    def test_simulate_multiple_draws(self):
        """simulate() with n_draws > 1 should return n_obs * n_draws rows."""
        ct, _, _ = _make_simple_data()
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        n_draws = 5
        simulated = model.simulate(ct, n_draws=n_draws, seed=42)

        assert len(simulated) == ct.n_observations * n_draws
        assert set(simulated["draw"].unique()) == set(range(n_draws))

    def test_simulate_reproducibility(self):
        """simulate() with same seed should produce identical results."""
        ct, _, _ = _make_simple_data()
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        sim1 = model.simulate(ct, n_draws=1, seed=123)
        sim2 = model.simulate(ct, n_draws=1, seed=123)

        pd.testing.assert_frame_equal(sim1, sim2)

    def test_simulate_different_seeds(self):
        """simulate() with different seeds should produce different results."""
        ct, _, _ = _make_simple_data()
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        sim1 = model.simulate(ct, n_draws=1, seed=123)
        sim2 = model.simulate(ct, n_draws=1, seed=456)

        # Different seeds should produce different choices (with high probability)
        assert not sim1["aid"].equals(sim2["aid"])

    def test_simulate_probabilities_are_valid(self):
        """Simulated choice probabilities should be valid (0, 1]."""
        ct, _, _ = _make_simple_data()
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        simulated = model.simulate(ct, n_draws=1, seed=42)

        assert (simulated["probability"] > 0).all()
        assert (simulated["probability"] <= 1).all()

    def test_simulate_chosen_alts_are_valid(self):
        """Simulated choices should be valid alternative IDs."""
        ct, _, _ = _make_simple_data()
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        simulated = model.simulate(ct, n_draws=1, seed=42)

        # All chosen alternatives should be in the data
        valid_alts = set(ct.to_frame()["aid"].unique())
        assert set(simulated["aid"].unique()).issubset(valid_alts)


# ---------------------------------------------------------------------------
# Test: Prediction on new data
# ---------------------------------------------------------------------------


class TestPredictionNewData:
    """Tests for prediction on data not used for estimation."""

    def test_predict_new_choosers(self):
        """Prediction on new choosers should produce valid probabilities."""
        ct, _, _ = _make_simple_data(n_obs=200)
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        # Create new choosers
        rng = np.random.default_rng(99)
        n_new = 50
        new_choosers = pd.DataFrame(
            {
                "income": rng.standard_normal(n_new),
            },
            index=pd.Index(np.arange(200, 200 + n_new), name="oid"),
        )

        alternatives = pd.DataFrame(
            {
                "cost": ct.to_frame().groupby("aid")["cost"].first(),
                "time": ct.to_frame().groupby("aid")["time"].first(),
            }
        ).reindex(range(5))

        # Predict on new data (no choices needed for prediction)
        new_ct = ChoiceTable.from_tables(
            new_choosers,
            alternatives,
            chosen_alternatives=pd.Series(rng.choice(5, size=n_new), index=new_choosers.index),
        )

        probs = model.probabilities(new_ct)

        # Probabilities should sum to ~1 per observation
        probs_2d = probs.reshape(n_new, 5)
        npt.assert_allclose(probs_2d.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_sampled_choice_sets(self):
        """Prediction with sampled choice sets should use inclusion_probs."""
        ct, _, _ = _make_sampled_data(n_obs=200, n_alts=20, sample_size=5)
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        # Probabilities should be valid
        probs = model.probabilities(ct)
        arrays = ct.to_arrays(formula="cost + time - 1")
        probs_2d = probs.reshape(arrays.n_obs, arrays.n_alts)

        # Probabilities should sum to 1 per observation
        npt.assert_allclose(probs_2d.sum(axis=1), 1.0, atol=1e-10)

        # All probabilities should be non-negative
        assert (probs >= 0).all()


# ---------------------------------------------------------------------------
# Test: Nested logit prediction
# ---------------------------------------------------------------------------


class TestNestedLogitPrediction:
    """Tests for nested logit prediction with inclusion_probs."""

    def test_nested_logit_probabilities_with_inclusion_probs(self):
        """NestedLogit.probabilities() should use inclusion_probs when available."""
        rng = np.random.default_rng(42)
        n_obs = 100
        n_alts = 5

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

        choices = rng.choice(n_alts, size=n_obs)

        # Use census data (no sampling) so probabilities sum to 1
        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
        )

        nests = NestingTree(
            [
                NestSpec(name="nest_a", alt_ids=[0, 1, 2], alpha=None),
                NestSpec(name="nest_b", alt_ids=[3, 4], alpha=None),
            ]
        )

        model = NestedMNL(ct, formula="cost + time - 1", nests=nests)
        model.fit()

        # probabilities() should work
        probs = model.probabilities()
        assert probs is not None
        assert probs.shape[0] == ct.n_observations
        assert probs.shape[1] == ct.n_alternatives

        # Probabilities should sum to ~1 per observation
        npt.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Test: Utilities with sampling correction
# ---------------------------------------------------------------------------


class TestUtilitiesSamplingCorrection:
    """Tests for utilities() with sampling correction."""

    def test_utilities_include_sampling_correction(self):
        """utilities() should include log(inclusion_probs) when present."""
        ct, _, _ = _make_sampled_data(n_obs=200, n_alts=20, sample_size=5)
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        utilities = model.utilities(ct)

        # Utilities should be finite
        assert np.all(np.isfinite(utilities))

    def test_utilities_without_sampling(self):
        """utilities() should work without sampling correction."""
        ct, _, _ = _make_simple_data()
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        utilities = model.utilities(ct)

        # Utilities should be finite
        assert np.all(np.isfinite(utilities))
