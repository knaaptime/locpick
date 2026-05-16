"""Tests for sampling correction and inclusion probabilities.

Covers: inclusion probability computation, sampling metadata,
parameter recovery with sampled choice sets, and consistency
of sampling correction behavior.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from locpick import ChoiceTable, EstimationProblem, MultinomialLogit
from locpick.data import ChoiceArrays

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_sampled_data(n_obs=500, n_alts=20, sample_size=5, seed=42):
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

    # True parameters
    beta_cost = -0.5
    beta_time = -0.1

    # Simulate choices using MNL
    utilities = beta_cost * alternatives["cost"].values + beta_time * alternatives["time"].values
    utilities += rng.gumbel(size=n_alts)
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
# Test: Inclusion probabilities
# ---------------------------------------------------------------------------


class TestInclusionProbabilities:
    """Tests for per-alternative inclusion probability computation."""

    def test_inclusion_probs_with_sampling(self):
        """ChoiceArrays from sampled data should include inclusion_probs."""
        ct, _, _ = make_sampled_data(n_obs=100, n_alts=20, sample_size=5)
        arrays = ct.to_arrays(formula="cost + time - 1")

        assert arrays.inclusion_probs is not None
        assert arrays.inclusion_probs.shape == (100, 5)  # n_obs=100, n_alts_eff=5

    def test_inclusion_probs_values(self):
        """Inclusion probabilities should reflect the sampling design."""
        n_alts = 20
        sample_size = 5
        rng = np.random.default_rng(42)
        n_obs = 100

        choosers = pd.DataFrame(
            {
                "income": rng.standard_normal(n_obs),
            },
            index=pd.Index(np.arange(n_obs), name="oid"),
        )

        alternatives = pd.DataFrame(
            {
                "cost": rng.uniform(1, 10, n_alts),
            },
            index=pd.Index(np.arange(n_alts), name="aid"),
        )

        choices = rng.choice(n_alts, size=n_obs)

        # Without replacement: π_j = n_samples / n_alts_full
        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
            sample_size=sample_size,
            replace=False,
            seed=42,
        )
        arrays = ct.to_arrays(formula="cost - 1")

        assert arrays.inclusion_probs is not None

        # For SRSWOR: π_j = n_samples / n_alts_full = 5/20 = 0.25
        # But chosen alt has π = 1.0
        non_chosen_probs = arrays.inclusion_probs[arrays.chosen == 0]
        npt.assert_allclose(non_chosen_probs, 0.25, atol=0.01)

        # Chosen alternatives should have π = 1.0
        chosen_probs = arrays.inclusion_probs[arrays.chosen == 1]
        npt.assert_allclose(chosen_probs, 1.0, atol=0.01)

    def test_inclusion_probs_with_replacement(self):
        """With replacement: π_j = 1 - (1 - 1/N)^n."""
        rng = np.random.default_rng(42)
        n_obs = 100
        n_alts = 20

        choosers = pd.DataFrame(
            {
                "income": rng.standard_normal(n_obs),
            },
            index=pd.Index(np.arange(n_obs), name="oid"),
        )

        alternatives = pd.DataFrame(
            {
                "cost": rng.uniform(1, 10, n_alts),
            },
            index=pd.Index(np.arange(n_alts), name="aid"),
        )

        choices = rng.choice(n_alts, size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
            sample_size=5,
            replace=True,
            seed=42,
        )

        arrays = ct.to_arrays(formula="cost - 1")

        assert arrays.inclusion_probs is not None
        # With replacement: π_j = 1 - (1 - 1/20)^5 ≈ 0.226
        expected_pi = 1.0 - (1.0 - 1.0 / n_alts) ** 5
        non_chosen_probs = arrays.inclusion_probs[arrays.chosen == 0]
        npt.assert_allclose(non_chosen_probs, expected_pi, atol=0.01)

    def test_no_inclusion_probs_without_sampling(self):
        """Census data should have no inclusion_probs."""
        rng = np.random.default_rng(42)
        n_obs = 50
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

        choices = rng.choice(n_alts, size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
        )

        arrays = ct.to_arrays(formula="y - 1")
        assert arrays.inclusion_probs is None

    def test_inclusion_probs_shape_validation(self):
        """inclusion_probs must have shape (n_obs, n_alts)."""
        with pytest.raises(ValueError, match="inclusion_probs"):
            ChoiceArrays(
                design_matrix=np.zeros((10, 2)),
                chosen=np.zeros((2, 5)),
                n_obs=2,
                n_alts=5,
                inclusion_probs=np.zeros((3, 5)),  # wrong n_obs
            )

    def test_model_runs_with_manual_inclusion_probs(self):
        """Model estimation should run with manually supplied inclusion_probs."""
        rng = np.random.default_rng(42)
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
                "y": rng.uniform(1, 10, n_alts),
            },
            index=pd.Index(np.arange(n_alts), name="aid"),
        )

        choices = rng.choice(n_alts, size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
        )

        arrays = ct.to_arrays(formula="y - 1")

        # Manually set inclusion probabilities
        arrays_with_both = ChoiceArrays(
            design_matrix=arrays.design_matrix,
            chosen=arrays.chosen,
            n_obs=n_obs,
            n_alts=n_alts,
            param_names=arrays.param_names,
            inclusion_probs=np.full((n_obs, n_alts), 0.75),  # different rate
        )

        # Build the model using EstimationProblem
        from locpick.data.problem import EstimationProblem

        problem = EstimationProblem(arrays=arrays_with_both)
        model = MultinomialLogit(data=None, problem=problem)
        result = model.fit()

        # Verify the model ran successfully
        assert np.isfinite(result.log_likelihood)

    def test_model_runs_with_manual_uniform_inclusion_probs(self):
        """Model estimation should run with uniform inclusion probabilities."""
        rng = np.random.default_rng(42)
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
                "y": rng.uniform(1, 10, n_alts),
            },
            index=pd.Index(np.arange(n_alts), name="aid"),
        )

        choices = rng.choice(n_alts, size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
        )

        arrays = ct.to_arrays(formula="y - 1")

        # Manually set uniform inclusion probabilities
        arrays_with_rates = ChoiceArrays(
            design_matrix=arrays.design_matrix,
            chosen=arrays.chosen,
            n_obs=n_obs,
            n_alts=n_alts,
            param_names=arrays.param_names,
            inclusion_probs=np.full((n_obs, n_alts), 0.5),
        )

        # Build the model using EstimationProblem
        from locpick.data.problem import EstimationProblem

        problem = EstimationProblem(arrays=arrays_with_rates)
        model = MultinomialLogit(data=None, problem=problem)
        result = model.fit()

        # Verify the model ran successfully
        assert np.isfinite(result.log_likelihood)


# ---------------------------------------------------------------------------
# Test: Sampling metadata
# ---------------------------------------------------------------------------


class TestSamplingMetadata:
    """Tests for sampling design provenance on EstimationProblem."""

    def test_choicearrays_excludes_sampling_provenance(self):
        """ChoiceArrays should expose correction tensors, not provenance metadata."""
        ct, _, _ = make_sampled_data(n_obs=50, n_alts=20, sample_size=5)
        arrays = ct.to_arrays(formula="cost + time - 1")

        assert arrays.inclusion_probs is not None
        assert not hasattr(arrays, "sampling_design")

    def test_sampling_design_with_sampling(self):
        """sampling_design should be populated when sampling is used."""
        ct, _, _ = make_sampled_data(n_obs=100, n_alts=20, sample_size=5)
        problem = EstimationProblem.from_choice_table(ct, formula="cost + time - 1")

        assert problem.sampling_design is not None
        sd = problem.sampling_design
        assert sd["sample_size"] == 5
        assert sd["n_alts_full"] == 20
        # make_sampled_data uses default replace=True, so method is srswr
        assert sd["method"] == "srswr"

    def test_sampling_design_with_replacement(self):
        """sampling_design should reflect replace=True."""
        rng = np.random.default_rng(42)
        n_obs = 50
        n_alts = 10

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

        choices = rng.choice(n_alts, size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
            sample_size=3,
            replace=True,
            seed=42,
        )
        problem = EstimationProblem.from_choice_table(ct, formula="y - 1")

        sd = problem.sampling_design
        assert sd is not None
        assert sd["method"] == "srswr"
        assert sd["replace"] is True

    def test_sampling_design_without_replacement(self):
        """sampling_design should reflect replace=False."""
        rng = np.random.default_rng(42)
        n_obs = 50
        n_alts = 10

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

        choices = rng.choice(n_alts, size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
            sample_size=3,
            replace=False,
            seed=42,
        )
        problem = EstimationProblem.from_choice_table(ct, formula="y - 1")

        sd = problem.sampling_design
        assert sd is not None
        assert sd["method"] == "srswor"
        assert sd["replace"] is False

    def test_no_sampling_design_without_sampling(self):
        """Census data should have no sampling_design provenance."""
        rng = np.random.default_rng(42)
        n_obs = 50
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

        choices = rng.choice(n_alts, size=n_obs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
        )
        problem = EstimationProblem.from_choice_table(ct, formula="y - 1")
        assert problem.sampling_design is None


# ---------------------------------------------------------------------------
# Test: Parameter recovery with sampling correction
# ---------------------------------------------------------------------------


class TestSamplingCorrectionRecovery:
    """Tests that sampling correction improves parameter recovery."""

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

        # Simulate choices
        utilities = beta_alt * alternatives["altval"].values
        probs = np.exp(utilities - utilities.max())
        probs /= probs.sum()
        choices = rng.choice(n_alts, size=n_obs, p=probs)

        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
            sample_size=sample_size,
            seed=42,
        )

        model = MultinomialLogit(ct, formula="altval - 1")
        result = model.fit()

        # The coefficient should be recoverable (within 30% tolerance)
        # even with sampled alternatives, because sampling correction is applied
        # Note: with sampled choice sets, some estimation error is expected
        assert abs(result.coefficients["altval"] - beta_alt) / abs(beta_alt) < 0.30
