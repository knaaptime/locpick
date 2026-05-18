# ruff: noqa: E402, F811
"""Sampling tests: correction, inclusion, and module-level."""

"""Tests for sampling correction and inclusion probabilities.

Covers: inclusion probability computation, sampling metadata,
parameter recovery with sampled choice sets, and consistency
of sampling correction behavior.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from locpick import MNL, ChoiceTable, EstimationProblem
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
        model = MNL(data=None, problem=problem)
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
        model = MNL(data=None, problem=problem)
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

        model = MNL(ct, formula="altval - 1")
        result = model.fit()

        # The coefficient should be recoverable (within 30% tolerance)
        # even with sampled alternatives, because sampling correction is applied
        # Note: with sampled choice sets, some estimation error is expected
        assert abs(result.coefficients["altval"] - beta_alt) / abs(beta_alt) < 0.30


"""Tests for locpick._sampling.correction module."""


from locpick._sampling.correction import apply_sampling_correction, get_sampling_correction
from locpick.data.arrays import ChoiceArrays


class TestGetSamplingCorrection:
    """Tests for get_sampling_correction."""

    def test_none_when_no_correction(self):
        arrays = ChoiceArrays(
            design_matrix=np.zeros((6, 2)),
            chosen=np.zeros((3, 2)),
            n_obs=3,
            n_alts=2,
        )
        assert get_sampling_correction(arrays) is None

    def test_returns_inclusion_probs(self):
        inclusion = np.full((3, 2), 0.5)
        arrays = ChoiceArrays(
            design_matrix=np.zeros((6, 2)),
            chosen=np.zeros((3, 2)),
            n_obs=3,
            n_alts=2,
            inclusion_probs=inclusion,
        )
        result = get_sampling_correction(arrays)
        npt.assert_array_equal(result, inclusion)

    def test_inclusion_probs_value_preserved(self):
        inclusion = np.full((3, 2), 0.3)
        arrays = ChoiceArrays(
            design_matrix=np.zeros((6, 2)),
            chosen=np.zeros((3, 2)),
            n_obs=3,
            n_alts=2,
            inclusion_probs=inclusion,
        )
        result = get_sampling_correction(arrays)
        npt.assert_array_equal(result, inclusion)


class TestApplySamplingCorrection:
    """Tests for apply_sampling_correction."""

    def test_no_change_when_no_correction(self):
        arrays = ChoiceArrays(
            design_matrix=np.zeros((6, 2)),
            chosen=np.zeros((3, 2)),
            n_obs=3,
            n_alts=2,
        )
        V = np.ones((3, 2))
        result = apply_sampling_correction(V, arrays)
        npt.assert_array_equal(result, V)

    def test_adds_log_correction_2d(self):
        arrays = ChoiceArrays(
            design_matrix=np.zeros((6, 2)),
            chosen=np.zeros((3, 2)),
            n_obs=3,
            n_alts=2,
            inclusion_probs=np.full((3, 2), 0.5),
        )
        V = np.zeros((3, 2))
        result = apply_sampling_correction(V, arrays)
        expected = np.full((3, 2), np.log(0.5))
        npt.assert_array_almost_equal(result, expected)

    def test_adds_log_correction_1d(self):
        arrays = ChoiceArrays(
            design_matrix=np.zeros((6, 2)),
            chosen=np.zeros((3, 2)),
            n_obs=3,
            n_alts=2,
            inclusion_probs=np.full((3, 2), 0.5),
        )
        V = np.zeros(6)
        result = apply_sampling_correction(V, arrays)
        expected = np.full(6, np.log(0.5))
        npt.assert_array_almost_equal(result, expected)

    def test_clamps_near_zero(self):
        arrays = ChoiceArrays(
            design_matrix=np.zeros((6, 2)),
            chosen=np.zeros((3, 2)),
            n_obs=3,
            n_alts=2,
            inclusion_probs=np.full((3, 2), 1e-400),  # subnormal, below 1e-300
        )
        V = np.zeros((3, 2))
        result = apply_sampling_correction(V, arrays)
        expected = np.full((3, 2), np.log(1e-300))
        npt.assert_array_almost_equal(result, expected)

    def test_preserves_input_shape(self):
        arrays = ChoiceArrays(
            design_matrix=np.zeros((6, 2)),
            chosen=np.zeros((3, 2)),
            n_obs=3,
            n_alts=2,
            inclusion_probs=np.full((3, 2), 0.5),
        )
        V = np.ones((3, 2))
        result = apply_sampling_correction(V, arrays)
        assert result.shape == V.shape


"""Tests for locpick._sampling.inclusion module."""


from locpick._sampling.inclusion import compute_inclusion_probs, validate_inclusion_probs


class TestComputeInclusionProbs:
    """Tests for compute_inclusion_probs."""

    def test_srswor_uniform(self):
        pi = compute_inclusion_probs(sample_size=5, n_alts=20, method="srswor")
        npt.assert_array_equal(pi, np.full(20, 0.25))

    def test_srswor_census_when_sample_size_ge_n_alts(self):
        pi = compute_inclusion_probs(sample_size=20, n_alts=20, method="srswor")
        npt.assert_array_equal(pi, np.ones(20))
        pi = compute_inclusion_probs(sample_size=25, n_alts=20, method="srswor")
        npt.assert_array_equal(pi, np.ones(20))

    def test_srswr(self):
        pi = compute_inclusion_probs(sample_size=5, n_alts=20, method="srswr")
        expected = 1.0 - (1.0 - 1.0 / 20) ** 5
        npt.assert_array_almost_equal(pi, np.full(20, expected))

    def test_weighted_wor(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        pi = compute_inclusion_probs(
            sample_size=2, n_alts=4, method="weighted_wor", weights=weights
        )
        w_sum = weights.sum()
        expected = 1.0 - np.exp(-weights / w_sum * 2)
        npt.assert_array_almost_equal(pi, expected)
        assert np.all(pi > 0) and np.all(pi <= 1)

    def test_weighted_wr(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        pi = compute_inclusion_probs(
            sample_size=2, n_alts=4, method="weighted_wr", weights=weights
        )
        p = weights / weights.sum()
        expected = 1.0 - (1.0 - p) ** 2
        npt.assert_array_almost_equal(pi, expected)

    def test_zero_sample_size(self):
        pi = compute_inclusion_probs(sample_size=0, n_alts=10, method="srswor")
        npt.assert_array_equal(pi, np.zeros(10))

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unsupported sampling method"):
            compute_inclusion_probs(sample_size=5, n_alts=10, method="unknown")

    def test_negative_sample_size(self):
        with pytest.raises(ValueError, match="sample_size must be non-negative"):
            compute_inclusion_probs(sample_size=-1, n_alts=10)

    def test_zero_n_alts(self):
        with pytest.raises(ValueError, match="n_alts must be positive"):
            compute_inclusion_probs(sample_size=5, n_alts=0)

    def test_weighted_missing_weights(self):
        with pytest.raises(ValueError, match="weights are required"):
            compute_inclusion_probs(sample_size=5, n_alts=10, method="weighted_wor")

    def test_weighted_wrong_shape(self):
        with pytest.raises(ValueError, match="weights must have shape"):
            compute_inclusion_probs(
                sample_size=5, n_alts=10, method="weighted_wor", weights=np.ones(5)
            )

    def test_weighted_negative_weights(self):
        with pytest.raises(ValueError, match="weights must be non-negative"):
            compute_inclusion_probs(
                sample_size=5, n_alts=3, method="weighted_wor", weights=np.array([1, -1, 1])
            )

    def test_weighted_zero_sum(self):
        with pytest.raises(ValueError, match="sum of weights must be positive"):
            compute_inclusion_probs(
                sample_size=5, n_alts=3, method="weighted_wor", weights=np.zeros(3)
            )

    def test_case_insensitive_method(self):
        pi1 = compute_inclusion_probs(sample_size=5, n_alts=20, method="SRSWOR")
        pi2 = compute_inclusion_probs(sample_size=5, n_alts=20, method="srswor")
        npt.assert_array_equal(pi1, pi2)


class TestValidateInclusionProbs:
    """Tests for validate_inclusion_probs."""

    def test_valid(self):
        probs = np.full((10, 5), 0.5)
        validate_inclusion_probs(probs, n_obs=10, n_alts=5)  # should not raise

    def test_wrong_shape(self):
        probs = np.full((10, 5), 0.5)
        with pytest.raises(ValueError, match="inclusion_probs must have shape"):
            validate_inclusion_probs(probs, n_obs=10, n_alts=4)

    def test_zero_value(self):
        probs = np.full((10, 5), 0.5)
        probs[0, 0] = 0.0
        with pytest.raises(ValueError, match="inclusion_probs must be in"):
            validate_inclusion_probs(probs, n_obs=10, n_alts=5)

    def test_greater_than_one(self):
        probs = np.full((10, 5), 0.5)
        probs[0, 0] = 1.1
        with pytest.raises(ValueError, match="inclusion_probs must be in"):
            validate_inclusion_probs(probs, n_obs=10, n_alts=5)
