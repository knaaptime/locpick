"""Tests for locpick._sampling.inclusion module."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

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
