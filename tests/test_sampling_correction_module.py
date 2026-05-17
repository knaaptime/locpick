"""Tests for locpick._sampling.correction module."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

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
