"""Sampling correction functions for choice models.

This module provides centralized functions for applying sampling corrections
to utility expressions. When alternatives are sampled rather than using the
full choice set, a correction term ``log(inclusion_probability)`` must be
added to each alternative's systematic utility to ensure consistent parameter
estimates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..data.arrays import ChoiceArrays


def get_sampling_correction(arrays: "ChoiceArrays") -> np.ndarray | None:
    """Return the effective sampling correction array.

    Uses per-alternative ``inclusion_probs``.

    Parameters
    ----------
    arrays : ChoiceArrays
        Estimation data container.

    Returns
    -------
    np.ndarray or None
        Array of shape ``(n_obs, n_alts)`` with the correction values,
        or ``None`` if no sampling correction is available.
    """
    correction = arrays.inclusion_probs
    if correction is None:
        return None
    return np.asarray(correction, dtype=np.float64)


def apply_sampling_correction(
    utilities: np.ndarray,
    arrays: "ChoiceArrays",
) -> np.ndarray:
    """Add the sampling correction term to systematic utilities.

    The correction term is ``log(inclusion_probability)``. A small floor of ``1e-300``
    is applied before taking the logarithm to avoid ``log(0)``.

    Parameters
    ----------
    utilities : np.ndarray
        Systematic utilities. May be 2-D with shape ``(n_obs, n_alts)``
        or 1-D with shape ``(n_obs * n_alts,)``.
    arrays : ChoiceArrays
        Estimation data container.

    Returns
    -------
    np.ndarray
        Utilities with sampling correction added, same shape as input.
    """
    correction = get_sampling_correction(arrays)
    if correction is None:
        return utilities

    n_obs = arrays.n_obs
    n_alts = arrays.n_alts
    correction_arr = correction.reshape(n_obs, n_alts)
    log_correction = np.log(np.maximum(correction_arr, 1e-300))

    if utilities.ndim == 1:
        return utilities + log_correction.ravel()
    return utilities + log_correction
