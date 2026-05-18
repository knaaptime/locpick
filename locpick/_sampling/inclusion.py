"""Inclusion probability computation for sampled choice sets.

This module provides functions for computing inclusion probabilities —
the probability that each alternative is included in a sampled choice set.
Inclusion probabilities are needed for sampling correction in estimation.
"""

from __future__ import annotations

import numpy as np


def compute_inclusion_probs(
    sample_size: int,
    n_alts: int,
    method: str = "srswor",
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-alternative inclusion probabilities.

    Parameters
    ----------
    sample_size : int
        Number of alternatives sampled per observation.
    n_alts : int
        Total number of alternatives in the full choice set.
    method : str, default "srswor"
        Sampling method. Supported values:

        - ``"srswor"`` — simple random sampling *without* replacement.
          Uniform inclusion probability: ``sample_size / n_alts``.
        - ``"srswr"`` — simple random sampling *with* replacement.
          Inclusion probability:
          ``1 - (1 - 1/n_alts) ** sample_size``.
        - ``"weighted_wor"`` — weighted sampling without replacement.
          Requires ``weights``. Uses a Poisson approximation:
          ``1 - exp(-w_i / sum(w) * sample_size)``.
        - ``"weighted_wr"`` — weighted sampling with replacement.
          Requires ``weights``. Inclusion probability:
          ``1 - (1 - w_i / sum(w)) ** sample_size``.

    weights : np.ndarray or None, optional
        Per-alternative weights, shape ``(n_alts,)``. Required for
        weighted methods.

    Returns
    -------
    np.ndarray
        Inclusion probabilities, shape ``(n_alts,)``. All values are
        in ``(0, 1]``.

    Raises
    ------
    ValueError
        If ``method`` is unsupported, ``sample_size`` is invalid, or
        ``weights`` is missing for a weighted method.
    """
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if n_alts <= 0:
        raise ValueError("n_alts must be positive")
    if sample_size == 0:
        return np.zeros(n_alts, dtype=np.float64)

    method = method.lower()

    if method in ("srswor", "census"):
        if sample_size >= n_alts:
            return np.ones(n_alts, dtype=np.float64)
        return np.full(n_alts, sample_size / n_alts, dtype=np.float64)

    if method == "srswr":
        p = 1.0 - (1.0 - 1.0 / n_alts) ** sample_size
        return np.full(n_alts, p, dtype=np.float64)

    if method in ("weighted_wor", "weighted_wr"):
        if weights is None:
            raise ValueError(f"weights are required for method={method!r}")
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n_alts,):
            raise ValueError(f"weights must have shape ({n_alts},), got {weights.shape}")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        w_sum = weights.sum()
        if w_sum == 0:
            raise ValueError("sum of weights must be positive")

        if method == "weighted_wor":
            # Poisson approximation for weighted sampling without replacement
            probs = 1.0 - np.exp(-weights / w_sum * sample_size)
        else:  # weighted_wr
            p_i = weights / w_sum
            probs = 1.0 - (1.0 - p_i) ** sample_size

        # Clamp to (0, 1] to avoid numerical issues
        return np.clip(probs, 1e-300, 1.0)

    raise ValueError(
        f"Unsupported sampling method: {method!r}. "
        f"Supported: 'srswor', 'srswr', 'weighted_wor', 'weighted_wr'."
    )


def validate_inclusion_probs(
    probs: np.ndarray,
    n_obs: int,
    n_alts: int,
) -> None:
    """Validate inclusion probability array shape and bounds.

    Parameters
    ----------
    probs : np.ndarray
        Inclusion probability array.
    n_obs : int
        Expected number of observations.
    n_alts : int
        Expected number of alternatives.

    Raises
    ------
    ValueError
        If shape is incorrect or values are outside ``(0, 1]``.
    """
    probs = np.asarray(probs)
    expected_shape = (n_obs, n_alts)
    if probs.shape != expected_shape:
        raise ValueError(f"inclusion_probs must have shape {expected_shape}, got {probs.shape}")
    if np.any(probs <= 0) or np.any(probs > 1):
        raise ValueError("inclusion_probs must be in (0, 1]")
