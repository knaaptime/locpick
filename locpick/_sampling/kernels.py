"""Sampling kernels for choice set construction.

This module provides compiled sampling kernels for drawing alternatives
without replacement, which is the core operation in constructing choice
tables for location choice models.

Two families of kernels are provided:

1. **ChoiceTable kernels** — operate on actual alternative IDs with
   per-observation exclusion support.  These are used by
   :class:`locpick.data.ChoiceTable` for sampling choice sets.

2. **Index-based kernels** — operate on 0-based indices without
   exclusion.  These are used by the public :func:`sample_alternatives`
   API for simple sampling tasks.

All kernels are implemented in Numba for performance, with pure-Python
fallbacks when Numba is not available.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def _decorator(func):
            return func

        return _decorator


# ------------------------------------------------------------------
# ChoiceTable sampling kernels (with exclusion support)
# ------------------------------------------------------------------

if HAS_NUMBA:

    @njit(cache=True)
    def _sample_unweighted_without_replacement_exclusion(
        alt_ids, excluded_alt_ids, sample_size
    ):
        """Sample alternatives without replacement for each observation.

        Parameters
        ----------
        alt_ids : 1D ndarray[int64]
            All alternative IDs.
        excluded_alt_ids : 1D ndarray[int64]
            Per-observation alt ID to exclude.  Use -1 for no exclusion.
        sample_size : int
            Number of alternatives to sample per observation.

        Returns
        -------
        1D ndarray[int64]
            Flat array of shape ``(n_obs * sample_size,)`` with sampled
            alternative IDs.
        """
        n_obs = excluded_alt_ids.shape[0]
        n_alts = alt_ids.shape[0]
        out = np.empty(n_obs * sample_size, dtype=np.int64)
        candidate_idx = np.empty(n_alts, dtype=np.int64)

        for i in range(n_obs):
            excluded = excluded_alt_ids[i]
            m = 0
            for j in range(n_alts):
                if excluded < 0 or alt_ids[j] != excluded:
                    candidate_idx[m] = j
                    m += 1

            for k in range(sample_size):
                r = np.random.randint(k, m)
                tmp = candidate_idx[k]
                candidate_idx[k] = candidate_idx[r]
                candidate_idx[r] = tmp
                out[i * sample_size + k] = alt_ids[candidate_idx[k]]

        return out

    @njit(cache=True)
    def _sample_weighted_without_replacement_1d_exclusion(
        alt_ids, alt_weights, alt_log_weights, excluded_alt_ids, sample_size
    ):
        """Sample alternatives without replacement using shared 1D weights.

        Weighted sampling is implemented with the Gumbel-top-k trick.
        Excludes a per-observation alternative (e.g., the chosen one).

        Parameters
        ----------
        alt_ids : 1D ndarray[int64]
            All alternative IDs.
        alt_weights : 1D ndarray[float64]
            Weight for each alternative.
        alt_log_weights : 1D ndarray[float64]
            Pre-computed ``log(max(weight, 1e-300))`` for each alternative.
        excluded_alt_ids : 1D ndarray[int64]
            Per-observation alt ID to exclude.  Use -1 for no exclusion.
        sample_size : int
            Number of alternatives to sample per observation.

        Returns
        -------
        1D ndarray[int64]
            Flat array of shape ``(n_obs * sample_size,)`` with sampled
            alternative IDs.
        """
        n_obs = excluded_alt_ids.shape[0]
        n_alts = alt_ids.shape[0]
        out = np.empty(n_obs * sample_size, dtype=np.int64)

        top_ids = np.empty(sample_size, dtype=np.int64)
        top_scores = np.empty(sample_size, dtype=np.float64)

        for i in range(n_obs):
            excluded = excluded_alt_ids[i]
            chosen_count = 0
            min_pos = 0
            min_score = 0.0

            for j in range(n_alts):
                if excluded >= 0 and alt_ids[j] == excluded:
                    continue

                w = alt_weights[j]
                if w <= 0.0:
                    continue

                u = np.random.random()
                if u < 1e-12:
                    u = 1e-12
                g = -np.log(-np.log(u))
                score = alt_log_weights[j] + g

                if chosen_count < sample_size:
                    top_ids[chosen_count] = alt_ids[j]
                    top_scores[chosen_count] = score
                    if chosen_count == 0 or score < min_score:
                        min_score = score
                        min_pos = chosen_count
                    chosen_count += 1
                elif score > min_score:
                    top_ids[min_pos] = alt_ids[j]
                    top_scores[min_pos] = score
                    min_pos = 0
                    min_score = top_scores[0]
                    for t in range(1, sample_size):
                        if top_scores[t] < min_score:
                            min_score = top_scores[t]
                            min_pos = t

            if chosen_count < sample_size:
                chosen_count = 0
                for j in range(n_alts):
                    if excluded < 0 or alt_ids[j] != excluded:
                        score = np.random.random()
                        if chosen_count < sample_size:
                            top_ids[chosen_count] = alt_ids[j]
                            top_scores[chosen_count] = score
                            if chosen_count == 0 or score < min_score:
                                min_score = score
                                min_pos = chosen_count
                            chosen_count += 1
                        elif score > min_score:
                            top_ids[min_pos] = alt_ids[j]
                            top_scores[min_pos] = score
                            min_pos = 0
                            min_score = top_scores[0]
                            for t in range(1, sample_size):
                                if top_scores[t] < min_score:
                                    min_score = top_scores[t]
                                    min_pos = t

            start = i * sample_size
            for k in range(sample_size):
                out[start + k] = top_ids[k]

        return out

else:

    def _sample_unweighted_without_replacement_exclusion(
        alt_ids, excluded_alt_ids, sample_size
    ):
        """Pure-Python fallback for unweighted sampling with exclusion."""
        n_obs = excluded_alt_ids.shape[0]
        out = np.empty(n_obs * sample_size, dtype=np.int64)
        rng = np.random.default_rng()

        for i in range(n_obs):
            excluded = excluded_alt_ids[i]
            if excluded >= 0:
                available = alt_ids[alt_ids != excluded]
            else:
                available = alt_ids.copy()
            sampled = rng.choice(available, size=sample_size, replace=False)
            out[i * sample_size : (i + 1) * sample_size] = sampled

        return out

    def _sample_weighted_without_replacement_1d_exclusion(
        alt_ids, alt_weights, alt_log_weights, excluded_alt_ids, sample_size
    ):
        """Pure-Python fallback for weighted sampling with exclusion (Gumbel-top-k)."""
        n_obs = excluded_alt_ids.shape[0]
        out = np.empty(n_obs * sample_size, dtype=np.int64)
        rng = np.random.default_rng()

        for i in range(n_obs):
            excluded = excluded_alt_ids[i]
            mask = np.ones(len(alt_ids), dtype=bool)
            if excluded >= 0:
                mask[alt_ids == excluded] = False
            mask[alt_weights <= 0] = False

            available_alts = alt_ids[mask]
            available_log_w = alt_log_weights[mask]

            if len(available_alts) <= sample_size:
                # Not enough alternatives — take all available
                chosen = available_alts
            else:
                # Gumbel-top-k trick
                gumbel = -np.log(-np.log(rng.random(len(available_alts))))
                scores = available_log_w + gumbel
                top_indices = np.argsort(scores)[-sample_size:]
                chosen = available_alts[top_indices]

            out[i * sample_size : i * sample_size + len(chosen)] = chosen
            # Pad if fewer alternatives than sample_size
            if len(chosen) < sample_size:
                out[i * sample_size + len(chosen) : (i + 1) * sample_size] = chosen[-1]

        return out


# ------------------------------------------------------------------
# Index-based sampling kernels (no exclusion)
# ------------------------------------------------------------------

if HAS_NUMBA:

    @njit(cache=True)
    def _sample_unweighted_without_replacement(
        n_choosers: int,
        n_alts: int,
        n_samples: int,
        seed: int,
    ) -> np.ndarray:
        """Sample alternatives without replacement (unweighted, index-based).

        Parameters
        ----------
        n_choosers : int
            Number of choosers.
        n_alts : int
            Number of alternatives per chooser.
        n_samples : int
            Number of alternatives to sample per chooser.
        seed : int
            Random seed.

        Returns
        -------
        np.ndarray
            Array of shape (n_choosers, n_samples) with sampled indices.
        """
        np.random.seed(seed)
        result = np.empty((n_choosers, n_samples), dtype=np.int64)

        for i in range(n_choosers):
            # Fisher-Yates partial shuffle
            pool = np.arange(n_alts)
            for j in range(n_samples):
                k = np.random.randint(j, n_alts)
                pool[j], pool[k] = pool[k], pool[j]
            result[i, :] = pool[:n_samples]

        return result

    @njit(cache=True)
    def _sample_weighted_without_replacement_1d(
        weights: np.ndarray,
        n_samples: int,
        seed: int,
    ) -> np.ndarray:
        """Sample alternatives without replacement with weights (1D, index-based).

        Uses the algorithm from Efraimidis and Spirakis (2006) for
        weighted sampling without replacement.

        Parameters
        ----------
        weights : np.ndarray
            Weight for each alternative (1D array).
        n_samples : int
            Number of alternatives to sample.
        seed : int
            Random seed.

        Returns
        -------
        np.ndarray
            Array of sampled indices (1D, length n_samples).
        """
        np.random.seed(seed)
        n = len(weights)
        result = np.empty(n_samples, dtype=np.int64)

        # Efraimidis-Spirakis: assign key = r^(1/w) to each item
        # then take the n_samples items with the largest keys
        keys = np.empty(n, dtype=np.float64)
        for i in range(n):
            keys[i] = np.random.random() ** (1.0 / weights[i])

        # Partial sort to find top n_samples
        indices = np.argsort(keys)[::-1][:n_samples]
        result[:] = indices

        return result

else:
    # Pure Python fallbacks for index-based kernels

    def _sample_unweighted_without_replacement(
        n_choosers: int,
        n_alts: int,
        n_samples: int,
        seed: int,
    ) -> np.ndarray:
        """Pure-Python fallback for unweighted sampling without replacement."""
        rng = np.random.default_rng(seed)
        result = np.empty((n_choosers, n_samples), dtype=np.int64)

        for i in range(n_choosers):
            pool = np.arange(n_alts)
            chosen = rng.choice(pool, size=n_samples, replace=False)
            result[i, :] = chosen

        return result

    def _sample_weighted_without_replacement_1d(
        weights: np.ndarray,
        n_samples: int,
        seed: int,
    ) -> np.ndarray:
        """Pure-Python fallback for weighted sampling without replacement."""
        rng = np.random.default_rng(seed)
        n = len(weights)

        # Efraimidis-Spirakis
        keys = rng.random(n) ** (1.0 / weights)
        indices = np.argsort(keys)[::-1][:n_samples]

        return indices


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def sample_alternatives(
    n_choosers: int,
    n_alts: int,
    n_samples: int,
    weights: np.ndarray | None = None,
    seed: int | None = None,
    replace: bool = False,
) -> np.ndarray:
    """Sample alternatives for choice set construction.

    Parameters
    ----------
    n_choosers : int
        Number of choosers.
    n_alts : int
        Number of alternatives per chooser.
    n_samples : int
        Number of alternatives to sample per chooser.
    weights : np.ndarray, optional
        Weights for each alternative. Shape (n_choosers, n_alts) for
        chooser-specific weights, or (n_alts,) for uniform weights.
    seed : int, optional
        Random seed for reproducibility.
    replace : bool
        Whether to sample with replacement. Default False.

    Returns
    -------
    np.ndarray
        Array of shape (n_choosers, n_samples) with sampled indices.
    """
    if seed is None:
        seed = np.random.randint(0, 2**31)

    if replace:
        rng = np.random.default_rng(seed)
        if weights is not None:
            # Weighted with replacement
            if weights.ndim == 1:
                probs = weights / weights.sum()
                return np.stack(
                    [
                        rng.choice(n_alts, size=n_samples, replace=True, p=probs)
                        for _ in range(n_choosers)
                    ]
                )
            else:
                result = np.empty((n_choosers, n_samples), dtype=np.int64)
                for i in range(n_choosers):
                    probs = weights[i] / weights[i].sum()
                    result[i] = rng.choice(n_alts, size=n_samples, replace=True, p=probs)
                return result
        else:
            return rng.integers(0, n_alts, size=(n_choosers, n_samples))

    else:
        if weights is not None:
            # Weighted without replacement
            rng = np.random.default_rng(seed)
            if weights.ndim == 1:
                return np.stack(
                    [
                        _sample_weighted_without_replacement_1d(weights, n_samples, seed=seed + i)
                        for i in range(n_choosers)
                    ]
                )
            else:
                result = np.empty((n_choosers, n_samples), dtype=np.int64)
                for i in range(n_choosers):
                    result[i] = _sample_weighted_without_replacement_1d(
                        weights[i], n_samples, seed=seed + i
                    )
                return result
        else:
            return _sample_unweighted_without_replacement(n_choosers, n_alts, n_samples, seed=seed)