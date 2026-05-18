"""NumPy MNL probability, log-likelihood, and gradient kernels.

This module provides the canonical NumPy implementation of the
Multinomial Logit (MNL) probability kernel.  It is the single
source of truth for the MNL computation pipeline:

1.  Compute systematic utility: ``V = X @ beta``
2.  Add sampling correction: ``V += log(inclusion_prob)`` if provided
3.  Mask unavailable alternatives: ``V[~available] = NEG_INF``
4.  Compute log-probabilities via numerically stable logsumexp
5.  Compute probabilities: ``P = exp(log_probs)``

Both the estimation objective (in
:class:`~locpick.models.mnl.MultinomialLogit`) and the prediction
methods (in :class:`~locpick.results.fit_result.FitResult`) should
delegate to these functions rather than re-implementing the pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp as scipy_logsumexp

from locpick._kernels.constants import NEG_INF

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Type for arrays that may be ``None`` (e.g. availability, weights).
OptionalArray = np.ndarray | None


# ---------------------------------------------------------------------------
# MNL probability kernel
# ---------------------------------------------------------------------------


def mnl_log_probs_numpy(
    utilities: np.ndarray,
    available: np.ndarray,
    inclusion_probs: OptionalArray = None,
) -> np.ndarray:
    """Compute MNL log-probabilities from systematic utilities.

    This is the canonical MNL probability kernel.  It applies sampling
    correction and availability masking, then computes log-probabilities
    via numerically stable logsumexp.

    Parameters
    ----------
    utilities : np.ndarray, shape (n_obs, n_alts)
        Systematic utilities ``V = X @ beta``, already reshaped.
    available : np.ndarray, shape (n_obs, n_alts)
        Binary availability mask (1 = available, 0 = unavailable).
        Use ``np.ones((n_obs, n_alts))`` if all alternatives are available.
    inclusion_probs : np.ndarray or None, shape (n_obs, n_alts)
        Inclusion probabilities for sampling correction.
        If provided, ``log(inclusion_probs)`` is added to utilities before
        the softmax.

    Returns
    -------
    np.ndarray, shape (n_obs, n_alts)
        Log-probabilities.  Unavailable alternatives receive ``NEG_INF``.
    """
    # Step 2: sampling correction
    V = utilities  # No copy needed: subsequent ops create new arrays
    if inclusion_probs is not None:
        V = V + np.log(np.asarray(inclusion_probs, dtype=np.float64))

    # Step 3: mask unavailable alternatives
    V = np.where(available > 0, V, NEG_INF)

    # Step 4: log-probabilities via stable logsumexp
    log_sum_exp = scipy_logsumexp(V, axis=1)
    log_probs = V - log_sum_exp[:, None]

    return log_probs


def mnl_probs_numpy(
    utilities: np.ndarray,
    available: np.ndarray,
    inclusion_probs: OptionalArray = None,
) -> np.ndarray:
    """Compute MNL choice probabilities from systematic utilities.

    Convenience wrapper around :func:`mnl_log_probs_numpy` that
    returns probabilities instead of log-probabilities.

    Parameters
    ----------
    utilities : np.ndarray, shape (n_obs, n_alts)
        Systematic utilities.
    available : np.ndarray, shape (n_obs, n_alts)
        Binary availability mask.
    inclusion_probs : np.ndarray or None, shape (n_obs, n_alts)
        Inclusion probabilities for sampling correction.

    Returns
    -------
    np.ndarray, shape (n_obs, n_alts)
        Choice probabilities.  Unavailable alternatives receive zero
        probability.
    """
    log_probs = mnl_log_probs_numpy(utilities, available, inclusion_probs)
    probs = np.exp(log_probs)

    # Ensure unavailable alternatives have exactly zero probability
    probs = probs * (available > 0)

    return probs


# ---------------------------------------------------------------------------
# MNL log-likelihood kernel
# ---------------------------------------------------------------------------


def mnl_log_likelihood_numpy(
    beta: np.ndarray,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    available: np.ndarray,
    n_obs: int,
    n_alts: int,
    weights: OptionalArray = None,
    inclusion_probs: OptionalArray = None,
    design_matrix_sparse=None,
) -> float:
    """Compute the MNL log-likelihood.

    Parameters
    ----------
    beta : np.ndarray, shape (n_params,)
        Parameter vector.
    design_matrix : np.ndarray, shape (n_obs * n_alts, n_params)
        Design matrix (flattened across observations and alternatives).
    chosen : np.ndarray, shape (n_obs, n_alts)
        Binary chosen-alternative indicators.
    available : np.ndarray, shape (n_obs, n_alts)
        Binary availability mask.
    n_obs : int
        Number of observations.
    n_alts : int
        Number of alternatives per observation.
    weights : np.ndarray or None, shape (n_obs,)
        Observation weights.
    inclusion_probs : np.ndarray or None, shape (n_obs, n_alts)
        Inclusion probabilities for sampling correction.
    design_matrix_sparse : scipy.sparse.spmatrix or None
        Sparse design matrix. If provided, used instead of dense
        ``design_matrix`` for the matrix-vector product.

    Returns
    -------
    float
        Weighted log-likelihood.
    """
    # Step 1: systematic utility
    if design_matrix_sparse is not None:
        utilities = design_matrix_sparse.dot(beta).reshape(n_obs, n_alts)
    else:
        utilities = (design_matrix @ beta).reshape(n_obs, n_alts)

    # Steps 2–4: log-probabilities
    log_probs = mnl_log_probs_numpy(utilities, available, inclusion_probs)

    # Step 5: chosen log-probabilities
    chosen_log_probs = (log_probs * chosen).sum(axis=1)

    if weights is not None:
        chosen_log_probs = chosen_log_probs * weights

    return float(np.sum(chosen_log_probs))


# ---------------------------------------------------------------------------
# MNL gradient kernel
# ---------------------------------------------------------------------------


def mnl_gradient_numpy(
    beta: np.ndarray,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    available: np.ndarray,
    n_obs: int,
    n_alts: int,
    weights: OptionalArray = None,
    inclusion_probs: OptionalArray = None,
    design_matrix_sparse=None,
) -> np.ndarray:
    """Compute the MNL gradient.

    Parameters
    ----------
    beta : np.ndarray, shape (n_params,)
        Parameter vector.
    design_matrix : np.ndarray, shape (n_obs * n_alts, n_params)
        Design matrix.
    chosen : np.ndarray, shape (n_obs, n_alts)
        Binary chosen-alternative indicators.
    available : np.ndarray, shape (n_obs, n_alts)
        Binary availability mask.
    n_obs : int
        Number of observations.
    n_alts : int
        Number of alternatives per observation.
    weights : np.ndarray or None, shape (n_obs,)
        Observation weights.
    inclusion_probs : np.ndarray or None, shape (n_obs, n_alts)
        Inclusion probabilities.
    design_matrix_sparse : scipy.sparse.spmatrix or None
        Sparse design matrix. If provided, used instead of dense
        ``design_matrix`` for the matrix-vector product.

    Returns
    -------
    np.ndarray, shape (n_params,)
        Gradient vector.
    """
    # Step 1: systematic utility
    if design_matrix_sparse is not None:
        utilities = design_matrix_sparse.dot(beta).reshape(n_obs, n_alts)
    else:
        utilities = (design_matrix @ beta).reshape(n_obs, n_alts)

    # Steps 2–4: probabilities
    probs = mnl_probs_numpy(utilities, available, inclusion_probs)

    # Step 5: residual = chosen - probs, masked by availability
    residual = chosen - probs
    residual = residual * available  # zero out unavailable

    if weights is not None:
        residual = residual * weights.reshape(n_obs, 1)

    if design_matrix_sparse is not None:
        grad = design_matrix_sparse.T.dot(residual.ravel())
    else:
        grad = design_matrix.T @ residual.ravel()
    return grad


# ---------------------------------------------------------------------------
# MNL observation-level scores kernel
# ---------------------------------------------------------------------------


def mnl_observation_scores_numpy(
    beta: np.ndarray,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    available: np.ndarray,
    n_obs: int,
    n_alts: int,
    weights: OptionalArray = None,
    inclusion_probs: OptionalArray = None,
) -> np.ndarray:
    """Compute observation-level score (gradient) vectors for MNL.

    Parameters
    ----------
    beta : np.ndarray, shape (n_params,)
        Parameter vector.
    design_matrix : np.ndarray, shape (n_obs * n_alts, n_params)
        Design matrix.
    chosen : np.ndarray, shape (n_obs, n_alts)
        Binary chosen-alternative indicators.
    available : np.ndarray, shape (n_obs, n_alts)
        Binary availability mask.
    n_obs : int
        Number of observations.
    n_alts : int
        Number of alternatives per observation.
    weights : np.ndarray or None, shape (n_obs,)
        Observation weights.
    inclusion_probs : np.ndarray or None, shape (n_obs, n_alts)
        Inclusion probabilities.

    Returns
    -------
    np.ndarray, shape (n_obs, n_params)
        Score vector for each observation.
    """
    # Step 1: systematic utility
    utilities = (design_matrix @ beta).reshape(n_obs, n_alts)

    # Steps 2–4: probabilities
    probs = mnl_probs_numpy(utilities, available, inclusion_probs)

    # Residual = chosen - probs, masked by availability
    residual = chosen - probs
    residual = residual * available

    if weights is not None:
        residual = residual * weights.reshape(n_obs, 1)

    # Reshape design matrix to (n_obs, n_alts, n_params)
    dm_3d = design_matrix.reshape(n_obs, n_alts, -1)

    # scores[i, k] = sum_j residual[i, j] * dm_3d[i, j, k]
    scores = np.einsum("ij,ijk->ik", residual, dm_3d)

    return scores
