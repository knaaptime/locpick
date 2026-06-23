"""Nested logit model for location choice estimation.

This module provides the ``NestedLogit`` class and supporting functions for
estimating nested logit models. The nested logit generalises the MNL by
grouping alternatives into nests, allowing correlated error terms within
each nest.

Mathematical formulation
------------------------
For alternative :math:`j` in nest :math:`m`, the choice probability is:

.. math::

    P_{ij} = \\frac{\\exp(V_{ij}/\\lambda_m)
             \\left(\\sum_{k \\in C_m} \\exp(V_{ik}/\\lambda_m)\\right)^{\\lambda_m - 1}}
            {\\sum_{m'} \\left(\\sum_{k \\in C_{m'}} \\exp(V_{ik}/\\lambda_{m'})\\right)^{\\lambda_{m'}}}

where :math:`\\lambda_m \\in (0, 1]` is the dissimilarity (nest) parameter
for nest :math:`m`. When :math:`\\lambda_m = 1` for all nests, the model
reduces to MNL.

Nest parameters are estimated via a logistic transform to enforce the
constraint :math:`\\lambda_m \\in (0, 1]`:

.. math::

    \\lambda_m = \\frac{1}{1 + \\exp(-\\alpha_m)}

where :math:`\\alpha_m` is the unconstrained parameter estimated by the
optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Nest specification
# ---------------------------------------------------------------------------


@dataclass
class NestSpec:
    """Specification of a nest in a nested logit model.

    Parameters
    ----------
    name : str
        Human-readable name for the nest (e.g., ``"transit"``, ``"auto"``).
    alt_ids : list of int
        Alternative IDs belonging to this nest.
    alpha : float or None
        Initial value for the unconstrained nest parameter
        (logit of the dissimilarity parameter). None means use 0.0
        (which corresponds to :math:`\\lambda = 0.5`).
    """

    name: str
    alt_ids: list[int]
    alpha: Optional[float] = None


@dataclass
class NestingTree:
    """Hierarchical nesting structure for nested logit models.

    A two-level nesting tree where each alternative belongs to exactly
    one nest. Alternatives not assigned to any nest are placed in a
    "root" nest with :math:`\\lambda = 1` (i.e., MNL behavior).

    Parameters
    ----------
    nests : list of NestSpec
        The nests in the tree. Each nest has a name, a list of
        alternative IDs, and an optional initial value for the
        unconstrained nest parameter.
    """

    nests: list[NestSpec]

    def __post_init__(self) -> None:
        """Validate the nesting tree."""
        if not self.nests:
            raise ValueError("NestingTree must have at least one nest.")

        # Check for duplicate alternative IDs across nests
        all_alt_ids = []
        for nest in self.nests:
            all_alt_ids.extend(nest.alt_ids)
        if len(all_alt_ids) != len(set(all_alt_ids)):
            raise ValueError(
                "Each alternative can belong to at most one nest. "
                "Found duplicate alternative IDs across nests."
            )

    @property
    def n_nests(self) -> int:
        """Number of nests."""
        return len(self.nests)

    @property
    def nest_names(self) -> list[str]:
        """Names of the nests."""
        return [n.name for n in self.nests]

    @property
    def all_alt_ids(self) -> list[int]:
        """All alternative IDs assigned to nests."""
        ids = []
        for nest in self.nests:
            ids.extend(nest.alt_ids)
        return ids

    def build_nest_matrix(self, alt_ids: list[int]) -> np.ndarray:
        """Build the nest membership matrix.

        Parameters
        ----------
        alt_ids : list of int
            The full list of alternative IDs in the choice set,
            in the order they appear in the design matrix.

        Returns
        -------
        np.ndarray
            Matrix of shape ``(n_rows, n_nests)`` where ``n_rows`` is
            the total number of (obs, alt) rows. Each column corresponds
            to a nest, with 1.0 if the alternative belongs to that nest
            and 0.0 otherwise. Alternatives not in any nest get a row
            of zeros (they are in the implicit root nest).
        """
        n_alts = len(alt_ids)
        n_nests = self.n_nests

        # Build per-alternative nest membership
        alt_to_nest = np.zeros((n_alts, n_nests), dtype=np.float64)
        for j, nest in enumerate(self.nests):
            for alt_id in nest.alt_ids:
                if alt_id in alt_ids:
                    idx = alt_ids.index(alt_id)
                    alt_to_nest[idx, j] = 1.0

        return alt_to_nest

    def initial_alphas(self) -> np.ndarray:
        """Return initial values for the unconstrained nest parameters.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_nests,)`` with initial alpha values.
        """
        return np.array([nest.alpha if nest.alpha is not None else 0.0 for nest in self.nests])


# ---------------------------------------------------------------------------
# Parameter transformation
# ---------------------------------------------------------------------------


def naturalize_nest_params(alpha: np.ndarray) -> np.ndarray:
    """Transform unconstrained nest parameters to natural (0, 1] scale.

    Parameters
    ----------
    alpha : np.ndarray
        Unconstrained nest parameters (logit scale).

    Returns
    -------
    np.ndarray
        Natural nest parameters :math:`\\lambda \\in (0, 1]`.
        Computed as :math:`\\lambda = 1 / (1 + \\exp(-\\alpha))`.
    """
    return 1.0 / (1.0 + np.exp(-alpha))


def constrain_nest_params(alpha: np.ndarray) -> np.ndarray:
    """Apply logistic transform to enforce :math:`\\lambda \\in (0, 1]`.

    Alias for :func:`naturalize_nest_params`.
    """
    return naturalize_nest_params(alpha)


# ---------------------------------------------------------------------------
# Nested logit probability kernel (NumPy)
# ---------------------------------------------------------------------------


def _nested_logit_probs_numpy(
    beta: np.ndarray,
    alpha: np.ndarray,
    design_matrix: np.ndarray,
    nest_matrix: np.ndarray,
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute nested logit probabilities (NumPy backend).

    Parameters
    ----------
    beta : np.ndarray, shape (k,)
        Utility coefficients.
    alpha : np.ndarray, shape (n_nests,)
        Unconstrained nest parameters (logit scale).
    design_matrix : np.ndarray, shape (n_obs * n_alts, k)
        Design matrix.
    nest_matrix : np.ndarray, shape (n_alts, n_nests)
        Alternative-to-nest membership matrix.
    n_obs : int
        Number of observations.
    n_alts : int
        Number of alternatives per observation.
    available : np.ndarray or None, shape (n_obs, n_alts)
        Availability matrix. 1.0 if available, 0.0 if not.
    inclusion_probs : np.ndarray or None, shape (n_obs, n_alts)
        Sampling rates for correction.

    Returns
    -------
    np.ndarray, shape (n_obs, n_alts)
        Choice probabilities for each (obs, alt) pair.
    """
    from scipy.special import logsumexp

    # Natural nest parameters: lambda_m in (0, 1]
    lambdas = naturalize_nest_params(alpha)  # (n_nests,)

    # Step 1: systematic utility
    utilities = (design_matrix @ beta).reshape(n_obs, n_alts)

    # Step 2: sampling correction
    if inclusion_probs is not None:
        sr = np.asarray(inclusion_probs, dtype=np.float64).reshape(n_obs, n_alts)
        utilities = utilities + np.log(np.maximum(sr, 1e-30))

    # Step 3: availability masking
    if available is not None:
        avail = np.asarray(available, dtype=np.float64).reshape(n_obs, n_alts)
    else:
        avail = np.ones((n_obs, n_alts), dtype=np.float64)

    from .._kernels.constants import NEG_INF

    utilities = np.where(avail > 0, utilities, NEG_INF)

    # Step 4: scaled utilities V_ij / lambda_m
    in_nest = nest_matrix.sum(axis=1) > 0  # (n_alts,) bool
    long_lambda = np.ones(n_alts, dtype=np.float64)
    for m in range(len(lambdas)):
        long_lambda[nest_matrix[:, m] > 0] = lambdas[m]
    long_lambda = long_lambda.reshape(1, n_alts)
    scaled_utilities = utilities / long_lambda  # (n_obs, n_alts)

    # Step 5: within-nest logsumexp for each nest
    n_nests = len(lambdas)
    nest_logsumexp = np.zeros((n_obs, n_nests), dtype=np.float64)
    for m in range(n_nests):
        nest_mask = nest_matrix[:, m] > 0  # (n_alts,)
        if not nest_mask.any():
            continue
        nest_utils = scaled_utilities.copy()
        nest_utils[:, ~nest_mask] = NEG_INF
        nest_utils = np.where(avail > 0, nest_utils, NEG_INF)
        nest_logsumexp[:, m] = logsumexp(nest_utils, axis=1)

    # Step 6: nest-level exponents
    nest_exponent = lambdas[None, :] * nest_logsumexp  # (n_obs, n_nests)

    # Step 7: root nest inclusive value (if any root alternatives)
    root_mask = ~in_nest  # (n_alts,)
    if root_mask.any():
        root_utils = utilities.copy()
        root_utils[:, ~root_mask] = NEG_INF
        root_utils = np.where(avail > 0, root_utils, NEG_INF)
        root_iv = logsumexp(root_utils, axis=1)  # (n_obs,)
        root_exponent = root_iv  # lambda=1, so exponent = 1 * IV
        all_exponents = np.column_stack([nest_exponent, root_exponent[:, None]])
    else:
        all_exponents = nest_exponent

    log_denom = logsumexp(all_exponents, axis=1)  # (n_obs,)

    # Step 8: unconditional probabilities (vectorised)
    # P(j) = exp(V_ij / lambda_m + (lambda_m - 1) * IV_m - log_denom)
    # For root nest: lambda=1, IV=V, so P(j) = exp(V_ij - log_denom)
    log_probs = np.full((n_obs, n_alts), NEG_INF, dtype=np.float64)

    for m in range(n_nests):
        nest_mask = nest_matrix[:, m] > 0  # (n_alts,)
        if not nest_mask.any():
            continue
        iv_m = nest_logsumexp[:, m]  # (n_obs,)
        lambda_m = lambdas[m]
        # Vectorised assignment for all alts in this nest
        log_probs[:, nest_mask] = (
            scaled_utilities[:, nest_mask] + (lambda_m - 1.0) * iv_m[:, None] - log_denom[:, None]
        )

    if root_mask.any():
        log_probs[:, root_mask] = utilities[:, root_mask] - log_denom[:, None]

    probs = np.exp(log_probs)
    probs = probs * (avail > 0)

    return probs


def _nested_logit_ll_numpy(
    beta: np.ndarray,
    alpha: np.ndarray,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    nest_matrix: np.ndarray,
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute nested logit log-likelihood (NumPy backend).

    Parameters
    ----------
    beta : np.ndarray, shape (k,)
        Utility coefficients.
    alpha : np.ndarray, shape (n_nests,)
        Unconstrained nest parameters.
    design_matrix, chosen, nest_matrix, n_obs, n_alts, available, inclusion_probs, weights
        See :func:`_nested_logit_probs_numpy`.

    Returns
    -------
    float
        Log-likelihood value.
    """
    probs = _nested_logit_probs_numpy(
        beta,
        alpha,
        design_matrix,
        nest_matrix,
        n_obs,
        n_alts,
        available=available,
        inclusion_probs=inclusion_probs,
    )

    # Chosen probabilities
    chosen_probs = (probs * chosen.reshape(n_obs, n_alts)).sum(axis=1)

    # Avoid log(0)
    chosen_probs = np.maximum(chosen_probs, 1e-30)
    log_chosen = np.log(chosen_probs)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(n_obs)
        log_chosen = log_chosen * w

    return float(log_chosen.sum())
