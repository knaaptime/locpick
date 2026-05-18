"""Spatially Correlated Logit (SCL) model for location choice estimation.

This module implements the SCL model proposed by Bhat and Guo (2004), which
captures spatial correlation between contiguous alternatives using a paired
Generalised Nested Logit (PGNL) structure.  The model is a member of the
GEV class and yields closed-form choice probabilities.

The SCL model uses a single dissimilarity parameter :math:`\\rho` that
governs the correlation between spatially adjacent alternatives.  When
:math:`\\rho = 1`, the model collapses to the Multinomial Logit (MNL).

References
----------
Bhat, C.R. and Guo, J.Y. (2004). A Mixed Spatially Correlated Logit Model:
    Formulation and Application to Residential Choice Modeling.
    *Transportation Research Part B*, 38(2), 147–168.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from locpick._compat import _JAX_AVAILABLE, _NUMBA_AVAILABLE, _NUMBA_PARALLEL
from locpick._jax.objective import Objective
from locpick._kernels.constants import NEG_INF
from locpick._sampling.correction import get_sampling_correction
from locpick._solvers import Solver, SolverResult
from locpick.data.arrays import ChoiceArrays
from locpick.models.base import (
    BaseChoiceModel,
    SpatialMixin,
    _compute_fit_statistics,
    _compute_null_ll,
)
from locpick.models.mixed import ParamDistribution, _resolve_draws
from locpick.models.nested import NestingTree, naturalize_nest_params
from locpick.results.fit_result import FitResult

# ---------------------------------------------------------------------------
# Numba imports (only if available)
# ---------------------------------------------------------------------------
if _NUMBA_AVAILABLE:
    from numba import njit, prange

# ---------------------------------------------------------------------------
# Spatial graph resolution
# ---------------------------------------------------------------------------


def _resolve_spatial_graph(
    graph: Any,
    alt_ids: Optional[list] = None,
) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    """Resolve a spatial connectivity object into SCL-ready arrays.

    Accepts a ``libpysal.graph.Graph``, a ``scipy.sparse`` array, or a
    dense NumPy array and returns the adjacency matrix, the allocation
    matrix, and the list of paired-nest edges needed by the SCL
    probability kernel.

    Parameters
    ----------
    graph : libpysal.graph.Graph, scipy.sparse array, or np.ndarray
        Spatial connectivity structure.  ``libpysal.graph.Graph`` objects
        are converted via their ``.sparse`` property.  ``scipy.sparse``
        arrays are used directly.  Dense NumPy arrays are converted to
        CSR format.  Weights are preserved (not binarised) so that
        distance- or kernel-weighted graphs can be used directly.
    alt_ids : list, optional
        Alternative IDs that index into the graph.  If *None*, the graph
        is assumed to be ordered consistently with the design matrix
        (i.e. row/column *i* of the graph corresponds to alternative *i*).

    Returns
    -------
    omega : np.ndarray, shape (n_alts, n_alts)
        Adjacency matrix with preserved weights.  ``omega[i, j] > 0`` if
        alternatives *i* and *j* are spatially connected, 0 otherwise.
        Diagonal is zero.
    allocation : np.ndarray, shape (n_alts, n_alts)
        Row-standardised allocation parameters.  ``allocation[i, j] =
        omega[i, j] / sum_k omega[i, k]``.  This is :math:`\\alpha_{i,j}`
        in Bhat & Guo (2004, Eq. 2).
    edge_list : list of (int, int)
        List of paired-nest edges ``(i, j)`` with ``i < j`` and
        ``omega[i, j] > 0``.  Each edge defines one paired nest in the
        PGNL structure.
    n_alts : int
        Number of alternatives (dimension of the graph).
    """
    import scipy.sparse as sp

    # --- Convert input to scipy sparse ---
    if hasattr(graph, "sparse"):
        # libpysal.graph.Graph — .sparse returns a csr_array
        sp_mat = sp.csr_array(graph.sparse, dtype=np.float64)
    elif sp.issparse(graph):
        sp_mat = sp.csr_array(graph, dtype=np.float64)
    else:
        sp_mat = sp.csr_array(np.asarray(graph, dtype=np.float64))

    n = sp_mat.shape[0]

    # --- Preserve weights, zero the diagonal ---
    omega_sp = sp.csr_array(sp_mat, dtype=np.float64)
    omega_sp.setdiag(0.0)
    omega_sp.eliminate_zeros()

    # Convert to dense for the SCL kernel (feasible for typical
    # residential-choice alternative sets of a few hundred zones)
    omega = omega_sp.toarray()

    # --- Allocation parameters: α_{i,j} = ω_{i,j} / Σ_k ω_{i,k} ---
    row_sums = omega.sum(axis=1, keepdims=True)
    # Avoid division by zero for isolated nodes (no neighbours)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    allocation = omega / row_sums

    # --- Edge list: paired nests (i, j) with i < j ---
    coo = sp.coo_array(omega_sp)
    edge_list = sorted((int(r), int(c)) for r, c in zip(coo.row, coo.col) if r < c)

    return omega, allocation, edge_list, n


# ---------------------------------------------------------------------------
# Parameter transformation
# ---------------------------------------------------------------------------


def naturalize_rho(alpha_rho: float) -> float:
    """Transform unconstrained parameter to natural scale :math:`\\rho \\in (0, 1]`.

    .. math::

        \\rho = \\frac{1}{1 + \\exp(-\\alpha_\\rho)}

    Parameters
    ----------
    alpha_rho : float
        Unconstrained dissimilarity parameter (logit scale).

    Returns
    -------
    float
        Natural dissimilarity parameter :math:`\\rho \\in (0, 1]`.
    """
    return 1.0 / (1.0 + np.exp(-alpha_rho))


def constrain_rho(alpha_rho: float) -> float:
    """Alias for :func:`naturalize_rho`."""
    return naturalize_rho(alpha_rho)


# ---------------------------------------------------------------------------
# Precomputed edge structure (Numba-friendly flat arrays)
# ---------------------------------------------------------------------------


class EdgeStructure:
    """Precomputed edge structure for fast SCL probability computation.

    Converts the Python-level ``edge_list`` and ``allocation`` matrix into
    flat NumPy arrays that Numba-JIT functions can consume without
    Python object overhead.

    Parameters
    ----------
    edge_list : list of (int, int)
        Paired-nest edges ``(i, j)`` with ``i < j``.
    n_alts : int
        Number of alternatives.
    allocation : np.ndarray, shape (n_alts, n_alts)
        Allocation matrix.
    """

    __slots__ = (
        "n_edges",
        "n_alts",
        "edge_i",
        "edge_j",
        "alt_edge_starts",
        "alt_edge_counts",
        "alt_edge_indices",
        "alt_edge_is_first",
        "connected",
        "isolated",
        "allocation",
    )

    def __init__(
        self,
        edge_list: list[tuple[int, int]],
        n_alts: int,
        allocation: np.ndarray,
    ):
        self.n_edges = len(edge_list)
        self.n_alts = n_alts
        self.allocation = np.ascontiguousarray(allocation, dtype=np.float64)

        # Flat edge arrays
        self.edge_i = np.zeros(self.n_edges, dtype=np.int64)
        self.edge_j = np.zeros(self.n_edges, dtype=np.int64)

        connected_set: set[int] = set()
        for idx, (i, j) in enumerate(edge_list):
            self.edge_i[idx] = i
            self.edge_j[idx] = j
            connected_set.add(i)
            connected_set.add(j)

        self.connected = np.array(sorted(connected_set), dtype=np.int64)
        self.isolated = np.array(sorted(set(range(n_alts)) - connected_set), dtype=np.int64)

        # Build alt → edge mapping as flat arrays
        # For each alt, store the indices of edges it participates in
        # and whether it is the "first" node (i) or "second" node (j).
        alt_edges: dict[int, list[tuple[int, bool]]] = {}
        for idx in range(self.n_edges):
            i = int(self.edge_i[idx])
            j = int(self.edge_j[idx])
            alt_edges.setdefault(i, []).append((idx, True))
            alt_edges.setdefault(j, []).append((idx, False))

        # Flatten into arrays: alt_edge_starts[alt] gives the start index
        # into alt_edge_indices / alt_edge_is_first; alt_edge_counts[alt]
        # gives the number of edges for that alt.
        self.alt_edge_starts = np.zeros(n_alts, dtype=np.int64)
        self.alt_edge_counts = np.zeros(n_alts, dtype=np.int64)

        total_entries = sum(len(v) for v in alt_edges.values())
        self.alt_edge_indices = np.zeros(total_entries, dtype=np.int64)
        self.alt_edge_is_first = np.zeros(total_entries, dtype=np.int64)

        pos = 0
        for alt in range(n_alts):
            edges = alt_edges.get(alt, [])
            self.alt_edge_starts[alt] = pos
            self.alt_edge_counts[alt] = len(edges)
            for edge_idx, is_first in edges:
                self.alt_edge_indices[pos] = edge_idx
                self.alt_edge_is_first[pos] = 1 if is_first else 0
                pos += 1


# ---------------------------------------------------------------------------
# Numba-JIT SCL probability kernel
# ---------------------------------------------------------------------------

if _NUMBA_AVAILABLE:

    @njit(cache=True, parallel=_NUMBA_PARALLEL)
    def _scl_log_probs_numba_core(
        V: np.ndarray,
        rho: float,
        allocation: np.ndarray,
        avail: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        alt_edge_starts: np.ndarray,
        alt_edge_counts: np.ndarray,
        alt_edge_indices: np.ndarray,
        alt_edge_is_first: np.ndarray,
        connected: np.ndarray,
        isolated: np.ndarray,
        n_edges: int,
    ) -> np.ndarray:
        """Numba-JIT SCL log-probability kernel (parallel over observations).

        All inputs are flat NumPy arrays — no Python objects.
        Uses sparse edge-based computation instead of full (n_alts, n_alts)
        allocation tensor for O(n_edges) per obs instead of O(n_alts²).
        Parallelized across observations with prange.
        """
        n_obs = V.shape[0]
        n_alts = V.shape[1]
        inv_rho = 1.0 / rho

        log_probs = np.full((n_obs, n_alts), NEG_INF)

        if n_edges == 0:
            # MNL fallback
            for n in prange(n_obs):
                max_v = NEG_INF
                for j in range(n_alts):
                    if V[n, j] > max_v:
                        max_v = V[n, j]
                log_sum = 0.0
                for j in range(n_alts):
                    log_sum += np.exp(V[n, j] - max_v)
                lse = max_v + np.log(log_sum)
                for j in range(n_alts):
                    log_probs[n, j] = V[n, j] - lse
            return log_probs

        n_isolated = isolated.shape[0]
        n_connected = connected.shape[0]

        # Main loop: parallelize over observations
        for n in prange(n_obs):
            # Pre-compute exp(V) for this observation
            exp_V_n = np.empty(n_alts)
            for j in range(n_alts):
                v = V[n, j]
                if v > 500.0:
                    exp_V_n[j] = np.exp(500.0)
                elif v < -500.0:
                    exp_V_n[j] = np.exp(-500.0)
                else:
                    exp_V_n[j] = np.exp(v)

            # Sparse edge terms for this observation
            edge_term_i_n = np.empty(n_edges)
            edge_term_j_n = np.empty(n_edges)
            nest_vals_n = np.empty(n_edges)

            for e in range(n_edges):
                i = edge_i[e]
                j = edge_j[e]
                a_ij = allocation[i, j]
                a_ji = allocation[j, i]

                val_i = a_ij * exp_V_n[i]
                if val_i < 1e-30:
                    val_i = 1e-30
                if val_i > 1e30:
                    val_i = 1e30
                edge_term_i_n[e] = np.power(val_i, inv_rho)

                val_j = a_ji * exp_V_n[j]
                if val_j < 1e-30:
                    val_j = 1e-30
                if val_j > 1e30:
                    val_j = 1e30
                edge_term_j_n[e] = np.power(val_j, inv_rho)

                s = edge_term_i_n[e] + edge_term_j_n[e]
                nest_vals_n[e] = np.power(s, rho)

            # Denominator
            denom = 0.0
            for e in range(n_edges):
                denom += nest_vals_n[e]
            for idx in range(n_isolated):
                denom += exp_V_n[isolated[idx]]
            log_denom = np.log(denom) if denom > 1e-300 else -690.0

            # Connected alternatives
            for c in range(n_connected):
                alt_i = connected[c]
                start = alt_edge_starts[alt_i]
                count = alt_edge_counts[alt_i]

                if count == 1:
                    e_idx = alt_edge_indices[start]
                    is_first = alt_edge_is_first[start]
                    if is_first:
                        my_term = edge_term_i_n[e_idx]
                        other_term = edge_term_j_n[e_idx]
                    else:
                        my_term = edge_term_j_n[e_idx]
                        other_term = edge_term_i_n[e_idx]

                    s = my_term + other_term
                    log_cond = np.log(my_term) - np.log(s) if s > 0.0 else 0.0
                    log_nest = (
                        np.log(nest_vals_n[e_idx]) - log_denom if nest_vals_n[e_idx] > 0.0 else 0.0
                    )
                    log_probs[n, alt_i] = log_cond + log_nest
                else:
                    # Multiple edges: need logsumexp
                    max_log = NEG_INF
                    contributions = np.empty(count)
                    for k in range(count):
                        e_idx = alt_edge_indices[start + k]
                        is_first = alt_edge_is_first[start + k]
                        if is_first:
                            my_term = edge_term_i_n[e_idx]
                            other_term = edge_term_j_n[e_idx]
                        else:
                            my_term = edge_term_j_n[e_idx]
                            other_term = edge_term_i_n[e_idx]

                        s = my_term + other_term
                        log_cond = np.log(my_term) - np.log(s) if s > 0.0 else 0.0
                        log_nest = (
                            np.log(nest_vals_n[e_idx]) - log_denom
                            if nest_vals_n[e_idx] > 0.0
                            else 0.0
                        )
                        contributions[k] = log_cond + log_nest
                        if contributions[k] > max_log:
                            max_log = contributions[k]

                    # logsumexp
                    lse_sum = 0.0
                    for k in range(count):
                        lse_sum += np.exp(contributions[k] - max_log)
                    log_probs[n, alt_i] = max_log + np.log(lse_sum)

            # Isolated alternatives
            for idx in range(n_isolated):
                alt_i = isolated[idx]
                log_probs[n, alt_i] = V[n, alt_i] - log_denom

        # Mask unavailable
        for n in prange(n_obs):
            for j in range(n_alts):
                if avail[n, j] <= 0.0:
                    log_probs[n, j] = NEG_INF

        return log_probs

    @njit(cache=True, parallel=_NUMBA_PARALLEL)
    def _scl_ll_numba_core(
        V: np.ndarray,
        rho: float,
        allocation: np.ndarray,
        chosen: np.ndarray,
        avail: np.ndarray,
        weights: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        alt_edge_starts: np.ndarray,
        alt_edge_counts: np.ndarray,
        alt_edge_indices: np.ndarray,
        alt_edge_is_first: np.ndarray,
        connected: np.ndarray,
        isolated: np.ndarray,
        n_edges: int,
    ) -> float:
        """Numba-JIT SCL log-likelihood kernel (parallel over observations)."""
        log_probs = _scl_log_probs_numba_core(
            V,
            rho,
            allocation,
            avail,
            edge_i,
            edge_j,
            alt_edge_starts,
            alt_edge_counts,
            alt_edge_indices,
            alt_edge_is_first,
            connected,
            isolated,
            n_edges,
        )
        n_obs = V.shape[0]
        n_alts = V.shape[1]
        ll = 0.0
        # Reduction over parallel loop
        ll_arr = np.zeros(n_obs)
        for n in prange(n_obs):
            for j in range(n_alts):
                if chosen[n, j] > 0.0:
                    ll_arr[n] += log_probs[n, j] * weights[n]
        for n in range(n_obs):
            ll += ll_arr[n]
        return ll


def _scl_log_probs_numba(
    V: np.ndarray,
    rho: float,
    edge_struct: EdgeStructure,
    avail: np.ndarray,
) -> np.ndarray:
    """Compute SCL log-probabilities using Numba-JIT backend.

    Parameters
    ----------
    V : np.ndarray, shape (n_obs, n_alts)
        Pre-masked systematic utilities (unavailable alts set to -1e30).
    rho : float
        Dissimilarity parameter.
    edge_struct : EdgeStructure
        Precomputed edge structure.
    avail : np.ndarray, shape (n_obs, n_alts)
        Availability matrix.

    Returns
    -------
    np.ndarray, shape (n_obs, n_alts)
    """
    return _scl_log_probs_numba_core(
        V,
        rho,
        edge_struct.allocation,
        avail,
        edge_struct.edge_i,
        edge_struct.edge_j,
        edge_struct.alt_edge_starts,
        edge_struct.alt_edge_counts,
        edge_struct.alt_edge_indices,
        edge_struct.alt_edge_is_first,
        edge_struct.connected,
        edge_struct.isolated,
        edge_struct.n_edges,
    )


def _scl_ll_numba(
    V: np.ndarray,
    rho: float,
    edge_struct: EdgeStructure,
    chosen: np.ndarray,
    avail: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute SCL log-likelihood using Numba-JIT backend."""
    return _scl_ll_numba_core(
        V,
        rho,
        edge_struct.allocation,
        chosen,
        avail,
        weights,
        edge_struct.edge_i,
        edge_struct.edge_j,
        edge_struct.alt_edge_starts,
        edge_struct.alt_edge_counts,
        edge_struct.alt_edge_indices,
        edge_struct.alt_edge_is_first,
        edge_struct.connected,
        edge_struct.isolated,
        edge_struct.n_edges,
    )


# ---------------------------------------------------------------------------
# SCL probability kernel (NumPy fallback)
# ---------------------------------------------------------------------------


def _scl_log_probs_numpy(
    beta: np.ndarray,
    rho: float,
    design_matrix: np.ndarray,
    allocation: np.ndarray,
    edge_list: list[tuple[int, int]],
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute SCL log-probabilities (NumPy backend).

    Implements the choice probability from Bhat & Guo (2004, Eq. 9):

    .. math::

        P_i = \\sum_{j \\neq i} P_{i|ij} \\times P_{ij}

    where the conditional probability within paired nest :math:`(i,j)` is:

    .. math::

        P_{i|ij} = \\frac{(\\alpha_{i,ij}\\, e^{V_i})^{1/\\rho}}
        {(\\alpha_{i,ij}\\, e^{V_i})^{1/\\rho} +
         (\\alpha_{j,ij}\\, e^{V_j})^{1/\\rho}}

    and the nest probability is:

    .. math::

        P_{ij} = \\frac{\\left[(\\alpha_{i,ij}\\, e^{V_i})^{1/\\rho} +
        (\\alpha_{j,ij}\\, e^{V_j})^{1/\\rho}\\right]^{\\rho}}
        {\\sum_{k<l}\\left[(\\alpha_{k,kl}\\, e^{V_k})^{1/\\rho} +
        (\\alpha_{l,kl}\\, e^{V_l})^{1/\\rho}\\right]^{\\rho}}

    Parameters
    ----------
    beta : np.ndarray, shape (k,)
        Utility coefficients.
    rho : float
        Dissimilarity parameter in (0, 1].  When ``rho = 1``, the model
        reduces to MNL.
    design_matrix : np.ndarray, shape (n_obs * n_alts, k)
        Design matrix of explanatory variables.
    allocation : np.ndarray, shape (n_alts, n_alts)
        Allocation matrix :math:`\\alpha_{i,j}`.
    edge_list : list of (int, int)
        Paired-nest edges defining the spatial correlation structure.
    n_obs : int
        Number of observations.
    n_alts : int
        Number of alternatives per observation.
    available : np.ndarray or None, shape (n_obs, n_alts)
        Binary availability matrix.
    inclusion_probs : np.ndarray or None, shape (n_obs, n_alts)
        Sampling rates for correction.

    Returns
    -------
    np.ndarray, shape (n_obs, n_alts)
        Log-probabilities for each (obs, alt) pair.
    """

    # Step 1: systematic utility
    V = (design_matrix @ beta).reshape(n_obs, n_alts)

    # Step 2: sampling correction
    if inclusion_probs is not None:
        sr = np.asarray(inclusion_probs, dtype=np.float64).reshape(n_obs, n_alts)
        V = V + np.log(np.maximum(sr, 1e-30))

    # Step 3: availability masking
    if available is not None:
        avail = np.asarray(available, dtype=np.float64).reshape(n_obs, n_alts)
    else:
        avail = np.ones((n_obs, n_alts), dtype=np.float64)

    V = np.where(avail > 0, V, NEG_INF)

    # Step 4: compute SCL probabilities
    inv_rho = 1.0 / rho
    exp_V = np.exp(np.clip(V, -500, 500))  # (n_obs, n_alts)

    # Pre-compute (α_{i,ij} * exp(V_i))^{1/ρ} for all (i, j) pairs
    # allocation: (n_alts, n_alts), exp_V: (n_obs, n_alts)
    # For each alt i: alloc_exp_V[obs, i, j] = allocation[i, j] * exp_V[obs, i]
    # Then raise to 1/ρ
    alloc_exp_V = allocation[None, :, :] * exp_V[:, :, None]  # (n_obs, n_alts, n_alts)
    alloc_exp_V = np.clip(alloc_exp_V, 1e-30, 1e30)
    alloc_exp_V_inv_rho = np.power(alloc_exp_V, inv_rho)  # (n_obs, n_alts, n_alts)

    n_edges = len(edge_list)
    if n_edges == 0:
        # No spatial correlation → MNL
        log_sum_exp_V = logsumexp(V, axis=1)
        log_probs = V - log_sum_exp_V[:, None]
        return log_probs

    # Identify connected vs isolated alternatives
    connected = set()
    for i, j in edge_list:
        connected.add(i)
        connected.add(j)
    isolated = sorted(set(range(n_alts)) - connected)

    # Compute nest values for each edge
    # nest_vals: (n_obs, n_edges)
    nest_vals = np.zeros((n_obs, n_edges), dtype=np.float64)

    for idx, (i, j) in enumerate(edge_list):
        term_i = alloc_exp_V_inv_rho[:, i, j]  # (α_{i,ij} * exp(V_i))^{1/ρ}
        term_j = alloc_exp_V_inv_rho[:, j, i]  # (α_{j,ij} * exp(V_j))^{1/ρ}
        nest_vals[:, idx] = np.power(term_i + term_j, rho)

    # Denominator: sum of nest values + exp(V_i) for isolated alternatives
    # In the SCL framework, isolated alts are "degenerate nests" with
    # a single member, so their nest value is just exp(V_i).
    if isolated:
        iso_exp_V = exp_V[:, isolated]  # (n_obs, n_isolated)
        denom_components = np.column_stack([nest_vals, iso_exp_V])
    else:
        denom_components = nest_vals

    # Use log of sum (not logsumexp) since nest_vals are already on natural scale
    log_denom = np.log(np.maximum(denom_components.sum(axis=1), 1e-300))  # (n_obs,)

    # Compute log-probabilities for each alternative
    log_probs = np.full((n_obs, n_alts), NEG_INF, dtype=np.float64)

    # Build a mapping from alt index to its edges
    alt_to_edges: dict[int, list[tuple[int, bool]]] = {}
    for idx, (i, j) in enumerate(edge_list):
        alt_to_edges.setdefault(i, []).append((idx, True))  # i is the first node
        alt_to_edges.setdefault(j, []).append((idx, False))  # j is the second node

    # Connected alternatives: P_i = Σ_{j ∈ N(i)} P_{i|ij} * P_{ij}
    for alt_i in connected:
        edge_contributions = []
        for edge_idx, is_first in alt_to_edges[alt_i]:
            i, j = edge_list[edge_idx]
            if is_first:
                # alt_i is node i in edge (i, j)
                # P_{i|ij} = (α_{i,ij} * exp(V_i))^{1/ρ} / [(α_{i,ij} * exp(V_i))^{1/ρ} + (α_{j,ij} * exp(V_j))^{1/ρ}]
                my_term = alloc_exp_V_inv_rho[:, alt_i, j]  # (α_{i,ij} * exp(V_i))^{1/ρ}
                other_term = alloc_exp_V_inv_rho[:, j, alt_i]  # (α_{j,ij} * exp(V_j))^{1/ρ}
            else:
                # alt_i is node j in edge (i, j)
                # P_{j|ij} = (α_{j,ij} * exp(V_j))^{1/ρ} / [(α_{i,ij} * exp(V_i))^{1/ρ} + (α_{j,ij} * exp(V_j))^{1/ρ}]
                my_term = alloc_exp_V_inv_rho[:, alt_i, i]  # (α_{j,ij} * exp(V_j))^{1/ρ}
                other_term = alloc_exp_V_inv_rho[:, i, alt_i]  # (α_{i,ij} * exp(V_i))^{1/ρ}

            # log P_{i|ij} = log(my_term) - log(my_term + other_term)
            log_cond = np.log(np.maximum(my_term, 1e-300)) - np.log(
                np.maximum(my_term + other_term, 1e-300)
            )
            # log P_{ij} = log(nest_val) - log_denom
            log_nest = np.log(np.maximum(nest_vals[:, edge_idx], 1e-300)) - log_denom

            edge_contributions.append(log_cond + log_nest)

        # log P_i = logsumexp over all edge contributions
        if len(edge_contributions) == 1:
            log_probs[:, alt_i] = edge_contributions[0]
        else:
            log_probs[:, alt_i] = logsumexp(np.column_stack(edge_contributions), axis=1)

    # Isolated alternatives: P_i = exp(V_i) / denom_total
    # These are "degenerate nests" with a single member
    for alt_i in isolated:
        log_probs[:, alt_i] = V[:, alt_i] - log_denom

    # Zero out unavailable alternatives
    log_probs = np.where(avail > 0, log_probs, NEG_INF)

    return log_probs


def _scl_log_probs_and_inclusive_value_numpy(
    beta: np.ndarray,
    rho: float,
    design_matrix: np.ndarray,
    allocation: np.ndarray,
    edge_list: list[tuple[int, int]],
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SCL log-probabilities and the inclusive value ln G_SCL.

    This is identical to :func:`_scl_log_probs_numpy` but also returns
    ``log_denom = ln G_SCL`` per observation, which serves as the inclusive
    value for a nested SCL upper level.

    Returns
    -------
    log_probs : np.ndarray, shape (n_obs, n_alts)
    log_inclusive_value : np.ndarray, shape (n_obs,)
        ``ln G_SCL`` for each observation.
    """
    # Step 1: systematic utility
    V = (design_matrix @ beta).reshape(n_obs, n_alts)

    # Step 2: sampling correction
    if inclusion_probs is not None:
        sr = np.asarray(inclusion_probs, dtype=np.float64).reshape(n_obs, n_alts)
        V = V + np.log(np.maximum(sr, 1e-30))

    # Step 3: availability masking
    if available is not None:
        avail = np.asarray(available, dtype=np.float64).reshape(n_obs, n_alts)
    else:
        avail = np.ones((n_obs, n_alts), dtype=np.float64)

    V = np.where(avail > 0, V, NEG_INF)

    # Step 4: compute SCL probabilities
    inv_rho = 1.0 / rho
    exp_V = np.exp(np.clip(V, -500, 500))  # (n_obs, n_alts)

    alloc_exp_V = allocation[None, :, :] * exp_V[:, :, None]  # (n_obs, n_alts, n_alts)
    alloc_exp_V = np.clip(alloc_exp_V, 1e-30, 1e30)
    alloc_exp_V_inv_rho = np.power(alloc_exp_V, inv_rho)  # (n_obs, n_alts, n_alts)

    n_edges = len(edge_list)
    if n_edges == 0:
        log_sum_exp_V = logsumexp(V, axis=1)
        log_probs = V - log_sum_exp_V[:, None]
        return log_probs, log_sum_exp_V

    # Identify connected vs isolated alternatives
    connected = set()
    for i, j in edge_list:
        connected.add(i)
        connected.add(j)
    isolated = sorted(set(range(n_alts)) - connected)

    # Compute nest values for each edge
    nest_vals = np.zeros((n_obs, n_edges), dtype=np.float64)
    for idx, (i, j) in enumerate(edge_list):
        term_i = alloc_exp_V_inv_rho[:, i, j]
        term_j = alloc_exp_V_inv_rho[:, j, i]
        nest_vals[:, idx] = np.power(term_i + term_j, rho)

    # Denominator
    if isolated:
        iso_exp_V = exp_V[:, isolated]
        denom_components = np.column_stack([nest_vals, iso_exp_V])
    else:
        denom_components = nest_vals

    log_denom = np.log(np.maximum(denom_components.sum(axis=1), 1e-300))  # (n_obs,)

    # Compute log-probabilities for each alternative
    log_probs = np.full((n_obs, n_alts), NEG_INF, dtype=np.float64)

    alt_to_edges: dict[int, list[tuple[int, bool]]] = {}
    for idx, (i, j) in enumerate(edge_list):
        alt_to_edges.setdefault(i, []).append((idx, True))
        alt_to_edges.setdefault(j, []).append((idx, False))

    for alt_i in connected:
        edge_contributions = []
        for edge_idx, is_first in alt_to_edges[alt_i]:
            i, j = edge_list[edge_idx]
            if is_first:
                my_term = alloc_exp_V_inv_rho[:, alt_i, j]
                other_term = alloc_exp_V_inv_rho[:, j, alt_i]
            else:
                my_term = alloc_exp_V_inv_rho[:, alt_i, i]
                other_term = alloc_exp_V_inv_rho[:, i, alt_i]

            log_cond = np.log(np.maximum(my_term, 1e-300)) - np.log(
                np.maximum(my_term + other_term, 1e-300)
            )
            log_nest = np.log(np.maximum(nest_vals[:, edge_idx], 1e-300)) - log_denom
            edge_contributions.append(log_cond + log_nest)

        if len(edge_contributions) == 1:
            log_probs[:, alt_i] = edge_contributions[0]
        else:
            log_probs[:, alt_i] = logsumexp(np.column_stack(edge_contributions), axis=1)

    for alt_i in isolated:
        log_probs[:, alt_i] = V[:, alt_i] - log_denom

    log_probs = np.where(avail > 0, log_probs, NEG_INF)

    return log_probs, log_denom


def _scl_ll_numpy(
    beta: np.ndarray,
    rho: float,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    allocation: np.ndarray,
    edge_list: list[tuple[int, int]],
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute SCL log-likelihood (NumPy backend).

    Parameters
    ----------
    beta : np.ndarray, shape (k,)
        Utility coefficients.
    rho : float
        Dissimilarity parameter in (0, 1].
    design_matrix, allocation, edge_list, n_obs, n_alts, available, inclusion_probs
        See :func:`_scl_log_probs_numpy`.
    chosen : np.ndarray, shape (n_obs, n_alts)
        Binary indicator matrix for chosen alternatives.
    weights : np.ndarray or None, shape (n_obs,)
        Observation-level weights.

    Returns
    -------
    float
        Log-likelihood value.
    """
    log_probs = _scl_log_probs_numpy(
        beta,
        rho,
        design_matrix,
        allocation,
        edge_list,
        n_obs,
        n_alts,
        available=available,
        inclusion_probs=inclusion_probs,
    )

    # Chosen log-probabilities
    chosen_2d = np.asarray(chosen, dtype=np.float64).reshape(n_obs, n_alts)
    chosen_log_probs = (log_probs * chosen_2d).sum(axis=1)

    # Avoid log(0)
    chosen_log_probs = np.maximum(chosen_log_probs, -1e30)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(n_obs)
        chosen_log_probs = chosen_log_probs * w

    return float(chosen_log_probs.sum())


def _scl_gradient_numpy(
    beta: np.ndarray,
    rho: float,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    allocation: np.ndarray,
    edge_list: list[tuple[int, int]],
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute SCL gradient via finite differences (NumPy backend).

    This is a fallback gradient that uses central finite differences.
    A proper analytical gradient will be implemented in a future version.

    Parameters
    ----------
    beta, rho, design_matrix, chosen, allocation, edge_list, n_obs, n_alts,
    available, inclusion_probs, weights
        See :func:`_scl_ll_numpy`.

    Returns
    -------
    np.ndarray, shape (k + 1,)
        Gradient of the log-likelihood with respect to [beta, alpha_rho].
    """
    params = np.concatenate([beta, [rho]])
    eps = 1e-5
    n_params = len(params)
    grad = np.zeros(n_params)

    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps

        ll_plus = _scl_ll_numpy(
            params_plus[: len(beta)],
            params_plus[len(beta)],
            design_matrix,
            chosen,
            allocation,
            edge_list,
            n_obs,
            n_alts,
            available=available,
            inclusion_probs=inclusion_probs,
            weights=weights,
        )
        ll_minus = _scl_ll_numpy(
            params_minus[: len(beta)],
            params_minus[len(beta)],
            design_matrix,
            chosen,
            allocation,
            edge_list,
            n_obs,
            n_alts,
            available=available,
            inclusion_probs=inclusion_probs,
            weights=weights,
        )

        grad[i] = (ll_plus - ll_minus) / (2 * eps)

    return grad


# ---------------------------------------------------------------------------
# Dispatch layer: JAX → Numba → NumPy
# ---------------------------------------------------------------------------


def _prepare_V(
    beta: np.ndarray,
    design_matrix: np.ndarray,
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute masked systematic utilities V and availability array.

    Returns
    -------
    V : np.ndarray, shape (n_obs, n_alts)
        Systematic utilities with unavailable alts set to -1e30.
    avail : np.ndarray, shape (n_obs, n_alts)
        Availability binary matrix.
    """
    V = (design_matrix @ beta).reshape(n_obs, n_alts)

    if inclusion_probs is not None:
        sr = np.asarray(inclusion_probs, dtype=np.float64).reshape(n_obs, n_alts)
        V = V + np.log(np.maximum(sr, 1e-30))

    if available is not None:
        avail = np.asarray(available, dtype=np.float64).reshape(n_obs, n_alts)
    else:
        avail = np.ones((n_obs, n_alts), dtype=np.float64)

    V = np.where(avail > 0, V, NEG_INF)
    return np.ascontiguousarray(V, dtype=np.float64), np.ascontiguousarray(avail, dtype=np.float64)


def _scl_log_probs_dispatch(
    beta: np.ndarray,
    rho: float,
    design_matrix: np.ndarray,
    allocation: np.ndarray,
    edge_list: list[tuple[int, int]],
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
    edge_struct: Optional[EdgeStructure] = None,
) -> np.ndarray:
    """Compute SCL log-probabilities, dispatching to Numba or NumPy."""
    V, avail = _prepare_V(beta, design_matrix, n_obs, n_alts, available, inclusion_probs)

    if _NUMBA_AVAILABLE and edge_struct is not None:
        return _scl_log_probs_numba(V, rho, edge_struct, avail)
    else:
        return _scl_log_probs_numpy(
            beta,
            rho,
            design_matrix,
            allocation,
            edge_list,
            n_obs,
            n_alts,
            available=available,
            inclusion_probs=inclusion_probs,
        )


def _scl_ll_dispatch(
    beta: np.ndarray,
    rho: float,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    allocation: np.ndarray,
    edge_list: list[tuple[int, int]],
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    edge_struct: Optional[EdgeStructure] = None,
) -> float:
    """Compute SCL log-likelihood, dispatching to Numba or NumPy."""
    V, avail = _prepare_V(beta, design_matrix, n_obs, n_alts, available, inclusion_probs)
    chosen_2d = np.ascontiguousarray(np.asarray(chosen, dtype=np.float64).reshape(n_obs, n_alts))
    if weights is not None:
        w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64).reshape(n_obs))
    else:
        w = np.ones(n_obs, dtype=np.float64)

    if _NUMBA_AVAILABLE and edge_struct is not None:
        return _scl_ll_numba(V, rho, edge_struct, chosen_2d, avail, w)
    else:
        return _scl_ll_numpy(
            beta,
            rho,
            design_matrix,
            chosen,
            allocation,
            edge_list,
            n_obs,
            n_alts,
            available=available,
            inclusion_probs=inclusion_probs,
            weights=weights,
        )


def _scl_gradient_dispatch(
    beta: np.ndarray,
    rho: float,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    allocation: np.ndarray,
    edge_list: list[tuple[int, int]],
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    edge_struct: Optional[EdgeStructure] = None,
) -> np.ndarray:
    """Compute SCL gradient via finite differences, using fastest backend.

    Uses Numba-JIT LL when available (each LL eval is much faster),
    otherwise falls back to NumPy LL.
    """
    params = np.concatenate([beta, [rho]])
    eps = 1e-5
    n_params = len(params)
    grad = np.zeros(n_params)

    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps

        ll_plus = _scl_ll_dispatch(
            params_plus[: len(beta)],
            params_plus[len(beta)],
            design_matrix,
            chosen,
            allocation,
            edge_list,
            n_obs,
            n_alts,
            available=available,
            inclusion_probs=inclusion_probs,
            weights=weights,
            edge_struct=edge_struct,
        )
        ll_minus = _scl_ll_dispatch(
            params_minus[: len(beta)],
            params_minus[len(beta)],
            design_matrix,
            chosen,
            allocation,
            edge_list,
            n_obs,
            n_alts,
            available=available,
            inclusion_probs=inclusion_probs,
            weights=weights,
            edge_struct=edge_struct,
        )

        grad[i] = (ll_plus - ll_minus) / (2 * eps)

    return grad


# ---------------------------------------------------------------------------
# SCL model class
# ---------------------------------------------------------------------------


class SCL(BaseChoiceModel, SpatialMixin):
    r"""Spatially Correlated Logit (SCL) model for location choice estimation.

    The SCL model captures spatial correlation between contiguous alternatives
    using a paired Generalised Nested Logit (PGNL) structure with a single
    dissimilarity parameter :math:`\\rho`.  When :math:`\\rho = 1`, the model
    reduces to the Multinomial Logit (MNL).

    The choice probability for alternative :math:`i` is:

    .. math::

        P_i = \\sum_{j \\neq i} P_{i|ij} \\times P_{ij}

    where the conditional probability within paired nest :math:`(i,j)` is:

    .. math::

        P_{i|ij} = \\frac{(\\alpha_{i,ij}\\, e^{V_i})^{1/\\rho}}
        {(\\alpha_{i,ij}\\, e^{V_i})^{1/\\rho} +
         (\\alpha_{j,ij}\\, e^{V_j})^{1/\\rho}}

    and the nest probability is:

    .. math::

        P_{ij} = \\frac{\\left[(\\alpha_{i,ij}\\, e^{V_i})^{1/\\rho} +
        (\\alpha_{j,ij}\\, e^{V_j})^{1/\\rho}\\right]^{\\rho}}
        {\\sum_{k<l}\\left[(\\alpha_{k,kl}\\, e^{V_k})^{1/\\rho} +
        (\\alpha_{l,kl}\\, e^{V_l})^{1/\\rho}\\right]^{\\rho}}

    The allocation parameters are derived from the spatial adjacency structure:

    .. math::

        \\alpha_{i,j} = \\frac{\\omega_{ij}}{\\sum_k \\omega_{ik}}

    where :math:`\\omega_{ij}` is 1 if zones :math:`i` and :math:`j` are
    spatially contiguous and 0 otherwise.

    The dissimilarity parameter is estimated via a logistic transform to
    enforce :math:`\\rho \\in (0, 1]`:

    .. math::

        \\rho = \\frac{1}{1 + \\exp(-\\alpha_\\rho)}

    Parameters
    ----------
    data : ChoiceTable
        The choice data to estimate on.
    formula : str, optional
        A formulaic formula string for the utility function.
    spec : ModelSpec, optional
        A ModelSpec object defining the utility function.
    graph : libpysal.graph.Graph, scipy.sparse array, or np.ndarray
        Spatial adjacency structure encoding contiguity between alternatives.
        Can be a ``libpysal.graph.Graph`` (recommended), a ``scipy.sparse``
        array, or a dense NumPy array.
    weights : str or array-like, optional
        Observation weights.
    availability : str or array-like, optional
        Alternative availability.
    solver : str or Solver, optional
        Solver name or instance.  Default ``"lbfgs"``.
    solver_options : dict, optional
        Additional options passed to the solver constructor.

    Examples
    --------
    >>> from locpick import ChoiceTable, SpatiallyCorrelatedLogit
    >>> from libpysal import graph
    >>> ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=10)
    >>> g = graph.Graph.build_contiguity(tracts_gdf, rook=False)
    >>> model = SpatiallyCorrelatedLogit(ct, formula="cost + time", graph=g)
    >>> result = model.fit()
    >>> print(result.summary())
    """

    def __init__(
        self,
        data,
        formula: Optional[str] = None,
        spec=None,
        graph: Any = None,
        nests: Optional[NestingTree] = None,
        random_params: Optional[dict[str, ParamDistribution]] = None,
        n_draws: Optional[int] = None,
        draw_type: Optional[str] = None,
        weights: Optional[Union[str, np.ndarray]] = None,
        availability: Optional[Union[str, np.ndarray]] = None,
        solver: Union[str, Solver] = None,
        solver_options: Optional[dict] = None,
        backend: Optional[str] = None,
    ):
        from locpick.config import config

        if graph is None:
            raise ValueError(
                "SpatiallyCorrelatedLogit requires a 'graph' argument "
                "specifying the spatial adjacency structure."
            )

        super().__init__(
            data=data,
            formula=formula,
            spec=spec,
            solver=solver,
            solver_options=solver_options,
            backend=backend,
            weights=weights,
            availability=availability,
        )
        self._graph_input = graph
        self._nests = nests
        self._random_params = random_params or {}
        self._n_draws = n_draws if n_draws is not None else config.default_n_draws_mixed
        self._draw_type = draw_type if draw_type is not None else config.default_qmc_engine
        self._allocation: Optional[np.ndarray] = None
        self._edge_list: Optional[list] = None
        self._n_alts_graph: Optional[int] = None
        self._edge_struct: Optional[EdgeStructure] = None
        self._nest_matrix: Optional[np.ndarray] = None
        self._edge_structs: Optional[list] = None
        self._edge_data_list: Optional[list] = None

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def _pre_fit(self, arrays: ChoiceArrays) -> None:
        """Resolve spatial graph and optional nest/random structures before objective construction."""
        self._resolve_spatial_graph()
        self._validate_graph_size(arrays)

        has_nests = self._nests is not None
        has_random = bool(self._random_params)

        # Build nest matrix and per-nest edge structures if nests are provided
        if has_nests:
            alt_ids = list(range(arrays.n_alts))
            self._nest_matrix = self._nests.build_nest_matrix(alt_ids)
            n_nests = self._nests.n_nests

            self._edge_structs = []
            self._edge_data_list = []

            for m in range(n_nests):
                nest_alts = np.where(self._nest_matrix[:, m] > 0)[0]
                n_nest_alts = len(nest_alts)

                if n_nest_alts == 0:
                    self._edge_structs.append(None)
                    self._edge_data_list.append(None)
                    continue

                # Extract subgraph for this nest
                nest_adj = self._omega[np.ix_(nest_alts, nest_alts)]
                _, nest_alloc, nest_edges, _ = _resolve_spatial_graph(nest_adj)

                # Build EdgeStructure for this nest
                edge_struct = EdgeStructure(nest_edges, n_nest_alts, nest_alloc)
                self._edge_structs.append(edge_struct)

                # Build EdgeDataJAX if JAX is available
                if _JAX_AVAILABLE:
                    from locpick._jax.data import EdgeDataJAX

                    edge_data = EdgeDataJAX.from_edge_structure(edge_struct)
                    self._edge_data_list.append(edge_data)
                else:
                    self._edge_data_list.append(None)

        # Set up random parameter structure if random_params are provided
        if has_random:
            param_names = list(arrays.param_names)
            random_param_names = list(self._random_params.keys())
            random_col_indices = [param_names.index(name) for name in random_param_names]
            random_distributions = [
                self._random_params[name].distribution for name in random_param_names
            ]
            k_random = len(random_col_indices)

            # Generate draws
            draws = _resolve_draws(self._draw_type, arrays.n_obs, self._n_draws, k_random, seed=42)

            # Cache random structure for objective/prediction paths
            self._draws = draws
            self._random_col_indices = random_col_indices
            self._random_distributions = random_distributions
            self._random_param_names = random_param_names

    def _get_solver_inputs(self, arrays: ChoiceArrays):
        """Get initial values, param names, bounds, and fixed mask.

        Parameter layout depends on which features are active:
        - SCL (no nests, no random): [beta, alpha_rho]
        - MSCL (no nests, random): [beta_fixed, alpha_rho, mean_*, sd_*]
        - NestedSCL (nests, no random): [beta, alpha_rho_1..M, alpha_lambda_1..M]
        - MNSCL (nests, random): [beta_fixed, alpha_rho_1..M, alpha_lambda_1..M, mean_*, sd_*]
        """
        has_nests = self._nests is not None
        has_random = bool(self._random_params)
        param_names_all = list(arrays.param_names)
        k_total = arrays.design_matrix.shape[1]

        if has_nests:
            n_nests = self._nests.n_nests
            nest_names = self._nests.nest_names
        else:
            n_nests = 0
            nest_names = []

        if has_random:
            random_param_names = list(self._random_params.keys())
            k_random = len(random_param_names)
            k_fixed = k_total - k_random
            fixed_param_names = [
                name for name in param_names_all if name not in random_param_names
            ]
        else:
            k_random = 0
            k_fixed = k_total
            fixed_param_names = list(param_names_all)
            random_param_names = []

        # Build parameter vector
        # Layout: [beta_fixed, rho_param(s), lambda_param(s)?, mean_*?, sd_*?]
        # rho_param: 1 scalar (alpha_rho) if no nests, M scalars (alpha_rho_m) if nests
        # lambda_param: M scalars (alpha_lambda_m) only if nests
        n_rho = n_nests if has_nests else 1  # 1 alpha_rho for plain SCL/MSCL
        n_lambda = n_nests if has_nests else 0  # lambda only for nested variants
        n_params = k_fixed + n_rho + n_lambda + 2 * k_random
        x0 = np.zeros(n_params)

        # Build parameter names
        display_names = list(fixed_param_names)
        if has_nests:
            display_names += [f"rho_{name}" for name in nest_names]
            display_names += [f"lambda_{name}" for name in nest_names]
        else:
            display_names += ["rho"]
        if has_random:
            display_names += [f"mean_{name}" for name in random_param_names]
            display_names += [f"sd_{name}" for name in random_param_names]

        return x0, display_names, None, None

    def _build_objective(self, arrays: ChoiceArrays) -> Objective:
        """Build optimization objective, dispatching to the appropriate JAX builder.

        Dispatches based on which features are active:
        - No nests, no random: SCL objective
        - No nests, random: MSCL objective
        - Nests, no random: Nested SCL objective
        - Nests, random: MNSCL objective
        """
        if self._allocation is None or self._edge_list is None:
            raise RuntimeError("Spatial graph must be resolved before building objective.")

        has_nests = self._nests is not None
        has_random = bool(self._random_params)

        # Dispatch to the appropriate JAX builder
        if has_nests and has_random:
            # MNSCL: Mixed Nested Spatially Correlated Logit
            return self._build_mnscl_objective(arrays)
        elif has_nests:
            # Nested SCL
            return self._build_nested_scl_objective(arrays)
        elif has_random:
            # MSCL: Mixed Spatially Correlated Logit
            return self._build_mscl_objective(arrays)
        else:
            # Plain SCL
            return self._build_scl_objective(arrays)

    def _build_scl_objective(self, arrays: ChoiceArrays) -> Objective:
        """Build SCL (plain) objective."""
        edge_struct = self._edge_struct

        backend = (self._backend or os.environ.get("CHOICEMODELS_SCL_BACKEND", "")).lower()
        if backend == "jax":
            use_jax = _JAX_AVAILABLE
        elif backend in {"numba", "numpy"}:
            use_jax = False
        else:
            # Default: Numba if available, else JAX, else NumPy
            use_jax = not _NUMBA_AVAILABLE and _JAX_AVAILABLE

        if use_jax:
            from locpick._jax.builders import build_scl_objective

            return build_scl_objective(arrays, edge_struct, self._allocation, self._edge_list)

        # NumPy/Numba fallback
        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        chosen = np.asarray(arrays.chosen, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts
        available = arrays.available
        weights = arrays.weights

        from locpick._sampling.correction import get_sampling_correction

        inclusion_probs = get_sampling_correction(arrays)
        k = dm.shape[1]

        def ll_fn(params):
            beta = params[:k]
            alpha_rho = params[k]
            rho = naturalize_rho(alpha_rho)
            return _scl_ll_dispatch(
                beta,
                rho,
                dm,
                chosen,
                self._allocation,
                self._edge_list,
                n_obs,
                n_alts,
                available=available,
                inclusion_probs=inclusion_probs,
                weights=weights,
                edge_struct=edge_struct,
            )

        def grad_fn(params):
            beta = params[:k]
            alpha_rho = params[k]
            rho = naturalize_rho(alpha_rho)
            grad = _scl_gradient_dispatch(
                beta,
                rho,
                dm,
                chosen,
                self._allocation,
                self._edge_list,
                n_obs,
                n_alts,
                available=available,
                inclusion_probs=inclusion_probs,
                weights=weights,
                edge_struct=edge_struct,
            )
            # Chain rule: dLL/d(alpha_rho) = dLL/d(rho) * d(rho)/d(alpha_rho)
            grad[-1] = grad[-1] * rho * (1.0 - rho)
            return grad

        return Objective.from_numpy(
            ll_fn=ll_fn,
            grad_fn=grad_fn,
            param_names=list(arrays.param_names) + ["rho"],
        )

    def _build_mscl_objective(self, arrays: ChoiceArrays) -> Objective:
        """Build MSCL (mixed) objective."""
        if not hasattr(self, "_random_col_indices"):
            raise RuntimeError(
                "Random parameter structure must be prepared before building objective."
            )

        backend = (self._backend or os.environ.get("CHOICEMODELS_MSCL_BACKEND", "")).lower()
        if backend == "jax":
            use_jax = _JAX_AVAILABLE
        elif backend in {"numba", "numpy"}:
            use_jax = False
        else:
            use_jax = _JAX_AVAILABLE

        if use_jax:
            from locpick._jax.builders import build_mscl_objective

            return build_mscl_objective(
                arrays,
                self._edge_struct,
                self._allocation,
                self._edge_list,
                self._random_col_indices,
                self._random_distributions,
                self._draws,
            )

        raise NotImplementedError(
            "Mixed Spatially Correlated Logit currently only supports the JAX backend. "
            f"Requested backend: {backend!r}."
        )

    def _build_nested_scl_objective(self, arrays: ChoiceArrays) -> Objective:
        """Build Nested SCL objective."""
        backend = (
            self._backend or os.environ.get("CHOICEMODELS_NESTED_SCL_BACKEND", "jax")
        ).lower()

        if backend == "jax" and _JAX_AVAILABLE:
            from locpick._jax.builders import build_nested_scl_objective

            return build_nested_scl_objective(arrays, self._nest_matrix, self._edge_data_list)

        raise NotImplementedError(
            "NestedSpatiallyCorrelatedLogit currently only supports the JAX backend. "
            f"Requested backend: {backend!r}."
        )

    def _build_mnscl_objective(self, arrays: ChoiceArrays) -> Objective:
        """Build MNSCL (mixed nested) objective."""
        backend = (self._backend or os.environ.get("CHOICEMODELS_MNSCL_BACKEND", "jax")).lower()

        if backend == "jax" and _JAX_AVAILABLE:
            from locpick._jax.builders import build_mnscl_objective

            return build_mnscl_objective(
                arrays,
                self._nest_matrix,
                self._edge_data_list,
                self._random_col_indices,
                self._random_distributions,
                self._draws,
            )

        raise NotImplementedError(
            "MixedNestedSpatiallyCorrelatedLogit currently only supports the JAX backend. "
            f"Requested backend: {backend!r}."
        )

    def _build_fit_result(
        self,
        solver_result: SolverResult,
        arrays: ChoiceArrays,
    ) -> FitResult:
        """Build a FitResult from solver output.

        Parameter layout depends on which features are active:
        - SCL (no nests, no random): [beta, alpha_rho]
        - MSCL (no nests, random): [beta_fixed, alpha_rho, mean_*, sd_*]
        - NestedSCL (nests, no random): [beta, alpha_rho_1..M, alpha_lambda_1..M]
        - MNSCL (nests, random): [beta_fixed, alpha_rho_1..M, alpha_lambda_1..M, mean_*, sd_*]
        """
        has_nests = self._nests is not None
        has_random = bool(self._random_params)
        all_params = solver_result.coefficients

        # Determine parameter layout
        param_names_all = list(arrays.param_names)
        k_total = arrays.design_matrix.shape[1]

        if has_random:
            random_param_names = list(self._random_params.keys())
            k_random = len(random_param_names)
            k_fixed = k_total - k_random
            fixed_param_names = [
                name for name in param_names_all if name not in random_param_names
            ]
        else:
            random_param_names = []
            k_random = 0
            k_fixed = k_total
            fixed_param_names = list(param_names_all)

        if has_nests:
            n_nests = self._nests.n_nests
            nest_names = self._nests.nest_names
        else:
            n_nests = 0
            nest_names = []

        # Number of rho and lambda parameters
        n_rho = n_nests if has_nests else 1  # 1 alpha_rho for plain SCL/MSCL
        n_lambda = n_nests if has_nests else 0  # lambda only for nested variants

        # Unpack parameters based on layout
        # Layout: [beta_fixed, rho_param(s), lambda_param(s)?, mean_*?, sd_*?]
        beta_fixed = all_params[:k_fixed]

        # Offset after rho and lambda parameters
        rho_offset = k_fixed
        if has_nests:
            alpha_rhos = all_params[rho_offset : rho_offset + n_nests]
            alpha_lambdas = all_params[rho_offset + n_nests : rho_offset + 2 * n_nests]
            rhos = naturalize_rho(alpha_rhos)
            lambdas = naturalize_nest_params(alpha_lambdas)
        else:
            alpha_rho = all_params[rho_offset]
            rho = naturalize_rho(alpha_rho)

        # Offset for random parameters (after rho and lambda)
        random_offset = rho_offset + n_rho + n_lambda
        if has_random:
            beta_random_means = all_params[random_offset : random_offset + k_random]
            beta_random_spreads = np.abs(
                all_params[random_offset + k_random : random_offset + 2 * k_random]
            )

        # Build display parameter names and values
        display_param_names = list(fixed_param_names)
        display_values = [beta_fixed]

        if has_nests:
            display_param_names += [f"rho_{name}" for name in nest_names]
            display_param_names += [f"lambda_{name}" for name in nest_names]
            display_values.append(rhos)
            display_values.append(lambdas)
        else:
            display_param_names += ["rho"]
            display_values.append([rho])

        if has_random:
            display_param_names += [f"mean_{name}" for name in random_param_names]
            display_param_names += [f"sd_{name}" for name in random_param_names]
            display_values.append(beta_random_means)
            display_values.append(beta_random_spreads)

        display_params = np.concatenate(display_values)

        # Standard errors with delta method for transformed parameters.
        # Prefer HVP-based Hessian (exact, via JAX autodiff) over the
        # solver's approximate inverse Hessian (e.g. L-BFGS-B hess_inv).
        # Note: _compute_hessian returns the Hessian of the log-likelihood
        # (negative definite), so we negate and invert for standard errors.
        std_errors = np.full(len(display_params), np.nan)
        try:
            hess = self._compute_hessian(all_params)
            se_all = self._compute_std_errors_from_hessian(hess)
            se_parts = [se_all[:k_fixed]]

            if has_nests:
                # Delta method for rho_m: SE(rho_m) = rho_m * (1 - rho_m) * SE(alpha_rho_m)
                se_rho = rhos * (1.0 - rhos) * se_all[rho_offset : rho_offset + n_nests]
                # Delta method for lambda_m: SE(lambda_m) = lambda_m * (1 - lambda_m) * SE(alpha_lambda_m)
                se_lambda = (
                    lambdas
                    * (1.0 - lambdas)
                    * se_all[rho_offset + n_nests : rho_offset + 2 * n_nests]
                )
                se_parts.append(se_rho)
                se_parts.append(se_lambda)
            else:
                # Delta method for rho: SE(rho) = rho * (1 - rho) * SE(alpha_rho)
                se_rho = rho * (1.0 - rho) * se_all[rho_offset]
                se_parts.append([se_rho])

            if has_random:
                se_parts.append(se_all[random_offset : random_offset + k_random])
                se_parts.append(se_all[random_offset + k_random : random_offset + 2 * k_random])

            std_errors = np.concatenate(se_parts)
        except Exception:
            # Fallback: try solver's Hessian (approximate, e.g. L-BFGS-B hess_inv)
            # Note: solver_result.hessian is the *inverse* of the negative Hessian
            # (positive definite), so sqrt(diag(hess)) gives standard errors directly.
            if solver_result.hessian is not None:
                try:
                    se_all = np.sqrt(np.abs(np.diag(solver_result.hessian)))
                    se_parts = [se_all[:k_fixed]]

                    if has_nests:
                        se_rho = rhos * (1.0 - rhos) * se_all[rho_offset : rho_offset + n_nests]
                        se_lambda = (
                            lambdas
                            * (1.0 - lambdas)
                            * se_all[rho_offset + n_nests : rho_offset + 2 * n_nests]
                        )
                        se_parts.append(se_rho)
                        se_parts.append(se_lambda)
                    else:
                        se_rho = rho * (1.0 - rho) * se_all[rho_offset]
                        se_parts.append([se_rho])

                    if has_random:
                        se_parts.append(se_all[random_offset : random_offset + k_random])
                        se_parts.append(
                            se_all[random_offset + k_random : random_offset + 2 * k_random]
                        )

                    std_errors = np.concatenate(se_parts)
                except Exception:
                    pass

        # Determine model type name
        if has_nests and has_random:
            model_type = "Mixed Nested Spatially Correlated Logit"
        elif has_nests:
            model_type = "Nested Spatially Correlated Logit"
        elif has_random:
            model_type = "Mixed Spatially Correlated Logit"
        else:
            model_type = "Spatially Correlated Logit"

        # Build result using shared helper
        coefficients = pd.Series(display_params, index=display_param_names, name="coefficient")
        std_err_series = pd.Series(std_errors, index=display_param_names, name="std_error")
        ll = solver_result.log_likelihood
        ll_null = _compute_null_ll(arrays)
        n_params = len(display_params)

        stats = _compute_fit_statistics(
            ll=ll,
            ll_null=ll_null,
            n_obs=arrays.n_obs,
            n_params=n_params,
            n_alts=arrays.n_alts,
            coefficients=coefficients,
            std_errors=std_err_series,
            model_type=model_type,
            solver_name=solver_result.solver_name,
            solver_result_raw=solver_result.raw,
        )

        return FitResult(
            spec=self._spec,
            **stats,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observation_scores(self, arrays) -> np.ndarray:
        """Compute observation-level score vectors via numerical differentiation.

        Currently only implemented for the plain SCL model (no nests, no random
        parameters).  For MSCL, NestedSCL, and MNSCL variants, this method
        raises ``NotImplementedError``.

        Parameters
        ----------
        arrays : ChoiceArrays
            The estimation data arrays.

        Returns
        -------
        np.ndarray, shape (n_obs, n_params)
            Score vector for each observation.

        Raises
        ------
        NotImplementedError
            If the model has nests or random parameters.
        """
        if self._result is None:
            raise RuntimeError("Model must be estimated first.")

        has_nests = self._nests is not None
        has_random = bool(self._random_params)
        if has_nests or has_random:
            raise NotImplementedError(
                "Observation-level scores are not yet implemented for "
                "MSCL, NestedSCL, or MNSCL variants. "
                "Use the default Hessian-based standard errors instead."
            )

        cache_key = id(arrays)
        if cache_key in self._observation_scores_cache:
            return self._observation_scores_cache[cache_key]

        eps = 1e-5
        k = arrays.design_matrix.shape[1]
        n_params = k + 1  # beta + rho
        n_obs = arrays.n_obs

        # Full parameter vector: [beta, rho]
        full_params = self._result.coefficients.values.copy()
        beta_hat = full_params[:k]
        rho_hat = full_params[k]

        # Chosen indicator
        chosen = np.asarray(arrays.chosen, dtype=np.float64).reshape(n_obs, arrays.n_alts)

        # Base probabilities and per-observation LL
        probs_base = self.probabilities(data=None, beta=beta_hat, rho=rho_hat)
        np.log(np.maximum(np.sum(probs_base * chosen, axis=1), 1e-30))

        scores = np.zeros((n_obs, n_params))

        for j in range(n_params):
            params_plus = full_params.copy()
            params_plus[j] += eps
            params_minus = full_params.copy()
            params_minus[j] -= eps

            beta_plus = params_plus[:k]
            rho_plus = params_plus[k]
            beta_minus = params_minus[:k]
            rho_minus = params_minus[k]

            probs_plus = self.probabilities(data=None, beta=beta_plus, rho=rho_plus)
            probs_minus = self.probabilities(data=None, beta=beta_minus, rho=rho_minus)

            ll_plus = np.log(np.maximum(np.sum(probs_plus * chosen, axis=1), 1e-30))
            ll_minus = np.log(np.maximum(np.sum(probs_minus * chosen, axis=1), 1e-30))

            scores[:, j] = (ll_plus - ll_minus) / (2 * eps)

        self._observation_scores_cache[cache_key] = scores
        return scores

    def utilities(self, data=None, beta=None):
        """Compute deterministic utilities.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to predict on.  If ``None``, uses estimation data.
        beta : np.ndarray or None
            Utility coefficients.  If ``None``, uses estimated values.

        Returns
        -------
        np.ndarray, shape (n_obs, n_alts)
            Deterministic utilities for each observation and alternative.
            Includes sampling correction (log of inclusion probability)
            when the data has sampling metadata.
        """
        if self._arrays is None:
            raise RuntimeError("Model must be estimated before prediction.")

        arrays = self._arrays
        if data is not None:
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )

        if beta is None:
            k = arrays.design_matrix.shape[1]
            beta = np.asarray(self._result.coefficients.values[:k], dtype=np.float64)

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        # Systematic utility
        V = (dm @ beta).reshape(n_obs, n_alts)

        # Add sampling correction if present
        from locpick._sampling.correction import apply_sampling_correction

        V = apply_sampling_correction(V, arrays)

        return V

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, data=None, n_draws: int = 1, seed: Optional[int] = None) -> pd.DataFrame:
        """Simulate choices from the estimated model.

        Draws random choices according to the model's predicted
        probabilities.  Useful for forecasting, validation, and Monte
        Carlo analysis.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to simulate on.  If ``None``, uses estimation data.
        n_draws : int, optional
            Number of simulation draws per observation.  Default 1.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Simulated choices with columns ``draw``, ``obs_id``,
            ``alt_id``, and ``probability``.
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated before simulation.")

        arrays = self._arrays
        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )
            ct = data

        rng = np.random.default_rng(seed)
        probs = self.probabilities(data=data)
        # Normalize probabilities to sum to 1 (numerical precision)
        probs = probs / probs.sum(axis=1, keepdims=True)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        # Get alternative IDs from the data
        df = ct.to_frame()
        alt_ids = df[ct.alt_id_col].values.reshape(n_obs, n_alts)
        obs_ids = df[ct.obs_id_col].values.reshape(n_obs, n_alts)[:, 0]

        results = []
        for draw in range(n_draws):
            chosen_indices = np.array([rng.choice(n_alts, p=probs[i]) for i in range(n_obs)])
            chosen_alts = alt_ids[np.arange(n_obs), chosen_indices]
            chosen_probs = probs[np.arange(n_obs), chosen_indices]

            for i in range(n_obs):
                results.append(
                    {
                        "draw": draw,
                        ct.obs_id_col: obs_ids[i],
                        ct.alt_id_col: chosen_alts[i],
                        "probability": chosen_probs[i],
                    }
                )

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Marginal Effects
    # ------------------------------------------------------------------

    def marginal_effect(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute direct marginal effects for a variable.

        For SCL, the direct marginal effect of variable *x* in alternative *i*
        is approximately :math:`(1 - P_{qi}) \\beta_x`.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute marginal effects on.  If ``None``, uses
            estimation data.
        variable : str
            Name of the variable to compute marginal effect for.

        Returns
        -------
        pd.Series
            Direct marginal effects, indexed by (obs_id, alt_id).
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated before computing marginal effects.")

        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )
            ct = data

        probs = self.probabilities(data=data if data is not None else None)
        beta = self._result.coefficients.get(variable, 0.0)
        df = ct.to_frame()

        me = (1 - probs.ravel()) * beta

        index = pd.MultiIndex.from_arrays(
            [df[ct.obs_id_col].values, df[ct.alt_id_col].values],
            names=[ct.obs_id_col, ct.alt_id_col],
        )
        return pd.Series(me, index=index, name=f"marginal_effect_{variable}")

    def cross_marginal_effect(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute cross-marginal effects for a variable.

        For SCL, the cross-marginal effect of variable *x* in alternative *j*
        with respect to a change in alternative *i* is approximately
        :math:`-P_i \\beta_x`.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute cross-marginal effects on.  If ``None``, uses
            estimation data.
        variable : str
            Name of the variable.

        Returns
        -------
        pd.Series
            Cross-marginal effects, indexed by (obs_id, alt_id).
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated before computing marginal effects.")

        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )
            ct = data

        probs = self.probabilities(data=data if data is not None else None)
        beta = self._result.coefficients.get(variable, 0.0)
        df = ct.to_frame()

        cross_me = -probs.ravel() * beta

        index = pd.MultiIndex.from_arrays(
            [df[ct.obs_id_col].values, df[ct.alt_id_col].values],
            names=[ct.obs_id_col, ct.alt_id_col],
        )
        return pd.Series(cross_me, index=index, name=f"cross_marginal_effect_{variable}")

    # ------------------------------------------------------------------
    # Elasticities
    # ------------------------------------------------------------------

    def elasticity(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute direct elasticities for a variable.

        The direct elasticity measures the percentage change in the
        probability of choosing an alternative with respect to a
        percentage change in a variable.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute elasticities on.  If ``None``, uses
            estimation data.
        variable : str
            Name of the variable to compute elasticity for.

        Returns
        -------
        pd.Series
            Direct elasticities, indexed by (obs_id, alt_id).
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated before computing elasticities.")

        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )
            ct = data

        probs = self.probabilities(data=data if data is not None else None)
        df = ct.to_frame()
        x = df[variable].values
        beta = self._result.coefficients.get(variable, 0.0)

        elasticities = (1 - probs.ravel()) * beta * x

        index = pd.MultiIndex.from_arrays(
            [df[ct.obs_id_col].values, df[ct.alt_id_col].values],
            names=[ct.obs_id_col, ct.alt_id_col],
        )
        return pd.Series(elasticities, index=index, name=f"elasticity_{variable}")

    def cross_elasticity(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute cross-elasticities for a variable.

        The cross-elasticity measures the percentage change in the
        probability of choosing one alternative with respect to a
        percentage change in a variable of another alternative.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute cross-elasticities on.  If ``None``, uses
            estimation data.
        variable : str
            Name of the variable.

        Returns
        -------
        pd.Series
            Cross-elasticities, indexed by (obs_id, alt_id).
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated before computing elasticities.")

        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )
            ct = data

        probs = self.probabilities(data=data if data is not None else None)
        beta = self._result.coefficients.get(variable, 0.0)
        df = ct.to_frame()
        x = df[variable].values

        cross_elast = -probs.ravel() * beta * x

        index = pd.MultiIndex.from_arrays(
            [df[ct.obs_id_col].values, df[ct.alt_id_col].values],
            names=[ct.obs_id_col, ct.alt_id_col],
        )
        return pd.Series(cross_elast, index=index, name=f"cross_elasticity_{variable}")

    # ------------------------------------------------------------------
    # Covariance estimation
    # ------------------------------------------------------------------

    def covariance_robust(self, data=None) -> np.ndarray:
        """Compute the sandwich (Huber-White) robust covariance matrix.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute covariance on.  If ``None``, uses
            estimation data.

        Returns
        -------
        np.ndarray, shape (n_parameters, n_parameters)
            Sandwich (robust) covariance matrix.
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated first.")

        arrays = self._arrays
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )

        scores = self._observation_scores(arrays)
        B = scores.T @ scores
        H_inv = self._get_hessian_inverse()

        if H_inv is None:
            try:
                return np.linalg.inv(B)
            except np.linalg.LinAlgError:
                return np.full_like(B, np.nan)

        return H_inv @ B @ H_inv

    def covariance_clustered(self, data=None, groups=None) -> np.ndarray:
        """Compute cluster-robust (Rogers) covariance matrix.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute covariance on.  If ``None``, uses
            estimation data.
        groups : array-like, shape (n_obs,)
            Cluster/group identifiers for each observation.

        Returns
        -------
        np.ndarray, shape (n_parameters, n_parameters)
            Cluster-robust covariance matrix.
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated first.")

        arrays = self._arrays
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )

        if groups is None:
            raise ValueError("groups must be provided for cluster-robust covariance.")

        scores = self._observation_scores(arrays)
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        n_params = scores.shape[1]

        B_clustered = np.zeros((n_params, n_params))
        for g in unique_groups:
            mask = groups == g
            g_c = scores[mask].sum(axis=0)
            B_clustered += np.outer(g_c, g_c)

        H_inv = self._get_hessian_inverse()

        if H_inv is None:
            try:
                return np.linalg.inv(B_clustered)
            except np.linalg.LinAlgError:
                return np.full_like(B_clustered, np.nan)

        return H_inv @ B_clustered @ H_inv

    def std_errors_robust(self, data=None) -> pd.Series:
        """Compute sandwich (Huber-White) robust standard errors.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute standard errors on.

        Returns
        -------
        pd.Series
            Robust standard errors, indexed by parameter name.
        """
        cov = self.covariance_robust(data=data)
        se = np.sqrt(np.abs(np.diag(cov)))
        return pd.Series(se, index=self._result.coefficients.index, name="std_error_robust")

    def std_errors_clustered(self, data=None, groups=None) -> pd.Series:
        """Compute cluster-robust standard errors.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute standard errors on.
        groups : array-like, shape (n_obs,)
            Cluster/group identifiers.

        Returns
        -------
        pd.Series
            Cluster-robust standard errors, indexed by parameter name.
        """
        cov = self.covariance_clustered(data=data, groups=groups)
        se = np.sqrt(np.abs(np.diag(cov)))
        return pd.Series(se, index=self._result.coefficients.index, name="std_error_clustered")

    def probabilities(self, data=None, beta=None, rho=None):
        """Compute choice probabilities.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to predict on.  If None, uses estimation data.
        beta : np.ndarray or None
            Utility coefficients.  If None, uses estimated values.
        rho : float or None
            Dissimilarity parameter.  If None, uses estimated value.

        Returns
        -------
        np.ndarray, shape (n_obs, n_alts)
            Choice probabilities.
        """
        if self._arrays is None:
            raise RuntimeError("Model must be estimated before prediction.")

        arrays = self._arrays
        if data is not None:
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )

        if beta is None:
            k = arrays.design_matrix.shape[1]
            beta = self._result.coefficients.values[:k]
        if rho is None:
            k = arrays.design_matrix.shape[1]
            rho = self._result.coefficients.values[k]

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts
        available = arrays.available
        inclusion_probs = get_sampling_correction(arrays)

        log_probs = _scl_log_probs_dispatch(
            beta,
            rho,
            dm,
            self._allocation,
            self._edge_list,
            n_obs,
            n_alts,
            available=available,
            inclusion_probs=inclusion_probs,
            edge_struct=self._edge_struct,
        )

        return np.exp(log_probs)
