r"""Mixed Spatially Correlated Logit (MSCL) model for location choice estimation.

This module implements the MSCL model proposed by Bhat and Guo (2004), which
combines a GEV-based Spatially Correlated Logit (SCL) structure with random
taste variation (mixed logit).  The SCL component captures spatial
correlation between contiguous alternatives via a paired Generalised Nested
Logit (PGNL) structure, while the mixing distribution captures unobserved
heterogeneity across decision-makers.

The key advantage of the MSCL model over a pure Mixed Multinomial Logit
(MMNL) is computational efficiency:  the SCL structure handles spatial
correlation in closed form, so the simulation dimension is limited to the
number of random parameters rather than the number of spatial error
components.

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
from locpick._solvers import Solver, SolverResult, get_solver
from locpick.data.arrays import ChoiceArrays
from locpick.data.problem import EstimationProblem
from locpick.models.base import BaseChoiceModel
from locpick.models.mixed import ParamDistribution
from locpick.models.scl import (
    EdgeStructure,
    _resolve_spatial_graph,
    naturalize_rho,
)
from locpick.results.fit_result import FitResult
from locpick.spec import ModelSpec

# ---------------------------------------------------------------------------
# MSCL probability kernel (Numba-JIT)
# ---------------------------------------------------------------------------

if _NUMBA_AVAILABLE:
    from numba import njit, prange

    # Import the JIT-compiled SCL core so Numba can resolve it at compile time.
    # Numba resolves @njit function references at compile time, so the import
    # must happen at module level (not inside a function).
    from locpick.models.scl import _scl_log_probs_numba_core  # noqa: F811

    @njit(cache=True, parallel=_NUMBA_PARALLEL)
    def _mscl_ll_numba_core(
        v_fixed: np.ndarray,
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
        dm_random: np.ndarray,
        draws: np.ndarray,
        beta_random_means: np.ndarray,
        beta_random_spreads: np.ndarray,
        random_distributions: np.ndarray,
        n_draws: int,
        k_random: int,
    ) -> float:
        """Numba-JIT MSCL simulated log-likelihood kernel.

        The draw loop is inside JIT — no Python overhead per draw.
        Calls the SCL Numba kernel directly (resolved at compile time).
        Parallelized over observations within each draw.
        """
        n_obs = v_fixed.shape[0]
        n_alts = v_fixed.shape[1]

        # Accumulate log-likelihood across draws using logsumexp trick
        # SLL = sum_n w_n * (logsumexp_r(log L_n(beta^r)) - log(R))
        max_log_L = np.full(n_obs, NEG_INF)
        sum_exp_L = np.zeros(n_obs)

        for r in range(n_draws):
            # Realise random coefficients for this draw (parallel over obs)
            beta_random_r = np.zeros((n_obs, k_random))
            for p in range(k_random):
                dist = random_distributions[p]
                mean_p = beta_random_means[p]
                spread_p = beta_random_spreads[p]
                for n in prange(n_obs):
                    z = draws[n, r, p]
                    if dist == 0:  # normal
                        beta_random_r[n, p] = mean_p + spread_p * z
                    elif dist == 1:  # lognormal
                        exponent = mean_p + spread_p * z
                        if exponent > 50.0:
                            exponent = 50.0
                        elif exponent < -50.0:
                            exponent = -50.0
                        beta_random_r[n, p] = np.exp(exponent)
                    else:  # triangular/uniform — approximate via normal CDF
                        # Abramowitz & Stegun approximation of Phi(z)
                        t = 1.0 / (1.0 + 0.2316419 * abs(z))
                        d = 0.3989422804014327  # 1/sqrt(2*pi)
                        poly = t * (
                            0.319381530
                            + t
                            * (
                                -0.356563782
                                + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))
                            )
                        )
                        if z >= 0:
                            phi_z = 1.0 - d * np.exp(-0.5 * z * z) * poly
                        else:
                            phi_z = d * np.exp(-0.5 * z * z) * poly
                        u = phi_z
                        beta_random_r[n, p] = mean_p + spread_p * (2.0 * u - 1.0)

            # Random utility component (parallel over obs)
            v_random = np.zeros((n_obs, n_alts))
            for n in prange(n_obs):
                for j in range(n_alts):
                    s = 0.0
                    for p in range(k_random):
                        s += dm_random[n * n_alts + j, p] * beta_random_r[n, p]
                    v_random[n, j] = s

            # Total utility
            V = v_fixed + v_random

            # Mask unavailable (parallel over obs)
            for n in prange(n_obs):
                for j in range(n_alts):
                    if avail[n, j] <= 0.0:
                        V[n, j] = NEG_INF

            # SCL log-probabilities for this draw (already parallel inside)
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

            # Chosen log-probability for this draw (parallel over obs)
            log_L_draw = np.empty(n_obs)
            for n in prange(n_obs):
                log_L_n = 0.0
                for j in range(n_alts):
                    if chosen[n, j] > 0.0:
                        log_L_n += log_probs[n, j]
                log_L_draw[n] = log_L_n

            # Online logsumexp accumulation (sequential — data dependency)
            for n in range(n_obs):
                log_L_n = log_L_draw[n]
                if log_L_n > max_log_L[n]:
                    sum_exp_L[n] *= np.exp(max_log_L[n] - log_L_n)
                    max_log_L[n] = log_L_n
                sum_exp_L[n] += np.exp(log_L_n - max_log_L[n])

        # Final SLL
        sll = 0.0
        log_R = np.log(float(n_draws))
        for n in range(n_obs):
            log_sim_prob = max_log_L[n] + np.log(sum_exp_L[n]) - log_R
            sll += log_sim_prob * weights[n]

        return sll


def _mscl_simulated_ll_numba(
    beta_fixed: np.ndarray,
    alpha_rho: float,
    beta_random_means: np.ndarray,
    beta_random_spreads: np.ndarray,
    random_distributions: list[str],
    draws: np.ndarray,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    edge_struct: "EdgeStructure",
    random_col_indices: list[int],
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute MSCL simulated log-likelihood using Numba-JIT backend."""
    rho = naturalize_rho(alpha_rho)
    n_draws = draws.shape[1]
    k_random = len(random_col_indices)

    # Build column indices
    all_col_indices = list(range(design_matrix.shape[1]))
    fixed_col_indices = [i for i in all_col_indices if i not in random_col_indices]

    dm_fixed = design_matrix[:, fixed_col_indices] if fixed_col_indices else None
    dm_random = design_matrix[:, random_col_indices]

    # Fixed utility component
    if dm_fixed is not None and len(beta_fixed) > 0:
        v_fixed = (dm_fixed @ beta_fixed).reshape(n_obs, n_alts).copy()
    else:
        v_fixed = np.zeros((n_obs, n_alts), dtype=np.float64)

    if available is not None:
        avail = np.asarray(available, dtype=np.float64).reshape(n_obs, n_alts)
    else:
        avail = np.ones((n_obs, n_alts), dtype=np.float64)

    if inclusion_probs is not None:
        sr = np.asarray(inclusion_probs, dtype=np.float64).reshape(n_obs, n_alts)
        v_fixed = v_fixed + np.log(np.maximum(sr, 1e-30))

    chosen_2d = np.asarray(chosen, dtype=np.float64).reshape(n_obs, n_alts)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(n_obs)
    else:
        w = np.ones(n_obs, dtype=np.float64)

    # Encode distributions as int array for Numba
    dist_map = {"normal": 0, "lognormal": 1, "triangular": 2, "uniform": 3}
    dist_codes = np.array([dist_map.get(d, 0) for d in random_distributions], dtype=np.int64)

    return _mscl_ll_numba_core(
        np.ascontiguousarray(v_fixed),
        rho,
        edge_struct.allocation,
        np.ascontiguousarray(chosen_2d),
        np.ascontiguousarray(avail),
        np.ascontiguousarray(w),
        edge_struct.edge_i,
        edge_struct.edge_j,
        edge_struct.alt_edge_starts,
        edge_struct.alt_edge_counts,
        edge_struct.alt_edge_indices,
        edge_struct.alt_edge_is_first,
        edge_struct.connected,
        edge_struct.isolated,
        edge_struct.n_edges,
        np.ascontiguousarray(dm_random),
        np.ascontiguousarray(draws),
        np.ascontiguousarray(beta_random_means),
        np.ascontiguousarray(beta_random_spreads),
        dist_codes,
        n_draws,
        k_random,
    )


# ---------------------------------------------------------------------------
# MSCL probability kernel (NumPy fallback)
# ---------------------------------------------------------------------------


def _mscl_simulated_ll_numpy(
    beta_fixed: np.ndarray,
    alpha_rho: float,
    beta_random_means: np.ndarray,
    beta_random_spreads: np.ndarray,
    random_distributions: list[str],
    draws: np.ndarray,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    allocation: np.ndarray,
    edge_list: list[tuple[int, int]],
    random_col_indices: list[int],
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute MSCL simulated log-likelihood (NumPy backend).

    The MSCL choice probability for alternative :math:`i` is:

    .. math::

        P_i = \\int (P_i \\mid \\beta) \\, f(\\beta \\mid \\theta) \\, d\\beta

    approximated by simulated maximum likelihood:

    .. math::

        \\tilde{P}_i = \\frac{1}{R} \\sum_{r=1}^{R} P_i(\\beta^r)

    where :math:`P_i(\\beta^r)` is the SCL choice probability conditional on
    the :math:`r`-th draw of the random coefficients, and :math:`f(\\beta
    \\mid \\theta)` is the mixing distribution (typically multivariate normal).

    The simulated log-likelihood is:

    .. math::

        \\text{SLL} = \\sum_{n=1}^{N} w_n \\log \\left(
        \\frac{1}{R} \\sum_{r=1}^{R} L_n(\\beta^r) \\right)

    Parameters
    ----------
    beta_fixed : np.ndarray, shape (k_fixed,)
        Coefficients for fixed (non-random) parameters.
    alpha_rho : float
        Unconstrained dissimilarity parameter (logit scale).
    beta_random_means : np.ndarray, shape (k_random,)
        Mean coefficients for random parameters.
    beta_random_spreads : np.ndarray, shape (k_random,)
        Spread (standard deviation) coefficients for random parameters.
    random_distributions : list of str
        Distribution name for each random parameter.
    draws : np.ndarray, shape (n_obs, n_draws, k_random)
        Standard normal draws for simulated integration.
    design_matrix : np.ndarray, shape (n_obs * n_alts, k_total)
        Design matrix.
    chosen : np.ndarray, shape (n_obs, n_alts)
        Binary indicator matrix for chosen alternatives.
    allocation : np.ndarray, shape (n_alts, n_alts)
        Allocation matrix :math:`\\alpha_{i,j}`.
    edge_list : list of (int, int)
        Paired-nest edges defining the spatial correlation structure.
    random_col_indices : list of int
        Column indices in the design matrix for random parameters.
    n_obs : int
        Number of observations.
    n_alts : int
        Number of alternatives per observation.
    available : np.ndarray or None, shape (n_obs, n_alts)
        Availability matrix.
    inclusion_probs : np.ndarray or None, shape (n_obs, n_alts)
        Sampling rates for correction.
    weights : np.ndarray or None, shape (n_obs,)
        Observation-level weights.

    Returns
    -------
    float
        Simulated log-likelihood value.
    """
    rho = naturalize_rho(alpha_rho)
    n_draws = draws.shape[1]
    k_random = len(random_col_indices)

    # Build column indices
    all_col_indices = list(range(design_matrix.shape[1]))
    fixed_col_indices = [i for i in all_col_indices if i not in random_col_indices]

    dm_fixed = design_matrix[:, fixed_col_indices] if fixed_col_indices else None
    dm_random = design_matrix[:, random_col_indices]

    # Fixed utility component
    if dm_fixed is not None and len(beta_fixed) > 0:
        v_fixed = (dm_fixed @ beta_fixed).reshape(n_obs, n_alts)
    else:
        v_fixed = np.zeros((n_obs, n_alts))

    # Availability mask
    if available is not None:
        avail = np.asarray(available, dtype=np.float64).reshape(n_obs, n_alts)
    else:
        avail = np.ones((n_obs, n_alts), dtype=np.float64)

    # Sampling correction
    if inclusion_probs is not None:
        sr = np.asarray(inclusion_probs, dtype=np.float64).reshape(n_obs, n_alts)
        v_fixed = v_fixed + np.log(np.maximum(sr, 1e-30))

    # Chosen indicator
    chosen_2d = np.asarray(chosen, dtype=np.float64).reshape(n_obs, n_alts)

    # Vectorised random coefficient generation across all draws at once
    beta_random_r = np.zeros((n_obs, n_draws, k_random), dtype=np.float64)
    for p in range(k_random):
        z_p = draws[:, :, p]  # (n_obs, n_draws)
        mean_p = beta_random_means[p]
        spread_p = beta_random_spreads[p]

        if random_distributions[p] == "normal":
            beta_random_r[:, :, p] = mean_p + spread_p * z_p
        elif random_distributions[p] == "lognormal":
            beta_random_r[:, :, p] = np.exp(np.clip(mean_p + spread_p * z_p, -50, 50))
        elif random_distributions[p] in ("triangular", "uniform"):
            from scipy.stats import norm as norm_dist

            u = norm_dist.cdf(z_p)
            beta_random_r[:, :, p] = mean_p + spread_p * (2 * u - 1)

    # Random utility component for all draws via einsum
    dm_random_3d = dm_random.reshape(n_obs, n_alts, k_random)
    v_random_all = np.einsum("njp,nrp->nrj", dm_random_3d, beta_random_r)

    # For each draw, compute conditional SLL
    log_L_draws = np.zeros((n_obs, n_draws), dtype=np.float64)

    for r in range(n_draws):
        V = v_fixed + v_random_all[:, r, :]  # (n_obs, n_alts)

        # Mask unavailable
        V = np.where(avail > 0, V, NEG_INF)

        # Compute SCL log-probabilities for this draw
        log_probs = _mscl_log_probs_from_V_numpy(
            V, rho, allocation, edge_list, n_obs, n_alts, avail
        )

        # Chosen log-probability for this draw
        log_L_draws[:, r] = (log_probs * chosen_2d).sum(axis=1)

    # Simulated log-likelihood
    # SLL = sum_n w_n * log( (1/R) * sum_r L_n(beta^r) )
    #      = sum_n w_n * (logsumexp_r(log L_n(beta^r)) - log(R))
    log_L_sim = logsumexp(log_L_draws, axis=1) - np.log(n_draws)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(n_obs)
        log_L_sim = log_L_sim * w

    return float(log_L_sim.sum())


def _mscl_log_probs_from_V_numpy(
    V: np.ndarray,
    rho: float,
    allocation: np.ndarray,
    edge_list: list[tuple[int, int]],
    n_obs: int,
    n_alts: int,
    avail: np.ndarray,
) -> np.ndarray:
    """Compute SCL log-probabilities from pre-computed utilities.

    This is a streamlined version of :func:`_scl_log_probs_numpy` that
    takes utilities directly rather than a design matrix and beta vector.
    Used by the MSCL kernel where utilities are computed per-draw.

    Parameters
    ----------
    V : np.ndarray, shape (n_obs, n_alts)
        Pre-computed systematic utilities.
    rho : float
        Dissimilarity parameter in (0, 1].
    allocation : np.ndarray, shape (n_alts, n_alts)
        Allocation matrix :math:`\\alpha_{i,j}`.
    edge_list : list of (int, int)
        Paired-nest edges.
    n_obs, n_alts : int
        Dimensions.
    avail : np.ndarray, shape (n_obs, n_alts)
        Availability matrix.

    Returns
    -------
    np.ndarray, shape (n_obs, n_alts)
        Log-probabilities.
    """
    inv_rho = 1.0 / rho
    exp_V = np.exp(np.clip(V, -500, 500))

    # Pre-compute (α_{i,ij} * exp(V_i))^{1/ρ}
    # alloc_exp_V[obs, i, j] = allocation[i, j] * exp_V[obs, i]
    alloc_exp_V = allocation[None, :, :] * exp_V[:, :, None]
    alloc_exp_V = np.clip(alloc_exp_V, 1e-30, 1e30)
    alloc_exp_V_inv_rho = np.power(alloc_exp_V, inv_rho)

    n_edges = len(edge_list)
    if n_edges == 0:
        log_sum_exp_V = logsumexp(V, axis=1)
        return V - log_sum_exp_V[:, None]

    # Identify connected vs isolated alternatives
    connected = set()
    for i, j in edge_list:
        connected.add(i)
        connected.add(j)
    isolated = sorted(set(range(n_alts)) - connected)

    # Nest values
    nest_vals = np.zeros((n_obs, n_edges), dtype=np.float64)
    for idx, (i, j) in enumerate(edge_list):
        term_i = alloc_exp_V_inv_rho[:, i, j]
        term_j = alloc_exp_V_inv_rho[:, j, i]
        nest_vals[:, idx] = np.power(term_i + term_j, rho)

    # Denominator: sum of nest values + exp(V_i) for isolated alternatives
    if isolated:
        iso_exp_V = exp_V[:, isolated]
        denom_components = np.column_stack([nest_vals, iso_exp_V])
    else:
        denom_components = nest_vals

    # Use log of sum (not logsumexp) since nest_vals are already on natural scale
    log_denom = np.log(np.maximum(denom_components.sum(axis=1), 1e-300))

    # Log-probabilities
    log_probs = np.full((n_obs, n_alts), NEG_INF, dtype=np.float64)

    alt_to_edges: dict[int, list[tuple[int, bool]]] = {}
    for idx, (i, j) in enumerate(edge_list):
        alt_to_edges.setdefault(i, []).append((idx, True))
        alt_to_edges.setdefault(j, []).append((idx, False))

    # Connected alternatives
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

    # Isolated alternatives: degenerate nests
    for alt_i in isolated:
        log_probs[:, alt_i] = V[:, alt_i] - log_denom

    log_probs = np.where(avail > 0, log_probs, NEG_INF)
    return log_probs


# ---------------------------------------------------------------------------
# MSCL model class
# ---------------------------------------------------------------------------


class MixedSpatiallyCorrelatedLogit(BaseChoiceModel):
    r"""Mixed Spatially Correlated Logit (MSCL) model for location choice.

    The MSCL model combines spatial correlation (SCL/GEV structure) with
    random taste variation (mixed logit), following Bhat & Guo (2004).

    The SCL component captures spatial correlation between contiguous
    alternatives via a paired Generalised Nested Logit (PGNL) structure
    with dissimilarity parameter :math:`\\rho`.  The mixing distribution
    captures unobserved heterogeneity across decision-makers.

    The MSCL choice probability is:

    .. math::

        P_i = \\int (P_i \\mid \\beta) \\, f(\\beta \\mid \\theta) \\, d\\beta

    approximated by simulated maximum likelihood:

    .. math::

        \\tilde{P}_i = \\frac{1}{R} \\sum_{r=1}^{R} P_i(\\beta^r)

    where :math:`P_i(\\beta^r)` is the SCL choice probability (see
    :class:`SpatiallyCorrelatedLogit`) conditional on the :math:`r`-th
    draw of the random coefficients.

    The key computational advantage over a pure MMNL is that the SCL
    structure handles spatial correlation in closed form, so the
    simulation dimension equals the number of random parameters (not the
    number of spatial error components).  In the empirical application
    of Bhat & Guo (2004), this reduces the integration dimension from
    ~500 (MMNL) to 3 (MSCL).

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
    random_params : dict, optional
        Mapping of parameter names to :class:`ParamDistribution` for
        mixed logit.  E.g. ``{"commute_time": ParamDistribution("normal")}``.
    n_draws : int, optional
        Number of Halton draws for simulated maximum likelihood.
        Default 250.
    draw_type : str, optional
        Type of draws: ``"halton"`` (default, legacy shuffled Halton),
        ``"random"``, ``"qmc"``/``"sobol"`` (scrambled Sobol), or
        ``"scrambled_halton"``.  The scrambled QMC variants typically
        reach the same simulation accuracy with substantially fewer
        draws.
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
    >>> from locpick import ChoiceTable, FitDiagnostics, MixedSpatiallyCorrelatedLogit
    >>> from locpick.models.mixed import ParamDistribution
    >>> from libpysal import graph
    >>> ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=10)
    >>> g = graph.Graph.build_contiguity(tracts_gdf, rook=False)
    >>> model = MixedSpatiallyCorrelatedLogit(
    ...     ct,
    ...     formula="commute_time + density + shopping_access",
    ...     graph=g,
    ...     random_params={"commute_time": ParamDistribution("normal")},
    ...     n_draws=250,
    ... )
    >>> result = model.fit()
    >>> print(FitDiagnostics.summary(result))
    """

    def __init__(
        self,
        data,
        formula: Optional[str] = None,
        spec: Optional[ModelSpec] = None,
        graph: Any = None,
        random_params: Optional[dict[str, ParamDistribution]] = None,
        n_draws: int = 250,
        draw_type: str = "halton",
        weights: Optional[Union[str, np.ndarray]] = None,
        availability: Optional[Union[str, np.ndarray]] = None,
        solver: Union[str, Solver] = "lbfgs",
        solver_options: Optional[dict] = None,
        backend: Optional[str] = None,
    ):
        if graph is None:
            raise ValueError(
                "MixedSpatiallyCorrelatedLogit requires a 'graph' argument "
                "specifying the spatial adjacency structure."
            )
        if formula is None and spec is None:
            raise ValueError("Either 'formula' or 'spec' must be provided.")
        if formula is not None and spec is not None:
            raise ValueError("Provide 'formula' or 'spec', not both.")

        self._data = data
        self._problem: Optional[EstimationProblem] = None
        self._formula = formula
        self._graph_input = graph
        self._random_params = random_params or {}
        self._n_draws = n_draws
        self._draw_type = draw_type
        self._weights = weights
        self._availability = availability
        self._solver_options = solver_options or {}
        self._backend = backend

        # Build ModelSpec from formula if needed
        if formula is not None:
            self._spec = ModelSpec(formula=formula)
        else:
            self._spec = spec

        # Resolve solver
        if isinstance(solver, str):
            self._solver = get_solver(solver, **self._solver_options)
        else:
            self._solver = solver

        # Lazy-initialized
        self._arrays: Optional[ChoiceArrays] = None
        self._result: Optional[FitResult] = None

        # Caches (cleared on re-estimation)
        self._hessian_inverse: Optional[np.ndarray] = None
        self._observation_scores_cache: dict = {}
        self._probabilities_cache: Optional[np.ndarray] = None
        self._utilities_cache: Optional[np.ndarray] = None
        self._covariance_bhhh_cache: Optional[np.ndarray] = None
        self._covariance_robust_cache: Optional[np.ndarray] = None
        self._allocation: Optional[np.ndarray] = None
        self._edge_list: Optional[list] = None
        self._n_alts_graph: Optional[int] = None
        self._edge_struct: Optional[EdgeStructure] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self):
        """The ChoiceTable data."""
        return self._data

    @property
    def spec(self) -> ModelSpec:
        """The ModelSpec used for estimation."""
        return self._spec

    @property
    def solver(self) -> Solver:
        """The solver used for estimation."""
        return self._solver

    @property
    def result(self) -> Optional[FitResult]:
        """The estimation result, or None if not yet estimated."""
        return self._result

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def fit(self) -> FitResult:
        """Estimate the MSCL model and return results.

        Returns
        -------
        FitResult
            Complete estimation results including coefficients, standard
            errors, the dissimilarity parameter :math:`\\rho`, random
            parameter means and standard deviations, and fit statistics.
        """
        # Build estimation arrays
        arrays = self._build_arrays()
        self._arrays = arrays

        # Resolve spatial graph
        omega, allocation, edge_list, n_alts_graph = _resolve_spatial_graph(self._graph_input)
        self._allocation = allocation
        self._edge_list = edge_list
        self._n_alts_graph = n_alts_graph

        if n_alts_graph != arrays.n_alts:
            raise ValueError(
                f"Spatial graph has {n_alts_graph} nodes but the choice "
                f"data has {arrays.n_alts} alternatives.  The graph must "
                f"cover exactly the same alternatives as the choice data."
            )

        # Precompute edge structure for Numba backend
        if _NUMBA_AVAILABLE:
            self._edge_struct = EdgeStructure(edge_list, n_alts_graph, allocation)
        else:
            self._edge_struct = None

        # Identify random parameter columns
        param_names = list(arrays.param_names)
        random_param_names = list(self._random_params.keys())
        random_col_indices = [param_names.index(name) for name in random_param_names]
        random_distributions = [
            self._random_params[name].distribution for name in random_param_names
        ]
        k_random = len(random_col_indices)
        k_total = arrays.design_matrix.shape[1]

        # Generate draws
        from locpick.models.mixed import _resolve_draws

        draws = _resolve_draws(self._draw_type, arrays.n_obs, self._n_draws, k_random, seed=42)

        # Cache random structure for objective/prediction paths
        self._draws = draws
        self._random_col_indices = random_col_indices
        self._random_distributions = random_distributions
        self._random_param_names = random_param_names

        # Parameter layout: [beta_fixed, alpha_rho, beta_random_means, beta_random_spreads]
        k_fixed = k_total - k_random
        n_params = k_fixed + 1 + 2 * k_random  # +1 for alpha_rho

        objective = self._build_objective(arrays)

        # Initial values
        x0 = np.zeros(n_params)
        # alpha_rho starts at 0 → rho ≈ 0.5
        x0[k_fixed] = 0.0

        # Parameter names for display
        fixed_param_names = [name for name in param_names if name not in random_param_names]
        display_param_names = (
            fixed_param_names
            + ["rho"]
            + [f"mean_{name}" for name in random_param_names]
            + [f"sd_{name}" for name in random_param_names]
        )

        # Solve
        solver_result = self._solver.solve(
            objective=objective,
            x0=x0,
            param_names=display_param_names,
        )

        # Build FitResult
        self._result = self._build_fit_result(
            solver_result, arrays, k_fixed, k_random, random_param_names
        )
        self._clear_caches()
        return self._result

    def _build_objective(self, arrays: ChoiceArrays) -> Objective:
        """Build optimization objective for MSCL estimation."""
        if self._allocation is None or self._edge_list is None:
            raise RuntimeError("Spatial graph must be resolved before building objective.")
        if not hasattr(self, "_random_col_indices"):
            raise RuntimeError(
                "Random parameter structure must be prepared before building objective."
            )

        random_col_indices = self._random_col_indices
        random_distributions = self._random_distributions
        random_param_names = self._random_param_names
        draws = self._draws

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        chosen = np.asarray(arrays.chosen, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts
        available = arrays.available
        weights = arrays.weights
        edge_struct = self._edge_struct

        from locpick._sampling.correction import get_sampling_correction

        inclusion_probs = get_sampling_correction(arrays)

        k_random = len(random_col_indices)
        k_total = dm.shape[1]
        k_fixed = k_total - k_random

        backend = (self._backend or os.environ.get("CHOICEMODELS_MSCL_BACKEND", "")).lower()
        if backend == "jax":
            use_jax = _JAX_AVAILABLE
        elif backend in {"numba", "numpy"}:
            use_jax = False
        else:
            # Default: JAX if available, else Numba, else NumPy
            use_jax = _JAX_AVAILABLE

        if use_jax:
            from locpick._jax.builders import build_mscl_objective

            return build_mscl_objective(
                arrays,
                edge_struct,
                self._allocation,
                self._edge_list,
                random_col_indices,
                random_distributions,
                draws,
            )

        def ll_fn(params):
            beta_fixed = params[:k_fixed]
            alpha_rho = params[k_fixed]
            beta_random_means = params[k_fixed + 1 : k_fixed + 1 + k_random]
            beta_random_spreads = np.abs(params[k_fixed + 1 + k_random :])

            if _NUMBA_AVAILABLE and edge_struct is not None:
                return _mscl_simulated_ll_numba(
                    beta_fixed,
                    alpha_rho,
                    beta_random_means,
                    beta_random_spreads,
                    random_distributions,
                    draws,
                    dm,
                    chosen,
                    edge_struct,
                    random_col_indices,
                    n_obs,
                    n_alts,
                    available=available,
                    inclusion_probs=inclusion_probs,
                    weights=weights,
                )
            return _mscl_simulated_ll_numpy(
                beta_fixed,
                alpha_rho,
                beta_random_means,
                beta_random_spreads,
                random_distributions,
                draws,
                dm,
                chosen,
                self._allocation,
                self._edge_list,
                random_col_indices,
                n_obs,
                n_alts,
                available=available,
                inclusion_probs=inclusion_probs,
                weights=weights,
            )

        def grad_fn(params):
            eps = 1e-5
            grad = np.zeros_like(params)
            for i in range(params.shape[0]):
                params_plus = params.copy()
                params_plus[i] += eps
                params_minus = params.copy()
                params_minus[i] -= eps
                grad[i] = (ll_fn(params_plus) - ll_fn(params_minus)) / (2 * eps)
            return grad

        fixed_param_names = [name for name in arrays.param_names if name not in random_param_names]
        display_param_names = (
            fixed_param_names
            + ["rho"]
            + [f"mean_{name}" for name in random_param_names]
            + [f"sd_{name}" for name in random_param_names]
        )
        return Objective.from_numpy(
            ll_fn=ll_fn,
            grad_fn=grad_fn,
            param_names=display_param_names,
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _clear_caches(self):
        """Clear all cached computation results."""
        self._hessian_inverse = None
        self._observation_scores_cache = {}
        self._probabilities_cache = None
        self._utilities_cache = None
        self._covariance_bhhh_cache = None
        self._covariance_robust_cache = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_hessian_inverse(self) -> np.ndarray | None:
        """Get the inverse Hessian from the solver result.

        Returns
        -------
        np.ndarray or None
            Inverse Hessian matrix, or None if not available.
        """
        if self._hessian_inverse is not None:
            return self._hessian_inverse

        if (
            self._result is not None
            and self._result.solver_result
            and "scipy_result" in self._result.solver_result
        ):
            scipy_result = self._result.solver_result["scipy_result"]
            if hasattr(scipy_result, "hess_inv"):
                try:
                    self._hessian_inverse = np.asarray(
                        scipy_result.hess_inv.todense()
                        if hasattr(scipy_result.hess_inv, "todense")
                        else scipy_result.hess_inv
                    )
                    return self._hessian_inverse
                except Exception:
                    pass

        if (
            self._result is not None
            and self._result.std_errors is not None
            and not self._result.std_errors.isna().all()
        ):
            variances = self._result.std_errors.values**2
            self._hessian_inverse = np.diag(variances)
            return self._hessian_inverse

        return None

    def _observation_scores(self, arrays) -> np.ndarray:
        """Compute observation-level score vectors via numerical differentiation.

        Parameters
        ----------
        arrays : ChoiceArrays
            The estimation data arrays.

        Returns
        -------
        np.ndarray, shape (n_obs, n_params)
            Score vector for each observation.
        """
        if self._result is None:
            raise RuntimeError("Model must be estimated first.")

        cache_key = id(arrays)
        if cache_key in self._observation_scores_cache:
            return self._observation_scores_cache[cache_key]

        eps = 1e-5
        param_names = list(arrays.param_names)
        random_col_indices = []
        for name, dist in self._random_params.items():
            random_col_indices.append(param_names.index(name))

        k_total = arrays.design_matrix.shape[1]
        k_random = len(random_col_indices)
        k_fixed = k_total - k_random
        n_params = k_fixed + 1 + 2 * k_random  # [beta_fixed, rho, means, spreads]
        n_obs = arrays.n_obs

        # Full parameter vector: [beta_fixed, rho, beta_random_means, beta_random_spreads]
        full_params = self._result.coefficients.values.copy()
        bf_hat = full_params[:k_fixed]
        rho_hat = full_params[k_fixed]
        brm_hat = full_params[k_fixed + 1 : k_fixed + 1 + k_random]
        brs_hat = full_params[k_fixed + 1 + k_random :]

        # Chosen indicator
        chosen = np.asarray(arrays.chosen, dtype=np.float64).reshape(n_obs, arrays.n_alts)

        # Base probabilities and per-observation LL
        probs_base = self.probabilities(
            data=None,
            beta_fixed=bf_hat,
            rho=rho_hat,
            beta_random_means=brm_hat,
            beta_random_spreads=brs_hat,
        )
        np.log(np.maximum(np.sum(probs_base * chosen, axis=1), 1e-30))

        scores = np.zeros((n_obs, n_params))

        for j in range(n_params):
            params_plus = full_params.copy()
            params_plus[j] += eps
            params_minus = full_params.copy()
            params_minus[j] -= eps

            bf_plus = params_plus[:k_fixed]
            rho_plus = params_plus[k_fixed]
            brm_plus = params_plus[k_fixed + 1 : k_fixed + 1 + k_random]
            brs_plus = params_plus[k_fixed + 1 + k_random :]
            bf_minus = params_minus[:k_fixed]
            rho_minus = params_minus[k_fixed]
            brm_minus = params_minus[k_fixed + 1 : k_fixed + 1 + k_random]
            brs_minus = params_minus[k_fixed + 1 + k_random :]

            probs_plus = self.probabilities(
                data=None,
                beta_fixed=bf_plus,
                rho=rho_plus,
                beta_random_means=brm_plus,
                beta_random_spreads=brs_plus,
            )
            probs_minus = self.probabilities(
                data=None,
                beta_fixed=bf_minus,
                rho=rho_minus,
                beta_random_means=brm_minus,
                beta_random_spreads=brs_minus,
            )

            ll_plus = np.log(np.maximum(np.sum(probs_plus * chosen, axis=1), 1e-30))
            ll_minus = np.log(np.maximum(np.sum(probs_minus * chosen, axis=1), 1e-30))

            scores[:, j] = (ll_plus - ll_minus) / (2 * eps)

        self._observation_scores_cache[cache_key] = scores
        return scores

    def utilities(self, data=None, beta_fixed=None, beta_random_means=None):
        """Compute deterministic utilities (expected utility at population means).

        For mixed spatially correlated logit models, the deterministic
        utility is computed using the expected (mean) coefficients:
        fixed coefficients for non-random variables and mean coefficients
        for random variables.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to predict on.  If ``None``, uses estimation data.
        beta_fixed : np.ndarray or None
            Fixed utility coefficients.  If ``None``, uses estimated values.
        beta_random_means : np.ndarray or None
            Random coefficient means.  If ``None``, uses estimated values.

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

        # Extract parameters
        param_names = list(arrays.param_names)
        random_col_indices = []
        for name, dist in self._random_params.items():
            random_col_indices.append(param_names.index(name))
        k_total = arrays.design_matrix.shape[1]
        k_random = len(random_col_indices)
        k_fixed = k_total - k_random

        if beta_fixed is None:
            coeffs = self._result.coefficients.values
            beta_fixed = np.asarray(coeffs[:k_fixed], dtype=np.float64)
        if beta_random_means is None:
            coeffs = self._result.coefficients.values
            beta_random_means = np.asarray(
                coeffs[k_fixed + 1 : k_fixed + 1 + k_random], dtype=np.float64
            )

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        # Build full coefficient vector: fixed + random means
        beta_full = np.zeros(dm.shape[1], dtype=np.float64)
        fixed_mask = np.ones(dm.shape[1], dtype=bool)
        fixed_mask[random_col_indices] = False
        beta_full[fixed_mask] = beta_fixed
        beta_full[random_col_indices] = beta_random_means

        # Systematic utility
        V = (dm @ beta_full).reshape(n_obs, n_alts)

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

        For MSCL, the direct marginal effect uses the mean coefficient
        value (``mean_{variable}`` in the results).

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

        For MSCL, the cross-marginal effect uses the mean coefficient
        value (``mean_{variable}`` in the results).

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

        For random parameters, the coefficient used is the mean value
        (``mean_{variable}`` in the results).

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
        # For random parameters, use the mean coefficient
        beta = self._result.coefficients.get(
            variable, self._result.coefficients.get(f"mean_{variable}", 0.0)
        )

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

        For random parameters, the coefficient used is the mean value
        (``mean_{variable}`` in the results).

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
        beta = self._result.coefficients.get(
            variable, self._result.coefficients.get(f"mean_{variable}", 0.0)
        )
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

    def covariance_bhhh(self, data=None) -> np.ndarray:
        """Compute the BHHH covariance matrix.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute covariance on.  If ``None``, uses
            estimation data.

        Returns
        -------
        np.ndarray, shape (n_parameters, n_parameters)
            BHHH covariance matrix.
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
        bhhh = scores.T @ scores
        try:
            return np.linalg.inv(bhhh)
        except np.linalg.LinAlgError:
            return np.full_like(bhhh, np.nan)

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

    def std_errors_bhhh(self, data=None) -> pd.Series:
        """Compute BHHH standard errors.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute standard errors on.

        Returns
        -------
        pd.Series
            BHHH standard errors, indexed by parameter name.
        """
        cov = self.covariance_bhhh(data=data)
        se = np.sqrt(np.diag(cov))
        return pd.Series(se, index=self._result.coefficients.index, name="std_error_bhhh")

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

    def probabilities(
        self,
        data=None,
        beta_fixed=None,
        rho=None,
        beta_random_means=None,
        beta_random_spreads=None,
    ):
        """Compute choice probabilities via simulation.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to predict on.  If ``None``, uses estimation data.
        beta_fixed : np.ndarray or None
            Fixed utility coefficients.  If ``None``, uses estimated values.
        rho : float or None
            Dissimilarity parameter.  If ``None``, uses estimated value.
        beta_random_means : np.ndarray or None
            Random coefficient means.  If ``None``, uses estimated values.
        beta_random_spreads : np.ndarray or None
            Random coefficient standard deviations.  If ``None``, uses
            estimated values.

        Returns
        -------
        np.ndarray, shape (n_obs, n_alts)
            Simulated choice probabilities averaged over random draws.
        """
        if self._arrays is None:
            raise RuntimeError("Model must be estimated before prediction.")

        arrays = self._arrays
        if data is not None:
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )

        # Extract parameters from result
        param_names = list(arrays.param_names)
        random_col_indices = []
        random_distributions = []
        for name, dist in self._random_params.items():
            random_col_indices.append(param_names.index(name))
            random_distributions.append(
                dist.distribution if hasattr(dist, "distribution") else dist
            )

        k_total = arrays.design_matrix.shape[1]
        k_random = len(random_col_indices)
        k_fixed = k_total - k_random

        if beta_fixed is None:
            # Display params: [beta_fixed, rho, mean_*, sd_*]
            coeffs = self._result.coefficients.values
            beta_fixed = coeffs[:k_fixed]
        if rho is None:
            coeffs = self._result.coefficients.values
            rho = coeffs[k_fixed]
        if beta_random_means is None:
            coeffs = self._result.coefficients.values
            beta_random_means = coeffs[k_fixed + 1 : k_fixed + 1 + k_random]
        if beta_random_spreads is None:
            coeffs = self._result.coefficients.values
            beta_random_spreads = coeffs[k_fixed + 1 + k_random :]

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts
        available = arrays.available
        inclusion_probs = get_sampling_correction(arrays)

        # Generate draws (same as estimation)
        from locpick.models.mixed import _resolve_draws

        draws = _resolve_draws(self._draw_type, n_obs, self._n_draws, k_random, seed=42)

        # Compute simulated probabilities
        alpha_rho = np.log(rho / (1.0 - rho)) if rho is not None else 0.0

        if self._edge_struct is not None and _NUMBA_AVAILABLE:
            _mscl_simulated_ll_numba(
                beta_fixed,
                alpha_rho,
                beta_random_means,
                np.abs(beta_random_spreads),
                random_distributions,
                draws,
                dm,
                np.zeros((n_obs, n_alts)),  # chosen not needed for probs
                self._edge_struct,
                random_col_indices,
                n_obs,
                n_alts,
                available=available,
                inclusion_probs=inclusion_probs,
                weights=None,
            )
        else:
            _mscl_simulated_ll_numpy(
                beta_fixed,
                alpha_rho,
                beta_random_means,
                np.abs(beta_random_spreads),
                random_distributions,
                draws,
                dm,
                np.zeros((n_obs, n_alts)),  # chosen not needed for probs
                self._allocation,
                self._edge_list,
                random_col_indices,
                n_obs,
                n_alts,
                available=available,
                inclusion_probs=inclusion_probs,
                weights=None,
            )

        # Average probabilities across draws
        # Use the SCL log-prob kernel directly for each draw.
        probs = np.zeros((n_obs, n_alts), dtype=np.float64)

        # Fixed utility component (same for all draws)
        fixed_col_indices = [i for i in range(dm.shape[1]) if i not in random_col_indices]
        if fixed_col_indices and len(beta_fixed) > 0:
            dm_fixed = dm[:, fixed_col_indices]
            V_fixed = (dm_fixed @ beta_fixed).reshape(n_obs, n_alts)
        else:
            V_fixed = np.zeros((n_obs, n_alts), dtype=np.float64)

        # Reshape random columns for vectorized computation
        dm_random = dm[:, random_col_indices].reshape(
            n_obs, n_alts, -1
        )  # (n_obs, n_alts, k_random)

        for r in range(self._n_draws):
            # Per-draw random coefficients: (n_obs, k_random)
            beta_random_draws = (
                beta_random_means[np.newaxis, :]
                + np.abs(beta_random_spreads)[np.newaxis, :] * draws[:, r, :]
            )

            # Per-observation random utility: V_random[i, j] = dm_random[i, j, :] @ beta_random_draws[i, :]
            V_random = np.einsum("ijk,ik->ij", dm_random, beta_random_draws)

            V = V_fixed + V_random
            if inclusion_probs is not None:
                V = V - np.log(np.maximum(inclusion_probs.reshape(n_obs, n_alts), 1e-300))

            log_probs = _mscl_log_probs_from_V_numpy(
                V,
                rho,
                self._allocation,
                self._edge_list,
                n_obs,
                n_alts,
                np.ones((n_obs, n_alts))
                if available is None
                else available.reshape(n_obs, n_alts),
            )
            probs += np.exp(log_probs)

        probs /= self._n_draws
        return probs

    def _build_arrays(self) -> ChoiceArrays:
        """Build ChoiceArrays from the data and spec."""
        if self._problem is None:
            spec = self._spec if self._formula is None else None
            self._problem = EstimationProblem.from_choice_table(
                self._data,
                spec=spec,
                formula=self._formula,
                weights=self._weights,
                available=self._availability,
                backend=self._backend or "auto",
                solver_name=getattr(self._solver, "name", "lbfgs"),
                solver_options=self._solver_options or None,
            )

        return self._problem.arrays

    def _build_fit_result(
        self,
        solver_result: SolverResult,
        arrays: ChoiceArrays,
        k_fixed: int,
        k_random: int,
        random_param_names: list[str],
    ) -> FitResult:
        """Build a FitResult from solver output."""
        all_params = solver_result.coefficients
        beta_fixed = all_params[:k_fixed]
        alpha_rho = all_params[k_fixed]
        rho = naturalize_rho(alpha_rho)
        beta_random_means = all_params[k_fixed + 1 : k_fixed + 1 + k_random]
        beta_random_spreads = np.abs(
            all_params[k_fixed + 1 + k_random : k_fixed + 1 + 2 * k_random]
        )

        # Parameter names
        param_names_all = list(arrays.param_names)
        fixed_param_names = [name for name in param_names_all if name not in random_param_names]
        display_param_names = (
            fixed_param_names
            + ["rho"]
            + [f"mean_{name}" for name in random_param_names]
            + [f"sd_{name}" for name in random_param_names]
        )

        # Display values: beta_fixed + rho + means + sds
        display_params = np.concatenate(
            [beta_fixed, [rho], beta_random_means, beta_random_spreads]
        )

        # Standard errors
        if solver_result.hessian is not None:
            try:
                se_all = np.sqrt(np.abs(np.diag(solver_result.hessian)))
                # Delta method for rho
                se_rho = rho * (1.0 - rho) * se_all[k_fixed]
                std_errors = np.concatenate(
                    [
                        se_all[:k_fixed],
                        [se_rho],
                        se_all[k_fixed + 1 :],
                    ]
                )
            except Exception:
                std_errors = np.full(len(display_params), np.nan)
        else:
            std_errors = np.full(len(display_params), np.nan)

        # T-values and p-values
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = np.where(std_errors > 0, display_params / std_errors, np.nan)
        from scipy import stats

        p_values = 2 * (1 - stats.norm.cdf(np.abs(np.nan_to_num(t_values))))

        # Confidence intervals
        z_crit = stats.norm.ppf(0.975)
        conf_lower = display_params - z_crit * std_errors
        conf_upper = display_params + z_crit * std_errors

        # Log-likelihood
        ll = solver_result.log_likelihood

        # Null log-likelihood
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts
        if arrays.available is not None:
            avail = np.asarray(arrays.available, dtype=np.float64).reshape(n_obs, -1)
            n_avail = avail.sum(axis=1)
            ll_null = -np.sum(np.log(n_avail))
        else:
            ll_null = -n_obs * np.log(n_alts)

        n_params = len(display_params)

        # Fit statistics
        aic = 2 * n_params - 2 * ll
        bic = n_params * np.log(n_obs) - 2 * ll
        rho_squared = 1 - ll / ll_null
        rho_bar_squared = 1 - (ll - n_params) / ll_null

        # Build pandas objects
        coefficients = pd.Series(display_params, index=display_param_names, name="coefficient")
        std_err_series = pd.Series(std_errors, index=display_param_names, name="std_error")
        t_series = pd.Series(t_values, index=display_param_names, name="t_value")
        p_series = pd.Series(p_values, index=display_param_names, name="p_value")
        conf_int = pd.DataFrame(
            {"lower": conf_lower, "upper": conf_upper},
            index=display_param_names,
        )

        return FitResult(
            coefficients=coefficients,
            std_errors=std_err_series,
            t_values=t_series,
            p_values=p_series,
            conf_int=conf_int,
            log_likelihood=ll,
            log_likelihood_null=ll_null,
            n_observations=n_obs,
            n_parameters=n_params,
            n_alts=n_alts,
            aic=aic,
            bic=bic,
            rho_squared=rho_squared,
            rho_bar_squared=rho_bar_squared,
            spec=self._spec,
            model_type="Mixed Spatially Correlated Logit",
            solver_name=solver_result.solver_name,
            solver_result=solver_result.raw,
        )
