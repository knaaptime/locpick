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

from typing import Any, Optional, Union

import numpy as np

from locpick._compat import _NUMBA_AVAILABLE, _NUMBA_PARALLEL
from locpick._solvers import Solver
from locpick.models.mixed import ParamDistribution
from locpick.models.scl import SCL

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


class MixedSCL(SCL):
    r"""Mixed Spatially Correlated Logit (MSCL) model for location choice.

    The MSCL model combines spatial correlation (SCL/GEV structure) with
    random taste variation (mixed logit), following Bhat & Guo (2004).

    The SCL component captures spatial correlation between contiguous
    alternatives via a paired Generalised Nested Logit (PGNL) structure
    with dissimilarity parameter :math:`\rho`.  The mixing distribution
    captures unobserved heterogeneity across decision-makers.

    The MSCL choice probability is:

    .. math::

        P_i = \int (P_i \mid \beta) \, f(\beta \mid \theta) \, d\beta

    approximated by simulated maximum likelihood:

    .. math::

        \tilde{P}_i = \frac{1}{R} \sum_{r=1}^{R} P_i(\beta^r)

    where :math:`P_i(\beta^r)` is the SCL choice probability (see
    :class:`SpatiallyCorrelatedLogit`) conditional on the :math:`r`-th
    draw of the random coefficients.

    This class is a thin convenience wrapper around
    :class:`SpatiallyCorrelatedLogit` that requires ``random_params``
    to be specified.

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
        Type of draws: ``"halton"`` (default) or ``"random"``.
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
    >>> from locpick import ChoiceTable, MixedSpatiallyCorrelatedLogit
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
    >>> print(result.summary())
    """

    def __init__(
        self,
        data,
        formula: Optional[str] = None,
        spec=None,
        graph: Any = None,
        random_params: Optional[dict[str, ParamDistribution]] = None,
        n_draws: Optional[int] = None,
        draw_type: Optional[str] = None,
        weights: Optional[Union[str, np.ndarray]] = None,
        availability: Optional[Union[str, np.ndarray]] = None,
        solver: Union[str, Solver] = None,
        solver_options: Optional[dict] = None,
        backend: Optional[str] = None,
    ):
        if graph is None:
            raise ValueError(
                "MixedSpatiallyCorrelatedLogit requires a 'graph' argument "
                "specifying the spatial adjacency structure."
            )

        super().__init__(
            data=data,
            formula=formula,
            spec=spec,
            graph=graph,
            nests=None,
            random_params=random_params,
            n_draws=n_draws,
            draw_type=draw_type,
            weights=weights,
            availability=availability,
            solver=solver,
            solver_options=solver_options,
            backend=backend,
        )

