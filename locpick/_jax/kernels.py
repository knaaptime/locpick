"""Pure JAX probability kernels for choice models.

These are standalone, testable, composable JAX functions that compute
log-probabilities for different choice model specifications.  They are
JIT-friendly and vmap-compatible — no side effects, no closures over
external data.

The kernels take pre-computed utilities ``V`` and model-specific parameters,
returning log-probabilities.  The log-likelihood is then computed by the
:class:`~locpick._jax.objective.Objective` builder, which handles weighting,
chosen-alternative selection, and gradient computation.

Kernel design principles
------------------------
1. **Pure functions** — no closures, no mutation, no I/O.
2. **JAX-native** — all inputs/outputs are ``jnp.ndarray``.
3. **Composable** — MSCL calls ``scl_log_probs`` directly.
4. **Vectorised** — no Python loops over observations or alternatives
   (except for isolated-alternative handling which uses ``jnp.where``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp
from jax.ops import segment_sum


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from locpick._kernels.constants import NEG_INF as _NEG_INF_FLOAT

_NEG_INF = jnp.array(_NEG_INF_FLOAT, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# MNL kernel
# ---------------------------------------------------------------------------


def mnl_log_probs(
    V: jnp.ndarray,
    available: jnp.ndarray,
    inclusion_probs: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute MNL log-probabilities from systematic utilities.

    Parameters
    ----------
    V : jnp.ndarray, shape (n_obs, n_alts)
        Systematic utilities.
    available : jnp.ndarray, shape (n_obs, n_alts)
        Binary availability matrix.
    inclusion_probs : jnp.ndarray or None, shape (n_obs, n_alts)
        Inclusion probabilities for sampling correction.
        If provided, ``log(inclusion_probs)`` is added to utilities before
        the softmax.

    Returns
    -------
    jnp.ndarray, shape (n_obs, n_alts)
        Log-probabilities for each (obs, alt) pair.
    """
    if inclusion_probs is not None:
        V = V + jnp.log(jnp.maximum(inclusion_probs, 1e-30))

    # Numerically stable logsumexp over available alternatives
    V_masked = jnp.where(available > 0, V, _NEG_INF)
    log_sum_exp = jax_logsumexp(V_masked, axis=1)
    log_probs = V_masked - log_sum_exp[:, None]
    # Zero out unavailable
    return jnp.where(available > 0, log_probs, _NEG_INF)


def mnl_probs(
    V: jnp.ndarray,
    available: jnp.ndarray,
    inclusion_probs: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute MNL probabilities from systematic utilities.

    Convenience wrapper around :func:`mnl_log_probs` that returns
    probabilities instead of log-probabilities.

    Parameters
    ----------
    V : jnp.ndarray, shape (n_obs, n_alts)
        Systematic utilities.
    available : jnp.ndarray, shape (n_obs, n_alts)
        Binary availability matrix.
    inclusion_probs : jnp.ndarray or None, shape (n_obs, n_alts)
        Inclusion probabilities for sampling correction.

    Returns
    -------
    jnp.ndarray, shape (n_obs, n_alts)
        Choice probabilities for each (obs, alt) pair.
    """
    probs = jnp.exp(mnl_log_probs(V, available, inclusion_probs=inclusion_probs))
    return probs * (available > 0)


# ---------------------------------------------------------------------------
# SCL kernel
# ---------------------------------------------------------------------------


def scl_log_probs(
    V: jnp.ndarray,
    rho: jnp.ndarray,
    edge_data: "EdgeDataJAX",
    available: jnp.ndarray,
) -> jnp.ndarray:
    """Compute SCL log-probabilities from systematic utilities.

    Implements the choice probability from Bhat & Guo (2004, Eq. 9)
    using fully vectorised JAX operations — no Python loops over
    observations or alternatives (except for isolated-alternative
    handling which uses ``jnp.where``).

    Parameters
    ----------
    V : jnp.ndarray, shape (n_obs, n_alts)
        Systematic utilities with unavailable alts set to -1e30.
    rho : jnp.ndarray, scalar
        Dissimilarity parameter in (0, 1].
    edge_data : EdgeDataJAX
        Precomputed spatial edge structure.
    available : jnp.ndarray, shape (n_obs, n_alts)
        Binary availability matrix.

    Returns
    -------
    jnp.ndarray, shape (n_obs, n_alts)
        Log-probabilities for each (obs, alt) pair.
    """
    inv_rho = 1.0 / rho
    n_obs = V.shape[0]
    n_alts = V.shape[1]
    n_edges = edge_data.n_edges

    # Mask unavailable
    V = jnp.where(available > 0, V, _NEG_INF)

    # exp(V) clipped for numerical stability
    exp_V = jnp.exp(jnp.clip(V, -500.0, 500.0))

    if n_edges == 0:
        # MNL fallback — no spatial correlation
        log_sum_exp = jax_logsumexp(V, axis=1)
        return V - log_sum_exp[:, None]

    # --- Sparse edge terms (vectorised over edges) ---
    exp_V_i = exp_V[:, edge_data.edge_i]   # (n_obs, n_edges)
    exp_V_j = exp_V[:, edge_data.edge_j]   # (n_obs, n_edges)

    alloc_ij = edge_data.allocation[edge_data.edge_i, edge_data.edge_j]  # (n_edges,)
    alloc_ji = edge_data.allocation[edge_data.edge_j, edge_data.edge_i]  # (n_edges,)

    term_i = jnp.power(jnp.clip(alloc_ij[None, :] * exp_V_i, 1e-30, 1e30), inv_rho)
    term_j = jnp.power(jnp.clip(alloc_ji[None, :] * exp_V_j, 1e-30, 1e30), inv_rho)

    # --- Nest values ---
    nest_vals = jnp.power(term_i + term_j, rho)  # (n_obs, n_edges)

    # --- Denominator ---
    denom = nest_vals.sum(axis=1)  # (n_obs,)
    if edge_data.isolated is not None and edge_data.isolated.shape[0] > 0:
        denom = denom + exp_V[:, edge_data.isolated].sum(axis=1)
    log_denom = jnp.log(jnp.maximum(denom, 1e-300))

    # --- Log-probabilities via flat alt-edge table (fully vectorised) ---
    e_idx = edge_data.flat_edge_idx
    is_first = edge_data.flat_is_first

    my_term = jnp.where(
        is_first[None, :] > 0,
        term_i[:, e_idx],
        term_j[:, e_idx],
    )
    other_term = jnp.where(
        is_first[None, :] > 0,
        term_j[:, e_idx],
        term_i[:, e_idx],
    )

    log_cond = jnp.log(jnp.maximum(my_term, 1e-300)) - jnp.log(
        jnp.maximum(my_term + other_term, 1e-300)
    )
    log_nest = jnp.log(jnp.maximum(nest_vals[:, e_idx], 1e-300)) - log_denom[:, None]

    contributions = log_cond + log_nest  # (n_obs, total_alt_edges)

    # Scatter contributions to alternatives using segment_sum + logsumexp
    flat_alt_idx = edge_data.flat_alt_idx

    # Numerically stable logsumexp via segment_sum
    max_contrib = jnp.full((n_obs, n_alts), _NEG_INF, dtype=jnp.float64)
    max_contrib = max_contrib.at[:, flat_alt_idx].max(contributions)

    flat_segment_ids = jnp.arange(n_obs)[:, None] * n_alts + flat_alt_idx[None, :]
    flat_segment_ids = flat_segment_ids.ravel()
    flat_exp_contrib = jnp.exp(
        (contributions - max_contrib[:, flat_alt_idx]).ravel()
    )
    flat_sum_exp = segment_sum(flat_exp_contrib, flat_segment_ids, n_obs * n_alts)
    sum_exp = flat_sum_exp.reshape(n_obs, n_alts)

    log_probs = max_contrib + jnp.log(jnp.maximum(sum_exp, 1e-300))

    # Isolated alternatives — vectorised via jnp.where
    if edge_data.isolated is not None and edge_data.isolated.shape[0] > 0:
        # Build a mask for isolated alternatives
        isolated_mask = jnp.zeros(n_alts, dtype=jnp.float64)
        isolated_mask = isolated_mask.at[edge_data.isolated].set(1.0)
        log_probs = jnp.where(
            isolated_mask[None, :] > 0,
            V - log_denom[:, None],
            log_probs,
        )

    # Mask unavailable
    log_probs = jnp.where(available > 0, log_probs, _NEG_INF)

    return log_probs


# ---------------------------------------------------------------------------
# Nested logit kernel
# ---------------------------------------------------------------------------


def nested_log_probs(
    V: jnp.ndarray,
    lambdas: jnp.ndarray,
    nest_matrix: jnp.ndarray,
    available: jnp.ndarray,
) -> jnp.ndarray:
    """Compute nested logit log-probabilities from systematic utilities.

    Parameters
    ----------
    V : jnp.ndarray, shape (n_obs, n_alts)
        Systematic utilities with unavailable alts set to -1e30.
    lambdas : jnp.ndarray, shape (n_nests,)
        Nest dissimilarity parameters in (0, 1].
    nest_matrix : jnp.ndarray, shape (n_alts, n_nests)
        Alternative-to-nest membership matrix.
    available : jnp.ndarray, shape (n_obs, n_alts)
        Binary availability matrix.

    Returns
    -------
    jnp.ndarray, shape (n_obs, n_alts)
        Log-probabilities for each (obs, alt) pair.
    """
    n_obs = V.shape[0]
    n_alts = V.shape[1]
    n_nests = lambdas.shape[0]

    # Mask unavailable
    V = jnp.where(available > 0, V, _NEG_INF)

    # Determine each alternative's nest lambda
    # long_lambda[j] = lambda_m if alt j is in nest m, else 1.0
    in_nest = nest_matrix.sum(axis=1) > 0  # (n_alts,) bool
    long_lambda = jnp.ones(n_alts, dtype=jnp.float64)
    for m in range(n_nests):
        mask_m = nest_matrix[:, m] > 0
        long_lambda = jnp.where(mask_m, lambdas[m], long_lambda)

    # Scaled utilities: V_ij / lambda_m(j)
    scaled_V = V / long_lambda[None, :]  # (n_obs, n_alts)

    # Inclusive values for each nest: IV_m = logsumexp(V_ij / lambda_m for j in C_m)
    # Build nest-specific masks and compute inclusive values
    nest_ivs = []
    for m in range(n_nests):
        mask_m = nest_matrix[:, m] > 0  # (n_alts,)
        # Mask out alternatives not in this nest
        nest_V = jnp.where(mask_m[None, :], scaled_V, _NEG_INF)
        nest_V = jnp.where(available > 0, nest_V, _NEG_INF)
        iv_m = jax_logsumexp(nest_V, axis=1)  # (n_obs,)
        nest_ivs.append(iv_m)
    nest_iv = jnp.stack(nest_ivs, axis=1)  # (n_obs, n_nests)

    # Nest exponents: lambda_m * IV_m
    nest_exponent = lambdas[None, :] * nest_iv  # (n_obs, n_nests)

    # Handle root nest (alternatives not in any nest)
    root_mask = ~in_nest  # (n_alts,) — alternatives not in any nest
    root_V = jnp.where(root_mask[None, :], V, _NEG_INF)
    root_V = jnp.where(available > 0, root_V, _NEG_INF)
    root_iv = jax_logsumexp(root_V, axis=1)  # (n_obs,)

    # Full denominator including root nest
    all_exponents = jnp.column_stack([nest_exponent, root_iv[:, None]])  # (n_obs, n_nests + 1)
    log_denom = jax_logsumexp(all_exponents, axis=1)  # (n_obs,)

    # Compute log-probabilities for each alternative
    log_probs = jnp.full((n_obs, n_alts), _NEG_INF, dtype=jnp.float64)

    for m in range(n_nests):
        mask_m = nest_matrix[:, m] > 0  # (n_alts,)
        iv_m = nest_iv[:, m]  # (n_obs,)
        lambda_m = lambdas[m]

        # log P(j) = V_ij / lambda_m + (lambda_m - 1) * IV_m - log_denom
        log_probs_m = scaled_V + (lambda_m - 1.0) * iv_m[:, None] - log_denom[:, None]
        log_probs = jnp.where(mask_m[None, :], log_probs_m, log_probs)

    # Root nest alternatives: log P(j) = V_ij - log_denom
    log_probs_root = V - log_denom[:, None]
    log_probs = jnp.where(root_mask[None, :], log_probs_root, log_probs)

    # Mask unavailable
    log_probs = jnp.where(available > 0, log_probs, _NEG_INF)

    return log_probs


# ---------------------------------------------------------------------------
# Mixed logit kernel (non-spatial)
# ---------------------------------------------------------------------------


def mixed_logit_ll(
    V_fixed: jnp.ndarray,
    dm_random: jnp.ndarray,
    beta_random_means: jnp.ndarray,
    beta_random_spreads: jnp.ndarray,
    dist_codes: jnp.ndarray,
    draws: jnp.ndarray,
    chosen: jnp.ndarray,
    weights: jnp.ndarray,
    available: jnp.ndarray,
    n_obs: int,
    n_alts: int,
    k_random: int,
    n_draws: int,
) -> jnp.ndarray:
    """Compute mixed logit simulated log-likelihood.

    Uses vmap over draws for vectorised simulation.

    Parameters
    ----------
    V_fixed : jnp.ndarray, shape (n_obs, n_alts)
        Fixed utility component.
    dm_random : jnp.ndarray, shape (n_obs * n_alts, k_random)
        Design matrix columns for random parameters.
    beta_random_means : jnp.ndarray, shape (k_random,)
        Mean coefficients for random parameters.
    beta_random_spreads : jnp.ndarray, shape (k_random,)
        Spread coefficients for random parameters.
    dist_codes : jnp.ndarray, shape (k_random,)
        Integer distribution codes (0=normal, 1=lognormal, 2=triangular, 3=uniform).
    draws : jnp.ndarray, shape (n_obs, n_draws, k_random)
        Standard normal draws for simulated integration.
    chosen : jnp.ndarray, shape (n_obs, n_alts)
        Binary indicator matrix for chosen alternatives.
    weights : jnp.ndarray, shape (n_obs,)
        Observation-level weights.
    available : jnp.ndarray, shape (n_obs, n_alts)
        Binary availability matrix.
    n_obs, n_alts, k_random, n_draws : int
        Problem dimensions.

    Returns
    -------
    jnp.ndarray, scalar
        Simulated log-likelihood.
    """
    # Vectorised random coefficient generation
    means = beta_random_means[None, :]   # (1, k_random)
    spreads = beta_random_spreads[None, :]  # (1, k_random)

    def _ll_single_draw(r):
        """Log-likelihood contribution for a single draw."""
        z_r = draws[:, r, :]  # (n_obs, k_random)

        # Normal: β = μ + σ * z
        beta_normal = means + spreads * z_r
        # Lognormal: β = exp(μ + σ * z)
        beta_lognormal = jnp.exp(jnp.clip(means + spreads * z_r, -50.0, 50.0))
        # Transform standard normal draws to uniform via CDF
        t = 1.0 / (1.0 + 0.2316419 * jnp.abs(z_r))
        d = 0.3989422804014327
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        phi_z = jnp.where(
            z_r >= 0,
            1.0 - d * jnp.exp(-0.5 * z_r * z_r) * poly,
            d * jnp.exp(-0.5 * z_r * z_r) * poly,
        )
        # Uniform on [μ - σ, μ + σ]
        beta_uniform = means + spreads * (2.0 * phi_z - 1.0)
        # Symmetric triangular on [μ - σ, μ + σ]
        # F^{-1}(u) = μ - σ + σ*sqrt(2u)   for u ≤ 0.5
        # F^{-1}(u) = μ + σ - σ*sqrt(2(1-u)) for u > 0.5
        mask = phi_z <= 0.5
        beta_triangular = jnp.where(
            mask,
            means + spreads * (jnp.sqrt(2.0 * phi_z) - 1.0),
            means + spreads * (1.0 - jnp.sqrt(2.0 * (1.0 - phi_z))),
        )

        # Select distribution per parameter — vectorised
        dist = dist_codes[None, :]  # (1, k_random)
        beta_random_r = jnp.where(
            dist == 0, beta_normal,
            jnp.where(dist == 1, beta_lognormal,
                jnp.where(dist == 2, beta_triangular, beta_uniform)),
        )  # (n_obs, k_random)

        # Random utility component
        v_random = jnp.sum(
            dm_random.reshape(n_obs, n_alts, k_random) * beta_random_r[:, None, :],
            axis=2,
        )

        # Total utility
        V = V_fixed + v_random

        # MNL log-probabilities
        log_probs = mnl_log_probs(V, available)

        # Chosen log-probability
        log_L_n = (log_probs * chosen).sum(axis=1)
        return log_L_n

    # vmap over draws
    log_L_all = jax.vmap(_ll_single_draw, in_axes=0)(jnp.arange(n_draws))

    # Simulated log-likelihood
    log_L_sim = jax_logsumexp(log_L_all, axis=0) - jnp.log(float(n_draws))
    return jnp.sum(log_L_sim * weights)


# ---------------------------------------------------------------------------
# Log-likelihood computation (shared across models)
# ---------------------------------------------------------------------------


def compute_ll(
    log_probs: jnp.ndarray,
    chosen: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Compute weighted log-likelihood from log-probabilities.

    Parameters
    ----------
    log_probs : jnp.ndarray, shape (n_obs, n_alts)
        Log-probabilities for each (obs, alt) pair.
    chosen : jnp.ndarray, shape (n_obs, n_alts)
        Binary indicator matrix for chosen alternatives.
    weights : jnp.ndarray, shape (n_obs,)
        Observation-level weights.

    Returns
    -------
    jnp.ndarray, scalar
        Weighted log-likelihood.
    """
    chosen_log_probs = (log_probs * chosen).sum(axis=1)
    return jnp.sum(chosen_log_probs * weights)


def compute_utilities(
    design_matrix: jnp.ndarray,
    beta: jnp.ndarray,
    n_obs: int,
    n_alts: int,
    inclusion_probs: jnp.ndarray | None = None,
    available: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute systematic utilities V = X @ beta with corrections.

    Parameters
    ----------
    design_matrix : jnp.ndarray, shape (n_obs * n_alts, k)
        Design matrix.
    beta : jnp.ndarray, shape (k,)
        Coefficient vector.
    n_obs : int
        Number of observations.
    n_alts : int
        Number of alternatives.
    inclusion_probs : jnp.ndarray or None, shape (n_obs, n_alts)
        Sampling correction rates.
    available : jnp.ndarray or None, shape (n_obs, n_alts)
        Binary availability matrix.

    Returns
    -------
    jnp.ndarray, shape (n_obs, n_alts)
        Systematic utilities with unavailable alts set to -1e30.
    """
    V = (design_matrix @ beta).reshape(n_obs, n_alts)

    if inclusion_probs is not None:
        V = V + jnp.log(jnp.maximum(inclusion_probs, 1e-30))

    if available is not None:
        V = jnp.where(available > 0, V, _NEG_INF)

    return V