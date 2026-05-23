"""Objective builders for choice models using the JAX backend.

These functions construct :class:`~locpick._jax.objective.Objective` instances
from model data, using the pure JAX kernels from :mod:`locpick._jax.kernels`.

Each builder:

1. Converts data to :class:`~locpick._jax.data.ChoiceDataJAX`
2. Constructs a JIT-compiled log-likelihood function using the kernels
3. Constructs a JIT-compiled gradient via ``jax.grad``
4. Wraps everything in an :class:`Objective` with proper transforms

The model classes call these builders instead of building closures inline.
"""

from __future__ import annotations

import functools

import numpy as np

from locpick._compat import _JAX_AVAILABLE
from locpick._jax.data import ChoiceDataJAX

if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as jax_logsumexp
from locpick._jax.kernels import (
    _NEG_INF,
    compute_ll,
    compute_ll_contribs,
    compute_utilities,
    mixed_logit_ll,
    mnl_log_probs,
    nested_log_probs,
    scl_log_probs,
)
from locpick._jax.objective import Objective
from locpick._jax.transforms import Identity, ParamTransform, Sigmoid, SoftPlus

# ---------------------------------------------------------------------------
# MNL objective
# ---------------------------------------------------------------------------


# Top-level JIT'd kernels — cached across all MNL objectives
@functools.partial(jax.jit, static_argnums=(6, 7))
def _mnl_ll_kernel(
    beta, design_matrix, available, chosen, weights, inclusion_probs, n_obs, n_alts
):
    """Pure JAX MNL log-likelihood (top-level for JIT caching)."""
    V = compute_utilities(design_matrix, beta, n_obs, n_alts)
    log_probs = mnl_log_probs(V, available, inclusion_probs=inclusion_probs)
    return compute_ll(log_probs, chosen, weights)


@functools.partial(jax.jit, static_argnums=(6, 7))
def _mnl_ll_contribs_kernel(
    beta, design_matrix, available, chosen, weights, inclusion_probs, n_obs, n_alts
):
    """Per-observation MNL log-likelihood contributions (top-level for JIT caching)."""
    V = compute_utilities(design_matrix, beta, n_obs, n_alts)
    log_probs = mnl_log_probs(V, available, inclusion_probs=inclusion_probs)
    return compute_ll_contribs(log_probs, chosen, weights)


# Pre-compute gradient of the kernel (also cached)
_mnl_grad_kernel = jax.jit(
    jax.grad(_mnl_ll_kernel, argnums=0),
    static_argnums=(6, 7),
)


def build_mnl_objective(arrays) -> Objective:
    """Build an Objective for MNL estimation using JAX.

    Parameters
    ----------
    arrays : ChoiceArrays
        Estimation data arrays.

    Returns
    -------
    Objective
        Objective with JIT-compiled LL, gradient, and Hessian.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX is required for MNL objective")

    data = ChoiceDataJAX.from_arrays(arrays)

    # Thin wrappers — JAX sees the same top-level kernel, so compilation is cached
    def _ll_jax(beta):
        return _mnl_ll_kernel(
            beta,
            data.design_matrix,
            data.available,
            data.chosen,
            data.weights,
            data.inclusion_probs,
            data.n_obs,
            data.n_alts,
        )

    def _ll_contribs_jax(beta):
        return _mnl_ll_contribs_kernel(
            beta,
            data.design_matrix,
            data.available,
            data.chosen,
            data.weights,
            data.inclusion_probs,
            data.n_obs,
            data.n_alts,
        )

    def _grad_jax(beta):
        return _mnl_grad_kernel(
            beta,
            data.design_matrix,
            data.available,
            data.chosen,
            data.weights,
            data.inclusion_probs,
            data.n_obs,
            data.n_alts,
        )

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        loglike_contribs_jax=_ll_contribs_jax,
        param_names=list(arrays.param_names),
    )


# ---------------------------------------------------------------------------
# SCL objective
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SCL objective
# ---------------------------------------------------------------------------


# Top-level JIT'd kernels — cached across all SCL objectives
@jax.jit
def _scl_ll_kernel(params, data):
    """Pure JAX SCL log-likelihood (top-level for JIT caching)."""
    k = data.design_matrix.shape[1]
    beta = params[:k]
    alpha_rho = params[k]
    rho = 1.0 / (1.0 + jnp.exp(-alpha_rho))

    V = compute_utilities(
        data.design_matrix,
        beta,
        data.n_obs,
        data.n_alts,
        inclusion_probs=data.inclusion_probs,
        available=data.available,
    )
    log_probs = scl_log_probs(V, rho, data.edge_data, data.available)
    return compute_ll(log_probs, data.chosen, data.weights)


@jax.jit
def _scl_ll_contribs_kernel(params, data):
    """Per-observation SCL log-likelihood contributions (top-level for JIT caching)."""
    k = data.design_matrix.shape[1]
    beta = params[:k]
    alpha_rho = params[k]
    rho = 1.0 / (1.0 + jnp.exp(-alpha_rho))

    V = compute_utilities(
        data.design_matrix,
        beta,
        data.n_obs,
        data.n_alts,
        inclusion_probs=data.inclusion_probs,
        available=data.available,
    )
    log_probs = scl_log_probs(V, rho, data.edge_data, data.available)
    return compute_ll_contribs(log_probs, data.chosen, data.weights)


# Pre-compute gradient of the kernel (also cached)
_scl_grad_kernel = jax.jit(jax.grad(_scl_ll_kernel, argnums=0))


def build_scl_objective(arrays, edge_struct, allocation, edge_list) -> Objective:
    """Build an Objective for SCL estimation using JAX.

    Parameters
    ----------
    arrays : ChoiceArrays
        Estimation data arrays.
    edge_struct : EdgeStructure
        Precomputed edge structure.
    allocation : np.ndarray
        Allocation matrix.
    edge_list : list of (int, int)
        Paired-nest edges.

    Returns
    -------
    Objective
        Objective with JIT-compiled LL, gradient, and Hessian.
        Includes a Sigmoid transform for the rho parameter.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX is required for SCL objective")

    data = ChoiceDataJAX.from_arrays(arrays, edge_struct=edge_struct)
    k = arrays.design_matrix.shape[1]

    # Thin wrappers — JAX sees the same top-level kernel, so compilation is cached
    def _ll_jax(params):
        return _scl_ll_kernel(params, data)

    def _ll_contribs_jax(params):
        return _scl_ll_contribs_kernel(params, data)

    def _grad_jax(params):
        return _scl_grad_kernel(params, data)

    param_names = list(arrays.param_names) + ["rho"]
    transform = ParamTransform.for_scl(k)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        loglike_contribs_jax=_ll_contribs_jax,
        param_names=param_names,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# MSCL objective
# ---------------------------------------------------------------------------


# Top-level JIT'd kernels — cached across all MSCL objectives
@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def _mscl_ll_kernel(params, data, k_fixed, k_random, n_draws):
    """Pure JAX MSCL simulated log-likelihood (top-level for JIT caching)."""
    from jax.scipy.special import logsumexp as jax_logsumexp

    beta_fixed = params[:k_fixed]
    alpha_rho = params[k_fixed]
    beta_random_means = params[k_fixed + 1 : k_fixed + 1 + k_random]
    beta_random_spreads_raw = params[k_fixed + 1 + k_random :]

    rho = 1.0 / (1.0 + jnp.exp(-alpha_rho))
    # Enforce non-negative spreads via softplus
    beta_random_spreads = jnp.log1p(jnp.exp(beta_random_spreads_raw))

    # Fixed utility component
    if data.dm_fixed is not None and k_fixed > 0:
        v_fixed = (data.dm_fixed @ beta_fixed).reshape(data.n_obs, data.n_alts)
    else:
        v_fixed = jnp.zeros((data.n_obs, data.n_alts), dtype=jnp.float64)

    # Sampling correction
    if data.inclusion_probs is not None:
        v_fixed = v_fixed + jnp.log(jnp.maximum(data.inclusion_probs, 1e-30))

    # Simulated log-likelihood via vmap over draws
    def _ll_single_draw(r):
        """Log-likelihood contribution for a single draw."""
        z_r = data.draws[:, r, :]  # (n_obs, k_random)

        # Vectorised random coefficient generation
        means = beta_random_means[None, :]  # (1, k_random)
        spreads = beta_random_spreads[None, :]  # (1, k_random)

        # Normal: β = μ + σ * z
        beta_normal = means + spreads * z_r
        # Lognormal: β = exp(μ + σ * z)
        beta_lognormal = jnp.exp(jnp.clip(means + spreads * z_r, -50.0, 50.0))
        # Uniform on [μ - σ, μ + σ]: transform standard normal CDF to U(-1,1)
        t = 1.0 / (1.0 + 0.2316419 * jnp.abs(z_r))
        d = 0.3989422804014327
        poly = t * (
            0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
        )
        phi_z = jnp.where(
            z_r >= 0,
            1.0 - d * jnp.exp(-0.5 * z_r * z_r) * poly,
            d * jnp.exp(-0.5 * z_r * z_r) * poly,
        )
        # Uniform on [μ - σ, μ + σ]
        beta_uniform = means + spreads * (2.0 * phi_z - 1.0)
        # Symmetric triangular on [μ - σ, μ + σ]
        mask = phi_z <= 0.5
        beta_triangular = jnp.where(
            mask,
            means + spreads * (jnp.sqrt(2.0 * phi_z) - 1.0),
            means + spreads * (1.0 - jnp.sqrt(2.0 * (1.0 - phi_z))),
        )

        # Select distribution per parameter — vectorised via jnp.where
        dist = data.dist_codes[None, :]  # (1, k_random)
        beta_random_r = jnp.where(
            dist == 0,
            beta_normal,
            jnp.where(
                dist == 1, beta_lognormal, jnp.where(dist == 2, beta_triangular, beta_uniform)
            ),
        )  # (n_obs, k_random)

        # Random utility component
        v_random = jnp.sum(
            data.dm_random.reshape(data.n_obs, data.n_alts, k_random) * beta_random_r[:, None, :],
            axis=2,
        )

        # Total utility
        V = v_fixed + v_random

        # SCL log-probabilities
        log_probs = scl_log_probs(V, rho, data.edge_data, data.available)

        # Chosen log-probability
        log_L_n = (log_probs * data.chosen).sum(axis=1)
        return log_L_n

    # vmap over draws
    log_L_all = jax.vmap(_ll_single_draw, in_axes=0)(jnp.arange(n_draws))

    # Simulated log-likelihood
    log_L_sim = jax_logsumexp(log_L_all, axis=0) - jnp.log(float(n_draws))
    return jnp.sum(log_L_sim * data.weights)


# Pre-compute gradient of the kernel (also cached)
_mscl_grad_kernel = jax.jit(
    jax.grad(_mscl_ll_kernel, argnums=0),
    static_argnums=(2, 3, 4),
)


# ---------------------------------------------------------------------------
# Mixed Nested SCL (MNSCL) objective
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(5, 6, 7, 8))
def _mnscl_ll_kernel(
    params,
    data,
    nest_matrix,
    edge_data_list,
    k,
    n_nests,
    k_fixed,
    k_random,
    n_draws,
    nest_alt_indices,
):
    """Pure JAX MNSCL simulated log-likelihood (top-level for JIT caching).

    Parameters
    ----------
    params : jnp.ndarray
        [beta_fixed, alpha_rho_1..M, alpha_lambda_1..M, beta_random_means, beta_random_spreads]
    data : ChoiceDataJAX
    nest_matrix : jnp.ndarray, shape (n_alts, n_nests)
    edge_data_list : list of EdgeDataJAX
        One per nest, containing the subgraph for that nest.
    k : int
        Number of utility coefficients (static arg).
    n_nests : int
        Number of nests (static arg).
    k_fixed : int
        Number of fixed coefficients (static arg).
    k_random : int
        Number of random coefficients (static arg).
    n_draws : int
        Number of simulation draws (static arg).
    nest_alt_indices : tuple of tuple of int
        Precomputed nest alt indices.
    """
    from locpick._jax.kernels import scl_log_probs_and_inclusive_value

    beta_fixed = params[:k_fixed]
    alpha_rhos = params[k_fixed : k_fixed + n_nests]
    alpha_lambdas = params[k_fixed + n_nests : k_fixed + 2 * n_nests]
    beta_random_means = params[k_fixed + 2 * n_nests : k_fixed + 2 * n_nests + k_random]
    beta_random_spreads_raw = params[k_fixed + 2 * n_nests + k_random :]

    rhos = 1.0 / (1.0 + jnp.exp(-alpha_rhos))
    lambdas = 1.0 / (1.0 + jnp.exp(-alpha_lambdas))
    beta_random_spreads = jnp.log1p(jnp.exp(beta_random_spreads_raw))

    # Fixed utility component
    if data.dm_fixed is not None and k_fixed > 0:
        v_fixed = (data.dm_fixed @ beta_fixed).reshape(data.n_obs, data.n_alts)
    else:
        v_fixed = jnp.zeros((data.n_obs, data.n_alts), dtype=jnp.float64)

    # Sampling correction
    if data.inclusion_probs is not None:
        v_fixed = v_fixed + jnp.log(jnp.maximum(data.inclusion_probs, 1e-30))

    def _ll_single_draw(r):
        """Log-likelihood contribution for a single draw."""
        z_r = data.draws[:, r, :]  # (n_obs, k_random)

        means = beta_random_means[None, :]  # (1, k_random)
        spreads = beta_random_spreads[None, :]  # (1, k_random)

        # Normal: β = μ + σ * z
        beta_normal = means + spreads * z_r
        # Lognormal: β = exp(μ + σ * z)
        beta_lognormal = jnp.exp(jnp.clip(means + spreads * z_r, -50.0, 50.0))
        # Uniform on [μ - σ, μ + σ]: transform standard normal CDF to U(-1,1)
        t = 1.0 / (1.0 + 0.2316419 * jnp.abs(z_r))
        d = 0.3989422804014327
        poly = t * (
            0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
        )
        phi_z = jnp.where(
            z_r >= 0,
            1.0 - d * jnp.exp(-0.5 * z_r * z_r) * poly,
            d * jnp.exp(-0.5 * z_r * z_r) * poly,
        )
        beta_uniform = means + spreads * (2.0 * phi_z - 1.0)
        mask = phi_z <= 0.5
        beta_triangular = jnp.where(
            mask,
            means + spreads * (jnp.sqrt(2.0 * phi_z) - 1.0),
            means + spreads * (1.0 - jnp.sqrt(2.0 * (1.0 - phi_z))),
        )

        dist = data.dist_codes[None, :]  # (1, k_random)
        beta_random_r = jnp.where(
            dist == 0,
            beta_normal,
            jnp.where(
                dist == 1, beta_lognormal, jnp.where(dist == 2, beta_triangular, beta_uniform)
            ),
        )  # (n_obs, k_random)

        v_random = jnp.sum(
            data.dm_random.reshape(data.n_obs, data.n_alts, k_random) * beta_random_r[:, None, :],
            axis=2,
        )

        V = v_fixed + v_random

        # Per-nest SCL probabilities and inclusive values
        nest_log_G = jnp.zeros((data.n_obs, n_nests), dtype=jnp.float64)
        log_probs_full = jnp.full((data.n_obs, data.n_alts), _NEG_INF, dtype=jnp.float64)

        for m in range(n_nests):
            nest_alts = nest_alt_indices[m]
            n_nest_alts = len(nest_alts)

            if n_nest_alts == 0:
                continue

            V_m = jnp.column_stack([V[:, alt] for alt in nest_alts])

            if data.available is not None:
                avail_m = jnp.column_stack([data.available[:, alt] for alt in nest_alts])
            else:
                avail_m = jnp.ones((data.n_obs, n_nest_alts), dtype=jnp.float64)

            log_probs_m, log_G_m = scl_log_probs_and_inclusive_value(
                V_m, rhos[m], edge_data_list[m], avail_m
            )

            nest_log_G = nest_log_G.at[:, m].set(log_G_m)

            for idx, alt_global in enumerate(nest_alts):
                log_probs_full = log_probs_full.at[:, alt_global].set(log_probs_m[:, idx])

        # Top-level NL: compute nest probabilities
        nest_exponents = lambdas[None, :] * nest_log_G  # (n_obs, n_nests)
        log_denom_top = jax_logsumexp(nest_exponents, axis=1)  # (n_obs,)
        log_P_nest = nest_exponents - log_denom_top[:, None]  # (n_obs, n_nests)

        # Combine: P_i = P_SCL(i|m) * P_NL(m)
        for m in range(n_nests):
            nest_alts = nest_alt_indices[m]
            for alt_global in nest_alts:
                old_log_prob = log_probs_full[:, alt_global]
                new_log_prob = old_log_prob + log_P_nest[:, m]
                log_probs_full = log_probs_full.at[:, alt_global].set(new_log_prob)

        # Chosen log-probability
        log_L_n = (log_probs_full * data.chosen).sum(axis=1)
        return log_L_n

    # vmap over draws
    log_L_all = jax.vmap(_ll_single_draw, in_axes=0)(jnp.arange(n_draws))

    # Simulated log-likelihood
    log_L_sim = jax_logsumexp(log_L_all, axis=0) - jnp.log(float(n_draws))
    return jnp.sum(log_L_sim * data.weights)


# Pre-compute gradient of the kernel (also cached)
_mnscl_grad_kernel = jax.jit(
    jax.grad(_mnscl_ll_kernel, argnums=0),
    static_argnums=(5, 6, 7, 8, 9),
)


def build_mnscl_objective(
    arrays,
    nest_matrix,
    edge_data_list,
    random_col_indices,
    random_distributions,
    draws,
) -> Objective:
    """Build an Objective for MNSCL estimation using JAX.

    Parameters
    ----------
    arrays : ChoiceArrays
        Estimation data arrays.
    nest_matrix : np.ndarray, shape (n_alts, n_nests)
        Alternative-to-nest membership matrix.
    edge_data_list : list of EdgeDataJAX
        One per nest, containing the subgraph for that nest.
    random_col_indices : list[int]
        Column indices of random parameters.
    random_distributions : list[str]
        Distribution names for random parameters.
    draws : np.ndarray
        Simulation draws, shape (n_obs, n_draws, k_random).

    Returns
    -------
    Objective
        Objective with JIT-compiled LL, gradient, and Hessian.
        Includes Sigmoid for rho/lambda and SoftPlus for random spreads.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX is required for MNSCL objective")

    data = ChoiceDataJAX.from_arrays(
        arrays,
        edge_struct=None,
        draws=draws,
        random_col_indices=random_col_indices,
        random_distributions=random_distributions,
    )

    nest_matrix_jax = jnp.asarray(nest_matrix, dtype=jnp.float64)
    n_nests = nest_matrix.shape[1]
    k = arrays.design_matrix.shape[1]
    k_random = len(random_col_indices)
    k_fixed = k - k_random
    n_draws = draws.shape[1]

    # Precompute nest alt indices (static, known at compile time)
    nest_alt_indices = tuple(
        tuple(int(i) for i in np.where(nest_matrix[:, m] > 0)[0]) for m in range(n_nests)
    )

    def _ll_jax(params):
        return _mnscl_ll_kernel(
            params,
            data,
            nest_matrix_jax,
            edge_data_list,
            k,
            n_nests,
            k_fixed,
            k_random,
            n_draws,
            nest_alt_indices,
        )

    def _grad_jax(params):
        return _mnscl_grad_kernel(
            params,
            data,
            nest_matrix_jax,
            edge_data_list,
            k,
            n_nests,
            k_fixed,
            k_random,
            n_draws,
            nest_alt_indices,
        )

    param_names_list = list(arrays.param_names)
    fixed_param_names = [
        name
        for name in param_names_list
        if name not in [param_names_list[i] for i in random_col_indices]
    ]
    random_param_names = [param_names_list[i] for i in random_col_indices]
    param_names = (
        fixed_param_names
        + [f"rho_{i}" for i in range(n_nests)]
        + [f"lambda_{i}" for i in range(n_nests)]
        + [f"mean_{name}" for name in random_param_names]
        + [f"sd_{name}" for name in random_param_names]
    )

    # Transforms: identity for beta, sigmoid for rho/lambda, softplus for spreads
    transforms = (
        [Identity() for _ in range(k_fixed)]
        + [Sigmoid() for _ in range(n_nests)]
        + [Sigmoid() for _ in range(n_nests)]
        + [Identity() for _ in range(k_random)]
        + [SoftPlus() for _ in range(k_random)]
    )
    transform = ParamTransform(transforms=transforms)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        param_names=param_names,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# MSCL objective builder
# ---------------------------------------------------------------------------


def build_mscl_objective(
    arrays,
    edge_struct,
    allocation,
    edge_list,
    random_col_indices,
    random_distributions,
    draws,
) -> Objective:
    """Build an Objective for MSCL estimation using JAX.

    Parameters
    ----------
    arrays : ChoiceArrays
        Estimation data arrays.
    edge_struct : EdgeStructure
        Precomputed edge structure.
    allocation : np.ndarray
        Allocation matrix.
    edge_list : list of (int, int)
        Paired-nest edges.
    random_col_indices : list[int]
        Column indices of random parameters.
    random_distributions : list[str]
        Distribution names for random parameters.
    draws : np.ndarray
        Simulation draws, shape (n_obs, n_draws, k_random).

    Returns
    -------
    Objective
        Objective with JIT-compiled LL, gradient, and Hessian.
        Includes Sigmoid for rho and SoftPlus for random spreads.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX is required for MSCL objective")

    data = ChoiceDataJAX.from_arrays(
        arrays,
        edge_struct=edge_struct,
        draws=draws,
        random_col_indices=random_col_indices,
        random_distributions=random_distributions,
    )

    k_fixed = len(data.fixed_col_indices) if data.fixed_col_indices else 0
    k_random = len(random_col_indices)
    n_draws = draws.shape[1]

    # Thin wrappers — JAX sees the same top-level kernel, so compilation is cached
    def _ll_jax(params):
        return _mscl_ll_kernel(params, data, k_fixed, k_random, n_draws)

    def _grad_jax(params):
        return _mscl_grad_kernel(params, data, k_fixed, k_random, n_draws)

    # Parameter names
    param_names_list = list(arrays.param_names)
    fixed_param_names = [
        name
        for name in param_names_list
        if name not in [param_names_list[i] for i in random_col_indices]
    ]
    random_param_names = [param_names_list[i] for i in random_col_indices]
    display_param_names = (
        fixed_param_names
        + ["rho"]
        + [f"mean_{name}" for name in random_param_names]
        + [f"sd_{name}" for name in random_param_names]
    )

    transform = ParamTransform.for_mscl(k_fixed, k_random)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        param_names=display_param_names,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# Nested logit objective
# ---------------------------------------------------------------------------


# Top-level JIT'd kernels — cached across all nested logit objectives
@functools.partial(jax.jit, static_argnums=(3,))
def _nested_ll_kernel(params, data, nest_matrix, k):
    """Pure JAX nested logit log-likelihood (top-level for JIT caching)."""
    beta = params[:k]
    alpha = params[k:]
    lambdas = 1.0 / (1.0 + jnp.exp(-alpha))

    V = compute_utilities(
        data.design_matrix,
        beta,
        data.n_obs,
        data.n_alts,
        inclusion_probs=data.inclusion_probs,
        available=data.available,
    )
    log_probs = nested_log_probs(V, lambdas, nest_matrix, data.available)
    return compute_ll(log_probs, data.chosen, data.weights)


# Pre-compute gradient of the kernel (also cached)
_nested_grad_kernel = jax.jit(
    jax.grad(_nested_ll_kernel, argnums=0),
    static_argnums=(3,),
)


def build_nested_objective(arrays, nest_matrix) -> Objective:
    """Build an Objective for nested logit estimation using JAX.

    Parameters
    ----------
    arrays : ChoiceArrays
        Estimation data arrays.
    nest_matrix : np.ndarray, shape (n_alts, n_nests)
        Alternative-to-nest membership matrix.

    Returns
    -------
    Objective
        Objective with JIT-compiled LL, gradient, and Hessian.
        Includes Sigmoid transforms for nest dissimilarity parameters.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX is required for nested logit objective")

    data = ChoiceDataJAX.from_arrays(arrays)
    nest_matrix_jax = jnp.asarray(nest_matrix, dtype=jnp.float64)
    n_nests = nest_matrix.shape[1]
    k = arrays.design_matrix.shape[1]

    # Thin wrappers — JAX sees the same top-level kernel, so compilation is cached
    def _ll_jax(params):
        return _nested_ll_kernel(params, data, nest_matrix_jax, k)

    def _grad_jax(params):
        return _nested_grad_kernel(params, data, nest_matrix_jax, k)

    param_names = list(arrays.param_names) + [f"nest_alpha_{i}" for i in range(n_nests)]

    # Sigmoid transform for nest parameters, identity for beta
    transforms = [Identity() for _ in range(k)] + [Sigmoid() for _ in range(n_nests)]
    transform = ParamTransform(transforms=transforms)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        param_names=param_names,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# Nested SCL objective
# ---------------------------------------------------------------------------


# Top-level JIT'd kernels — cached across all nested SCL objectives
@functools.partial(jax.jit, static_argnums=(4, 5))
def _nested_scl_ll_kernel(params, data, nest_matrix, edge_data_list, k, nest_alt_indices):
    """Pure JAX nested SCL log-likelihood (top-level for JIT caching).

    Parameters
    ----------
    params : jnp.ndarray
        [beta, alpha_rho_1, ..., alpha_rho_M, alpha_lambda_1, ..., alpha_lambda_M]
    data : ChoiceDataJAX
    nest_matrix : jnp.ndarray, shape (n_alts, n_nests)
    edge_data_list : list of EdgeDataJAX
        One per nest, containing the subgraph for that nest.
    k : int
        Number of utility coefficients (static arg).
    nest_alt_indices : tuple of tuple of int
        Precomputed nest alt indices: ``nest_alt_indices[m][i]`` = global alt index
        of the i-th alternative in nest m.
    """
    from locpick._jax.kernels import scl_log_probs_and_inclusive_value

    beta = params[:k]
    n_nests = nest_matrix.shape[1]
    alpha_rhos = params[k : k + n_nests]
    alpha_lambdas = params[k + n_nests :]

    rhos = 1.0 / (1.0 + jnp.exp(-alpha_rhos))
    lambdas = 1.0 / (1.0 + jnp.exp(-alpha_lambdas))

    V = compute_utilities(
        data.design_matrix,
        beta,
        data.n_obs,
        data.n_alts,
        inclusion_probs=data.inclusion_probs,
        available=data.available,
    )

    # Per-nest SCL probabilities and inclusive values
    nest_log_G = jnp.zeros((data.n_obs, n_nests), dtype=jnp.float64)
    log_probs_full = jnp.full((data.n_obs, data.n_alts), _NEG_INF, dtype=jnp.float64)

    for m in range(n_nests):
        nest_alts = nest_alt_indices[m]
        n_nest_alts = len(nest_alts)

        if n_nest_alts == 0:
            continue

        # Extract utilities for this nest using static indices
        V_m = jnp.column_stack([V[:, alt] for alt in nest_alts])

        # Build availability for this nest
        if data.available is not None:
            avail_m = jnp.column_stack([data.available[:, alt] for alt in nest_alts])
        else:
            avail_m = jnp.ones((data.n_obs, n_nest_alts), dtype=jnp.float64)

        # Call SCL kernel for this nest
        log_probs_m, log_G_m = scl_log_probs_and_inclusive_value(
            V_m, rhos[m], edge_data_list[m], avail_m
        )

        # Store inclusive value
        nest_log_G = nest_log_G.at[:, m].set(log_G_m)

        # Scatter probabilities back to full array
        for idx, alt_global in enumerate(nest_alts):
            log_probs_full = log_probs_full.at[:, alt_global].set(log_probs_m[:, idx])

    # Top-level NL: compute nest probabilities
    nest_exponents = lambdas[None, :] * nest_log_G  # (n_obs, n_nests)
    log_denom_top = jax_logsumexp(nest_exponents, axis=1)  # (n_obs,)
    log_P_nest = nest_exponents - log_denom_top[:, None]  # (n_obs, n_nests)

    # Combine: P_i = P_SCL(i|m) * P_NL(m)
    for m in range(n_nests):
        nest_alts = nest_alt_indices[m]
        for alt_global in nest_alts:
            old_log_prob = log_probs_full[:, alt_global]
            new_log_prob = old_log_prob + log_P_nest[:, m]
            log_probs_full = log_probs_full.at[:, alt_global].set(new_log_prob)

    return compute_ll(log_probs_full, data.chosen, data.weights)


# Pre-compute gradient of the kernel (also cached)
_nested_scl_grad_kernel = jax.jit(
    jax.grad(_nested_scl_ll_kernel, argnums=0),
    static_argnums=(4, 5),
)


def build_nested_scl_objective(arrays, nest_matrix, edge_data_list) -> Objective:
    """Build an Objective for nested SCL estimation using JAX.

    Parameters
    ----------
    arrays : ChoiceArrays
        Estimation data arrays.
    nest_matrix : np.ndarray, shape (n_alts, n_nests)
        Alternative-to-nest membership matrix.
    edge_data_list : list of EdgeDataJAX
        One per nest, containing the subgraph for that nest.

    Returns
    -------
    Objective
        Objective with JIT-compiled LL, gradient, and Hessian.
        Includes Sigmoid transforms for rho and lambda parameters.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX is required for nested SCL objective")

    data = ChoiceDataJAX.from_arrays(arrays)
    nest_matrix_jax = jnp.asarray(nest_matrix, dtype=jnp.float64)
    n_nests = nest_matrix.shape[1]
    k = arrays.design_matrix.shape[1]

    # Precompute nest alt indices (static, known at compile time)
    nest_alt_indices = tuple(
        tuple(int(i) for i in np.where(nest_matrix[:, m] > 0)[0]) for m in range(n_nests)
    )

    # Thin wrappers — JAX sees the same top-level kernel, so compilation is cached
    def _ll_jax(params):
        return _nested_scl_ll_kernel(
            params, data, nest_matrix_jax, edge_data_list, k, nest_alt_indices
        )

    def _grad_jax(params):
        return _nested_scl_grad_kernel(
            params, data, nest_matrix_jax, edge_data_list, k, nest_alt_indices
        )

    param_names = (
        list(arrays.param_names)
        + [f"rho_{i}" for i in range(n_nests)]
        + [f"nest_lambda_{i}" for i in range(n_nests)]
    )

    # Sigmoid transform for rho and lambda, identity for beta
    transforms = (
        [Identity() for _ in range(k)]
        + [Sigmoid() for _ in range(n_nests)]
        + [Sigmoid() for _ in range(n_nests)]
    )
    transform = ParamTransform(transforms=transforms)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        param_names=param_names,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# Mixed logit objective (non-spatial)
# ---------------------------------------------------------------------------


# Top-level JIT'd kernels — cached across all mixed logit objectives
@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def _mixed_ll_kernel(params, data, k_fixed, k_random, n_draws):
    """Pure JAX mixed logit simulated log-likelihood (top-level for JIT caching)."""
    beta_fixed = params[:k_fixed]
    beta_random_means = params[k_fixed : k_fixed + k_random]
    beta_random_spreads_raw = params[k_fixed + k_random :]

    # Enforce non-negative spreads via softplus
    beta_random_spreads = jnp.log1p(jnp.exp(beta_random_spreads_raw))

    # Fixed utility component
    if data.dm_fixed is not None and k_fixed > 0:
        v_fixed = (data.dm_fixed @ beta_fixed).reshape(data.n_obs, data.n_alts)
    else:
        v_fixed = jnp.zeros((data.n_obs, data.n_alts), dtype=jnp.float64)

    # Sampling correction
    if data.inclusion_probs is not None:
        v_fixed = v_fixed + jnp.log(jnp.maximum(data.inclusion_probs, 1e-30))

    return mixed_logit_ll(
        V_fixed=v_fixed,
        dm_random=data.dm_random,
        beta_random_means=beta_random_means,
        beta_random_spreads=beta_random_spreads,
        dist_codes=data.dist_codes,
        draws=data.draws,
        chosen=data.chosen,
        weights=data.weights,
        available=data.available,
        n_obs=data.n_obs,
        n_alts=data.n_alts,
        k_random=k_random,
        n_draws=n_draws,
    )


# Pre-compute gradient of the kernel (also cached)
_mixed_grad_kernel = jax.jit(
    jax.grad(_mixed_ll_kernel, argnums=0),
    static_argnums=(2, 3, 4),
)


def build_mixed_logit_objective(
    arrays,
    random_col_indices,
    random_distributions,
    draws,
) -> Objective:
    """Build an Objective for mixed logit estimation using JAX.

    Parameters
    ----------
    arrays : ChoiceArrays
        Estimation data arrays.
    random_col_indices : list[int]
        Column indices of random parameters in the design matrix.
    random_distributions : list[str]
        Distribution names for random parameters ('normal', 'lognormal',
        'triangular', 'uniform').
    draws : np.ndarray, shape (n_obs, n_draws, k_random)
        Standard normal draws for simulated integration.

    Returns
    -------
    Objective
        Objective with JIT-compiled LL, gradient, and Hessian.
        Includes SoftPlus transforms for random parameter spreads.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX is required for mixed logit objective")

    data = ChoiceDataJAX.from_arrays(
        arrays,
        draws=draws,
        random_col_indices=random_col_indices,
        random_distributions=random_distributions,
    )

    k_fixed = len(data.fixed_col_indices) if data.fixed_col_indices else 0
    k_random = len(random_col_indices)
    n_draws = draws.shape[1]

    # Thin wrappers — JAX sees the same top-level kernel, so compilation is cached
    def _ll_jax(params):
        return _mixed_ll_kernel(params, data, k_fixed, k_random, n_draws)

    def _grad_jax(params):
        return _mixed_grad_kernel(params, data, k_fixed, k_random, n_draws)

    # Parameter names
    param_names_list = list(arrays.param_names)
    fixed_param_names = [
        name for i, name in enumerate(param_names_list) if i not in random_col_indices
    ]
    random_param_names = [param_names_list[i] for i in random_col_indices]
    display_param_names = (
        fixed_param_names
        + [f"mean_{name}" for name in random_param_names]
        + [f"sd_{name}" for name in random_param_names]
    )

    # Transforms: identity for fixed, identity for means, softplus for spreads
    transforms = (
        [Identity() for _ in range(k_fixed)]
        + [Identity() for _ in range(k_random)]
        + [Identity() for _ in range(k_random)]  # softplus applied inside _ll_jax
    )
    transform = ParamTransform(transforms=transforms)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        param_names=display_param_names,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# Mixed Nested Logit objective
# ---------------------------------------------------------------------------


# Top-level JIT'd kernels — cached across all mixed nested objectives
@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def _mixed_nested_ll_kernel(
    params,
    data,
    nest_matrix,
    k_fixed,
    n_nests,
    k_random,
    n_draws,
    nest_alt_indices,
):
    """Pure JAX mixed nested logit simulated log-likelihood (top-level for JIT caching).

    Parameters
    ----------
    params : jnp.ndarray
        [beta_fixed, alpha_nest_1..M, beta_random_means, beta_random_spreads_raw]
    data : ChoiceDataJAX
    nest_matrix : jnp.ndarray, shape (n_alts, n_nests)
    k_fixed : int
        Number of fixed utility coefficients (static arg).
    n_nests : int
        Number of nests (static arg).
    k_random : int
        Number of random coefficients (static arg).
    n_draws : int
        Number of simulation draws (static arg).
    nest_alt_indices : tuple of tuple of int
        Precomputed nest alt indices.
    """
    from locpick._jax.kernels import mixed_nested_logit_ll

    beta_fixed = params[:k_fixed]
    alpha_nest = params[k_fixed : k_fixed + n_nests]
    beta_random_means = params[k_fixed + n_nests : k_fixed + n_nests + k_random]
    beta_random_spreads_raw = params[k_fixed + n_nests + k_random :]

    # Naturalize nest parameters: alpha -> lambda
    lambdas = 1.0 / (1.0 + jnp.exp(-alpha_nest))

    # Enforce non-negative spreads via softplus
    beta_random_spreads = jnp.log1p(jnp.exp(beta_random_spreads_raw))

    # Fixed utility component
    if data.dm_fixed is not None and k_fixed > 0:
        v_fixed = (data.dm_fixed @ beta_fixed).reshape(data.n_obs, data.n_alts)
    else:
        v_fixed = jnp.zeros((data.n_obs, data.n_alts), dtype=jnp.float64)

    # Sampling correction
    if data.inclusion_probs is not None:
        v_fixed = v_fixed + jnp.log(jnp.maximum(data.inclusion_probs, 1e-30))

    return mixed_nested_logit_ll(
        V_fixed=v_fixed,
        dm_random=data.dm_random,
        beta_random_means=beta_random_means,
        beta_random_spreads=beta_random_spreads,
        dist_codes=data.dist_codes,
        draws=data.draws,
        lambdas=lambdas,
        nest_matrix=nest_matrix,
        chosen=data.chosen,
        weights=data.weights,
        available=data.available,
        n_obs=data.n_obs,
        n_alts=data.n_alts,
        k_random=k_random,
        n_draws=n_draws,
        n_nests=n_nests,
    )


# Pre-compute gradient of the kernel (also cached)
_mixed_nested_grad_kernel = jax.jit(
    jax.grad(_mixed_nested_ll_kernel, argnums=0),
    static_argnums=(3, 4, 5, 6, 7),
)


def build_mixed_nested_objective(
    arrays,
    nest_matrix,
    random_col_indices,
    random_distributions,
    draws,
) -> Objective:
    """Build an Objective for mixed nested logit estimation using JAX.

    Parameters
    ----------
    arrays : ChoiceArrays
        Estimation data arrays.
    nest_matrix : np.ndarray, shape (n_alts, n_nests)
        Alternative-to-nest membership matrix.
    random_col_indices : list[int]
        Column indices of random parameters in the design matrix.
    random_distributions : list[str]
        Distribution names for random parameters ('normal', 'lognormal',
        'triangular', 'uniform').
    draws : np.ndarray, shape (n_obs, n_draws, k_random)
        Standard normal draws for simulated integration.

    Returns
    -------
    Objective
        Objective with JIT-compiled LL, gradient, and Hessian.
        Includes Sigmoid transforms for nest parameters and SoftPlus
        for random parameter spreads.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX is required for mixed nested logit objective")

    data = ChoiceDataJAX.from_arrays(
        arrays,
        draws=draws,
        random_col_indices=random_col_indices,
        random_distributions=random_distributions,
    )

    nest_matrix_jax = jnp.asarray(nest_matrix, dtype=jnp.float64)
    n_nests = nest_matrix.shape[1]
    k = arrays.design_matrix.shape[1]
    k_random = len(random_col_indices)
    k_fixed = k - k_random
    n_draws = draws.shape[1]

    # Precompute nest alt indices (static, known at compile time)
    nest_alt_indices = tuple(
        tuple(int(i) for i in np.where(nest_matrix[:, m] > 0)[0]) for m in range(n_nests)
    )

    def _ll_jax(params):
        return _mixed_nested_ll_kernel(
            params,
            data,
            nest_matrix_jax,
            k_fixed,
            n_nests,
            k_random,
            n_draws,
            nest_alt_indices,
        )

    def _grad_jax(params):
        return _mixed_nested_grad_kernel(
            params,
            data,
            nest_matrix_jax,
            k_fixed,
            n_nests,
            k_random,
            n_draws,
            nest_alt_indices,
        )

    param_names_list = list(arrays.param_names)
    fixed_param_names = [
        name
        for name in param_names_list
        if name not in [param_names_list[i] for i in random_col_indices]
    ]
    random_param_names = [param_names_list[i] for i in random_col_indices]
    display_param_names = (
        fixed_param_names
        + [f"lambda_{i}" for i in range(n_nests)]
        + [f"mean_{name}" for name in random_param_names]
        + [f"sd_{name}" for name in random_param_names]
    )

    # Transforms: identity for fixed beta, sigmoid for nest params, identity for means, softplus for spreads
    transforms = (
        [Identity() for _ in range(k_fixed)]
        + [Sigmoid() for _ in range(n_nests)]
        + [Identity() for _ in range(k_random)]
        + [SoftPlus() for _ in range(k_random)]
    )
    transform = ParamTransform(transforms=transforms)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        param_names=display_param_names,
        transform=transform,
    )
