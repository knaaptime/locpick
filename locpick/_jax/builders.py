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

import numpy as np

from locpick._compat import _JAX_AVAILABLE
from locpick._jax.data import ChoiceDataJAX, EdgeDataJAX

if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
from locpick._jax.kernels import (
    scl_log_probs,
    mnl_log_probs,
    nested_log_probs,
    mixed_logit_ll,
    compute_ll,
    compute_ll_contribs,
    compute_utilities,
    _NEG_INF,
)
from locpick._jax.objective import Objective
from locpick._jax.transforms import ParamTransform, Sigmoid, Identity


# ---------------------------------------------------------------------------
# MNL objective
# ---------------------------------------------------------------------------


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

    def _ll_jax(beta):
        """Pure JAX MNL log-likelihood."""
        V = compute_utilities(data.design_matrix, beta, data.n_obs, data.n_alts)
        log_probs = mnl_log_probs(V, data.available, inclusion_probs=data.inclusion_probs)
        return compute_ll(log_probs, data.chosen, data.weights)

    def _ll_contribs_jax(beta):
        """Per-observation MNL log-likelihood contributions."""
        V = compute_utilities(data.design_matrix, beta, data.n_obs, data.n_alts)
        log_probs = mnl_log_probs(V, data.available, inclusion_probs=data.inclusion_probs)
        return compute_ll_contribs(log_probs, data.chosen, data.weights)

    _ll_jit = jax.jit(_ll_jax)
    _grad_jit = jax.jit(jax.grad(_ll_jax))
    _ll_contribs_jit = jax.jit(_ll_contribs_jax)

    return Objective.from_jax(
        ll_fn=_ll_jit,
        grad_fn=_grad_jit,
        loglike_contribs_jax=_ll_contribs_jit,
        param_names=list(arrays.param_names),
    )


# ---------------------------------------------------------------------------
# SCL objective
# ---------------------------------------------------------------------------


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

    def _ll_jax(params):
        """Pure JAX SCL log-likelihood."""
        beta = params[:k]
        alpha_rho = params[k]
        rho = 1.0 / (1.0 + jnp.exp(-alpha_rho))  # sigmoid transform

        V = compute_utilities(
            data.design_matrix, beta, data.n_obs, data.n_alts,
            inclusion_probs=data.inclusion_probs,
            available=data.available,
        )
        log_probs = scl_log_probs(V, rho, data.edge_data, data.available)
        return compute_ll(log_probs, data.chosen, data.weights)

    def _ll_contribs_jax(params):
        """Per-observation SCL log-likelihood contributions."""
        beta = params[:k]
        alpha_rho = params[k]
        rho = 1.0 / (1.0 + jnp.exp(-alpha_rho))

        V = compute_utilities(
            data.design_matrix, beta, data.n_obs, data.n_alts,
            inclusion_probs=data.inclusion_probs,
            available=data.available,
        )
        log_probs = scl_log_probs(V, rho, data.edge_data, data.available)
        return compute_ll_contribs(log_probs, data.chosen, data.weights)

    _ll_jit = jax.jit(_ll_jax)
    _grad_jit = jax.jit(jax.grad(_ll_jax))
    _ll_contribs_jit = jax.jit(_ll_contribs_jax)

    param_names = list(arrays.param_names) + ["rho"]
    transform = ParamTransform.for_scl(k)

    return Objective.from_jax(
        ll_fn=_ll_jit,
        grad_fn=_grad_jit,
        loglike_contribs_jax=_ll_contribs_jit,
        param_names=param_names,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# MSCL objective
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

    from jax.scipy.special import logsumexp as jax_logsumexp

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

    def _ll_jax(params):
        """Pure JAX MSCL simulated log-likelihood."""
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

            # Vectorised random coefficient generation — no Python loop
            # means: (k_random,), spreads: (k_random,), z_r: (n_obs, k_random)
            means = beta_random_means[None, :]   # (1, k_random)
            spreads = beta_random_spreads[None, :]  # (1, k_random)

            # Normal: β = μ + σ * z
            beta_normal = means + spreads * z_r
            # Lognormal: β = exp(μ + σ * z)
            beta_lognormal = jnp.exp(jnp.clip(means + spreads * z_r, -50.0, 50.0))
            # Uniform on [μ - σ, μ + σ]: transform standard normal CDF to U(-1,1)
            # NOTE: "triangular" is currently a synonym for "uniform" in this
            # implementation. A true triangular distribution would require a
            # different inverse-CDF (piecewise sqrt, not linear).
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
            mask = phi_z <= 0.5
            beta_triangular = jnp.where(
                mask,
                means + spreads * (jnp.sqrt(2.0 * phi_z) - 1.0),
                means + spreads * (1.0 - jnp.sqrt(2.0 * (1.0 - phi_z))),
            )

            # Select distribution per parameter — vectorised via jnp.where
            # dist_codes: (k_random,) → broadcast against (n_obs, k_random)
            dist = data.dist_codes[None, :]  # (1, k_random)
            beta_random_r = jnp.where(
                dist == 0, beta_normal,
                jnp.where(dist == 1, beta_lognormal,
                    jnp.where(dist == 2, beta_triangular, beta_uniform)),
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

    _ll_jit = jax.jit(_ll_jax)
    _grad_jit = jax.jit(jax.grad(_ll_jax))

    # Parameter names
    param_names_list = list(arrays.param_names)
    fixed_param_names = [name for name in param_names_list if name not in
                        [param_names_list[i] for i in random_col_indices]]
    random_param_names = [param_names_list[i] for i in random_col_indices]
    display_param_names = (
        fixed_param_names
        + ["rho"]
        + [f"mean_{name}" for name in random_param_names]
        + [f"sd_{name}" for name in random_param_names]
    )

    transform = ParamTransform.for_mscl(k_fixed, k_random)

    return Objective.from_jax(
        ll_fn=_ll_jit,
        grad_fn=_grad_jit,
        param_names=display_param_names,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# Nested logit objective
# ---------------------------------------------------------------------------


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

    def _ll_jax(params):
        """Pure JAX nested logit log-likelihood."""
        beta = params[:k]
        alpha = params[k:]  # unconstrained nest params
        lambdas = 1.0 / (1.0 + jnp.exp(-alpha))  # sigmoid transform

        V = compute_utilities(
            data.design_matrix, beta, data.n_obs, data.n_alts,
            inclusion_probs=data.inclusion_probs,
            available=data.available,
        )
        log_probs = nested_log_probs(V, lambdas, nest_matrix_jax, data.available)
        return compute_ll(log_probs, data.chosen, data.weights)

    _ll_jit = jax.jit(_ll_jax)
    _grad_jit = jax.jit(jax.grad(_ll_jax))

    param_names = list(arrays.param_names) + [f"nest_alpha_{i}" for i in range(n_nests)]

    # Sigmoid transform for nest parameters, identity for beta
    transforms = [Identity() for _ in range(k)] + [Sigmoid() for _ in range(n_nests)]
    transform = ParamTransform(transforms=transforms)

    return Objective.from_jax(
        ll_fn=_ll_jit,
        grad_fn=_grad_jit,
        param_names=param_names,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# Mixed logit objective (non-spatial)
# ---------------------------------------------------------------------------


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

    def _ll_jax(params):
        """Pure JAX mixed logit simulated log-likelihood."""
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

    _ll_jit = jax.jit(_ll_jax)
    _grad_jit = jax.jit(jax.grad(_ll_jax))

    # Parameter names
    param_names_list = list(arrays.param_names)
    fixed_param_names = [
        name for i, name in enumerate(param_names_list)
        if i not in random_col_indices
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
        ll_fn=_ll_jit,
        grad_fn=_grad_jit,
        param_names=display_param_names,
        transform=transform,
    )