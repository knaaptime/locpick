"""JAX kernels and objective builder for SAR-MNL PML estimation.

Implements the pseudo maximum likelihood (PML) estimator from
Smirnov (2010): spatially-filtered utilities with variance
normalisation by ``diag((I - ρW)^{-1})``, then standard MNL softmax.

The spatial solve ``(I - ρW) V* = V_base`` is done via LU factorisation
(dense, for moderate J) — the same matrix ``A = I - ρW`` is factorised
once and reused for all choosers.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from locpick._jax.data import ChoiceDataJAX
from locpick._jax.kernels import (
    _NEG_INF,
    compute_ll,
    compute_ll_contribs,
    compute_utilities,
    mnl_log_probs,
)
from locpick._jax.objective import Objective
from locpick._jax.transforms import Identity, ParamTransform, Tanh


# ---------------------------------------------------------------------------
# Core PML kernel
# ---------------------------------------------------------------------------


def _sar_mnl_ll_core(params, design_matrix, available, chosen, weights,
                     inclusion_probs, W_dense, n_obs, n_alts):
    """SAR-MNL PML log-likelihood (Smirnov 2010).

    Parameters
    ----------
    params : jnp.ndarray, shape (k+1,)
        [beta_1..k, alpha_rho] where rho = tanh(alpha_rho).
    design_matrix : jnp.ndarray, shape (n_obs * n_alts, k)
    available : jnp.ndarray, shape (n_obs, n_alts)
    chosen : jnp.ndarray, shape (n_obs, n_alts)
    weights : jnp.ndarray, shape (n_obs,)
    inclusion_probs : jnp.ndarray or None
    W_dense : jnp.ndarray, shape (n_alts, n_alts)
        Dense spatial weights matrix (row-standardised, zero diagonal).
    n_obs : int
    n_alts : int
    """
    k = design_matrix.shape[1]
    beta = params[:k]
    alpha_rho = params[k]
    rho = jnp.tanh(alpha_rho)

    # Base utilities: V_base (n_obs, n_alts)
    V_base = compute_utilities(
        design_matrix, beta, n_obs, n_alts,
        inclusion_probs=inclusion_probs, available=available,
    )

    # Spatial filter: solve (I - rho*W) V_filtered^T = V_base^T
    # A is (n_alts, n_alts), same for all choosers — solve once for all RHS
    A = jnp.eye(n_alts) - rho * W_dense
    V_filtered = jax.scipy.linalg.solve(A, V_base.T).T  # (n_obs, n_alts)

    # Variance normalisation: D = diag(A^{-1})
    A_inv = jax.scipy.linalg.inv(A)
    D = jnp.diag(A_inv)  # (n_alts,)
    V_star = V_filtered / D[None, :]  # normalise each alternative by d_jj

    # MNL log-probabilities
    log_probs = mnl_log_probs(V_star, available)
    return compute_ll(log_probs, chosen, weights)


def _sar_mnl_ll_contribs_core(params, design_matrix, available, chosen, weights,
                               inclusion_probs, W_dense, n_obs, n_alts):
    """Per-observation SAR-MNL PML log-likelihood contributions."""
    k = design_matrix.shape[1]
    beta = params[:k]
    alpha_rho = params[k]
    rho = jnp.tanh(alpha_rho)

    V_base = compute_utilities(
        design_matrix, beta, n_obs, n_alts,
        inclusion_probs=inclusion_probs, available=available,
    )

    A = jnp.eye(n_alts) - rho * W_dense
    V_filtered = jax.scipy.linalg.solve(A, V_base.T).T
    A_inv = jax.scipy.linalg.inv(A)
    D = jnp.diag(A_inv)
    V_star = V_filtered / D[None, :]

    log_probs = mnl_log_probs(V_star, available)
    return compute_ll_contribs(log_probs, chosen, weights)


# ---------------------------------------------------------------------------
# Objective builder
# ---------------------------------------------------------------------------


def build_sar_mnl_objective(arrays, W_sparse: sp.csr_array) -> Objective:
    """Build an Objective for SAR-MNL PML estimation (Smirnov 2010).

    Parameters
    ----------
    arrays : ChoiceArrays
        Estimation data arrays.
    W_sparse : scipy.sparse.csr_array
        Row-standardised alt×alt spatial weights matrix (zero diagonal).

    Returns
    -------
    Objective
        Objective with JIT-compiled LL, gradient, and Hessian.
        Includes a Tanh transform for the rho parameter.
    """
    data = ChoiceDataJAX.from_arrays(arrays)
    W_dense = jnp.array(W_sparse.toarray(), dtype=jnp.float64)
    n_obs = arrays.n_obs
    n_alts = arrays.n_alts
    k = arrays.design_matrix.shape[1]

    # JIT-compiled closures — data and W are captured, only params is dynamic
    @jax.jit
    def _ll_jax(params):
        return _sar_mnl_ll_core(
            params, data.design_matrix, data.available, data.chosen,
            data.weights, data.inclusion_probs, W_dense, n_obs, n_alts,
        )

    @jax.jit
    def _ll_contribs_jax(params):
        return _sar_mnl_ll_contribs_core(
            params, data.design_matrix, data.available, data.chosen,
            data.weights, data.inclusion_probs, W_dense, n_obs, n_alts,
        )

    @jax.jit
    def _grad_jax(params):
        return jax.grad(_sar_mnl_ll_core, argnums=0)(
            params, data.design_matrix, data.available, data.chosen,
            data.weights, data.inclusion_probs, W_dense, n_obs, n_alts,
        )

    param_names = list(arrays.param_names) + ["rho"]
    transform = ParamTransform.for_sar_mnl(k)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        loglike_contribs_jax=_ll_contribs_jax,
        param_names=param_names,
        transform=transform,
    )