"""JAX kernels and objective builder for SAR-MNL PML estimation.

Implements the pseudo maximum likelihood (PML) estimator from
Smirnov (2010): spatially-filtered utilities with variance
normalisation by ``diag((I - ρW)^{-1})``, then standard MNL softmax.

Two solve paths are available:

- **Dense** (default for ``n_alts ≤ 2000``): LU factorisation via
  ``jax.scipy.linalg.solve`` / ``inv``.  The same matrix ``A = I - ρW``
  is factorised once and reused for all choosers.
- **Conjugate gradient** (for ``n_alts > 2000``): iterative solve via
  ``jax.scipy.sparse.linalg.cg``.  Avoids materialising the dense
  inverse; the diagonal of ``A^{-1}`` is estimated via a power-series
  approximation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import scipy.sparse as sp

from .data import ChoiceDataJAX
from .kernels import (
    compute_ll,
    compute_ll_contribs,
    compute_utilities,
    mnl_log_probs,
)
from .objective import Objective
from .transforms import ParamTransform

# Threshold for switching from dense solve to conjugate gradient.
_DENSE_CUTOFF = 2000


# ---------------------------------------------------------------------------
# Dense solve path
# ---------------------------------------------------------------------------


def _sar_mnl_ll_core(
    params, design_matrix, available, chosen, weights, inclusion_probs, W_dense, n_obs, n_alts
):
    """SAR-MNL PML log-likelihood — dense solve path (Smirnov 2010).

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
        design_matrix,
        beta,
        n_obs,
        n_alts,
        inclusion_probs=inclusion_probs,
        available=available,
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


def _sar_mnl_ll_contribs_core(
    params, design_matrix, available, chosen, weights, inclusion_probs, W_dense, n_obs, n_alts
):
    """Per-observation SAR-MNL PML log-likelihood contributions — dense path."""
    k = design_matrix.shape[1]
    beta = params[:k]
    alpha_rho = params[k]
    rho = jnp.tanh(alpha_rho)

    V_base = compute_utilities(
        design_matrix,
        beta,
        n_obs,
        n_alts,
        inclusion_probs=inclusion_probs,
        available=available,
    )

    A = jnp.eye(n_alts) - rho * W_dense
    V_filtered = jax.scipy.linalg.solve(A, V_base.T).T
    A_inv = jax.scipy.linalg.inv(A)
    D = jnp.diag(A_inv)
    V_star = V_filtered / D[None, :]

    log_probs = mnl_log_probs(V_star, available)
    return compute_ll_contribs(log_probs, chosen, weights)


# ---------------------------------------------------------------------------
# Conjugate-gradient solve path (for large n_alts)
# ---------------------------------------------------------------------------


def _cg_solve(A, B, n_alts):
    """Solve A @ X = B via conjugate gradient, vectorised over columns of B.

    Uses ``jax.scipy.sparse.linalg.cg`` per column.  JAX autodiff
    works through CG via implicit differentiation.
    """

    def solve_one(b):
        x, _ = jax.scipy.sparse.linalg.cg(A, b)
        return x

    # vmap over columns of B (n_alts, n_rhs)
    return jax.vmap(solve_one, in_axes=1, out_axes=1)(B)


def _diag_inv_power_series(rho, W_dense, n_alts, n_terms=20):
    """Estimate diag((I - rho*W)^{-1}) via power series.

    Since W has zero diagonal, odd powers also have zero diagonal.
    Only even powers contribute: d_jj = 1 + rho^2 (W^2)_jj +
    rho^4 (W^4)_jj + ...  Converges for |rho| < 1/omega_max.
    """
    d = jnp.ones(n_alts)  # first term: diag(I) = 1
    W_power = W_dense @ W_dense  # W^2
    rho_sq = rho * rho
    coeff = rho_sq
    for _ in range(n_terms):
        d = d + coeff * jnp.diag(W_power)
        W_power = W_power @ W_power  # W^{2k}
        coeff = coeff * rho_sq
    return d


def _sar_mnl_ll_cg_core(
    params, design_matrix, available, chosen, weights, inclusion_probs, W_dense, n_obs, n_alts
):
    """SAR-MNL PML log-likelihood — conjugate-gradient path.

    Uses CG for the spatial solve and a power-series approximation
    for the variance normalisation diagonal.
    """
    k = design_matrix.shape[1]
    beta = params[:k]
    alpha_rho = params[k]
    rho = jnp.tanh(alpha_rho)

    V_base = compute_utilities(
        design_matrix,
        beta,
        n_obs,
        n_alts,
        inclusion_probs=inclusion_probs,
        available=available,
    )

    A = jnp.eye(n_alts) - rho * W_dense
    # CG solve: A @ V_filtered^T = V_base^T
    V_filtered = _cg_solve(A, V_base.T, n_alts).T  # (n_obs, n_alts)

    # Variance normalisation via power series
    D = _diag_inv_power_series(rho, W_dense, n_alts)
    V_star = V_filtered / D[None, :]

    log_probs = mnl_log_probs(V_star, available)
    return compute_ll(log_probs, chosen, weights)


def _sar_mnl_ll_contribs_cg_core(
    params, design_matrix, available, chosen, weights, inclusion_probs, W_dense, n_obs, n_alts
):
    """Per-observation SAR-MNL PML log-likelihood — CG path."""
    k = design_matrix.shape[1]
    beta = params[:k]
    alpha_rho = params[k]
    rho = jnp.tanh(alpha_rho)

    V_base = compute_utilities(
        design_matrix,
        beta,
        n_obs,
        n_alts,
        inclusion_probs=inclusion_probs,
        available=available,
    )

    A = jnp.eye(n_alts) - rho * W_dense
    V_filtered = _cg_solve(A, V_base.T, n_alts).T
    D = _diag_inv_power_series(rho, W_dense, n_alts)
    V_star = V_filtered / D[None, :]

    log_probs = mnl_log_probs(V_star, available)
    return compute_ll_contribs(log_probs, chosen, weights)


# ---------------------------------------------------------------------------
# Objective builder
# ---------------------------------------------------------------------------


def build_sar_mnl_objective(arrays, W_sparse: sp.csr_array, use_cg: bool = False) -> Objective:
    """Build an Objective for SAR-MNL PML estimation (Smirnov 2010).

    Parameters
    ----------
    arrays : ChoiceArrays
        Estimation data arrays.
    W_sparse : scipy.sparse.csr_array
        Row-standardised alt×alt spatial weights matrix (zero diagonal).
    use_cg : bool, default False
        If True, use conjugate-gradient solve (for large n_alts > 2000).
        If False, use dense LU solve (faster for moderate n_alts).

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

    # Select solve path
    if use_cg:
        ll_core = _sar_mnl_ll_cg_core
        ll_contribs_core = _sar_mnl_ll_contribs_cg_core
    else:
        ll_core = _sar_mnl_ll_core
        ll_contribs_core = _sar_mnl_ll_contribs_core

    # JIT-compiled closures — data and W are captured, only params is dynamic
    @jax.jit
    def _ll_jax(params):
        return ll_core(
            params,
            data.design_matrix,
            data.available,
            data.chosen,
            data.weights,
            data.inclusion_probs,
            W_dense,
            n_obs,
            n_alts,
        )

    @jax.jit
    def _ll_contribs_jax(params):
        return ll_contribs_core(
            params,
            data.design_matrix,
            data.available,
            data.chosen,
            data.weights,
            data.inclusion_probs,
            W_dense,
            n_obs,
            n_alts,
        )

    @jax.jit
    def _grad_jax(params):
        return jax.grad(ll_core, argnums=0)(
            params,
            data.design_matrix,
            data.available,
            data.chosen,
            data.weights,
            data.inclusion_probs,
            W_dense,
            n_obs,
            n_alts,
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
