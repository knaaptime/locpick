"""JAX kernels and objective builder for SAR-MNL PML estimation.

Implements the pseudo maximum likelihood (PML) estimator from
Smirnov (2010): spatially-filtered utilities with variance
normalisation by ``diag((I - ρW)^{-1})``, then standard MNL softmax.

The spatial filter ``(I - ρW)^{-1} V_base`` is applied by
:func:`_sar_filter`, shared across all SAR families.  When the model
supplies a ``sparse_solve_fn`` (cholgraph/klujax; see
:mod:`locpick._jax.sparse_backends`) the solve is sparse, differentiable
and JIT-native; otherwise it is a dense LU solve.

The variance-normalisation diagonal ``diag((I - ρW)^{-1})`` comes from a
precomputed Chebyshev/AAA interpolant (see
:mod:`locpick._jax.diag_precompute`), falling back to an exact dense
inverse when no interpolant is supplied.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .data import ChoiceDataJAX
from .kernels import (
    compute_ll,
    compute_ll_contribs,
    compute_utilities,
    mnl_log_probs,
    nested_log_probs,
)
from .objective import Objective
from .transforms import Identity, ParamTransform, Sigmoid, Tanh

# ---------------------------------------------------------------------------
# Dense solve path
# ---------------------------------------------------------------------------


def _sar_mnl_ll_core(
    params,
    design_matrix,
    available,
    chosen,
    weights,
    inclusion_probs,
    W_dense,
    n_obs,
    n_alts,
    diag_eval_fn=None,
    sparse_solve_fn=None,
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
    diag_eval_fn : callable or None
        If provided, evaluates ``diag((I - ρW)^{-1})`` via precomputed
        interpolation (Chebyshev or AAA).  If None, computes the
        exact diagonal via a dense inverse.
    sparse_solve_fn : callable or None
        If provided, uses sparse solve with custom VJP instead of dense LU.
        Callable signature: ``(rho, V_base) -> V_filtered``.
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

    V_star = _sar_filter(rho, V_base, W_dense, n_alts, diag_eval_fn, sparse_solve_fn)

    # MNL log-probabilities
    log_probs = mnl_log_probs(V_star, available)
    return compute_ll(log_probs, chosen, weights)


def _sar_mnl_ll_contribs_core(
    params,
    design_matrix,
    available,
    chosen,
    weights,
    inclusion_probs,
    W_dense,
    n_obs,
    n_alts,
    diag_eval_fn=None,
    sparse_solve_fn=None,
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

    V_star = _sar_filter(rho, V_base, W_dense, n_alts, diag_eval_fn, sparse_solve_fn)

    log_probs = mnl_log_probs(V_star, available)
    return compute_ll_contribs(log_probs, chosen, weights)


def _diag_inv_exact_dense(rho, W_dense, n_alts):
    """Compute ``diag((I - rho*W)^{-1})`` exactly via a dense inverse.

    Used when no precomputed interpolant is supplied (small ``n_alts``,
    where the O(n^3) inverse is cheaper than fitting an interpolant).
    For larger problems ``build_sar_*`` passes ``diag_eval_fn`` from
    :mod:`locpick._jax.diag_precompute` instead.
    """
    A = jnp.eye(n_alts) - rho * W_dense
    return jnp.diag(jnp.linalg.inv(A))


def _sar_filter(rho, V_base, W_dense, n_alts, diag_eval_fn=None, sparse_solve_fn=None):
    """Apply the SAR spatial filter and variance normalisation.

    Returns ``V_star = V_filtered / D`` where
    ``V_filtered = (I - ρW)^{-1} V_base`` and ``D = diag((I - ρW)^{-1})``.

    ``sparse_solve_fn`` (cholgraph/klujax, differentiable and JIT-native)
    replaces the dense factorisation when supplied; otherwise the solve is
    dense.  The diagonal comes from the precomputed interpolant when
    ``diag_eval_fn`` is given, else an exact dense inverse.  Shared by every
    SAR family so the solve path lives in one place.
    """
    if sparse_solve_fn is not None:
        V_filtered = sparse_solve_fn(rho, V_base)
    else:
        A = jnp.eye(n_alts) - rho * W_dense
        V_filtered = jax.scipy.linalg.solve(A, V_base.T).T
    if diag_eval_fn is not None:
        D = diag_eval_fn(rho)
    else:
        D = _diag_inv_exact_dense(rho, W_dense, n_alts)
    return V_filtered / D[None, :]


def build_sar_mnl_objective(
    arrays,
    W_sparse: sp.csr_array,
    diag_precompute=None,
    sparse_solve_fn=None,
) -> Objective:
    """Build an Objective for SAR-MNL PML estimation (Smirnov 2010).

    Parameters
    ----------
    arrays : ChoiceArrays
        Estimation data arrays.
    W_sparse : scipy.sparse.csr_array
        Row-standardised alt×alt spatial weights matrix (zero diagonal).
    diag_precompute : DiagPrecompute or None, default None
        Precomputed diagonal interpolation for ``diag((I - ρW)^{-1})``.
        If provided, replaces the power-series approximation with an
        exact Chebyshev/AAA interpolation (pure JAX, differentiable).
        If None, falls back to the 20-term power series.
    sparse_solve_fn : callable or None, default None
        Differentiable sparse solve ``(rho, V_base) -> V_filtered``
        (cholgraph/klujax).  When None, a dense solve is used.

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
    ll_core = _sar_mnl_ll_core
    ll_contribs_core = _sar_mnl_ll_contribs_core

    # Select diagonal computation path
    if diag_precompute is not None:
        diag_eval_fn = diag_precompute.eval_jax
    else:
        diag_eval_fn = None

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
            diag_eval_fn=diag_eval_fn,
            sparse_solve_fn=sparse_solve_fn,
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
            diag_eval_fn=diag_eval_fn,
            sparse_solve_fn=sparse_solve_fn,
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
            diag_eval_fn=diag_eval_fn,
            sparse_solve_fn=sparse_solve_fn,
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


# ---------------------------------------------------------------------------
# SAR-Nested Logit
# ---------------------------------------------------------------------------


def _sar_nested_ll_core(
    params,
    design_matrix,
    available,
    chosen,
    weights,
    inclusion_probs,
    W_dense,
    n_obs,
    n_alts,
    nest_matrix,
    k,
    n_nests,
    diag_eval_fn=None,
    sparse_solve_fn=None,
):
    """SAR-Nested PML log-likelihood — dense solve path."""
    beta = params[:k]
    alpha_rho = params[k]
    alpha_lambdas = params[k + 1 : k + 1 + n_nests]
    rho = jnp.tanh(alpha_rho)
    lambdas = 1.0 / (1.0 + jnp.exp(-alpha_lambdas))

    V_base = compute_utilities(
        design_matrix,
        beta,
        n_obs,
        n_alts,
        inclusion_probs=inclusion_probs,
        available=available,
    )
    V_star = _sar_filter(rho, V_base, W_dense, n_alts, diag_eval_fn, sparse_solve_fn)

    log_probs = nested_log_probs(V_star, lambdas, nest_matrix, available)
    return compute_ll(log_probs, chosen, weights)


def _sar_nested_ll_contribs_core(
    params,
    design_matrix,
    available,
    chosen,
    weights,
    inclusion_probs,
    W_dense,
    n_obs,
    n_alts,
    nest_matrix,
    k,
    n_nests,
    diag_eval_fn=None,
    sparse_solve_fn=None,
):
    """SAR-Nested per-observation LL contributions — dense path."""
    beta = params[:k]
    alpha_rho = params[k]
    alpha_lambdas = params[k + 1 : k + 1 + n_nests]
    rho = jnp.tanh(alpha_rho)
    lambdas = 1.0 / (1.0 + jnp.exp(-alpha_lambdas))

    V_base = compute_utilities(
        design_matrix,
        beta,
        n_obs,
        n_alts,
        inclusion_probs=inclusion_probs,
        available=available,
    )
    V_star = _sar_filter(rho, V_base, W_dense, n_alts, diag_eval_fn, sparse_solve_fn)

    log_probs = nested_log_probs(V_star, lambdas, nest_matrix, available)
    return compute_ll_contribs(log_probs, chosen, weights)


def build_sar_nested_objective(
    arrays,
    W_sparse: sp.csr_array,
    nest_matrix: np.ndarray,
    diag_precompute=None,
    sparse_solve_fn=None,
) -> Objective:
    """Build an Objective for SAR-Nested PML estimation.

    Applies the SAR spatial filter globally, then applies nested logit
    nesting structure to the spatially-filtered utilities.

    Parameters
    ----------
    arrays : ChoiceArrays
    W_sparse : scipy.sparse.csr_array
        Row-standardised alt×alt spatial weights matrix.
    nest_matrix : np.ndarray, shape (n_alts, n_nests)
        Alternative-to-nest membership matrix.
    diag_precompute : DiagPrecompute or None, default None
        Precomputed diagonal interpolation for variance normalization.

    Returns
    -------
    Objective
    """
    data = ChoiceDataJAX.from_arrays(arrays)
    W_dense = jnp.array(W_sparse.toarray(), dtype=jnp.float64)
    n_obs = arrays.n_obs
    n_alts = arrays.n_alts
    k = arrays.design_matrix.shape[1]
    nest_matrix_jax = jnp.asarray(nest_matrix, dtype=jnp.float64)
    n_nests = nest_matrix.shape[1]

    ll_core = _sar_nested_ll_core
    ll_contribs_core = _sar_nested_ll_contribs_core

    diag_eval_fn = diag_precompute.eval_jax if diag_precompute is not None else None

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
            nest_matrix_jax,
            k,
            n_nests,
            diag_eval_fn=diag_eval_fn,
            sparse_solve_fn=sparse_solve_fn,
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
            nest_matrix_jax,
            k,
            n_nests,
            diag_eval_fn=diag_eval_fn,
            sparse_solve_fn=sparse_solve_fn,
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
            nest_matrix_jax,
            k,
            n_nests,
            diag_eval_fn=diag_eval_fn,
            sparse_solve_fn=sparse_solve_fn,
        )

    param_names = list(arrays.param_names) + ["rho"] + [f"lambda_{i}" for i in range(n_nests)]
    transforms = [Identity() for _ in range(k)] + [Tanh()] + [Sigmoid() for _ in range(n_nests)]
    transform = ParamTransform(transforms=transforms)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        loglike_contribs_jax=_ll_contribs_jax,
        param_names=param_names,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# SAR-Mixed Logit
# ---------------------------------------------------------------------------


def _sar_mixed_ll_core(
    params,
    data,
    W_dense,
    n_obs,
    n_alts,
    k_fixed,
    k_random,
    n_draws,
    diag_eval_fn=None,
    sparse_solve_fn=None,
):
    """SAR-Mixed simulated PML log-likelihood."""
    from .kernels import mixed_logit_ll

    beta_fixed = params[:k_fixed]
    alpha_rho = params[k_fixed]
    beta_random_means = params[k_fixed + 1 : k_fixed + 1 + k_random]
    beta_random_spreads_raw = params[k_fixed + 1 + k_random :]
    rho = jnp.tanh(alpha_rho)
    beta_random_spreads = jnp.log1p(jnp.exp(beta_random_spreads_raw))

    # Fixed utility
    if data.dm_fixed is not None and k_fixed > 0:
        v_fixed = (data.dm_fixed @ beta_fixed).reshape(n_obs, n_alts)
    else:
        v_fixed = jnp.zeros((n_obs, n_alts), dtype=jnp.float64)

    if data.inclusion_probs is not None:
        v_fixed = v_fixed + jnp.log(jnp.maximum(data.inclusion_probs, 1e-30))

    # Apply SAR filter to fixed utility
    v_fixed_star = _sar_filter(rho, v_fixed, W_dense, n_alts, diag_eval_fn, sparse_solve_fn)

    return mixed_logit_ll(
        V_fixed=v_fixed_star,
        dm_random=data.dm_random,
        beta_random_means=beta_random_means,
        beta_random_spreads=beta_random_spreads,
        dist_codes=data.dist_codes,
        draws=data.draws,
        chosen=data.chosen,
        weights=data.weights,
        available=data.available,
        n_obs=n_obs,
        n_alts=n_alts,
        k_random=k_random,
        n_draws=n_draws,
    )


def _sar_mixed_ll_contribs_core(
    params,
    data,
    W_dense,
    n_obs,
    n_alts,
    k_fixed,
    k_random,
    n_draws,
    diag_eval_fn=None,
    sparse_solve_fn=None,
):
    """SAR-Mixed per-observation LL contributions."""
    from .kernels import mixed_logit_ll_contribs

    beta_fixed = params[:k_fixed]
    alpha_rho = params[k_fixed]
    beta_random_means = params[k_fixed + 1 : k_fixed + 1 + k_random]
    beta_random_spreads_raw = params[k_fixed + 1 + k_random :]
    rho = jnp.tanh(alpha_rho)
    beta_random_spreads = jnp.log1p(jnp.exp(beta_random_spreads_raw))

    if data.dm_fixed is not None and k_fixed > 0:
        v_fixed = (data.dm_fixed @ beta_fixed).reshape(n_obs, n_alts)
    else:
        v_fixed = jnp.zeros((n_obs, n_alts), dtype=jnp.float64)

    if data.inclusion_probs is not None:
        v_fixed = v_fixed + jnp.log(jnp.maximum(data.inclusion_probs, 1e-30))

    v_fixed_star = _sar_filter(rho, v_fixed, W_dense, n_alts, diag_eval_fn, sparse_solve_fn)

    return mixed_logit_ll_contribs(
        V_fixed=v_fixed_star,
        dm_random=data.dm_random,
        beta_random_means=beta_random_means,
        beta_random_spreads=beta_random_spreads,
        dist_codes=data.dist_codes,
        draws=data.draws,
        chosen=data.chosen,
        weights=data.weights,
        available=data.available,
        n_obs=n_obs,
        n_alts=n_alts,
        k_random=k_random,
        n_draws=n_draws,
    )


def build_sar_mixed_objective(
    arrays,
    W_sparse: sp.csr_array,
    random_col_indices,
    random_distributions,
    draws,
    diag_precompute=None,
    sparse_solve_fn=None,
) -> Objective:
    """Build an Objective for SAR-Mixed PML estimation.

    Applies the SAR spatial filter to the fixed utility component, then
    adds random utility per draw and computes simulated MNL probabilities.

    Parameters
    ----------
    arrays : ChoiceArrays
    W_sparse : scipy.sparse.csr_array
    random_col_indices : list[int]
    random_distributions : list[str]
    draws : np.ndarray, shape (n_obs, n_draws, k_random)
    diag_precompute : DiagPrecompute or None, default None

    Returns
    -------
    Objective
    """
    data = ChoiceDataJAX.from_arrays(
        arrays,
        draws=draws,
        random_col_indices=random_col_indices,
        random_distributions=random_distributions,
    )
    W_dense = jnp.array(W_sparse.toarray(), dtype=jnp.float64)
    n_obs = arrays.n_obs
    n_alts = arrays.n_alts
    k_fixed = len(data.fixed_col_indices) if data.fixed_col_indices else 0
    k_random = len(random_col_indices)
    n_draws = draws.shape[1]

    diag_eval_fn = diag_precompute.eval_jax if diag_precompute is not None else None

    @jax.jit
    def _ll_jax(params):
        return _sar_mixed_ll_core(
            params,
            data,
            W_dense,
            n_obs,
            n_alts,
            k_fixed,
            k_random,
            n_draws,
            diag_eval_fn=diag_eval_fn,
            sparse_solve_fn=sparse_solve_fn,
        )

    @jax.jit
    def _ll_contribs_jax(params):
        return _sar_mixed_ll_contribs_core(
            params,
            data,
            W_dense,
            n_obs,
            n_alts,
            k_fixed,
            k_random,
            n_draws,
            diag_eval_fn=diag_eval_fn,
            sparse_solve_fn=sparse_solve_fn,
        )

    @jax.jit
    def _grad_jax(params):
        return jax.grad(_sar_mixed_ll_core, argnums=0)(
            params,
            data,
            W_dense,
            n_obs,
            n_alts,
            k_fixed,
            k_random,
            n_draws,
            diag_eval_fn=diag_eval_fn,
            sparse_solve_fn=sparse_solve_fn,
        )

    param_names_list = list(arrays.param_names)
    fixed_param_names = [
        name for i, name in enumerate(param_names_list) if i not in random_col_indices
    ]
    random_param_names = [param_names_list[i] for i in random_col_indices]
    display_param_names = (
        fixed_param_names
        + ["rho"]
        + [f"mean_{name}" for name in random_param_names]
        + [f"sd_{name}" for name in random_param_names]
    )

    transforms = (
        [Identity() for _ in range(k_fixed)]
        + [Tanh()]
        + [Identity() for _ in range(k_random)]
        + [Identity() for _ in range(k_random)]  # softplus applied inside kernel
    )
    transform = ParamTransform(transforms=transforms)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        loglike_contribs_jax=_ll_contribs_jax,
        param_names=display_param_names,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# SAR-Mixed-Nested Logit
# ---------------------------------------------------------------------------


def _sar_mixed_nested_ll_core(
    params,
    data,
    W_dense,
    n_obs,
    n_alts,
    nest_matrix,
    k_fixed,
    n_nests,
    k_random,
    n_draws,
    nest_alt_indices,
    diag_eval_fn=None,
    sparse_solve_fn=None,
):
    """SAR-Mixed-Nested simulated PML log-likelihood."""
    from .kernels import mixed_nested_logit_ll

    beta_fixed = params[:k_fixed]
    alpha_rho = params[k_fixed]
    alpha_lambdas = params[k_fixed + 1 : k_fixed + 1 + n_nests]
    beta_random_means = params[k_fixed + 1 + n_nests : k_fixed + 1 + n_nests + k_random]
    beta_random_spreads_raw = params[k_fixed + 1 + n_nests + k_random :]
    rho = jnp.tanh(alpha_rho)
    lambdas = 1.0 / (1.0 + jnp.exp(-alpha_lambdas))
    beta_random_spreads = jnp.log1p(jnp.exp(beta_random_spreads_raw))

    # Fixed utility
    if data.dm_fixed is not None and k_fixed > 0:
        v_fixed = (data.dm_fixed @ beta_fixed).reshape(n_obs, n_alts)
    else:
        v_fixed = jnp.zeros((n_obs, n_alts), dtype=jnp.float64)

    if data.inclusion_probs is not None:
        v_fixed = v_fixed + jnp.log(jnp.maximum(data.inclusion_probs, 1e-30))

    # Apply SAR filter to fixed utility
    v_fixed_star = _sar_filter(rho, v_fixed, W_dense, n_alts, diag_eval_fn, sparse_solve_fn)

    return mixed_nested_logit_ll(
        V_fixed=v_fixed_star,
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
        n_obs=n_obs,
        n_alts=n_alts,
        k_random=k_random,
        n_draws=n_draws,
        n_nests=n_nests,
    )


def _sar_mixed_nested_ll_contribs_core(
    params,
    data,
    W_dense,
    n_obs,
    n_alts,
    nest_matrix,
    k_fixed,
    n_nests,
    k_random,
    n_draws,
    nest_alt_indices,
    diag_eval_fn=None,
    sparse_solve_fn=None,
):
    """SAR-Mixed-Nested per-observation LL contributions."""
    from .kernels import mixed_nested_logit_ll_contribs

    beta_fixed = params[:k_fixed]
    alpha_rho = params[k_fixed]
    alpha_lambdas = params[k_fixed + 1 : k_fixed + 1 + n_nests]
    beta_random_means = params[k_fixed + 1 + n_nests : k_fixed + 1 + n_nests + k_random]
    beta_random_spreads_raw = params[k_fixed + 1 + n_nests + k_random :]
    rho = jnp.tanh(alpha_rho)
    lambdas = 1.0 / (1.0 + jnp.exp(-alpha_lambdas))
    beta_random_spreads = jnp.log1p(jnp.exp(beta_random_spreads_raw))

    if data.dm_fixed is not None and k_fixed > 0:
        v_fixed = (data.dm_fixed @ beta_fixed).reshape(n_obs, n_alts)
    else:
        v_fixed = jnp.zeros((n_obs, n_alts), dtype=jnp.float64)

    if data.inclusion_probs is not None:
        v_fixed = v_fixed + jnp.log(jnp.maximum(data.inclusion_probs, 1e-30))

    v_fixed_star = _sar_filter(rho, v_fixed, W_dense, n_alts, diag_eval_fn, sparse_solve_fn)

    return mixed_nested_logit_ll_contribs(
        V_fixed=v_fixed_star,
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
        n_obs=n_obs,
        n_alts=n_alts,
        k_random=k_random,
        n_draws=n_draws,
        n_nests=n_nests,
    )


def build_sar_mixed_nested_objective(
    arrays,
    W_sparse: sp.csr_array,
    nest_matrix: np.ndarray,
    random_col_indices,
    random_distributions,
    draws,
    diag_precompute=None,
    sparse_solve_fn=None,
) -> Objective:
    """Build an Objective for SAR-Mixed-Nested PML estimation.

    Applies the SAR spatial filter to the fixed utility, adds random
    utility per draw, then applies nested logit nesting.

    Parameters
    ----------
    arrays : ChoiceArrays
    W_sparse : scipy.sparse.csr_array
    nest_matrix : np.ndarray, shape (n_alts, n_nests)
    random_col_indices : list[int]
    random_distributions : list[str]
    draws : np.ndarray, shape (n_obs, n_draws, k_random)
    diag_precompute : DiagPrecompute or None, default None

    Returns
    -------
    Objective
    """
    data = ChoiceDataJAX.from_arrays(
        arrays,
        draws=draws,
        random_col_indices=random_col_indices,
        random_distributions=random_distributions,
    )
    W_dense = jnp.array(W_sparse.toarray(), dtype=jnp.float64)
    n_obs = arrays.n_obs
    n_alts = arrays.n_alts
    nest_matrix_jax = jnp.asarray(nest_matrix, dtype=jnp.float64)
    n_nests = nest_matrix.shape[1]
    k = arrays.design_matrix.shape[1]
    k_random = len(random_col_indices)
    k_fixed = k - k_random
    n_draws = draws.shape[1]

    nest_alt_indices = tuple(
        tuple(int(i) for i in np.where(nest_matrix[:, m] > 0)[0]) for m in range(n_nests)
    )

    diag_eval_fn = diag_precompute.eval_jax if diag_precompute is not None else None

    @jax.jit
    def _ll_jax(params):
        return _sar_mixed_nested_ll_core(
            params,
            data,
            W_dense,
            n_obs,
            n_alts,
            nest_matrix_jax,
            k_fixed,
            n_nests,
            k_random,
            n_draws,
            nest_alt_indices,
            diag_eval_fn=diag_eval_fn,
            sparse_solve_fn=sparse_solve_fn,
        )

    @jax.jit
    def _ll_contribs_jax(params):
        return _sar_mixed_nested_ll_contribs_core(
            params,
            data,
            W_dense,
            n_obs,
            n_alts,
            nest_matrix_jax,
            k_fixed,
            n_nests,
            k_random,
            n_draws,
            nest_alt_indices,
            diag_eval_fn=diag_eval_fn,
            sparse_solve_fn=sparse_solve_fn,
        )

    @jax.jit
    def _grad_jax(params):
        return jax.grad(_sar_mixed_nested_ll_core, argnums=0)(
            params,
            data,
            W_dense,
            n_obs,
            n_alts,
            nest_matrix_jax,
            k_fixed,
            n_nests,
            k_random,
            n_draws,
            nest_alt_indices,
            diag_eval_fn=diag_eval_fn,
            sparse_solve_fn=sparse_solve_fn,
        )

    param_names_list = list(arrays.param_names)
    fixed_param_names = [
        name for i, name in enumerate(param_names_list) if i not in random_col_indices
    ]
    random_param_names = [param_names_list[i] for i in random_col_indices]
    display_param_names = (
        fixed_param_names
        + ["rho"]
        + [f"lambda_{i}" for i in range(n_nests)]
        + [f"mean_{name}" for name in random_param_names]
        + [f"sd_{name}" for name in random_param_names]
    )

    transforms = (
        [Identity() for _ in range(k_fixed)]
        + [Tanh()]
        + [Sigmoid() for _ in range(n_nests)]
        + [Identity() for _ in range(k_random)]
        + [Identity() for _ in range(k_random)]  # softplus applied inside kernel
    )
    transform = ParamTransform(transforms=transforms)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        loglike_contribs_jax=_ll_contribs_jax,
        param_names=display_param_names,
        transform=transform,
    )
