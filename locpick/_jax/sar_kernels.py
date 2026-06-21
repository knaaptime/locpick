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
    nested_log_probs,
)
from .objective import Objective
from .transforms import Identity, ParamTransform, Sigmoid, SoftPlus, Tanh

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


# ---------------------------------------------------------------------------
# SAR + Nested
# ---------------------------------------------------------------------------


def _sar_nested_ll_core(
    params,
    design_matrix,
    available,
    chosen,
    weights,
    inclusion_probs,
    W_dense,
    nest_matrix,
    n_obs,
    n_alts,
    n_nests,
):
    """SAR + Nested PML log-likelihood.

    Layout: [beta_1..k, alpha_rho, alpha_lambda_1..M]
    Spatial filter applied to utilities, then nested GEV.
    """
    k = design_matrix.shape[1]
    beta = params[:k]
    alpha_rho = params[k]
    alpha_lambdas = params[k + 1 : k + 1 + n_nests]
    rho = jnp.tanh(alpha_rho)
    lambdas = 1.0 / (1.0 + jnp.exp(-alpha_lambdas))  # sigmoid → (0, 1]

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
    D = jnp.diag(jax.scipy.linalg.inv(A))
    V_star = V_filtered / D[None, :]

    log_probs = nested_log_probs(V_star, lambdas, nest_matrix, available)
    return compute_ll(log_probs, chosen, weights)


def build_sar_nested_objective(arrays, W_sparse, nest_matrix) -> Objective:
    """Build Objective for SAR + Nested estimation."""
    data = ChoiceDataJAX.from_arrays(arrays)
    W_dense = jnp.array(W_sparse.toarray(), dtype=jnp.float64)
    n_obs = arrays.n_obs
    n_alts = arrays.n_alts
    k = arrays.design_matrix.shape[1]
    n_nests = nest_matrix.shape[1]
    nest_matrix_jax = jnp.array(nest_matrix, dtype=jnp.float64)

    @jax.jit
    def _ll_jax(params):
        return _sar_nested_ll_core(
            params,
            data.design_matrix,
            data.available,
            data.chosen,
            data.weights,
            data.inclusion_probs,
            W_dense,
            nest_matrix_jax,
            n_obs,
            n_alts,
            n_nests,
        )

    @jax.jit
    def _grad_jax(params):
        return jax.grad(_sar_nested_ll_core, argnums=0)(
            params,
            data.design_matrix,
            data.available,
            data.chosen,
            data.weights,
            data.inclusion_probs,
            W_dense,
            nest_matrix_jax,
            n_obs,
            n_alts,
            n_nests,
        )

    param_names = list(arrays.param_names) + ["rho"]
    param_names += [f"lambda_{i}" for i in range(n_nests)]
    # Transform: Identity for beta, Tanh for rho, Sigmoid for lambdas
    transforms = [Identity()] * k + [Tanh()] + [Sigmoid(0, 1)] * n_nests
    transform = ParamTransform(transforms)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        param_names=param_names,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# SAR + Mixed
# ---------------------------------------------------------------------------


def _sar_mixed_ll_core(
    params,
    dm_fixed,
    dm_random,
    available,
    chosen,
    weights,
    inclusion_probs,
    W_dense,
    dist_codes,
    draws,
    n_obs,
    n_alts,
    k_fixed,
    k_random,
    n_draws,
):
    """SAR + Mixed simulated PML log-likelihood.

    Layout: [beta_fixed, alpha_rho, mean_*, sd_*]
    Spatial filter applied to fixed utility; random part added after.
    """
    beta_fixed = params[:k_fixed]
    alpha_rho = params[k_fixed]
    beta_random_means = params[k_fixed + 1 : k_fixed + 1 + k_random]
    beta_random_spreads_raw = params[k_fixed + 1 + k_random :]
    rho = jnp.tanh(alpha_rho)
    spreads = jnp.log1p(jnp.exp(beta_random_spreads_raw))  # softplus

    A = jnp.eye(n_alts) - rho * W_dense
    A_inv = jax.scipy.linalg.inv(A)
    D = jnp.diag(A_inv)

    # Fixed utility (spatially filtered + normalised)
    if dm_fixed is not None and k_fixed > 0:
        V_fixed_base = (dm_fixed @ beta_fixed).reshape(n_obs, n_alts)
    else:
        V_fixed_base = jnp.zeros((n_obs, n_alts), dtype=jnp.float64)

    if inclusion_probs is not None:
        V_fixed_base = V_fixed_base + jnp.log(jnp.maximum(inclusion_probs, 1e-30))

    V_fixed_filtered = jax.scipy.linalg.solve(A, V_fixed_base.T).T
    V_fixed_star = V_fixed_filtered / D[None, :]

    # Simulated likelihood: average over draws
    means = beta_random_means[None, :]

    def _prob_single_draw(r):
        z_r = draws[:, r, :]
        beta_normal = means + spreads * z_r
        beta_lognormal = jnp.exp(jnp.clip(means + spreads * z_r, -50, 50))
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
        abs_z = jnp.abs(z_r)
        tri_sign = jnp.where(z_r >= 0, 1.0, -1.0)
        beta_triangular = means + spreads * tri_sign * (jnp.sqrt(2.0 * abs_z) - 1.0)

        beta_r = jnp.where(
            dist_codes == 0,
            beta_normal,
            jnp.where(
                dist_codes == 1,
                beta_lognormal,
                jnp.where(dist_codes == 2, beta_triangular, beta_uniform),
            ),
        )  # (n_obs, k_random)

        # Random utility: broadcast per-obs random coefficients with design matrix
        V_random = jnp.sum(
            dm_random.reshape(n_obs, n_alts, k_random) * beta_r[:, None, :],
            axis=2,
        )
        V_total = V_fixed_star + V_random
        V_masked = jnp.where(available > 0, V_total, -1e30)
        log_sum_exp = jax.scipy.special.logsumexp(V_masked, axis=1)
        log_probs = V_masked - log_sum_exp[:, None]
        log_probs = jnp.where(available > 0, log_probs, -1e30)
        # Per-obs probability for this draw
        return jnp.exp((log_probs * chosen).sum(axis=1))

    probs_sim = jax.vmap(_prob_single_draw)(jnp.arange(n_draws)).mean(axis=0)
    ll = jnp.sum(jnp.log(jnp.maximum(probs_sim, 1e-30)) * weights)
    return ll


def build_sar_mixed_objective(
    arrays, W_sparse, random_col_indices, random_distributions, draws
) -> Objective:
    """Build Objective for SAR + Mixed estimation."""
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

    @jax.jit
    def _ll_jax(params):
        return _sar_mixed_ll_core(
            params,
            data.dm_fixed,
            data.dm_random,
            data.available,
            data.chosen,
            data.weights,
            data.inclusion_probs,
            W_dense,
            data.dist_codes,
            data.draws,
            n_obs,
            n_alts,
            k_fixed,
            k_random,
            n_draws,
        )

    @jax.jit
    def _grad_jax(params):
        return jax.grad(_sar_mixed_ll_core, argnums=0)(
            params,
            data.dm_fixed,
            data.dm_random,
            data.available,
            data.chosen,
            data.weights,
            data.inclusion_probs,
            W_dense,
            data.dist_codes,
            data.draws,
            n_obs,
            n_alts,
            k_fixed,
            k_random,
            n_draws,
        )

    param_names_list = list(arrays.param_names)
    fixed_names = [name for i, name in enumerate(param_names_list) if i not in random_col_indices]
    random_names = [param_names_list[i] for i in random_col_indices]
    display_names = (
        fixed_names
        + ["rho"]
        + [f"mean_{n}" for n in random_names]
        + [f"sd_{n}" for n in random_names]
    )

    transforms = (
        [Identity()] * k_fixed + [Tanh()] + [Identity()] * k_random + [SoftPlus()] * k_random
    )
    transform = ParamTransform(transforms)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        param_names=display_names,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# SAR + Mixed + Nested
# ---------------------------------------------------------------------------


def _sar_mixed_nested_ll_core(
    params,
    dm_fixed,
    dm_random,
    available,
    chosen,
    weights,
    inclusion_probs,
    W_dense,
    nest_matrix,
    dist_codes,
    draws,
    n_obs,
    n_alts,
    k_fixed,
    k_random,
    n_draws,
    n_nests,
):
    """SAR + Mixed + Nested simulated PML log-likelihood.

    Layout: [beta_fixed, alpha_rho, alpha_lambda_1..M, mean_*, sd_*]
    """
    beta_fixed = params[:k_fixed]
    alpha_rho = params[k_fixed]
    alpha_lambdas = params[k_fixed + 1 : k_fixed + 1 + n_nests]
    beta_random_means = params[k_fixed + 1 + n_nests : k_fixed + 1 + n_nests + k_random]
    beta_random_spreads_raw = params[k_fixed + 1 + n_nests + k_random :]
    rho = jnp.tanh(alpha_rho)
    lambdas = 1.0 / (1.0 + jnp.exp(-alpha_lambdas))
    spreads = jnp.log1p(jnp.exp(beta_random_spreads_raw))

    A = jnp.eye(n_alts) - rho * W_dense
    A_inv = jax.scipy.linalg.inv(A)
    D = jnp.diag(A_inv)

    if dm_fixed is not None and k_fixed > 0:
        V_fixed_base = (dm_fixed @ beta_fixed).reshape(n_obs, n_alts)
    else:
        V_fixed_base = jnp.zeros((n_obs, n_alts), dtype=jnp.float64)
    if inclusion_probs is not None:
        V_fixed_base = V_fixed_base + jnp.log(jnp.maximum(inclusion_probs, 1e-30))

    V_fixed_filtered = jax.scipy.linalg.solve(A, V_fixed_base.T).T
    V_fixed_star = V_fixed_filtered / D[None, :]

    means = beta_random_means[None, :]

    def _prob_single_draw(r):
        z_r = draws[:, r, :]
        beta_normal = means + spreads * z_r
        beta_lognormal = jnp.exp(jnp.clip(means + spreads * z_r, -50, 50))
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
        abs_z = jnp.abs(z_r)
        tri_sign = jnp.where(z_r >= 0, 1.0, -1.0)
        beta_triangular = means + spreads * tri_sign * (jnp.sqrt(2.0 * abs_z) - 1.0)

        beta_r = jnp.where(
            dist_codes == 0,
            beta_normal,
            jnp.where(
                dist_codes == 1,
                beta_lognormal,
                jnp.where(dist_codes == 2, beta_triangular, beta_uniform),
            ),
        )  # (n_obs, k_random)

        # Random utility: broadcast per-obs random coefficients with design matrix
        V_random = jnp.sum(
            dm_random.reshape(n_obs, n_alts, k_random) * beta_r[:, None, :],
            axis=2,
        )
        V_total = V_fixed_star + V_random
        log_probs = nested_log_probs(V_total, lambdas, nest_matrix, available)
        return jnp.exp((log_probs * chosen).sum(axis=1))

    probs_sim = jax.vmap(_prob_single_draw)(jnp.arange(n_draws)).mean(axis=0)
    ll = jnp.sum(jnp.log(jnp.maximum(probs_sim, 1e-30)) * weights)
    return ll


def build_sar_mixed_nested_objective(
    arrays, W_sparse, nest_matrix, random_col_indices, random_distributions, draws
) -> Objective:
    """Build Objective for SAR + Mixed + Nested estimation."""
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
    n_nests = nest_matrix.shape[1]
    nest_matrix_jax = jnp.array(nest_matrix, dtype=jnp.float64)

    @jax.jit
    def _ll_jax(params):
        return _sar_mixed_nested_ll_core(
            params,
            data.dm_fixed,
            data.dm_random,
            data.available,
            data.chosen,
            data.weights,
            data.inclusion_probs,
            W_dense,
            nest_matrix_jax,
            data.dist_codes,
            data.draws,
            n_obs,
            n_alts,
            k_fixed,
            k_random,
            n_draws,
            n_nests,
        )

    @jax.jit
    def _grad_jax(params):
        return jax.grad(_sar_mixed_nested_ll_core, argnums=0)(
            params,
            data.dm_fixed,
            data.dm_random,
            data.available,
            data.chosen,
            data.weights,
            data.inclusion_probs,
            W_dense,
            nest_matrix_jax,
            data.dist_codes,
            data.draws,
            n_obs,
            n_alts,
            k_fixed,
            k_random,
            n_draws,
            n_nests,
        )

    param_names_list = list(arrays.param_names)
    fixed_names = [name for i, name in enumerate(param_names_list) if i not in random_col_indices]
    random_names = [param_names_list[i] for i in random_col_indices]
    display_names = (
        fixed_names
        + ["rho"]
        + [f"lambda_{i}" for i in range(n_nests)]
        + [f"mean_{n}" for n in random_names]
        + [f"sd_{n}" for n in random_names]
    )

    transforms = (
        [Identity()] * k_fixed
        + [Tanh()]
        + [Sigmoid(0, 1)] * n_nests
        + [Identity()] * k_random
        + [SoftPlus()] * k_random
    )
    transform = ParamTransform(transforms)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        param_names=display_names,
        transform=transform,
    )
