"""JAX kernels and objective builder for SAR-MNL PML estimation.

Implements the pseudo maximum likelihood (PML) estimator from
Smirnov (2010): spatially-filtered utilities with variance
normalisation by ``diag((I - ρW)^{-1})``, then standard MNL softmax.

The spatial filter ``(I - ρW)^{-1} V_base`` is applied by
:func:`_sar_filter`, shared across all SAR families.  When the model
supplies a ``sparse_solve_fn`` (sparsax CHOLMOD/KLU; see
:mod:`locpick._jax.sparse_backends`) the solve is sparse, differentiable
and JIT-native; otherwise it is a dense LU solve.

The variance-normalisation diagonal ``diag((I - ρW)^{-1})`` comes from a
precomputed Chebyshev/AAA interpolant (see
:mod:`locpick._jax.diag_precompute`), falling back to an exact dense
inverse when no interpolant is supplied.
"""

from __future__ import annotations

import dataclasses

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
    random_spread,
)
from .objective import Objective, make_log_probs_fn
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
    normalize=True,
    lowrank=None,
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
    # Unavailable alternatives still exist spatially, so utilities are
    # filtered unmasked; availability is applied to the filtered values.
    V_star = _sar_filtered_utilities(
        rho,
        beta,
        design_matrix,
        n_obs,
        n_alts,
        W_dense,
        diag_eval_fn,
        sparse_solve_fn,
        normalize,
        inclusion_probs=inclusion_probs,
        lowrank=lowrank,
    )

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
    normalize=True,
    lowrank=None,
):
    """Per-observation SAR-MNL PML log-likelihood contributions — dense path."""
    k = design_matrix.shape[1]
    beta = params[:k]
    alpha_rho = params[k]
    rho = jnp.tanh(alpha_rho)

    # Unavailable alternatives still exist spatially, so utilities are
    # filtered unmasked; availability is applied to the filtered values.
    V_star = _sar_filtered_utilities(
        rho,
        beta,
        design_matrix,
        n_obs,
        n_alts,
        W_dense,
        diag_eval_fn,
        sparse_solve_fn,
        normalize,
        inclusion_probs=inclusion_probs,
        lowrank=lowrank,
    )

    log_probs = mnl_log_probs(V_star, available)
    return compute_ll_contribs(log_probs, chosen, weights)


def _jit_with_operands(fn, *operands):
    """JIT ``params -> fn(params, *operands)``, capturing ``operands``.

    The data, dense ``W`` and low-rank design factors are closed over, so XLA
    treats them as compile-time constants and may fold work on them.  Passing
    them as traced arguments instead was benchmarked and made no consistent
    difference to compile or per-iteration time.
    """
    return jax.jit(lambda params: fn(params, *operands))


def _diag_inv_exact_dense(rho, W_dense, n_alts):
    """Compute ``diag((I - rho*W)^{-1})`` exactly via a dense inverse.

    Used when no precomputed interpolant is supplied (small ``n_alts``,
    where the O(n^3) inverse is cheaper than fitting an interpolant).
    For larger problems ``build_sar_*`` passes ``diag_eval_fn`` from
    :mod:`locpick._jax.diag_precompute` instead.
    """
    A = jnp.eye(n_alts) - rho * W_dense
    return jnp.diag(jnp.linalg.inv(A))


def _maybe_dense_W(W_sparse, sparse_solve_fn, diag_precompute, normalize=True):
    """Densify ``W`` only when a dense path will actually read it.

    :func:`_sar_filter` touches ``W_dense`` in exactly two places: the LU
    solve, when no ``sparse_solve_fn`` is supplied, and
    :func:`_diag_inv_exact_dense`, when normalising without an interpolant.
    When those are covered the dense matrix is never read, so materialising it
    would pin ``n_alts**2`` float64 on device for nothing — and that is
    precisely the large-``n_alts`` regime the sparse backends exist to serve
    (3.2 GB at ``n_alts`` = 20k).  Returns None when it is not needed.
    """
    if sparse_solve_fn is None:
        return jnp.array(W_sparse.toarray(), dtype=jnp.float64)
    # Sparse solve covers the filter; the diagonal is the only other reader,
    # and the reduced form never asks for it.
    if not normalize or diag_precompute is not None:
        return None
    return jnp.array(W_sparse.toarray(), dtype=jnp.float64)


def _sar_solve(rho, B, W_dense, n_alts, sparse_solve_fn=None):
    """Apply ``(I - ρW)^{-1}`` along the alternative axis of every row of ``B``.

    ``sparse_solve_fn`` (sparsax, differentiable and JIT-native) replaces the
    dense factorisation when supplied.  Rows are independent right-hand
    sides, so ``B`` may hold observations or low-rank design factors.
    """
    if sparse_solve_fn is not None:
        return sparse_solve_fn(rho, B)
    A = jnp.eye(n_alts) - rho * W_dense
    return jax.scipy.linalg.solve(A, B.T).T


def _sar_normalise(rho, V_filtered, W_dense, n_alts, diag_eval_fn=None, normalize=True):
    """Divide filtered utilities by ``D = diag((I - ρW)^{-1})`` under PML."""
    if not normalize:
        return V_filtered
    if diag_eval_fn is not None:
        D = diag_eval_fn(rho)
    else:
        D = _diag_inv_exact_dense(rho, W_dense, n_alts)
    return V_filtered / D[None, :]


def _sar_filter(
    rho, V_base, W_dense, n_alts, diag_eval_fn=None, sparse_solve_fn=None, normalize=True
):
    """Apply the SAR spatial filter, optionally with variance normalisation.

    Two SAR specifications share this filter, differing only in ``normalize``:

    ``normalize=True`` (Smirnov 2010 PML) returns ``V_star = V_filtered / D``
    with ``D = diag((I - ρW)^{-1})``.  It is the pseudo-likelihood for the model
    in which the *disturbances* are filtered too, so ``Var(u_j)`` varies by
    alternative; dividing by ``D`` standardises that heteroskedasticity before
    the softmax, which logit cannot otherwise accommodate.

    ``normalize=False`` (reduced form) returns ``V_filtered`` directly.  This is
    the model in which only *systematic* utility reaches a spatial equilibrium,
    ``ψ = ρWψ + Xβ``, while the idiosyncratic errors stay iid EV1 — so the
    softmax over ``(I - ρW)^{-1} Xβ`` is the *exact* likelihood, not a
    pseudo-likelihood.  It needs no diagonal at all, which removes the
    interpolant precompute from the critical path.

    Both give the same score at ``ρ = 0`` (``∂ψ/∂ρ = Wψ₀``), because
    ``d/dρ diag((I - ρW)^{-1})|₀ = diag(W) = 0``.
    """
    V_filtered = _sar_solve(rho, V_base, W_dense, n_alts, sparse_solve_fn)
    return _sar_normalise(rho, V_filtered, W_dense, n_alts, diag_eval_fn, normalize)


def _sar_filtered_utilities(
    rho,
    beta,
    design_matrix,
    n_obs,
    n_alts,
    W_dense,
    diag_eval_fn=None,
    sparse_solve_fn=None,
    normalize=True,
    inclusion_probs=None,
    lowrank=None,
):
    """Filtered (and, under PML, normalised) utilities ``(I - ρW)^{-1} Xβ``.

    With ``lowrank = (U, S_T, owner)`` from :func:`design_lowrank`, the base
    utilities are ``V_base = (U * beta[owner]) @ S_T``, and since the filter
    is linear, ``(I - ρW)^{-1}`` is applied to the ``R`` rows of ``S_T``
    instead of the ``n_obs`` rows of ``V_base``: ``R`` solves per evaluation
    rather than one per chooser.  The result is identical.

    Utilities are never masked for availability here: an unavailable
    alternative still exists spatially and passes its utility to its
    neighbours.  Callers apply availability to the filtered values.
    """
    if lowrank is not None:
        U, S_T, owner = lowrank
        F = _sar_solve(rho, S_T, W_dense, n_alts, sparse_solve_fn)
        V_filtered = (U * beta[owner][None, :]) @ F
    else:
        if design_matrix is None:
            V_base = jnp.zeros((n_obs, n_alts), dtype=jnp.float64)
            if inclusion_probs is not None:
                V_base = V_base + jnp.log(jnp.maximum(inclusion_probs, 1e-30))
        else:
            V_base = compute_utilities(
                design_matrix, beta, n_obs, n_alts, inclusion_probs=inclusion_probs
            )
        V_filtered = _sar_solve(rho, V_base, W_dense, n_alts, sparse_solve_fn)
    return _sar_normalise(rho, V_filtered, W_dense, n_alts, diag_eval_fn, normalize)


def _sar_filtered_columns(
    rho,
    dm_cols,
    n_obs,
    n_alts,
    W_dense,
    diag_eval_fn=None,
    sparse_solve_fn=None,
    normalize=True,
    lowrank=None,
):
    """Filter each design column separately: column ``p`` becomes ``L(X_p)``.

    ``L`` (the solve, then division by ``D`` under PML) is linear along the
    alternative axis, and a random coefficient is constant across a
    chooser's alternatives, so ``L(b X_p) = b L(X_p)``: filtering the random
    design once per evaluation filters every draw's random utility exactly.

    Parameters
    ----------
    dm_cols : jnp.ndarray, shape (n_obs * n_alts, k)
    lowrank : tuple or None
        Factors of these columns from :func:`design_lowrank`.

    Returns
    -------
    jnp.ndarray, shape (n_obs * n_alts, k)
    """
    k = dm_cols.shape[1]
    if lowrank is not None:
        U, S_T, owner = lowrank
        F = _sar_solve(rho, S_T, W_dense, n_alts, sparse_solve_fn)  # (R, n_alts)
        onehot = (owner[:, None] == jnp.arange(k)[None, :]).astype(jnp.float64)  # (R, k)
        X_f = jnp.einsum("nr,rp,rj->njp", U, onehot, F)
    else:
        X = dm_cols.reshape(n_obs, n_alts, k).transpose(2, 0, 1).reshape(k * n_obs, n_alts)
        X_f = _sar_solve(rho, X, W_dense, n_alts, sparse_solve_fn)
        X_f = X_f.reshape(k, n_obs, n_alts).transpose(1, 2, 0)
    if normalize:
        D = (
            diag_eval_fn(rho)
            if diag_eval_fn is not None
            else _diag_inv_exact_dense(rho, W_dense, n_alts)
        )
        X_f = X_f / D[None, :, None]
    return X_f.reshape(n_obs * n_alts, k)


def design_lowrank(arrays, columns=None, max_rank_frac=0.25):
    """Exact low-rank factors of design columns in obs × alt layout.

    Each design column ``X_k`` (``n_obs × n_alts``) is written exactly as
    ``U_k S_k'``:

    - rank one, ``X_k = u v'``, covers alternative attributes (``u = 1``),
      chooser × alternative interactions and chooser attributes;
    - otherwise its distinct rows, with ``U_k`` the row-membership indicator,
      covers variables such as distance from a limited set of origins.

    The stacked factors give ``V_base = (U * beta[owner]) @ S_T``.  Returns
    None, so callers filter every observation as before, when some column
    has neither structure, when the total rank is not well below ``n_obs``,
    or when a sampling correction enters the base utilities.

    Parameters
    ----------
    arrays : ChoiceArrays
    columns : list of int or None
        Design columns to factor, in order; all columns when None.
    max_rank_frac : float, default 0.25
        Use the factors only when their total rank is below this fraction of
        ``n_obs``.

    Returns
    -------
    tuple of jnp.ndarray or None
        ``(U, S_T, owner)`` with shapes ``(n_obs, R)``, ``(R, n_alts)`` and
        ``(R,)``, where ``owner`` indexes positions in ``columns``, or None.
    """
    if arrays.inclusion_probs is not None:
        return None
    n_obs, n_alts = arrays.n_obs, arrays.n_alts
    dm = np.asarray(arrays.design_matrix, dtype=np.float64)
    cols = list(range(dm.shape[1])) if columns is None else list(columns)
    if not cols:
        return None
    budget = max_rank_frac * n_obs

    U_parts, S_parts, owner = [], [], []
    for pos, k in enumerate(cols):
        X = dm[:, k].reshape(n_obs, n_alts)
        scale = np.abs(X).max()
        if scale == 0.0:
            continue  # contributes nothing to utility
        v = X[np.argmax(np.abs(X).sum(axis=1))]
        u = X @ v / (v @ v)
        if np.abs(X - np.outer(u, v)).max() <= 1e-12 * scale * max(1.0, np.abs(u).max()):
            U_parts.append(u[:, None])
            S_parts.append(v[None, :])
            owner.append(pos)
            continue
        uniq, inv = np.unique(X, axis=0, return_inverse=True)
        if len(uniq) >= budget:
            return None
        ind = np.zeros((n_obs, len(uniq)))
        ind[np.arange(n_obs), inv.ravel()] = 1.0
        U_parts.append(ind)
        S_parts.append(uniq)
        owner.extend([pos] * len(uniq))

    if not U_parts or len(owner) >= budget:
        return None
    return (
        jnp.asarray(np.hstack(U_parts), dtype=jnp.float64),
        jnp.asarray(np.vstack(S_parts), dtype=jnp.float64),
        jnp.asarray(np.array(owner, dtype=np.int32)),
    )


def build_sar_mnl_objective(
    arrays,
    W_sparse: sp.csr_array,
    diag_precompute=None,
    sparse_solve_fn=None,
    normalize=True,
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
        (sparsax CHOLMOD/KLU).  When None, a dense solve is used.

    Returns
    -------
    Objective
        Objective with JIT-compiled LL, gradient, and Hessian.
        Includes a Tanh transform for the rho parameter.
    """
    data = ChoiceDataJAX.from_arrays(arrays)
    W_dense = _maybe_dense_W(W_sparse, sparse_solve_fn, diag_precompute, normalize)
    lowrank = design_lowrank(arrays)
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

    # Data, W and the low-rank factors are captured (see _jit_with_operands)
    def _ll(params, data, W_dense, lowrank):
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
            normalize=normalize,
            lowrank=lowrank,
        )

    def _contribs(params, data, W_dense, lowrank):
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
            normalize=normalize,
            lowrank=lowrank,
        )

    def _grad(params, data, W_dense, lowrank):
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
            normalize=normalize,
            lowrank=lowrank,
        )

    param_names = list(arrays.param_names) + ["rho"]
    transform = ParamTransform.for_sar_mnl(k)

    _ll_jax = _jit_with_operands(_ll, data, W_dense, lowrank)
    _grad_jax = _jit_with_operands(_grad, data, W_dense, lowrank)
    _ll_contribs_jax = _jit_with_operands(_contribs, data, W_dense, lowrank)

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
    normalize=True,
    lowrank=None,
):
    """SAR-Nested PML log-likelihood — dense solve path."""
    beta = params[:k]
    alpha_rho = params[k]
    alpha_lambdas = params[k + 1 : k + 1 + n_nests]
    rho = jnp.tanh(alpha_rho)
    lambdas = 1.0 / (1.0 + jnp.exp(-alpha_lambdas))

    # Unavailable alternatives still exist spatially, so utilities are
    # filtered unmasked; availability is applied to the filtered values.
    V_star = _sar_filtered_utilities(
        rho,
        beta,
        design_matrix,
        n_obs,
        n_alts,
        W_dense,
        diag_eval_fn,
        sparse_solve_fn,
        normalize,
        inclusion_probs=inclusion_probs,
        lowrank=lowrank,
    )

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
    normalize=True,
    lowrank=None,
):
    """SAR-Nested per-observation LL contributions — dense path."""
    beta = params[:k]
    alpha_rho = params[k]
    alpha_lambdas = params[k + 1 : k + 1 + n_nests]
    rho = jnp.tanh(alpha_rho)
    lambdas = 1.0 / (1.0 + jnp.exp(-alpha_lambdas))

    # Unavailable alternatives still exist spatially, so utilities are
    # filtered unmasked; availability is applied to the filtered values.
    V_star = _sar_filtered_utilities(
        rho,
        beta,
        design_matrix,
        n_obs,
        n_alts,
        W_dense,
        diag_eval_fn,
        sparse_solve_fn,
        normalize,
        inclusion_probs=inclusion_probs,
        lowrank=lowrank,
    )

    log_probs = nested_log_probs(V_star, lambdas, nest_matrix, available)
    return compute_ll_contribs(log_probs, chosen, weights)


def build_sar_nested_objective(
    arrays,
    W_sparse: sp.csr_array,
    nest_matrix: np.ndarray,
    diag_precompute=None,
    sparse_solve_fn=None,
    normalize=True,
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
    W_dense = _maybe_dense_W(W_sparse, sparse_solve_fn, diag_precompute, normalize)
    lowrank = design_lowrank(arrays)
    n_obs = arrays.n_obs
    n_alts = arrays.n_alts
    k = arrays.design_matrix.shape[1]
    nest_matrix_jax = jnp.asarray(nest_matrix, dtype=jnp.float64)
    n_nests = nest_matrix.shape[1]

    ll_core = _sar_nested_ll_core
    ll_contribs_core = _sar_nested_ll_contribs_core

    diag_eval_fn = diag_precompute.eval_jax if diag_precompute is not None else None

    def _ll(params, data, W_dense, lowrank):
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
            normalize=normalize,
            lowrank=lowrank,
        )

    def _contribs(params, data, W_dense, lowrank):
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
            normalize=normalize,
            lowrank=lowrank,
        )

    def _contribs_cw(params, chosen, weights, available=None):
        return ll_contribs_core(
            params,
            data.design_matrix,
            data.available if available is None else available,
            chosen,
            weights,
            data.inclusion_probs,
            W_dense,
            n_obs,
            n_alts,
            nest_matrix_jax,
            k,
            n_nests,
            diag_eval_fn=diag_eval_fn,
            sparse_solve_fn=sparse_solve_fn,
            normalize=normalize,
            lowrank=lowrank,
        )

    def _grad(params, data, W_dense, lowrank):
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
            normalize=normalize,
            lowrank=lowrank,
        )

    param_names = list(arrays.param_names) + ["rho"] + [f"lambda_{i}" for i in range(n_nests)]
    transforms = [Identity() for _ in range(k)] + [Tanh()] + [Sigmoid() for _ in range(n_nests)]
    transform = ParamTransform(transforms=transforms)

    _ll_jax = _jit_with_operands(_ll, data, W_dense, lowrank)
    _grad_jax = _jit_with_operands(_grad, data, W_dense, lowrank)
    _ll_contribs_jax = _jit_with_operands(_contribs, data, W_dense, lowrank)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        loglike_contribs_jax=_ll_contribs_jax,
        param_names=param_names,
        transform=transform,
        log_probs_jax=make_log_probs_fn(_contribs_cw, n_obs, n_alts, mixed=False),
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
    normalize=True,
    lowrank=None,
    lowrank_random=None,
):
    """SAR-Mixed simulated PML log-likelihood."""
    from .kernels import mixed_logit_ll

    beta_fixed = params[:k_fixed]
    alpha_rho = params[k_fixed]
    beta_random_means = params[k_fixed + 1 : k_fixed + 1 + k_random]
    beta_random_spreads_raw = params[k_fixed + 1 + k_random :]
    rho = jnp.tanh(alpha_rho)
    beta_random_spreads = random_spread(beta_random_spreads_raw, k_random)

    # Filtered fixed utility (see _sar_filtered_utilities)
    v_fixed_star = _sar_filtered_utilities(
        rho,
        beta_fixed,
        data.dm_fixed if k_fixed > 0 else None,
        n_obs,
        n_alts,
        W_dense,
        diag_eval_fn,
        sparse_solve_fn,
        normalize,
        inclusion_probs=data.inclusion_probs,
        lowrank=lowrank,
    )

    return mixed_logit_ll(
        V_fixed=v_fixed_star,
        dm_random=_sar_filtered_columns(
            rho,
            data.dm_random,
            n_obs,
            n_alts,
            W_dense,
            diag_eval_fn,
            sparse_solve_fn,
            normalize,
            lowrank=lowrank_random,
        ),
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
        panel_codes=data.panel_codes,
        n_panels=data.n_panels,
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
    normalize=True,
    lowrank=None,
    lowrank_random=None,
):
    """SAR-Mixed per-observation LL contributions."""
    from .kernels import mixed_logit_ll_contribs

    beta_fixed = params[:k_fixed]
    alpha_rho = params[k_fixed]
    beta_random_means = params[k_fixed + 1 : k_fixed + 1 + k_random]
    beta_random_spreads_raw = params[k_fixed + 1 + k_random :]
    rho = jnp.tanh(alpha_rho)
    beta_random_spreads = random_spread(beta_random_spreads_raw, k_random)

    # Filtered fixed utility (see _sar_filtered_utilities)
    v_fixed_star = _sar_filtered_utilities(
        rho,
        beta_fixed,
        data.dm_fixed if k_fixed > 0 else None,
        n_obs,
        n_alts,
        W_dense,
        diag_eval_fn,
        sparse_solve_fn,
        normalize,
        inclusion_probs=data.inclusion_probs,
        lowrank=lowrank,
    )

    return mixed_logit_ll_contribs(
        V_fixed=v_fixed_star,
        dm_random=_sar_filtered_columns(
            rho,
            data.dm_random,
            n_obs,
            n_alts,
            W_dense,
            diag_eval_fn,
            sparse_solve_fn,
            normalize,
            lowrank=lowrank_random,
        ),
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
        panel_codes=data.panel_codes,
        n_panels=data.n_panels,
    )


def build_sar_mixed_objective(
    arrays,
    W_sparse: sp.csr_array,
    random_col_indices,
    random_distributions,
    draws,
    diag_precompute=None,
    sparse_solve_fn=None,
    normalize=True,
    panel=None,
) -> Objective:
    """Build an Objective for SAR-Mixed PML estimation.

    The SAR filter applies to the whole systematic utility, random part
    included.  Because a random coefficient is constant across alternatives
    for a given chooser and draw, filtering ``b * X_p`` equals ``b`` times the
    filtered ``X_p``, so the random design columns are filtered once per
    evaluation rather than once per draw (see :func:`_sar_filtered_columns`).

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
        panel=panel,
    )
    W_dense = _maybe_dense_W(W_sparse, sparse_solve_fn, diag_precompute, normalize)
    fixed_cols = [
        i for i in range(arrays.design_matrix.shape[1]) if i not in set(random_col_indices)
    ]
    lowrank = design_lowrank(arrays, columns=fixed_cols)
    lowrank_random = design_lowrank(arrays, columns=list(random_col_indices))
    n_obs = arrays.n_obs
    n_alts = arrays.n_alts
    k_fixed = len(data.fixed_col_indices) if data.fixed_col_indices else 0
    k_random = len(random_col_indices)
    n_draws = draws.shape[1]

    diag_eval_fn = diag_precompute.eval_jax if diag_precompute is not None else None

    def _ll(params, data, W_dense, lowrank):
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
            normalize=normalize,
            lowrank=lowrank,
            lowrank_random=lowrank_random,
        )

    def _contribs(params, data, W_dense, lowrank):
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
            normalize=normalize,
            lowrank=lowrank,
            lowrank_random=lowrank_random,
        )

    def _contribs_cw(params, chosen, weights, available=None):
        return _sar_mixed_ll_contribs_core(
            params,
            dataclasses.replace(
                data,
                chosen=chosen,
                weights=weights,
                available=data.available if available is None else available,
                panel_codes=None,
                n_panels=None,
            ),
            W_dense,
            n_obs,
            n_alts,
            k_fixed,
            k_random,
            n_draws,
            diag_eval_fn=diag_eval_fn,
            sparse_solve_fn=sparse_solve_fn,
            normalize=normalize,
            lowrank=lowrank,
            lowrank_random=lowrank_random,
        )

    def _grad(params, data, W_dense, lowrank):
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
            normalize=normalize,
            lowrank=lowrank,
            lowrank_random=lowrank_random,
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

    _ll_jax = _jit_with_operands(_ll, data, W_dense, lowrank)
    _grad_jax = _jit_with_operands(_grad, data, W_dense, lowrank)
    _ll_contribs_jax = _jit_with_operands(_contribs, data, W_dense, lowrank)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        loglike_contribs_jax=_ll_contribs_jax,
        param_names=display_param_names,
        transform=transform,
        log_probs_jax=make_log_probs_fn(_contribs_cw, n_obs, n_alts, mixed=True),
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
    normalize=True,
    lowrank=None,
    lowrank_random=None,
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
    beta_random_spreads = random_spread(beta_random_spreads_raw, k_random)

    # Filtered fixed utility (see _sar_filtered_utilities)
    v_fixed_star = _sar_filtered_utilities(
        rho,
        beta_fixed,
        data.dm_fixed if k_fixed > 0 else None,
        n_obs,
        n_alts,
        W_dense,
        diag_eval_fn,
        sparse_solve_fn,
        normalize,
        inclusion_probs=data.inclusion_probs,
        lowrank=lowrank,
    )

    return mixed_nested_logit_ll(
        V_fixed=v_fixed_star,
        dm_random=_sar_filtered_columns(
            rho,
            data.dm_random,
            n_obs,
            n_alts,
            W_dense,
            diag_eval_fn,
            sparse_solve_fn,
            normalize,
            lowrank=lowrank_random,
        ),
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
        panel_codes=data.panel_codes,
        n_panels=data.n_panels,
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
    normalize=True,
    lowrank=None,
    lowrank_random=None,
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
    beta_random_spreads = random_spread(beta_random_spreads_raw, k_random)

    # Filtered fixed utility (see _sar_filtered_utilities)
    v_fixed_star = _sar_filtered_utilities(
        rho,
        beta_fixed,
        data.dm_fixed if k_fixed > 0 else None,
        n_obs,
        n_alts,
        W_dense,
        diag_eval_fn,
        sparse_solve_fn,
        normalize,
        inclusion_probs=data.inclusion_probs,
        lowrank=lowrank,
    )

    return mixed_nested_logit_ll_contribs(
        V_fixed=v_fixed_star,
        dm_random=_sar_filtered_columns(
            rho,
            data.dm_random,
            n_obs,
            n_alts,
            W_dense,
            diag_eval_fn,
            sparse_solve_fn,
            normalize,
            lowrank=lowrank_random,
        ),
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
        panel_codes=data.panel_codes,
        n_panels=data.n_panels,
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
    normalize=True,
    panel=None,
) -> Objective:
    """Build an Objective for SAR-Mixed-Nested PML estimation.

    The SAR filter applies to the whole systematic utility, random part
    included (see :func:`build_sar_mixed_objective`); nested logit is then
    applied to the filtered utilities.

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
        panel=panel,
    )
    W_dense = _maybe_dense_W(W_sparse, sparse_solve_fn, diag_precompute, normalize)
    fixed_cols = [
        i for i in range(arrays.design_matrix.shape[1]) if i not in set(random_col_indices)
    ]
    lowrank = design_lowrank(arrays, columns=fixed_cols)
    lowrank_random = design_lowrank(arrays, columns=list(random_col_indices))
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

    def _ll(params, data, W_dense, lowrank):
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
            normalize=normalize,
            lowrank=lowrank,
            lowrank_random=lowrank_random,
        )

    def _contribs(params, data, W_dense, lowrank):
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
            normalize=normalize,
            lowrank=lowrank,
            lowrank_random=lowrank_random,
        )

    def _contribs_cw(params, chosen, weights, available=None):
        return _sar_mixed_nested_ll_contribs_core(
            params,
            dataclasses.replace(
                data,
                chosen=chosen,
                weights=weights,
                available=data.available if available is None else available,
                panel_codes=None,
                n_panels=None,
            ),
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
            normalize=normalize,
            lowrank=lowrank,
            lowrank_random=lowrank_random,
        )

    def _grad(params, data, W_dense, lowrank):
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
            normalize=normalize,
            lowrank=lowrank,
            lowrank_random=lowrank_random,
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

    _ll_jax = _jit_with_operands(_ll, data, W_dense, lowrank)
    _grad_jax = _jit_with_operands(_grad, data, W_dense, lowrank)
    _ll_contribs_jax = _jit_with_operands(_contribs, data, W_dense, lowrank)

    return Objective.from_jax(
        ll_fn=_ll_jax,
        grad_fn=_grad_jax,
        loglike_contribs_jax=_ll_contribs_jax,
        param_names=display_param_names,
        transform=transform,
        log_probs_jax=make_log_probs_fn(_contribs_cw, n_obs, n_alts, mixed=True),
    )
