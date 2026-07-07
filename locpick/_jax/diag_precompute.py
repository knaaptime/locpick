"""Precomputed diagonal interpolation for SAR variance normalization.

The variance-normalization diagonal ``D(ρ) = diag((I - ρW)^{-1})`` depends
only on ρ (not on β).  It is evaluated at every solver iteration but
changes only when ρ changes.  This module precomputes ``D(ρ)`` at a set
of nodes and interpolates via:

- **Chebyshev polynomial** (symmetric W, smooth function): Evaluate at
  Chebyshev nodes via sparse Cholesky, fit Chebyshev coefficients via
  DCT-I, evaluate per-iteration via Clenshaw recurrence in pure JAX.
- **AAA rational approximation** (non-symmetric W, singularities): Evaluate
  at a coarse grid via sparse LU, fit AAA support points/weights, evaluate
  per-iteration via barycentric formula in pure JAX.

Both evaluation paths are **pure JAX** (differentiable, JIT-compatible) —
no custom VJP needed for D(ρ).  JAX autodiff computes ∂D/∂ρ automatically.

This replaces the 20-term power series approximation
(``_diag_inv_power_series``) which:
- Fails for large ρ (converges only for |ρ| < 1/ω_max)
- Creates a large JIT graph (20 unrolled iterations)
- Is approximate, not exact
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .sparse_solve import evaluate_diagonal_at_nodes, is_symmetric

# ---------------------------------------------------------------------------
# Chebyshev interpolation (symmetric W)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChebyshevDiagPrecompute:
    """Precomputed Chebyshev coefficients for ``diag((I - ρW)^{-1})``.

    Attributes
    ----------
    coeffs : np.ndarray, shape (order, n_alts)
        Chebyshev coefficients per alternative.
    rho_min : float
        Lower bound of the ρ approximation interval.
    rho_max : float
        Upper bound of the ρ approximation interval.
    order : int
        Chebyshev polynomial degree.
    n_alts : int
        Number of alternatives.
    """

    coeffs: np.ndarray
    rho_min: float
    rho_max: float
    order: int
    n_alts: int


def chebyshev_diag_precompute(
    W_sparse: sp.csr_matrix,
    order: int = 20,
    rho_min: float = -0.95,
    rho_max: float = 0.95,
    use_cholmod: bool = True,
) -> ChebyshevDiagPrecompute:
    """Precompute Chebyshev coefficients for ``diag((I - ρW)^{-1})``.

    Evaluates the diagonal exactly at ``order`` Chebyshev nodes via sparse
    Cholesky (with symbolic reuse), then fits Chebyshev coefficients via DCT-I.

    Parameters
    ----------
    W_sparse : scipy.sparse.csr_matrix
        Row-standardised spatial weights matrix.
    order : int, default 20
        Chebyshev polynomial degree (number of nodes).
    rho_min : float, default -0.95
        Lower bound of the ρ approximation interval.
    rho_max : float, default 0.95
        Upper bound of the ρ approximation interval.
    use_cholmod : bool, default True
        Whether to use CHOLMOD (if available).

    Returns
    -------
    ChebyshevDiagPrecompute
    """
    n = W_sparse.shape[0]

    # Chebyshev nodes (Clenshaw-Curtis): cos((2j-1)π/(2*order))
    k = np.arange(1, order + 1)
    nodes_cos = np.cos((2 * k - 1) * np.pi / (2 * order))
    rho_nodes = 0.5 * (rho_max - rho_min) * nodes_cos + 0.5 * (rho_max + rho_min)

    # Evaluate diagonal at each node via sparse factorization
    D_samples = evaluate_diagonal_at_nodes(W_sparse, rho_nodes, use_cholmod=use_cholmod)
    # D_samples: (order, n_alts)

    # DCT-I → Chebyshev coefficients per alternative
    coeffs = np.zeros((order, n), dtype=np.float64)
    for j in range(order):
        scale = 2.0 / order if j > 0 else 1.0 / order
        coeffs[j] = scale * np.sum(
            D_samples * np.cos(j * (2 * k - 1) * np.pi / (2 * order))[:, None],
            axis=0,
        )

    return ChebyshevDiagPrecompute(
        coeffs=coeffs,
        rho_min=rho_min,
        rho_max=rho_max,
        order=order,
        n_alts=n,
    )


def chebyshev_diag_eval_jax(pre: ChebyshevDiagPrecompute, rho):
    """Evaluate ``D(ρ)`` from Chebyshev coefficients in pure JAX.

    Uses Clenshaw recurrence: O(order) per evaluation.
    Fully differentiable and JIT-compatible.

    Parameters
    ----------
    pre : ChebyshevDiagPrecompute
        Precomputed coefficients.
    rho : jnp.ndarray (scalar)
        Spatial autoregressive parameter.

    Returns
    -------
    jnp.ndarray, shape (n_alts,)
        Diagonal of ``(I - ρW)^{-1}``.
    """
    import jax.numpy as jnp

    x = (2.0 * rho - pre.rho_max - pre.rho_min) / (pre.rho_max - pre.rho_min)
    m = pre.order
    c = jnp.asarray(pre.coeffs, dtype=jnp.float64)  # (order, n_alts)

    if m == 0:
        return jnp.zeros(pre.n_alts)
    if m == 1:
        return c[0]

    # Clenshaw recurrence, vectorized over alternatives
    b_next = jnp.zeros(pre.n_alts)
    b_curr = c[m - 1]
    for k in range(m - 2, 0, -1):
        b_new = 2.0 * x * b_curr - b_next + c[k]
        b_next = b_curr
        b_curr = b_new

    return c[0] + x * b_curr - b_next


# ---------------------------------------------------------------------------
# AAA rational approximation (non-symmetric W)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AAADiagPrecompute:
    """Precomputed AAA rational approximant for ``diag((I - ρW)^{-1})``.

    Attributes
    ----------
    support_points : np.ndarray, shape (m, n_alts)
        ρ values where exact diagonal was computed (AAA support points).
    support_values : np.ndarray, shape (m, n_alts)
        Exact diagonal values at support points.
    weights : np.ndarray, shape (m, n_alts)
        Barycentric weights from the AAA algorithm.
    rho_min : float
    rho_max : float
    n_alts : int
    """

    support_points: np.ndarray
    support_values: np.ndarray
    weights: np.ndarray
    rho_min: float
    rho_max: float
    n_alts: int


def _aaa_algorithm(
    z: np.ndarray,
    f: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core AAA algorithm for rational approximation.

    Given sample points ``z`` and function values ``f``, find support
    points, values, and barycentric weights for a rational approximant.

    Parameters
    ----------
    z : np.ndarray, shape (M,)
        Sample points.
    f : np.ndarray, shape (M,)
        Function values at sample points.
    tol : float
        Relative tolerance for convergence.
    max_iter : int
        Maximum number of support points.

    Returns
    -------
    support_points, support_values, weights : np.ndarray
    """
    z = np.asarray(z, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    M = len(z)

    is_support = np.zeros(M, dtype=bool)
    residual = f.copy()
    weights_list = [1.0]

    for m in range(1, min(max_iter, M // 2) + 1):
        candidate_residual = np.abs(residual).copy()
        candidate_residual[is_support] = -1
        next_idx = np.argmax(candidate_residual)

        if candidate_residual[next_idx] < tol * np.max(np.abs(f)):
            break

        is_support[next_idx] = True
        sp_z = z[is_support]
        sp_f = f[is_support]
        m_curr = len(sp_z)

        non_support = ~is_support
        z_ns = z[non_support]
        f_ns = f[non_support]

        if m_curr == 1:
            weights_list = [1.0]
            residual = f - sp_f[0]
            continue

        # Loewner matrix
        n_ns = len(z_ns)
        A = np.zeros((n_ns, m_curr), dtype=np.float64)
        for j in range(m_curr):
            A[:, j] = (f_ns - sp_f[j]) / (z_ns - sp_z[j])

        # SVD: min ||A @ w|| s.t. ||w|| = 1
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        w = Vt[-1]
        weights_list = w

        # Residual
        n_val = np.zeros(n_ns)
        d_val = np.zeros(n_ns)
        for j in range(m_curr):
            diff = z_ns - sp_z[j]
            n_val += w[j] * sp_f[j] / diff
            d_val += w[j] / diff

        r_ns = np.where(np.abs(d_val) > 1e-15, n_val / d_val, sp_f[0])
        residual = f.copy()
        residual[non_support] = f_ns - r_ns

    return z[is_support], f[is_support], np.array(weights_list, dtype=np.float64)


def aaa_diag_precompute(
    W_sparse: sp.csr_matrix,
    rho_min: float = -0.95,
    rho_max: float = 0.95,
    n_coarse: int = 30,
    tol: float = 1e-10,
    max_iter: int = 30,
    use_cholmod: bool = False,
) -> AAADiagPrecompute:
    """Precompute AAA rational approximants for ``diag((I - ρW)^{-1})``.

    Uses the lazy AAA strategy: evaluate at ``n_coarse`` Chebyshev-spaced
    points (not 200), then run AAA on those.  The ~30 sparse LU
    factorizations are shared across all ``n_alts`` diagonal components.

    Parameters
    ----------
    W_sparse : scipy.sparse.csr_matrix
        Row-standardised spatial weights matrix.
    rho_min : float, default -0.95
    rho_max : float, default 0.95
    n_coarse : int, default 30
        Number of Chebyshev-spaced evaluation points.
    tol : float
        AAA convergence tolerance.
    max_iter : int
        Maximum AAA support points.
    use_cholmod : bool, default False
        Whether to use CHOLMOD (typically False for non-symmetric W).

    Returns
    -------
    AAADiagPrecompute
    """
    n = W_sparse.shape[0]

    # Chebyshev-spaced coarse grid
    k = np.arange(1, n_coarse + 1)
    coarse_cos = np.cos((2 * k - 1) * np.pi / (2 * n_coarse))
    rho_coarse = 0.5 * (rho_max - rho_min) * coarse_cos + 0.5 * (rho_max + rho_min)

    # Evaluate diagonal at coarse points (shared across all components)
    D_coarse = evaluate_diagonal_at_nodes(W_sparse, rho_coarse, use_cholmod=use_cholmod)
    # D_coarse: (n_coarse, n_alts)

    # Fit AAA per component
    all_sp_z = []
    all_sp_f = []
    all_w = []

    for j in range(n):
        sp_z, sp_f, w = _aaa_algorithm(rho_coarse, D_coarse[:, j], tol=tol, max_iter=max_iter)
        all_sp_z.append(sp_z)
        all_sp_f.append(sp_f)
        all_w.append(w)

    # Pad to same length for vectorized JAX evaluation
    max_m = max(len(s) for s in all_sp_z)
    support_points = np.zeros((max_m, n), dtype=np.float64)
    support_values = np.zeros((max_m, n), dtype=np.float64)
    weights = np.zeros((max_m, n), dtype=np.float64)
    for j in range(n):
        m = len(all_sp_z[j])
        support_points[:m, j] = all_sp_z[j]
        support_values[:m, j] = all_sp_f[j]
        weights[:m, j] = all_w[j]
        # Pad weights with 0 so they don't contribute

    return AAADiagPrecompute(
        support_points=support_points,
        support_values=support_values,
        weights=weights,
        rho_min=rho_min,
        rho_max=rho_max,
        n_alts=n,
    )


def aaa_diag_eval_jax(pre: AAADiagPrecompute, rho):
    """Evaluate ``D(ρ)`` from AAA fits in pure JAX.

    Uses the barycentric formula: O(m) per component.
    Fully differentiable and JIT-compatible.

    Parameters
    ----------
    pre : AAADiagPrecompute
        Precomputed approximant.
    rho : jnp.ndarray (scalar)
        Spatial autoregressive parameter.

    Returns
    -------
    jnp.ndarray, shape (n_alts,)
        Diagonal of ``(I - ρW)^{-1}``.
    """
    import jax.numpy as jnp

    sp_z = jnp.asarray(pre.support_points, dtype=jnp.float64)  # (m, n_alts)
    sp_f = jnp.asarray(pre.support_values, dtype=jnp.float64)  # (m, n_alts)
    w = jnp.asarray(pre.weights, dtype=jnp.float64)  # (m, n_alts)

    # Barycentric formula, vectorized over alternatives
    diff = rho - sp_z  # (m, n_alts)
    n_val = jnp.sum(w * sp_f / diff, axis=0)  # (n_alts,)
    d_val = jnp.sum(w / diff, axis=0)  # (n_alts,)

    return n_val / d_val


# ---------------------------------------------------------------------------
# Auto-selecting factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagPrecompute:
    """Precomputed diagonal interpolation (Chebyshev or AAA).

    Auto-selects Chebyshev for symmetric W, AAA for non-symmetric W.
    """

    method: str  # "chebyshev" or "aaa"
    cheb_pre: ChebyshevDiagPrecompute | None = None
    aaa_pre: AAADiagPrecompute | None = None
    n_alts: int = 0

    def eval_jax(self, rho):
        """Evaluate D(ρ) in pure JAX."""
        if self.method == "chebyshev":
            return chebyshev_diag_eval_jax(self.chb_pre, rho)
        return aaa_diag_eval_jax(self.aaa_pre, rho)


def precompute_diagonal(
    W_sparse: sp.csr_matrix,
    rho_min: float = -0.95,
    rho_max: float = 0.95,
    order: int = 20,
    n_coarse: int = 30,
) -> DiagPrecompute:
    """Auto-select and precompute diagonal interpolation.

    Uses Chebyshev for symmetric W (contiguity, rook, queen) and AAA
    for non-symmetric W (KNN, directed graphs).

    Parameters
    ----------
    W_sparse : scipy.sparse.csr_matrix
        Row-standardised spatial weights matrix.
    rho_min : float, default -0.95
    rho_max : float, default 0.95
    order : int, default 20
        Chebyshev order (for symmetric W).
    n_coarse : int, default 30
        AAA coarse grid size (for non-symmetric W).

    Returns
    -------
    DiagPrecompute
    """
    n = W_sparse.shape[0]

    if is_symmetric(W_sparse):
        pre = chebyshev_diag_precompute(W_sparse, order=order, rho_min=rho_min, rho_max=rho_max)
        return DiagPrecompute(method="chebyshev", chb_pre=pre, n_alts=n)

    pre = aaa_diag_precompute(W_sparse, rho_min=rho_min, rho_max=rho_max, n_coarse=n_coarse)
    return DiagPrecompute(method="aaa", aaa_pre=pre, n_alts=n)
