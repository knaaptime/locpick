"""Pure-JAX sparse solves for the SAR spatial filter.

The SAR-family kernels apply the spatial filter ``V = (I - ρW)^{-1} V_base``
at every optimiser step.  Done densely this is ``O(n_alts^3)`` per evaluation
and the conjugate-gradient path still materialises ``I - ρW``.  This module
provides sparse, JIT-compatible, autodiff-friendly replacements that scale
with the number of nonzeros in ``W``.

Both paths come from **sparsax**, which bundles CHOLMOD and KLU:

- **CHOLMOD**: used when ``W`` is *symmetrizable* — that is, its adjacency
  pattern is symmetric, as for contiguity/distance-band/KNN-made-symmetric
  graphs.  ``I - ρW`` is then similar to the SPD matrix ``I - ρW_sym``
  (``W_sym = D^{1/2} W D^{-1/2}``), which CHOLMOD factorises.
- **KLU**: used for genuinely asymmetric ``W`` (directed graphs, raw KNN), a
  sparse LU that needs no symmetry.

Both back the differentiable ``(rho, V_base) -> V_filtered`` closure returned
by :func:`make_sparse_solve_fn`, which slots into the ``sparse_solve_fn`` hook
of the SAR kernels.  The dependency is optional: when sparsax is absent the
factory returns ``None`` and the caller falls back to the dense path.

Symmetrisation is recovered from ``W`` itself, so it is correct whether or not
the caller has already row-standardised the weights (a row-standardised ``W``
no longer exposes the raw degrees).  ``W_sym = sqrt(W ∘ Wᵀ)`` gives the
symmetric similar matrix directly; the diagonal scaling ``s = D^{1/2}`` that
the solve needs is recovered up to a per-component constant (which cancels)
from ``log s_i - log s_j = ½ log(W_ji / W_ij)`` over a spanning forest.
"""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import scipy.sparse as sp


def sparsax_available() -> bool:
    """Whether the sparsax (CHOLMOD) backend is importable."""
    return find_spec("sparsax") is not None


def is_symmetrizable(W_sparse: sp.spmatrix, tol: float = 1e-10) -> bool:
    """Whether ``W``'s adjacency pattern is structurally symmetric.

    A row-standardised ``W = D^{-1}A`` with symmetric ``A`` is similar to the
    symmetric ``W_sym = D^{-1/2} A D^{-1/2}``, so ``I - ρW`` can be solved via a
    symmetric-positive-definite factorisation.  Structural (pattern) symmetry
    is what the CHOLMOD path needs; the numeric similarity transform then makes
    the values symmetric exactly.
    """
    W = sp.csr_matrix(W_sparse)
    W.setdiag(0.0)
    W.eliminate_zeros()
    pattern = (np.abs(W) > tol).astype(np.int8)
    return (pattern != pattern.T).nnz == 0


def symmetrize(W_row_std: sp.csr_matrix):
    """Return ``(W_sym, s)`` making ``I - ρW`` a scaled SPD system.

    ``W_sym = sqrt(W ∘ Wᵀ)`` is symmetric and similar to ``W``; ``s = D^{1/2}``
    (per connected component, up to a positive scale that cancels in the
    solve) satisfies ``W_sym = diag(s) W diag(1/s)``, i.e.::

        (I - ρW)^{-1} = diag(1/s) (I - ρW_sym)^{-1} diag(s).

    Requires a structurally symmetric pattern (see :func:`is_symmetrizable`).
    """
    W = sp.csr_matrix(W_row_std, dtype=np.float64)
    W.setdiag(0.0)
    W.eliminate_zeros()
    n = W.shape[0]

    # W_sym = sqrt(W_ij * W_ji), vectorised via the elementwise product with Wᵀ.
    W_sym = W.multiply(W.T)
    W_sym.data = np.sqrt(W_sym.data)
    W_sym = sp.csr_matrix(W_sym)

    # Recover log s over a spanning forest: log s_i - log s_j = ½ log(W_ji/W_ij).
    coo = W.tocoo()
    wji = {(r, c): v for r, c, v in zip(coo.row, coo.col, coo.data)}
    # The DFS below walks each component from its own root, so no separate
    # connected-components pass is needed.
    log_s = np.zeros(n, dtype=np.float64)
    seen = np.zeros(n, dtype=bool)
    indptr, indices, data = W.indptr, W.indices, W.data
    for root in range(n):
        if seen[root]:
            continue
        seen[root] = True
        stack = [root]
        while stack:
            i = stack.pop()
            for k in range(indptr[i], indptr[i + 1]):
                j = indices[k]
                if seen[j]:
                    continue
                w_ij = data[k]
                w_ji = wji.get((j, i), 0.0)
                if w_ij > 0.0 and w_ji > 0.0:
                    log_s[j] = log_s[i] - 0.5 * np.log(w_ji / w_ij)
                seen[j] = True
                stack.append(j)
    return W_sym, np.exp(log_s)


def _fixed_coo_pattern(M: sp.csr_matrix, n: int):
    """Static COO pattern of ``I - ρM`` plus its ρ-independent value pieces.

    The sparsity pattern of ``I - ρM`` does not depend on ρ, so it is captured
    once and only the values ``i_vals - ρ * m_vals`` are rebuilt per call.
    """
    pattern = sp.coo_matrix(sp.eye(n) + M)
    row = pattern.row.astype(np.int32)
    col = pattern.col.astype(np.int32)
    i_vals = (row == col).astype(np.float64)

    coo = sp.coo_matrix(M)
    lookup = {(r, c): v for r, c, v in zip(coo.row, coo.col, coo.data)}
    m_vals = np.array([lookup.get((r, c), 0.0) for r, c in zip(row, col)], dtype=np.float64)
    return row, col, i_vals, m_vals


def sparsax_node_diagonals(W_row_std: sp.csr_matrix, rho_nodes: np.ndarray) -> np.ndarray:
    """Exact ``diag((I - ρW)^{-1})`` at each ρ node via CHOLMOD selected inverse.

    Uses ``diag((I - ρW)^{-1}) = diag((I - ρW_sym)^{-1})`` and sparsax's
    ``selinv`` (Takahashi recurrence on the Cholesky factor), which reads off
    the diagonal without forming a dense inverse.  Intended for the interpolant
    precompute, so it runs outside autodiff.
    """
    import sparsax

    n = W_row_std.shape[0]
    W_sym, _ = symmetrize(W_row_std)
    row, col, i_vals, m_vals = _fixed_coo_pattern(W_sym, n)
    diag_mask = row == col

    out = np.zeros((len(rho_nodes), n), dtype=np.float64)
    for k, rho in enumerate(rho_nodes):
        z = np.asarray(sparsax.selinv(row, col, i_vals - float(rho) * m_vals, n))
        out[k, row[diag_mask]] = z[diag_mask]
    return out


def make_sparse_solve_fn(W_row_std: sp.csr_matrix, prefer: str = "auto"):
    """Build a differentiable sparse spatial-filter solve, or ``None``.

    Parameters
    ----------
    W_row_std : scipy.sparse matrix
        Row-standardised spatial weights (zero diagonal).
    prefer : {"auto", "sparsax", "sparsax_lu"}, default "auto"
        Force a path for testing.  ``"auto"`` picks sparsax's CHOLMOD solve for
        symmetrizable ``W`` and its KLU solve otherwise; ``"sparsax_lu"``
        forces KLU even when ``W`` is symmetrizable.

    Returns
    -------
    (callable, str) or (None, None)
        A JIT/autodiff-compatible ``solve(rho, V_base) -> V_filtered`` (both
        ``(n_obs, n_alts)``) and the path name — ``"sparsax"`` (CHOLMOD) or
        ``"sparsax_lu"`` (KLU) — or ``(None, None)`` when sparsax is absent.

        The two paths are named separately because only CHOLMOD admits
        :func:`sparsax_node_diagonals`: the selected inverse runs on the
        D-symmetrisation, which does not exist for asymmetric ``W``.
    """
    W = sp.csr_matrix(W_row_std, dtype=np.float64)
    W.setdiag(0.0)
    W.eliminate_zeros()
    n = W.shape[0]

    if not sparsax_available():
        return None, None

    if is_symmetrizable(W) and prefer in ("auto", "sparsax"):
        return _sparsax_solve_fn(W, n), "sparsax"
    return _sparsax_lu_solve_fn(W, n), "sparsax_lu"


def _sparsax_solve_fn(W_row_std: sp.csr_matrix, n: int):
    """CHOLMOD-backed filter via the SPD D-symmetrisation of ``I - ρW``."""
    import jax.numpy as jnp
    import sparsax

    W_sym, s = symmetrize(W_row_std)
    row, col, i_vals, m_vals = _fixed_coo_pattern(W_sym, n)

    ri = jnp.asarray(row)
    ci = jnp.asarray(col)
    iv = jnp.asarray(i_vals)
    wv = jnp.asarray(m_vals)
    sj = jnp.asarray(s)
    s_inv = jnp.asarray(1.0 / s)

    def solve(rho, V_base):
        # (I - ρW)^{-1} = diag(1/s) (I - ρW_sym)^{-1} diag(s); the filter acts on
        # the alternative axis, so solve with V_base transposed to (n_alts, n_obs).
        ax = iv - rho * wv
        x = sparsax.solve(ri, ci, ax, sj[:, None] * V_base.T)
        return (s_inv[:, None] * x).T

    return solve


def _sparsax_lu_solve_fn(W_row_std: sp.csr_matrix, n: int):
    """KLU-backed filter for asymmetric ``W``, via sparsax's LU path.

    The non-symmetrizable counterpart of :func:`_sparsax_solve_fn`: no
    D-symmetrisation exists, so ``I - ρW`` is solved directly by KLU rather
    than CHOLMOD.  Keeping both paths in sparsax means one native library
    serves every ``W``.
    """
    import jax.numpy as jnp
    import sparsax

    row, col, i_vals, m_vals = _fixed_coo_pattern(W_row_std, n)
    ri = jnp.asarray(row)
    ci = jnp.asarray(col)
    iv = jnp.asarray(i_vals)
    wv = jnp.asarray(m_vals)

    def solve(rho, V_base):
        # The filter acts on the alternative axis, so solve with V_base
        # transposed to (n_alts, n_obs), matching the CHOLMOD path.
        ax = iv - rho * wv
        return sparsax.lu_solve(ri, ci, ax, V_base.T).T

    return solve
