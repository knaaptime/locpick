"""Regression tests for SAR variance-normalization diagonal interpolation.

Guards the ``precompute_diagonal`` / ``DiagPrecompute.eval_jax`` path used
by SAR models when ``n_alts > 50``.  A field-name mismatch previously
crashed the symmetric-W (Chebyshev) branch; the existing SAR recovery
tests use ``n_alts == 50`` and never exercised it.

Both branches are validated against the exact dense diagonal
``diag((I - rho W)^{-1})``.
"""

import numpy as np
import numpy.testing as npt
import scipy.sparse as sp

from locpick._jax.diag_precompute import precompute_diagonal


def _circular_row_std_W(n: int) -> sp.csr_matrix:
    """Row-standardised circular adjacency (degree 2 everywhere → symmetric)."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, (i - 1) % n] = 1.0
        A[i, (i + 1) % n] = 1.0
    W = A / A.sum(axis=1, keepdims=True)
    return sp.csr_matrix(W)


def _path_row_std_W(n: int) -> sp.csr_matrix:
    """Row-standardised path graph (endpoints degree 1 → non-symmetric)."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if i > 0:
            A[i, i - 1] = 1.0
        if i < n - 1:
            A[i, i + 1] = 1.0
    W = A / A.sum(axis=1, keepdims=True)
    return sp.csr_matrix(W)


def _dense_diag(W: sp.csr_matrix, rho: float) -> np.ndarray:
    n = W.shape[0]
    return np.diag(np.linalg.inv(np.eye(n) - rho * W.toarray()))


def test_precompute_diagonal_symmetric_large_matches_dense():
    """Symmetric W with n_alts > 50 selects Chebyshev and matches dense diag."""
    n = 60
    W = _circular_row_std_W(n)
    pre = precompute_diagonal(W)

    assert pre.method == "chebyshev"
    for rho in [-0.9, -0.5, 0.0, 0.5, 0.9]:
        approx = np.asarray(pre.eval_jax(rho))
        npt.assert_allclose(approx, _dense_diag(W, rho), rtol=5e-3, atol=1e-6)


def test_precompute_diagonal_nonsymmetric_large_matches_dense():
    """Non-symmetric W with n_alts > 50 selects AAA and matches dense diag."""
    n = 60
    W = _path_row_std_W(n)
    pre = precompute_diagonal(W)

    assert pre.method == "aaa"
    for rho in [-0.9, -0.5, 0.0, 0.5, 0.9]:
        approx = np.asarray(pre.eval_jax(rho))
        npt.assert_allclose(approx, _dense_diag(W, rho), rtol=5e-3, atol=1e-6)
