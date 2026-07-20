"""Regression tests for SAR variance-normalization diagonal interpolation.

Guards the ``precompute_diagonal`` / ``DiagPrecompute.eval_jax`` path used
by SAR models when ``n_alts > 50``.  A field-name mismatch previously
crashed the symmetric-W (Chebyshev) branch; the existing SAR recovery
tests use ``n_alts == 50`` and never exercised it.

Both interpolation families are validated against the exact dense
diagonal ``diag((I - rho W)^{-1})``.
"""

import numpy as np
import numpy.testing as npt
import pytest
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


@pytest.mark.parametrize("builder", [_circular_row_std_W, _path_row_std_W])
def test_precompute_diagonal_matches_dense(builder):
    """The default (AAA) interpolant matches the exact diagonal closely."""
    n = 60
    W = builder(n)
    pre = precompute_diagonal(W)

    assert pre.method == "aaa"
    for rho in [-0.9, -0.5, 0.0, 0.5, 0.9]:
        approx = np.asarray(pre.eval_jax(rho))
        npt.assert_allclose(approx, _dense_diag(W, rho), rtol=1e-6, atol=1e-8)


def test_chebyshev_still_available_for_symmetric_w():
    """The Chebyshev family remains selectable for symmetric W."""
    n = 60
    W = _circular_row_std_W(n)
    pre = precompute_diagonal(W, method="chebyshev")

    assert pre.method == "chebyshev"
    for rho in [-0.9, -0.5, 0.0, 0.5, 0.9]:
        approx = np.asarray(pre.eval_jax(rho))
        npt.assert_allclose(approx, _dense_diag(W, rho), rtol=5e-3, atol=1e-6)


def test_chebyshev_rejects_nonsymmetric_w():
    with pytest.raises(ValueError, match="symmetric"):
        precompute_diagonal(_path_row_std_W(60), method="chebyshev")


@pytest.mark.parametrize("builder", [_circular_row_std_W, _path_row_std_W])
def test_eval_clamps_outside_fitted_interval(builder):
    """rho beyond the fitted range clamps instead of extrapolating."""
    W = builder(60)
    pre = precompute_diagonal(W)

    npt.assert_allclose(np.asarray(pre.eval_jax(0.99)), np.asarray(pre.eval_jax(0.95)))
    npt.assert_allclose(np.asarray(pre.eval_jax(-0.99)), np.asarray(pre.eval_jax(-0.95)))


@pytest.mark.parametrize("builder", [_circular_row_std_W, _path_row_std_W])
def test_eval_and_gradient_finite_at_zero(builder):
    """rho == 0 collides with the AAA zero padding; it must stay finite.

    rho = tanh(0) = 0 is the default starting point, so a 0/0 in the
    barycentric formula would poison the very first solver iteration.
    """
    import jax
    import jax.numpy as jnp

    W = builder(60)
    pre = precompute_diagonal(W)

    value = np.asarray(pre.eval_jax(0.0))
    assert np.all(np.isfinite(value))
    npt.assert_allclose(value, np.ones(60), rtol=1e-8, atol=1e-8)

    grad = jax.grad(lambda r: jnp.sum(pre.eval_jax(r)))(0.0)
    assert np.isfinite(grad)
