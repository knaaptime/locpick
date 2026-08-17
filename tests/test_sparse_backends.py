"""Tests for the sparse SAR spatial-filter backends.

Covers the JIT/autodiff-native solves (sparsax CHOLMOD for symmetrizable W, KLU
for asymmetric W) and the scipy sparse factorisations (CHOLMOD / KLU) used off
the JAX path, all validated against a dense reference.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp

_HAS_JAX = importlib.util.find_spec("jax") is not None
pytestmark = pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")

if _HAS_JAX:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

from locpick._jax.sparse_backends import (  # noqa: E402
    is_symmetrizable,
    make_sparse_solve_fn,
    sparsax_available,
    symmetrize,
)


def _row_std(A: sp.csr_matrix) -> sp.csr_matrix:
    A = sp.csr_matrix(A)
    A.setdiag(0.0)
    A.eliminate_zeros()
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1.0)
    return sp.csr_matrix(sp.diags(1.0 / deg) @ A)


def _symmetric_W(n=120, seed=1):
    A = sp.random(n, n, density=0.06, random_state=seed)
    A = ((A + A.T) > 0).astype(float)
    return _row_std(A)


def _asymmetric_W(n=120, seed=1):
    A = (sp.random(n, n, density=0.06, random_state=seed) > 0).astype(float)
    return _row_std(A)


def _dense_filter(W, rho, V_base):
    A = np.eye(W.shape[0]) - rho * W.toarray()
    return np.linalg.solve(A, V_base.T).T


# ---------------------------------------------------------------------------
# Symmetrisation
# ---------------------------------------------------------------------------


def test_symmetrize_recovers_symmetry_from_row_standardized_W():
    """W_sym must be symmetric even though row-standardised W is not.

    The raw degrees are gone after row-standardisation, so symmetrisation is
    recovered from W itself.
    """
    W = _symmetric_W()
    assert is_symmetrizable(W)
    W_sym, s = symmetrize(W)
    assert abs(W_sym - W_sym.T).max() < 1e-12

    # (I - ρW)^{-1} = diag(1/s)(I - ρW_sym)^{-1}diag(s)
    n = W.shape[0]
    rho = 0.6
    b = np.random.default_rng(0).standard_normal((n, 3))
    ref = np.linalg.solve(np.eye(n) - rho * W.toarray(), b)
    got = (1.0 / s)[:, None] * np.linalg.solve(np.eye(n) - rho * W_sym.toarray(), s[:, None] * b)
    npt.assert_allclose(got, ref, atol=1e-10)


def test_asymmetric_not_symmetrizable():
    assert not is_symmetrizable(_asymmetric_W())


# ---------------------------------------------------------------------------
# JAX-native solves
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not sparsax_available(), reason="sparsax not installed")
def test_sparsax_backend_matches_dense_and_differentiates():
    W = _symmetric_W()
    solve_fn, backend = make_sparse_solve_fn(W)
    assert backend == "sparsax"

    n = W.shape[0]
    V = jnp.asarray(np.random.default_rng(0).standard_normal((80, n)))
    for rho in [0.0, 0.4, -0.6, 0.9]:
        npt.assert_allclose(np.asarray(solve_fn(rho, V)), _dense_filter(W, rho, V), atol=1e-9)

    # JIT + gradient wrt rho against a finite difference.
    g = float(jax.grad(lambda r: jnp.sum(jax.jit(solve_fn)(r, V) ** 2))(0.4))
    eps = 1e-6
    obj = lambda r: np.sum(_dense_filter(W, r, V) ** 2)  # noqa: E731
    npt.assert_allclose(g, (obj(0.4 + eps) - obj(0.4 - eps)) / (2 * eps), rtol=1e-5)


@pytest.mark.skipif(not sparsax_available(), reason="sparsax not installed")
def test_sparsax_lu_backend_matches_dense_and_differentiates():
    W = _asymmetric_W()
    solve_fn, backend = make_sparse_solve_fn(W)
    assert backend == "sparsax_lu"

    n = W.shape[0]
    V = jnp.asarray(np.random.default_rng(1).standard_normal((80, n)))
    for rho in [0.0, 0.4, -0.6, 0.9]:
        npt.assert_allclose(np.asarray(solve_fn(rho, V)), _dense_filter(W, rho, V), atol=1e-9)

    g = float(jax.grad(lambda r: jnp.sum(jax.jit(solve_fn)(r, V) ** 2))(0.4))
    eps = 1e-6
    obj = lambda r: np.sum(_dense_filter(W, r, V) ** 2)  # noqa: E731
    npt.assert_allclose(g, (obj(0.4 + eps) - obj(0.4 - eps)) / (2 * eps), rtol=1e-5)


# ---------------------------------------------------------------------------
# scipy sparse factorisations (off the JAX path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", [_symmetric_W, _asymmetric_W])
def test_scipy_factorization_matches_dense(builder):
    """CHOLMOD (symmetric) and KLU/SuperLU (asymmetric) match dense solve+diag."""
    from locpick._jax.sparse_solve import create_factorization, refactorize

    W = builder(120)
    Wd = W.toarray()
    n = W.shape[0]
    b = np.random.default_rng(0).standard_normal((n, 3))

    fact = create_factorization(W, 0.3)
    for rho in [0.3, 0.7, -0.5]:
        fact = refactorize(fact, W, rho)
        A = np.eye(n) - rho * Wd
        npt.assert_allclose(fact.solve(b), np.linalg.solve(A, b), atol=1e-10)
        npt.assert_allclose(fact.diagonal_inverse(), np.diag(np.linalg.inv(A)), atol=1e-10)


def test_symmetrizable_W_uses_cholmod_when_available():
    """A row-standardised symmetric-adjacency W should hit the CHOLMOD path."""
    pytest.importorskip("sksparse.cholmod")
    from locpick._jax.sparse_solve import CholmodFactorization, create_factorization

    fact = create_factorization(_symmetric_W(80), 0.5)
    assert isinstance(fact, CholmodFactorization)
