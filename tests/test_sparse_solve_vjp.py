"""Tests for the sparse solve custom VJP.

Verifies that the sparse solve (CHOLMOD/SuperLU with custom VJP) produces
identical results to the dense JAX solve for both forward values and
gradients.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

import scipy.sparse as sp

from locpick._jax.sparse_solve import (
    SparseSolveContext,
    make_sparse_solve_fn,
)


def _make_circular_W(n_alts):
    """Build a circular adjacency matrix (symmetric, row-standardised)."""
    W = np.zeros((n_alts, n_alts))
    for i in range(n_alts):
        W[i, (i - 1) % n_alts] = 1.0
        W[i, (i + 1) % n_alts] = 1.0
    # Row-standardise
    row_sums = W.sum(axis=1, keepdims=True)
    W = W / row_sums
    return sp.csr_matrix(W, dtype=np.float64)


def _make_knn_W(n_alts, k=3):
    """Build an asymmetric KNN-style W (non-symmetric, row-standardised)."""
    rng = np.random.default_rng(42)
    # Random coordinates
    coords = rng.standard_normal((n_alts, 2))
    W = np.zeros((n_alts, n_alts))
    for i in range(n_alts):
        dists = np.sum((coords - coords[i]) ** 2, axis=1)
        dists[i] = np.inf  # exclude self
        neighbors = np.argsort(dists)[:k]
        W[i, neighbors] = 1.0
    # Row-standardise
    row_sums = W.sum(axis=1, keepdims=True)
    W = W / row_sums
    return sp.csr_matrix(W, dtype=np.float64)


class TestSparseSolveVJP:
    """Test that sparse solve with custom VJP matches dense JAX solve."""

    @pytest.mark.parametrize("n_alts", [10, 20, 50])
    def test_forward_matches_dense(self, n_alts):
        """Sparse solve forward should match dense solve."""
        W_sp = _make_circular_W(n_alts)
        W_dense = jnp.array(W_sp.toarray(), dtype=jnp.float64)

        rho_val = 0.3
        n_obs = 5
        V_base = jnp.array(
            np.random.default_rng(0).standard_normal((n_obs, n_alts)), dtype=jnp.float64
        )

        # Dense solve
        A = jnp.eye(n_alts) - rho_val * W_dense
        V_filtered_dense = jax.scipy.linalg.solve(A, V_base.T).T

        # Sparse solve
        ctx = SparseSolveContext(W_sp, use_cholmod=True)
        sparse_solve = make_sparse_solve_fn(ctx)
        V_filtered_sparse = sparse_solve(jnp.array(rho_val), V_base)

        npt.assert_allclose(
            np.asarray(V_filtered_sparse),
            np.asarray(V_filtered_dense),
            rtol=1e-8,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n_alts", [10, 20, 50])
    def test_gradient_matches_dense(self, n_alts):
        """Sparse solve gradient (w.r.t. rho) should match dense gradient."""
        W_sp = _make_circular_W(n_alts)
        W_dense = jnp.array(W_sp.toarray(), dtype=jnp.float64)

        n_obs = 5
        V_base_np = np.random.default_rng(0).standard_normal((n_obs, n_alts))
        V_base = jnp.array(V_base_np, dtype=jnp.float64)

        # Dense solve + gradient
        def _dense_ll(rho):
            A = jnp.eye(n_alts) - rho * W_dense
            V_filtered = jax.scipy.linalg.solve(A, V_base.T).T
            return jnp.sum(V_filtered)  # scalar for gradient check

        grad_dense = jax.grad(_dense_ll)(jnp.array(0.3))

        # Sparse solve + gradient
        ctx = SparseSolveContext(W_sp, use_cholmod=True)
        sparse_solve = make_sparse_solve_fn(ctx)

        def _sparse_ll(rho):
            V_filtered = sparse_solve(rho, V_base)
            return jnp.sum(V_filtered)

        grad_sparse = jax.grad(_sparse_ll)(jnp.array(0.3))

        npt.assert_allclose(
            float(grad_sparse),
            float(grad_dense),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_forward_knn_non_symmetric(self):
        """Sparse solve should work with non-symmetric KNN W (SuperLU path)."""
        n_alts = 20
        W_sp = _make_knn_W(n_alts, k=3)
        W_dense = jnp.array(W_sp.toarray(), dtype=jnp.float64)

        rho_val = 0.2
        n_obs = 5
        V_base = jnp.array(
            np.random.default_rng(0).standard_normal((n_obs, n_alts)), dtype=jnp.float64
        )

        # Dense solve
        A = jnp.eye(n_alts) - rho_val * W_dense
        V_filtered_dense = jax.scipy.linalg.solve(A, V_base.T).T

        # Sparse solve (will use SuperLU since W is non-symmetric)
        ctx = SparseSolveContext(W_sp, use_cholmod=True)
        sparse_solve = make_sparse_solve_fn(ctx)
        V_filtered_sparse = sparse_solve(jnp.array(rho_val), V_base)

        npt.assert_allclose(
            np.asarray(V_filtered_sparse),
            np.asarray(V_filtered_dense),
            rtol=1e-8,
            atol=1e-10,
        )

    def test_gradient_knn_non_symmetric(self):
        """Gradient should work with non-symmetric KNN W."""
        n_alts = 20
        W_sp = _make_knn_W(n_alts, k=3)
        W_dense = jnp.array(W_sp.toarray(), dtype=jnp.float64)

        n_obs = 5
        V_base = jnp.array(
            np.random.default_rng(0).standard_normal((n_obs, n_alts)), dtype=jnp.float64
        )

        # Dense gradient
        def _dense_ll(rho):
            A = jnp.eye(n_alts) - rho * W_dense
            V_filtered = jax.scipy.linalg.solve(A, V_base.T).T
            return jnp.sum(V_filtered)

        grad_dense = jax.grad(_dense_ll)(jnp.array(0.2))

        # Sparse gradient
        ctx = SparseSolveContext(W_sp, use_cholmod=True)
        sparse_solve = make_sparse_solve_fn(ctx)

        def _sparse_ll(rho):
            V_filtered = sparse_solve(rho, V_base)
            return jnp.sum(V_filtered)

        grad_sparse = jax.grad(_sparse_ll)(jnp.array(0.2))

        npt.assert_allclose(
            float(grad_sparse),
            float(grad_dense),
            rtol=1e-5,
            atol=1e-7,
        )
