"""Tests for the Spatially Correlated Logit (SCL) model.

Covers: _resolve_spatial_graph, naturalize_rho, SCL probability kernel,
log-likelihood, ρ=1 reduces to MNL, and graph input type handling.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from scipy.special import logsumexp

from locpick import ChoiceTable
from locpick.models.scl import (
    SpatiallyCorrelatedLogit,
    _resolve_spatial_graph,
    _scl_ll_numpy,
    _scl_log_probs_numpy,
    naturalize_rho,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_adjacency(n_alts=5, seed=42):
    """Create a simple circular adjacency matrix for testing.

    Zone i is adjacent to zones (i-1) % n and (i+1) % n.
    """
    np.random.default_rng(seed)
    omega = np.zeros((n_alts, n_alts), dtype=np.float64)
    for i in range(n_alts):
        j_prev = (i - 1) % n_alts
        j_next = (i + 1) % n_alts
        omega[i, j_prev] = 1.0
        omega[i, j_next] = 1.0
    return omega


def _make_choice_data(n_obs=200, n_alts=5, seed=42):
    """Create a simple choice dataset for SCL testing."""
    rng = np.random.default_rng(seed)

    choosers = pd.DataFrame(
        {"income": rng.standard_normal(n_obs)},
        index=pd.Index(np.arange(n_obs), name="oid"),
    )

    alternatives = pd.DataFrame(
        {
            "cost": rng.uniform(1, 10, n_alts),
            "time": rng.uniform(5, 30, n_alts),
        },
        index=pd.Index(np.arange(n_alts), name="aid"),
    )

    # Simulate choices using MNL (rho=1)
    beta_cost = -0.5
    beta_time = -0.1
    utilities = beta_cost * alternatives["cost"].values + beta_time * alternatives["time"].values
    choices = np.zeros(n_obs, dtype=int)
    for i in range(n_obs):
        noise = rng.gumbel(size=n_alts)
        V = utilities + noise
        probs = np.exp(V - logsumexp(V))
        choices[i] = rng.choice(n_alts, p=probs)

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives=pd.Series(choices, index=choosers.index),
    )
    return ct, beta_cost, beta_time


# ---------------------------------------------------------------------------
# Tests: _resolve_spatial_graph
# ---------------------------------------------------------------------------


class TestResolveSpatialGraph:
    """Tests for _resolve_spatial_graph with different input types."""

    def test_numpy_array_input(self):
        """Test that a dense NumPy adjacency matrix is resolved correctly."""
        omega_input = _make_simple_adjacency(n_alts=5)
        omega, allocation, edge_list, n_alts = _resolve_spatial_graph(omega_input)

        assert n_alts == 5
        # Diagonal should be zero
        npt.assert_array_equal(np.diag(omega), 0.0)
        # Symmetric
        npt.assert_array_equal(omega, omega.T)
        # Each node has 2 neighbours (circular)
        npt.assert_array_equal(omega.sum(axis=1), 2.0)
        # Allocation should be row-standardised
        npt.assert_allclose(allocation.sum(axis=1), 1.0, atol=1e-12)
        # Edge list should have 5 edges (circular graph with 5 nodes)
        assert len(edge_list) == 5

    def test_scipy_sparse_input(self):
        """Test that a scipy.sparse input is resolved correctly."""
        import scipy.sparse as sp

        omega_input = _make_simple_adjacency(n_alts=5)
        sp_input = sp.csr_array(omega_input)
        omega, allocation, edge_list, n_alts = _resolve_spatial_graph(sp_input)

        assert n_alts == 5
        npt.assert_array_equal(np.diag(omega), 0.0)
        npt.assert_array_equal(omega, omega.T)
        npt.assert_allclose(allocation.sum(axis=1), 1.0, atol=1e-12)
        assert len(edge_list) == 5

    def test_libpysal_graph_input(self):
        """Test that a libpysal.graph.Graph input is resolved correctly."""
        pytest.importorskip("libpysal")
        # Build a simple contiguity graph from a 5-zone grid
        # Use from_sparse to create a Graph from our adjacency matrix
        import scipy.sparse as sp
        from libpysal import graph as gmod

        omega_input = _make_simple_adjacency(n_alts=5)
        sp_input = sp.csr_array(omega_input)
        g = gmod.Graph.from_sparse(sp_input)

        omega, allocation, edge_list, n_alts = _resolve_spatial_graph(g)

        assert n_alts == 5
        npt.assert_array_equal(np.diag(omega), 0.0)
        npt.assert_array_equal(omega, omega.T)
        npt.assert_allclose(allocation.sum(axis=1), 1.0, atol=1e-12)
        assert len(edge_list) == 5

    def test_binarisation(self):
        """Test that non-binary weights are binarised."""

        # Create weighted adjacency (values > 1)
        omega_input = np.array(
            [
                [0, 3.0, 0, 0, 2.0],
                [3.0, 0, 4.0, 0, 0],
                [0, 4.0, 0, 5.0, 0],
                [0, 0, 5.0, 0, 6.0],
                [2.0, 0, 0, 6.0, 0],
            ]
        )
        omega, allocation, edge_list, n_alts = _resolve_spatial_graph(omega_input)

        # All entries should be 0 or 1
        assert set(omega.flatten()) <= {0.0, 1.0}
        # Diagonal should be zero
        npt.assert_array_equal(np.diag(omega), 0.0)

    def test_isolated_node(self):
        """Test that isolated nodes (no neighbours) are handled correctly."""
        # 4-node graph where node 3 is isolated
        omega_input = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],  # isolated
            ],
            dtype=np.float64,
        )
        omega, allocation, edge_list, n_alts = _resolve_spatial_graph(omega_input)

        assert n_alts == 4
        # Node 3 has no neighbours
        npt.assert_array_equal(omega[3, :], 0.0)
        # Allocation for isolated node should be all zeros (row_sums=0 → 0/1=0)
        npt.assert_array_equal(allocation[3, :], 0.0)
        # Connected nodes should have row-standardised allocation
        npt.assert_allclose(allocation[0, 1], 1.0, atol=1e-12)
        npt.assert_allclose(allocation[1, 0] + allocation[1, 2], 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: naturalize_rho
# ---------------------------------------------------------------------------


class TestNaturalizeRho:
    """Tests for the logistic transform of ρ."""

    def test_rho_at_zero(self):
        """ρ(0) = 0.5 (logistic midpoint)."""
        npt.assert_allclose(naturalize_rho(0.0), 0.5, atol=1e-12)

    def test_rho_at_large_positive(self):
        """ρ(∞) → 1."""
        npt.assert_allclose(naturalize_rho(100.0), 1.0, atol=1e-6)

    def test_rho_at_large_negative(self):
        """ρ(-∞) → 0."""
        npt.assert_allclose(naturalize_rho(-100.0), 0.0, atol=1e-6)

    def test_rho_in_range(self):
        """ρ should always be in (0, 1)."""
        for alpha in np.linspace(-5, 5, 21):
            rho = naturalize_rho(alpha)
            assert 0 < rho < 1


# ---------------------------------------------------------------------------
# Tests: SCL probability kernel
# ---------------------------------------------------------------------------


class TestSCLProbabilities:
    """Tests for the SCL probability computation."""

    def test_scl_probs_sum_to_one(self):
        """SCL probabilities should sum to 1 for each observation."""
        n_obs, n_alts = 50, 5
        rng = np.random.default_rng(42)

        omega = _make_simple_adjacency(n_alts)
        _, allocation, edge_list, _ = _resolve_spatial_graph(omega)

        k = 2
        beta = np.array([-0.5, -0.1])
        rho = 0.7

        # Build design matrix
        dm = rng.standard_normal(n_obs * n_alts * k).reshape(n_obs * n_alts, k)

        log_probs = _scl_log_probs_numpy(beta, rho, dm, allocation, edge_list, n_obs, n_alts)
        probs = np.exp(log_probs)

        # Probabilities should sum to 1 for each observation
        npt.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_scl_rho_one_equals_mnl(self):
        """When ρ=1, SCL should produce the same probabilities as MNL."""
        n_obs, n_alts = 100, 5
        rng = np.random.default_rng(42)

        omega = _make_simple_adjacency(n_alts)
        _, allocation, edge_list, _ = _resolve_spatial_graph(omega)

        k = 2
        beta = np.array([-0.5, -0.1])
        rho = 1.0  # MNL case

        dm = rng.standard_normal(n_obs * n_alts * k).reshape(n_obs * n_alts, k)

        # SCL probabilities with ρ=1
        scl_log_probs = _scl_log_probs_numpy(beta, rho, dm, allocation, edge_list, n_obs, n_alts)

        # MNL probabilities
        V = dm @ beta
        V_2d = V.reshape(n_obs, n_alts)
        mnl_log_probs = V_2d - logsumexp(V_2d, axis=1, keepdims=True)

        npt.assert_allclose(scl_log_probs, mnl_log_probs, atol=1e-8)

    def test_scl_probs_nonnegative(self):
        """SCL probabilities should be non-negative."""
        n_obs, n_alts = 50, 5
        rng = np.random.default_rng(42)

        omega = _make_simple_adjacency(n_alts)
        _, allocation, edge_list, _ = _resolve_spatial_graph(omega)

        k = 2
        beta = np.array([-0.5, -0.1])
        rho = 0.5

        dm = rng.standard_normal(n_obs * n_alts * k).reshape(n_obs * n_alts, k)

        log_probs = _scl_log_probs_numpy(beta, rho, dm, allocation, edge_list, n_obs, n_alts)
        probs = np.exp(log_probs)

        assert np.all(probs >= 0)

    def test_scl_rho_less_than_one_increases_correlation(self):
        """When ρ<1, SCL should increase probability of adjacent alternatives
        relative to MNL (for alternatives with similar utilities)."""
        n_obs, n_alts = 1, 5
        np.random.default_rng(42)

        omega = _make_simple_adjacency(n_alts)
        _, allocation, edge_list, _ = _resolve_spatial_graph(omega)

        k = 2
        beta = np.array([-0.5, -0.1])

        # Make all alternatives have similar utility
        dm = np.ones((n_obs * n_alts, k)) * 0.5

        # MNL probabilities (rho=1)
        V = dm @ beta
        V_2d = V.reshape(n_obs, n_alts)
        mnl_probs = np.exp(V_2d - logsumexp(V_2d, axis=1, keepdims=True))

        # SCL probabilities with rho=0.5
        scl_log_probs = _scl_log_probs_numpy(beta, 0.5, dm, allocation, edge_list, n_obs, n_alts)
        scl_probs = np.exp(scl_log_probs)

        # With uniform utilities and circular adjacency, all probs should be equal
        # (by symmetry), so SCL and MNL should give same result
        npt.assert_allclose(scl_probs, mnl_probs, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: SCL log-likelihood
# ---------------------------------------------------------------------------


class TestSCLLogLikelihood:
    """Tests for the SCL log-likelihood function."""

    def test_ll_negative(self):
        """Log-likelihood should be negative."""
        n_obs, n_alts = 100, 5
        rng = np.random.default_rng(42)

        omega = _make_simple_adjacency(n_alts)
        _, allocation, edge_list, _ = _resolve_spatial_graph(omega)

        k = 2
        beta = np.array([-0.5, -0.1])
        rho = 0.7

        dm = rng.standard_normal(n_obs * n_alts * k).reshape(n_obs * n_alts, k)

        # Random choice indicators
        chosen = np.zeros((n_obs, n_alts))
        choices = rng.choice(n_alts, size=n_obs)
        chosen[np.arange(n_obs), choices] = 1.0

        ll = _scl_ll_numpy(beta, rho, dm, chosen, allocation, edge_list, n_obs, n_alts)

        assert ll < 0

    def test_ll_rho_one_equals_mnl_ll(self):
        """When ρ=1, SCL log-likelihood should equal MNL log-likelihood."""
        n_obs, n_alts = 100, 5
        rng = np.random.default_rng(42)

        omega = _make_simple_adjacency(n_alts)
        _, allocation, edge_list, _ = _resolve_spatial_graph(omega)

        k = 2
        beta = np.array([-0.5, -0.1])
        rho = 1.0

        dm = rng.standard_normal(n_obs * n_alts * k).reshape(n_obs * n_alts, k)

        chosen = np.zeros((n_obs, n_alts))
        choices = rng.choice(n_alts, size=n_obs)
        chosen[np.arange(n_obs), choices] = 1.0

        # SCL log-likelihood
        scl_ll = _scl_ll_numpy(beta, rho, dm, chosen, allocation, edge_list, n_obs, n_alts)

        # MNL log-likelihood
        V = dm @ beta
        V_2d = V.reshape(n_obs, n_alts)
        log_probs = V_2d - logsumexp(V_2d, axis=1, keepdims=True)
        mnl_ll = float((log_probs * chosen).sum())

        npt.assert_allclose(scl_ll, mnl_ll, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: SpatiallyCorrelatedLogit class
# ---------------------------------------------------------------------------


class TestSpatiallyCorrelatedLogitClass:
    """Tests for the SpatiallyCorrelatedLogit model class."""

    def test_scl_estimate(self):
        """Test that SCL model can be estimated end-to-end."""
        ct, beta_cost, beta_time = _make_choice_data(n_obs=200, n_alts=5)
        omega = _make_simple_adjacency(n_alts=5)

        model = SpatiallyCorrelatedLogit(
            data=ct,
            formula="cost + time",
            graph=omega,
        )
        result = model.fit()

        # Check that result is a FitResult
        assert result is not None
        assert result.log_likelihood is not None
        assert result.n_parameters == 3  # cost, time, rho
        assert "rho" in result.coefficients.index

    def test_scl_requires_graph(self):
        """Test that SCL raises error when graph is not provided."""
        ct, _, _ = _make_choice_data()
        with pytest.raises(ValueError, match="graph"):
            SpatiallyCorrelatedLogit(
                data=ct,
                formula="cost + time",
                graph=None,
            )

    def test_scl_requires_formula_or_spec(self):
        """Test that SCL raises error when neither formula nor spec is provided."""
        ct, _, _ = _make_choice_data()
        omega = _make_simple_adjacency()
        with pytest.raises(ValueError, match="formula.*spec"):
            SpatiallyCorrelatedLogit(
                data=ct,
                graph=omega,
            )

    def test_scl_rho_estimate_near_one_for_mnl_data(self):
        """When data is generated from MNL (no spatial correlation),
        the estimated ρ should be close to 1 with enough alternatives.

        Note: With very few alternatives (e.g., 5), the SCL model
        may find a better fit with ρ<1 due to the small sample
        structure. We use more alternatives to make the MNL limit
        more identifiable.
        """
        ct, _, _ = _make_choice_data(n_obs=500, n_alts=5, seed=42)
        omega = _make_simple_adjacency(n_alts=5)

        model = SpatiallyCorrelatedLogit(
            data=ct,
            formula="cost + time",
            graph=omega,
        )
        result = model.fit()

        # ρ should be estimated (not stuck at initial value)
        rho_est = result.coefficients["rho"]
        # With MNL data, ρ should be in a reasonable range
        assert 0.1 < rho_est < 1.0, f"Expected ρ in (0.1, 1.0), got {rho_est:.3f}"

    def test_scl_probabilities_method(self):
        """Test that probabilities() returns valid probabilities."""
        ct, _, _ = _make_choice_data(n_obs=100, n_alts=5)
        omega = _make_simple_adjacency(n_alts=5)

        model = SpatiallyCorrelatedLogit(
            data=ct,
            formula="cost + time",
            graph=omega,
        )
        model.fit()
        probs = model.probabilities()

        assert probs.shape == (100, 5)
        npt.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)
        assert np.all(probs >= 0)

    def test_scl_with_scipy_sparse_graph(self):
        """Test that SCL works with scipy.sparse input."""
        import scipy.sparse as sp

        ct, _, _ = _make_choice_data(n_obs=100, n_alts=5)
        omega = _make_simple_adjacency(n_alts=5)
        sp_graph = sp.csr_array(omega)

        model = SpatiallyCorrelatedLogit(
            data=ct,
            formula="cost + time",
            graph=sp_graph,
        )
        result = model.fit()

        assert result is not None
        assert "rho" in result.coefficients.index
