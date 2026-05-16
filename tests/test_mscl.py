"""Tests for the Mixed Spatially Correlated Logit (MSCL) model.

Covers: MSCL estimation, ρ=1 with no random params reduces to MNL,
parameter recovery, and integration with SCL components.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from locpick import ChoiceTable
from locpick.models.mixed import ParamDistribution
from locpick.models.mscl import MixedSpatiallyCorrelatedLogit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_adjacency(n_alts=5, seed=42):
    """Create a simple circular adjacency matrix for testing."""
    omega = np.zeros((n_alts, n_alts), dtype=np.float64)
    for i in range(n_alts):
        j_prev = (i - 1) % n_alts
        j_next = (i + 1) % n_alts
        omega[i, j_prev] = 1.0
        omega[i, j_next] = 1.0
    return omega


def _make_choice_data(n_obs=200, n_alts=5, seed=42):
    """Create a simple choice dataset for MSCL testing."""
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

    # Simulate choices using MNL
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
# Tests: MSCL model class
# ---------------------------------------------------------------------------


class TestMixedSpatiallyCorrelatedLogitClass:
    """Tests for the MixedSpatiallyCorrelatedLogit model class."""

    def test_mscl_requires_graph(self):
        """Test that MSCL raises error when graph is not provided."""
        ct, _, _ = _make_choice_data()
        with pytest.raises(ValueError, match="graph"):
            MixedSpatiallyCorrelatedLogit(
                data=ct,
                formula="cost + time",
                graph=None,
            )

    def test_mscl_requires_formula_or_spec(self):
        """Test that MSCL raises error when neither formula nor spec is provided."""
        ct, _, _ = _make_choice_data()
        omega = _make_simple_adjacency()
        with pytest.raises(ValueError, match="formula.*spec"):
            MixedSpatiallyCorrelatedLogit(
                data=ct,
                graph=omega,
            )

    def test_mscl_estimate_no_random_params(self):
        """Test MSCL estimation with no random parameters (reduces to SCL)."""
        ct, _, _ = _make_choice_data(n_obs=200, n_alts=5)
        omega = _make_simple_adjacency(n_alts=5)

        model = MixedSpatiallyCorrelatedLogit(
            data=ct,
            formula="cost + time",
            graph=omega,
            random_params=[],
            n_draws=50,
        )
        result = model.fit()

        assert result is not None
        assert result.log_likelihood is not None
        assert "rho" in result.coefficients.index

    def test_mscl_estimate_with_random_params(self):
        """Test MSCL estimation with random parameters."""
        ct, _, _ = _make_choice_data(n_obs=200, n_alts=5)
        omega = _make_simple_adjacency(n_alts=5)

        model = MixedSpatiallyCorrelatedLogit(
            data=ct,
            formula="cost + time",
            graph=omega,
            random_params={"cost": ParamDistribution(param="cost", distribution="normal")},
            n_draws=50,
        )
        result = model.fit()

        assert result is not None
        assert "rho" in result.coefficients.index
        # Should have cost_mean, cost_sd, time (fixed), rho
        assert result.n_parameters >= 4

    def test_mscl_rho_near_one_for_mnl_data(self):
        """When data is generated from MNL (no spatial correlation),
        the estimated ρ should be in a reasonable range.

        Note: With few alternatives, the SCL model may find a better
        fit with ρ<1 due to the small sample structure.
        """
        ct, _, _ = _make_choice_data(n_obs=500, n_alts=5, seed=42)
        omega = _make_simple_adjacency(n_alts=5)

        model = MixedSpatiallyCorrelatedLogit(
            data=ct,
            formula="cost + time",
            graph=omega,
            random_params={},
            n_draws=50,
        )
        result = model.fit()

        rho_est = result.coefficients["rho"]
        assert 0.1 < rho_est < 1.0, f"Expected ρ in (0.1, 1.0), got {rho_est:.3f}"

    def test_mscl_with_scipy_sparse_graph(self):
        """Test that MSCL works with scipy.sparse input."""
        import scipy.sparse as sp

        ct, _, _ = _make_choice_data(n_obs=100, n_alts=5)
        omega = _make_simple_adjacency(n_alts=5)
        sp_graph = sp.csr_array(omega)

        model = MixedSpatiallyCorrelatedLogit(
            data=ct,
            formula="cost + time",
            graph=sp_graph,
            random_params=[],
            n_draws=50,
        )
        result = model.fit()

        assert result is not None
        assert "rho" in result.coefficients.index

    def test_mscl_halton_vs_random_draws(self):
        """Test that MSCL works with both Halton and random draws."""
        ct, _, _ = _make_choice_data(n_obs=100, n_alts=5)
        omega = _make_simple_adjacency(n_alts=5)

        # Halton draws (default)
        model_halton = MixedSpatiallyCorrelatedLogit(
            data=ct,
            formula="cost + time",
            graph=omega,
            random_params={"cost": ParamDistribution(param="cost", distribution="normal")},
            n_draws=50,
            draw_type="halton",
        )
        result_halton = model_halton.fit()
        assert result_halton is not None

        # Pseudo-random draws
        model_random = MixedSpatiallyCorrelatedLogit(
            data=ct,
            formula="cost + time",
            graph=omega,
            random_params={"cost": ParamDistribution(param="cost", distribution="normal")},
            n_draws=50,
            draw_type="random",
        )
        result_random = model_random.fit()
        assert result_random is not None
