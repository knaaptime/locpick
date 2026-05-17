"""Tests for the trust-region Newton-CG solvers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from locpick import ChoiceTable, MultinomialLogit
from locpick._solvers.lbfgs import LBFGSSolver
from locpick._solvers.protocol import get_solver, list_solvers
from locpick._solvers.trust_ncg import TrustKrylovSolver, TrustNCGSolver


@pytest.fixture(scope="module")
def mnl_table():
    rng = np.random.default_rng(0)
    n_obs, n_alts = 800, 8
    obs = pd.DataFrame({"obs_id": np.arange(n_obs)}).set_index("obs_id")
    alts = pd.DataFrame(
        {
            "alt_id": np.arange(n_alts),
            "rent": rng.normal(2000.0, 500.0, n_alts),
            "jobs": rng.normal(0.0, 1.0, n_alts),
        }
    ).set_index("alt_id")
    util = (
        -0.001 * alts["rent"].to_numpy()[None, :]
        + 0.5 * alts["jobs"].to_numpy()[None, :]
        + rng.gumbel(size=(n_obs, n_alts))
    )
    obs["choice"] = util.argmax(axis=1)
    return ChoiceTable.from_tables(
        choosers=obs,
        alternatives=alts,
        chosen_alternatives="choice",
        sample_size=n_alts - 1,
        replace=False,
    )


def test_trust_ncg_registered():
    assert "trust-ncg" in list_solvers()
    assert "trust-krylov" in list_solvers()
    assert isinstance(get_solver("trust-ncg"), TrustNCGSolver)
    assert isinstance(get_solver("trust-krylov"), TrustKrylovSolver)


def test_trust_ncg_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        TrustNCGSolver(method="not-a-method")


@pytest.mark.parametrize(
    "solver_cls",
    [TrustNCGSolver, TrustKrylovSolver],
)
def test_trust_ncg_matches_lbfgs_on_mnl(mnl_table, solver_cls):
    baseline = MultinomialLogit(mnl_table, formula="rent + jobs", solver=LBFGSSolver())
    baseline.fit()
    ll_base = baseline._result.log_likelihood

    model = MultinomialLogit(mnl_table, formula="rent + jobs", solver=solver_cls())
    model.fit()
    ll = model._result.log_likelihood

    assert np.isclose(ll, ll_base, atol=1e-3)
    assert np.allclose(
        model._result.coefficients,
        baseline._result.coefficients,
        atol=5e-3,
    )


def test_trust_ncg_matches_lbfgs_on_scl():
    """Sanity check on a spatial model: trust-ncg + JAX HVP should reach
    the same SCL optimum as scipy L-BFGS-B."""
    from locpick import SpatiallyCorrelatedLogit
    from locpick.dgp import simulate_scl

    ds = simulate_scl(n_obs=600, n_alts=12, seed=11)

    base = SpatiallyCorrelatedLogit(
        data=ds.choice_table,
        formula="cost + time + income_x_cost",
        graph=ds.adjacency,
        solver=LBFGSSolver(),
        backend="jax",
    )
    base.fit()

    test = SpatiallyCorrelatedLogit(
        data=ds.choice_table,
        formula="cost + time + income_x_cost",
        graph=ds.adjacency,
        solver=TrustNCGSolver(),
        backend="jax",
    )
    test.fit()

    assert np.isclose(
        test._result.log_likelihood,
        base._result.log_likelihood,
        atol=1e-2,
    )
    assert np.allclose(
        test._result.coefficients.values,
        base._result.coefficients.values,
        atol=5e-3,
    )


def test_trust_ncg_rejects_bounds():
    """Box-constrained problems should error early."""
    from locpick._jax.objective import Objective

    obj = Objective(
        fn=lambda x: -float(np.sum(x**2)),
        grad=lambda x: -2.0 * np.asarray(x, dtype=float),
        jax_fn=lambda x: -float(np.sum(np.asarray(x) ** 2)),
        jax_grad=lambda x: -2.0 * np.asarray(x, dtype=float),
    )
    solver = TrustNCGSolver()
    with pytest.raises(ValueError, match="does not support box constraints"):
        solver.solve(
            obj,
            x0=np.zeros(3),
            param_names=["a", "b", "c"],
            bounds=[(-1.0, 1.0)] * 3,
        )
