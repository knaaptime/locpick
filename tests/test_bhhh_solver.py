"""Tests for the BHHH solver."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from locpick import ChoiceTable, MultinomialLogit, SpatiallyCorrelatedLogit
from locpick._solvers.bhhh import BHHHSolver
from locpick._solvers.lbfgs import LBFGSSolver
from locpick._solvers.protocol import get_solver, list_solvers
from locpick.dgp import simulate_scl


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


def test_bhhh_registered():
    assert "bhhh" in list_solvers()
    assert isinstance(get_solver("bhhh"), BHHHSolver)


def test_bhhh_matches_lbfgs_on_mnl(mnl_table):
    base = MultinomialLogit(mnl_table, formula="rent + jobs", solver=LBFGSSolver())
    base.fit()

    model = MultinomialLogit(mnl_table, formula="rent + jobs", solver=BHHHSolver())
    model.fit()

    assert np.isclose(
        model._result.log_likelihood,
        base._result.log_likelihood,
        atol=1e-3,
    )
    assert np.allclose(
        model._result.coefficients,
        base._result.coefficients,
        atol=5e-3,
    )


def test_bhhh_matches_lbfgs_on_scl():
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
        solver=BHHHSolver(),
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


def test_bhhh_requires_loglike_contribs():
    """A bare Objective without per-obs LL must raise a clear error."""
    from locpick._jax.objective import Objective

    obj = Objective(
        fn=lambda x: -float(np.sum(x**2)),
        grad=lambda x: -2.0 * np.asarray(x, dtype=float),
    )
    solver = BHHHSolver()
    with pytest.raises(ValueError, match="loglike_contribs_jax"):
        solver.solve(obj, x0=np.zeros(3), param_names=["a", "b", "c"])


def test_bhhh_rejects_bounds(mnl_table):
    from locpick._jax.objective import Objective

    obj = Objective(
        fn=lambda x: -float(np.sum(x**2)),
        grad=lambda x: -2.0 * np.asarray(x, dtype=float),
        loglike_contribs_jax=lambda x: x,  # arbitrary, just must be non-None
    )
    solver = BHHHSolver()
    with pytest.raises(ValueError, match="does not support box constraints"):
        solver.solve(
            obj,
            x0=np.zeros(3),
            param_names=["a", "b", "c"],
            bounds=[(-1.0, 1.0)] * 3,
        )
