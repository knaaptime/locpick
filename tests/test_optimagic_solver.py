"""Tests for the OptimagicSolver wrapper around ``optimagic.minimize``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optimagic")

from locpick import ChoiceTable, MultinomialLogit
from locpick._solvers.lbfgs import LBFGSSolver
from locpick._solvers.optimagic import OptimagicSolver
from locpick._solvers.protocol import get_solver, list_solvers


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


def test_optimagic_registered_lazily():
    assert "optimagic" in list_solvers()
    solver = get_solver("optimagic", algorithm="scipy_lbfgsb")
    assert isinstance(solver, OptimagicSolver)


@pytest.mark.parametrize(
    "algorithm",
    ["scipy_lbfgsb", "nlopt_lbfgsb", "scipy_trust_constr"],
)
def test_optimagic_matches_lbfgs(mnl_table, algorithm):
    """OptimagicSolver should reach the same LL as scipy L-BFGS-B."""
    baseline = MultinomialLogit(mnl_table, formula="rent + jobs", solver=LBFGSSolver())
    baseline.fit()
    ll_base = baseline._result.log_likelihood

    model = MultinomialLogit(
        mnl_table,
        formula="rent + jobs",
        solver=OptimagicSolver(algorithm=algorithm),
    )
    model.fit()
    ll_opt = model._result.log_likelihood

    assert np.isclose(ll_opt, ll_base, atol=1e-3)
    assert np.allclose(
        model._result.coefficients,
        baseline._result.coefficients,
        atol=5e-3,
    )


def test_optimagic_unknown_algorithm_raises(mnl_table):
    solver = OptimagicSolver(algorithm="not_a_real_algorithm")
    model = MultinomialLogit(mnl_table, formula="rent + jobs", solver=solver)
    with pytest.raises(Exception):
        model.fit()
