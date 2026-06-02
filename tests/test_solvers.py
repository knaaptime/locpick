# ruff: noqa: E402, F811
"""Solver tests: protocol, trust-ncg, optimagic."""

"""Tests for the unified Objective-first solver interface."""

import inspect

import numpy as np
import pytest

from locpick._jax.objective import Objective
from locpick._solvers.lbfgs import LBFGSSolver
from locpick._solvers.optax import OptaxSolver


def test_solver_signatures_are_objective_first():
    """All solvers should expose the same Objective-first solve signature."""
    expected = ["self", "objective", "x0", "param_names", "bounds", "fixed_mask"]

    for solver_cls in (LBFGSSolver, OptaxSolver):
        names = list(inspect.signature(solver_cls.solve).parameters.keys())[:6]
        assert names == expected


@pytest.mark.parametrize("solver_cls", [LBFGSSolver, OptaxSolver])
def test_solvers_require_objective_instance(solver_cls):
    """Solvers should reject legacy callable-only inputs."""
    solver = solver_cls(maxiter=1)

    with pytest.raises(TypeError, match="Objective"):
        solver.solve(
            objective=lambda x: -float(np.sum(np.asarray(x, dtype=np.float64) ** 2)),
            x0=np.array([0.0], dtype=np.float64),
            param_names=["beta"],
        )


def test_solver_result_hessian_is_none_by_default():
    """Solvers should not compute Hessian eagerly; lazy computation is preferred."""
    pytest.importorskip("optax")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def ll_jax(x):
        return -jnp.sum((x - 1.0) ** 2)

    objective = Objective.from_jax(ll_fn=ll_jax, grad_fn=jax.grad(ll_jax))
    solver = OptaxSolver(maxiter=200)
    result = solver.solve(
        objective=objective,
        x0=np.array([0.0], dtype=np.float64),
        param_names=["beta"],
    )

    # Optax no longer computes Hessian eagerly; model computes it lazily
    assert result.hessian is None


def test_optax_solver_emits_diagnostics_in_raw_payload():
    """Optax should surface status/iteration diagnostics in SolverResult.raw."""
    pytest.importorskip("optax")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def ll_jax(x):
        return -jnp.sum((x - 1.0) ** 2)

    objective = Objective.from_jax(ll_fn=ll_jax, grad_fn=jax.grad(ll_jax))
    solver = OptaxSolver(maxiter=200)
    result = solver.solve(
        objective=objective,
        x0=np.array([0.0], dtype=np.float64),
        param_names=["beta"],
    )

    assert "optax" in result.solver_name.lower()
    assert result.n_iterations > 0


"""Tests for the trust-region Newton-CG solvers."""


import pandas as pd
import pytest

from locpick import MNL, ChoiceTable
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
    baseline = MNL(mnl_table, formula="rent + jobs", solver=LBFGSSolver())
    baseline.fit()
    ll_base = baseline._result.log_likelihood

    model = MNL(mnl_table, formula="rent + jobs", solver=solver_cls())
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
    from locpick import MNL
    from locpick.dgp import simulate_scl

    ds = simulate_scl(n_obs=600, n_alts=12, seed=11)

    base = MNL(
        data=ds.choice_table,
        formula="cost + time + income_x_cost",
        graph=ds.adjacency,
        solver=LBFGSSolver(),
        backend="jax",
    )
    base.fit()

    test = MNL(
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


"""Tests for the OptimagicSolver wrapper around ``optimagic.minimize``."""


import pytest

pytest.importorskip("optimagic")

from locpick._solvers.lbfgs import LBFGSSolver
from locpick._solvers.optimagic import OptimagicSolver


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
    baseline = MNL(mnl_table, formula="rent + jobs", solver=LBFGSSolver())
    baseline.fit()
    ll_base = baseline._result.log_likelihood

    model = MNL(
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
    model = MNL(mnl_table, formula="rent + jobs", solver=solver)
    with pytest.raises(Exception):
        model.fit()
