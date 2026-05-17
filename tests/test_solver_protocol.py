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
    optax = pytest.importorskip("optax")
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
    optax = pytest.importorskip("optax")
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
