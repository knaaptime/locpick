"""Tests for the unified Objective-first solver interface."""

import inspect

import numpy as np
import pytest

from locpick._jax.objective import Objective
from locpick._solvers.lbfgs import LBFGSSolver
from locpick._solvers.optax import OptaxSolver
from locpick._solvers.optimistix import OptimistixSolver


def test_solver_signatures_are_objective_first():
    """All solvers should expose the same Objective-first solve signature."""
    expected = ["self", "objective", "x0", "param_names", "bounds", "fixed_mask"]

    for solver_cls in (LBFGSSolver, OptaxSolver, OptimistixSolver):
        names = list(inspect.signature(solver_cls.solve).parameters.keys())[:6]
        assert names == expected


@pytest.mark.parametrize("solver_cls", [LBFGSSolver, OptaxSolver, OptimistixSolver])
def test_solvers_require_objective_instance(solver_cls):
    """Solvers should reject legacy callable-only inputs."""
    solver = solver_cls(maxiter=1)

    with pytest.raises(TypeError, match="Objective"):
        solver.solve(
            objective=lambda x: -float(np.sum(np.asarray(x, dtype=np.float64) ** 2)),
            x0=np.array([0.0], dtype=np.float64),
            param_names=["beta"],
        )


def test_optimistix_defaults_preserve_inference_outputs():
    """Optimistix defaults should preserve Hessian-based inference outputs."""
    solver = OptimistixSolver()

    assert solver.rtol == pytest.approx(1e-6)
    assert solver.atol == pytest.approx(1e-6)
    assert solver.compute_hessian is True


def test_optimistix_emits_solver_diagnostics_in_raw_payload():
    """Optimistix should surface status/iteration diagnostics in SolverResult.raw."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def ll_jax(x):
        return -jnp.sum((x - 1.0) ** 2)

    objective = Objective.from_jax(ll_fn=ll_jax, grad_fn=jax.grad(ll_jax))
    solver = OptimistixSolver(maxiter=200)
    result = solver.solve(
        objective=objective,
        x0=np.array([0.0], dtype=np.float64),
        param_names=["beta"],
    )

    assert "optimistix_result" in result.raw
    assert "optimistix_message" in result.raw
    assert "optimistix_code" in result.raw
    assert "num_accepted_steps" in result.raw
    assert "run_diagnostics" in result.raw
    assert "compute_hessian" in result.raw
    assert result.raw["method"] == "lbfgs"
    assert result.n_iterations == int(result.raw["num_accepted_steps"])
    assert len(result.raw["run_diagnostics"]) == 1
