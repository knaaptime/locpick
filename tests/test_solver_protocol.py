"""Tests for the unified Objective-first solver interface."""

import inspect

import numpy as np
import pytest

from locpick._solvers.lbfgs import LBFGSSolver
from locpick._solvers.optax import OptaxSolver
from locpick._solvers.optimagic import OptimagicSolver


def test_solver_signatures_are_objective_first():
    """All solvers should expose the same Objective-first solve signature."""
    expected = ["self", "objective", "x0", "param_names", "bounds", "fixed_mask"]

    for solver_cls in (LBFGSSolver, OptaxSolver, OptimagicSolver):
        names = list(inspect.signature(solver_cls.solve).parameters.keys())[:6]
        assert names == expected


@pytest.mark.parametrize("solver_cls", [LBFGSSolver, OptaxSolver, OptimagicSolver])
def test_solvers_require_objective_instance(solver_cls):
    """Solvers should reject legacy callable-only inputs."""
    solver = solver_cls(maxiter=1) if solver_cls is not OptimagicSolver else solver_cls()

    with pytest.raises(TypeError, match="Objective"):
        solver.solve(
            objective=lambda x: -float(np.sum(np.asarray(x, dtype=np.float64) ** 2)),
            x0=np.array([0.0], dtype=np.float64),
            param_names=["beta"],
        )
