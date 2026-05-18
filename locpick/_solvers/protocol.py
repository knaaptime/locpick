"""Solver protocol and result dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Solver(Protocol):
    """Protocol for choice model solvers.

    Any object that implements ``solve`` can be used as a solver for
    choice model estimation.
    """

    def solve(
        self,
        objective,
        x0: np.ndarray,
        param_names: list[str],
        bounds=None,
        fixed_mask=None,
        **kwargs,
    ) -> "SolverResult":
        """Estimate parameters by maximizing the log-likelihood.

        Parameters
        ----------
        objective : Objective
            An :class:`locpick._jax.objective.Objective` instance.
            Solvers extract ``fn``, ``grad``, ``jax_fn``, ``jax_grad``,
            ``param_names``, and ``bounds`` from this object.
        x0 : np.ndarray
            Initial parameter values.
        param_names : list[str]
            Names for each parameter.
        bounds : list of tuple or None
            Per-parameter bounds ``(min, max)``. ``None`` means unbounded.
        fixed_mask : np.ndarray or None
            Boolean mask — ``True`` marks parameters held fixed at their
            initial values.
        **kwargs
            Additional solver-specific options.

        Returns
        -------
        SolverResult
        """
        ...


@dataclass
class SolverResult:
    """Raw output from a solver.

    Parameters
    ----------
    coefficients : np.ndarray
        Estimated parameter values.
    hessian : np.ndarray or None
        Inverse Hessian (variance-covariance matrix) at the optimum.
    log_likelihood : float
        Log-likelihood at convergence.
    n_iterations : int
        Number of solver iterations.
    converged : bool
        Whether the solver converged.
    message : str
        Solver exit message.
    solver_name : str
        Name of the solver used.
    raw : dict
        Raw solver output for debugging.
    """

    coefficients: np.ndarray
    hessian: np.ndarray | None = None
    log_likelihood: float = 0.0
    n_iterations: int = 0
    converged: bool = True
    message: str = ""
    solver_name: str = ""
    raw: dict = field(default_factory=dict)


# ------------------------------------------------------------------
# Solver registry
# ------------------------------------------------------------------

_SOLVERS: dict[str, type] = {}


def _register_builtins() -> None:
    """Register built-in solvers."""
    from .lbfgs import LBFGSSolver

    _SOLVERS.setdefault("lbfgs", LBFGSSolver)

    # Trust-region Newton (optional — requires JAX + scipy)
    try:
        from .trust_ncg import TrustKrylovSolver, TrustNCGSolver

        _SOLVERS.setdefault("trust-ncg", TrustNCGSolver)
        _SOLVERS.setdefault("trust-krylov", TrustKrylovSolver)
    except ImportError:
        pass

    # Optimagic (optional — requires optimagic)
    try:
        from .optimagic import OptimagicSolver

        _SOLVERS.setdefault("optimagic", OptimagicSolver)
    except ImportError:
        pass

    # Optimistix (optional — requires optimistix)
    try:
        from .optimistix import OptimistixSolver

        _SOLVERS.setdefault("optimistix", OptimistixSolver)
    except ImportError:
        pass


def register_solver(name: str, solver_cls: type) -> None:
    """Register a custom solver class.

    Parameters
    ----------
    name : str
        Name to register the solver under.
    solver_cls : type
        Solver class implementing the Solver protocol.
    """
    _SOLVERS[name] = solver_cls


def get_solver(name: str, **kwargs) -> Solver:
    """Get a solver instance by name.

    Parameters
    ----------
    name : str
        Name of the solver (e.g., ``"lbfgs"``, ``"optax"``).
    **kwargs
        Additional keyword arguments passed to the solver constructor.

    Returns
    -------
    Solver
        An instance of the requested solver.

    Raises
    ------
    KeyError
        If the solver name is not registered.
    """
    _register_builtins()
    if name not in _SOLVERS:
        available = ", ".join(sorted(_SOLVERS.keys()))
        raise KeyError(f"Unknown solver '{name}'. Available solvers: {available}")
    return _SOLVERS[name](**kwargs)


def list_solvers() -> list[str]:
    """Return a list of registered solver names."""
    _register_builtins()
    return sorted(_SOLVERS.keys())
