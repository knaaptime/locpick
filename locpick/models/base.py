"""Base class and protocol for choice model classes.

This module defines the :class:`ChoiceModel` protocol and the
:class:`BaseChoiceModel` abstract base class that all concrete
model classes implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from locpick.data.arrays import ChoiceArrays
from locpick.data.choicetable import ChoiceTable
from locpick.data.problem import EstimationProblem
from locpick.results.fit_result import FitResult
from locpick._jax.objective import Objective
from locpick._solvers.protocol import SolverResult, get_solver


@runtime_checkable
class ChoiceModel(Protocol):
    """Protocol for discrete choice model classes.

    All concrete model classes (``MultinomialLogit``, ``NestedLogit``,
    ``MixedLogit``, ``SpatiallyCorrelatedLogit``,
    ``MixedSpatiallyCorrelatedLogit``) implement this protocol.
    """

    def fit(self, **kwargs) -> FitResult:
        """Estimate the model and return results."""
        ...

    def probabilities(self, data=None, **params) -> np.ndarray:
        """Compute choice probabilities."""
        ...

    def utilities(self, data=None, **params) -> np.ndarray:
        """Compute choice utilities."""
        ...


class BaseChoiceModel(ABC):
    """Abstract base class for discrete choice model estimators.

    Provides common infrastructure for model estimation, result caching,
    and prediction. Subclasses must implement model-specific methods
    for building objectives and computing probabilities.
    """

    def __init__(
        self,
        data: ChoiceTable | EstimationProblem,
        formula: str | None = None,
        spec=None,
        solver: str = "lbfgs",
        backend: str | None = None,
        weights=None,
        availability=None,
    ):
        from locpick.spec.model_spec import ModelSpec

        # Handle EstimationProblem path
        if isinstance(data, EstimationProblem):
            self._problem = data
            self._data = None
            self._formula = None
            self._spec = None
            self._weights = None
            self._availability = None
        else:
            self._problem = None
            self._data = data
            self._formula = formula
            self._spec = spec
            self._weights = weights
            self._availability = availability

        # Build ModelSpec from formula if needed
        if formula is not None:
            self._spec = ModelSpec(formula=formula)

        # Lazy-initialized estimation arrays (for non-problem path)
        self._arrays: Optional[ChoiceArrays] = None
        self._result: Optional[FitResult] = None

        # Caches
        self._hessian_inverse: Optional[np.ndarray] = None
        self._observation_scores_cache: dict = {}
        self._probabilities_cache: Optional[np.ndarray] = None
        self._utilities_cache: Optional[np.ndarray] = None

        # Solver
        self._solver = get_solver(solver)
        self._backend = backend

    @property
    def data(self):
        """The ChoiceTable data."""
        return self._data

    @property
    def spec(self):
        """The ModelSpec used for estimation."""
        return self._spec

    @property
    def solver(self):
        """The solver used for estimation."""
        return self._solver

    @property
    def result(self) -> Optional[FitResult]:
        """The estimation result, or None if not yet estimated."""
        return self._result

    def fit(self, **kwargs) -> FitResult:
        """Estimate the model and return results.

        Returns
        -------
        FitResult
        """
        arrays = self._get_arrays()
        self._arrays = arrays
        objective = self._build_objective(arrays)
        x0, param_names, bounds, fixed_mask = self._get_solver_inputs(arrays)

        solver_result = self._solver.solve(
            objective=objective,
            x0=x0,
            param_names=param_names,
            bounds=bounds,
            fixed_mask=fixed_mask,
        )

        self._result = self._build_fit_result(solver_result, arrays)
        self._clear_caches()
        return self._result

    def _get_arrays(self) -> ChoiceArrays:
        """Get estimation arrays from problem or build from data."""
        if self._problem is not None:
            return self._problem.arrays
        return self._build_arrays()

    def _get_solver_inputs(self, arrays: ChoiceArrays):
        """Get initial values, param names, bounds, and fixed mask."""
        if self._problem is not None:
            return (
                self._problem.initial_values,
                list(arrays.param_names),
                self._problem.bounds,
                self._problem.fixed_mask,
            )
        return (
            np.zeros(arrays.design_matrix.shape[1]),
            list(arrays.param_names),
            None,
            None,
        )

    @abstractmethod
    def _build_arrays(self) -> ChoiceArrays:
        """Build ChoiceArrays from the data and spec."""
        ...

    @abstractmethod
    def _build_objective(self, arrays: ChoiceArrays) -> Objective:
        """Build the optimization objective."""
        ...

    @abstractmethod
    def _build_fit_result(self, solver_result: SolverResult, arrays: ChoiceArrays) -> FitResult:
        """Build a FitResult from solver output."""
        ...

    @abstractmethod
    def probabilities(self, data=None, **params) -> np.ndarray:
        """Compute choice probabilities."""
        ...

    @abstractmethod
    def utilities(self, data=None, **params) -> np.ndarray:
        """Compute choice utilities."""
        ...

    def _clear_caches(self):
        """Clear all cached computation results."""
        self._hessian_inverse = None
        self._observation_scores_cache = {}
        self._probabilities_cache = None
        self._utilities_cache = None
