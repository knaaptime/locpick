"""Base class and protocol for choice model classes.

This module defines the :class:`ChoiceModel` protocol and the
:class:`BaseChoiceModel` abstract base class that all concrete
model classes implement.  It also provides :class:`SpatialMixin`,
a mixin for models that require a spatial adjacency graph.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, Union, runtime_checkable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cho_factor, cho_solve

from locpick._jax.objective import Objective
from locpick._solvers.protocol import Solver, SolverResult, get_solver
from locpick.data.arrays import ChoiceArrays
from locpick.data.choicetable import ChoiceTable
from locpick.data.problem import EstimationProblem
from locpick.results.fit_result import FitResult

# ------------------------------------------------------------------
# Cholesky-based linear-algebra helpers
# ------------------------------------------------------------------
# These replace np.linalg.inv calls with Cholesky decomposition, which
# is both faster and more numerically stable for positive-definite (PD)
# and positive-semidefinite (PSD) matrices.


def _safe_inv(A: np.ndarray) -> np.ndarray:
    """Invert a symmetric positive-(semi)definite matrix via Cholesky.

    Falls back to LU decomposition if the matrix is not PD.
    Returns an array of NaN on failure.
    """
    n = A.shape[0]
    try:
        return cho_solve(cho_factor(A), np.eye(n))
    except np.linalg.LinAlgError:
        # Not PD — fall back to general inverse
        try:
            return np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return np.full_like(A, np.nan)


def _sandwich_inv(H_inv: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the sandwich covariance H⁻¹ B H⁻¹.

    When ``H_inv`` is already available (cached from the Hessian inverse),
    the sandwich is computed directly as a matrix product.  This is the
    common path after model estimation.

    Parameters
    ----------
    H_inv : np.ndarray
        Inverse of the negative Hessian (covariance matrix), already computed.
    B : np.ndarray
        Bread matrix (positive semi-definite), e.g. scores.T @ scores.

    Returns
    -------
    np.ndarray
        H⁻¹ B H⁻¹, the sandwich covariance.
    """
    return H_inv @ B @ H_inv


def _aggregate_per_obs_alt(s: pd.Series, by: str, name: str):
    """Aggregate a per-(obs, alt) Series into AME-style summaries."""
    if by == "alt":
        alt_level = s.index.names[1]
        out = s.groupby(level=alt_level).mean()
        out.name = name
        return out
    if by == "obs":
        obs_level = s.index.names[0]
        out = s.groupby(level=obs_level).mean()
        out.name = name
        return out
    if by == "overall":
        return float(s.mean())
    raise ValueError(f"by must be 'alt', 'obs', or 'overall'; got {by!r}")


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


# ---------------------------------------------------------------------------
# Helpers shared across all models
# ---------------------------------------------------------------------------


def _compute_null_ll(arrays: ChoiceArrays) -> float:
    """Compute the null log-likelihood (equal-choice-probability model).

    Parameters
    ----------
    arrays : ChoiceArrays
        Estimation data arrays.

    Returns
    -------
    float
        Null log-likelihood.
    """
    n_obs = arrays.n_obs
    n_alts = arrays.n_alts
    if arrays.available is not None:
        avail = np.asarray(arrays.available, dtype=np.float64).reshape(n_obs, -1)
        n_avail = avail.sum(axis=1)
        return -np.sum(np.log(n_avail))
    return -n_obs * np.log(n_alts)


def _compute_fit_statistics(
    ll: float,
    ll_null: float,
    n_obs: int,
    n_params: int,
    n_alts: int,
    coefficients: pd.Series,
    std_errors: pd.Series,
    model_type: str,
    solver_name: str,
    solver_result_raw: Any,
) -> dict:
    """Compute standard fit statistics from estimation results.

    Parameters
    ----------
    ll : float
        Log-likelihood at estimated parameters.
    ll_null : float
        Null log-likelihood.
    n_obs : int
        Number of observations.
    n_params : int
        Number of estimated parameters.
    n_alts : int
        Number of alternatives.
    coefficients : pd.Series
        Estimated coefficients (display-scale).
    std_errors : pd.Series
        Standard errors.
    model_type : str
        Human-readable model name.
    solver_name : str
        Name of the solver used.
    solver_result_raw : Any
        Raw solver result for storage.

    Returns
    -------
    dict
        Dictionary with keys: ``coefficients``, ``std_errors``, ``t_values``,
        ``p_values``, ``conf_int``, ``log_likelihood``, ``log_likelihood_null``,
        ``n_observations``, ``n_parameters``, ``n_alts``, ``aic``, ``bic``,
        ``rho_squared``, ``rho_bar_squared``, ``model_type``, ``solver_name``,
        ``solver_result``.
    """
    t_values = np.where(
        std_errors.values > 0,
        coefficients.values / std_errors.values,
        np.nan,
    )
    p_values = 2 * (1 - stats.norm.cdf(np.abs(np.nan_to_num(t_values))))

    z_crit = stats.norm.ppf(0.975)
    conf_lower = coefficients.values - z_crit * std_errors.values
    conf_upper = coefficients.values + z_crit * std_errors.values

    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(n_obs) - 2 * ll
    rho_squared = 1 - ll / ll_null
    rho_bar_squared = 1 - (ll - n_params) / ll_null

    return {
        "coefficients": coefficients,
        "std_errors": std_errors,
        "t_values": pd.Series(t_values, index=coefficients.index, name="t_value"),
        "p_values": pd.Series(p_values, index=coefficients.index, name="p_value"),
        "conf_int": pd.DataFrame(
            {"lower": conf_lower, "upper": conf_upper},
            index=coefficients.index,
        ),
        "log_likelihood": ll,
        "log_likelihood_null": ll_null,
        "n_observations": n_obs,
        "n_parameters": n_params,
        "n_alts": n_alts,
        "aic": aic,
        "bic": bic,
        "rho_squared": rho_squared,
        "rho_bar_squared": rho_bar_squared,
        "model_type": model_type,
        "solver_name": solver_name,
        "solver_result": solver_result_raw,
    }


# ---------------------------------------------------------------------------
# Base model class
# ---------------------------------------------------------------------------


class BaseChoiceModel(ABC):
    """Abstract base class for discrete choice model estimators.

    Provides common infrastructure for model estimation, result caching,
    and prediction. Subclasses must implement model-specific methods
    for building objectives and computing probabilities.

    Parameters
    ----------
    data : ChoiceTable or EstimationProblem
        The choice data to estimate on.
    formula : str, optional
        A formulaic formula string.
    spec : ModelSpec, optional
        A ModelSpec object.  Mutually exclusive with ``formula``.
    solver : str or Solver, optional
        Solver name or instance.  Default ``"lbfgs"``.
    solver_options : dict, optional
        Additional options passed to the solver constructor.
    backend : str, optional
        Computation backend hint.
    weights : str or array-like, optional
        Observation weights.
    availability : str or array-like, optional
        Alternative availability.
    """

    def __init__(
        self,
        data: Union[ChoiceTable, EstimationProblem],
        formula: Optional[str] = None,
        spec=None,
        solver: Union[str, Solver] = None,
        solver_options: Optional[dict] = None,
        backend: Optional[str] = None,
        weights: Optional[Union[str, np.ndarray]] = None,
        availability: Optional[Union[str, np.ndarray]] = None,
    ):
        from locpick.spec.model_spec import ModelSpec

        self._solver_options = solver_options or {}
        self._backend = backend

        # Resolve solver
        solver_name = solver if solver is not None else "lbfgs"
        if isinstance(solver_name, str):
            self._solver = get_solver(solver_name, **self._solver_options)
        else:
            self._solver = solver_name

        # Handle EstimationProblem path
        if isinstance(data, EstimationProblem):
            self._problem = data
            self._data = None
            self._formula = None
            self._spec = None
            self._weights = None
            self._availability = None
            self._arrays = data.arrays
        else:
            self._problem = None
            self._data = data
            self._formula = formula
            self._weights = weights
            self._availability = availability

            # Build ModelSpec from formula if needed
            if formula is not None and spec is not None:
                raise ValueError("Provide 'formula' or 'spec', not both.")
            if formula is None and spec is None:
                raise ValueError("Either 'formula' or 'spec' must be provided.")
            if formula is not None:
                self._spec = ModelSpec(formula=formula)
            else:
                self._spec = spec

            # Lazy-initialized estimation arrays
            self._arrays: Optional[ChoiceArrays] = None

        self._result: Optional[FitResult] = None
        self._objective: Optional[Objective] = None

        # Caches (cleared on re-estimation)
        self._hessian_inverse: Optional[np.ndarray] = None
        self._observation_scores_cache: dict = {}
        self._probabilities_cache: Optional[np.ndarray] = None
        self._utilities_cache: Optional[np.ndarray] = None
        self._covariance_robust_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self):
        """The ChoiceTable data."""
        return self._data

    @property
    def spec(self):
        """The ModelSpec used for estimation."""
        return self._spec

    @property
    def solver(self) -> Solver:
        """The solver used for estimation."""
        return self._solver

    @property
    def result(self) -> Optional[FitResult]:
        """The estimation result, or None if not yet estimated."""
        return self._result

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def fit(self, **kwargs) -> FitResult:
        """Estimate the model and return results.

        Subclasses that need to resolve spatial graphs or set up
        model-specific data should override ``_pre_fit()`` instead
        of this method.

        Returns
        -------
        FitResult
        """
        arrays = self._get_arrays()
        self._arrays = arrays

        # Hook for subclasses to resolve spatial graphs, etc.
        self._pre_fit(arrays)

        objective = self._build_objective(arrays)
        self._objective = objective
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

    def _pre_fit(self, arrays: ChoiceArrays) -> None:
        """Hook called before objective construction in ``fit()``.

        Subclasses can override this to resolve spatial graphs, build
        edge structures, or perform other pre-estimation setup.  The
        default implementation does nothing.
        """
        pass

    def _get_arrays(self) -> ChoiceArrays:
        """Get estimation arrays from problem or build from data."""
        if self._problem is not None:
            return self._problem.arrays
        return self._build_arrays()

    def _build_arrays(self) -> ChoiceArrays:
        """Build ChoiceArrays from the data and spec.

        This is the same across all model classes — it delegates to
        ``EstimationProblem.from_choice_table()`` with the model's
        stored configuration.
        """
        if self._problem is not None:
            return self._problem.arrays

        spec = self._spec if self._formula is None else None
        self._problem = EstimationProblem.from_choice_table(
            self._data,
            spec=spec,
            formula=self._formula,
            weights=self._weights,
            available=self._availability,
            backend=self._backend or "auto",
            solver_name=getattr(self._solver, "name", "lbfgs"),
            solver_options=self._solver_options or None,
        )
        return self._problem.arrays

    def _get_solver_inputs(self, arrays: ChoiceArrays):
        """Get initial values, param names, bounds, and fixed mask.

        Subclasses that add extra parameters (rho, lambda, sigma)
        should override this to extend the parameter vector.
        """
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

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _clear_caches(self):
        """Clear all cached computation results."""
        self._hessian_inverse = None
        self._observation_scores_cache = {}
        self._probabilities_cache = None
        self._utilities_cache = None
        self._covariance_robust_cache = None

    def _compute_hessian(self, beta: np.ndarray) -> np.ndarray:
        """Compute the Hessian at the given parameter values.

        Uses the most accurate method available:
        1. HVP-based Hessian (JAX autodiff) when JAX objective is available
        2. Gradient finite differences when only gradient is available
        3. Full finite differences as last resort

        Parameters
        ----------
        beta : np.ndarray
            Parameter vector (natural scale).

        Returns
        -------
        np.ndarray, shape (n_params, n_params)
            Hessian matrix of the log-likelihood (negative definite).
        """
        if self._objective is not None:
            return self._objective.hessian(beta)
        # Fallback: should not normally reach here
        return self._finite_diff_hessian(beta)

    def _compute_std_errors_from_hessian(self, hess: np.ndarray) -> np.ndarray:
        """Compute standard errors from a log-likelihood Hessian.

        The Hessian of the log-likelihood is negative definite (since we
        maximize).  Standard errors are the square root of the diagonal
        of the inverse of the *negative* Hessian:

            SE = sqrt(diag(inv(-H)))

        Parameters
        ----------
        hess : np.ndarray, shape (n_params, n_params)
            Hessian matrix of the log-likelihood (negative definite).

        Returns
        -------
        np.ndarray, shape (n_params,)
            Standard errors for each parameter.
        """
        try:
            neg_hess = -hess
            cov = cho_solve(cho_factor(neg_hess), np.eye(neg_hess.shape[0]))
            diag_cov = np.diag(cov)
            # Clamp tiny negative values (numerical noise) to zero,
            # but treat zero-variance parameters as unidentified (SE=0
            # is never meaningful) and mark them NaN instead.
            se = np.sqrt(np.maximum(diag_cov, 0))
            se[se == 0] = np.nan
            return se
        except np.linalg.LinAlgError:
            # Not PD — fall back to general inverse
            try:
                cov = np.linalg.inv(-hess)
                diag_cov = np.diag(cov)
                se = np.sqrt(np.maximum(diag_cov, 0))
                se[se == 0] = np.nan
                return se
            except np.linalg.LinAlgError:
                return np.full(hess.shape[0], np.nan)

    def _finite_diff_hessian(self, beta: np.ndarray) -> np.ndarray:
        """Compute Hessian via central finite differences (fallback)."""
        n = len(beta)
        h = 1e-5
        hess = np.zeros((n, n))
        ll_fn = self._objective.fn if self._objective is not None else None
        if ll_fn is None:
            return np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i, n):
                x_pp = beta.copy()
                x_pp[i] += h
                x_pp[j] += h
                x_pm = beta.copy()
                x_pm[i] += h
                x_pm[j] -= h
                x_mp = beta.copy()
                x_mp[i] -= h
                x_mp[j] += h
                x_mm = beta.copy()
                x_mm[i] -= h
                x_mm[j] -= h
                hess[i, j] = (ll_fn(x_pp) - ll_fn(x_pm) - ll_fn(x_mp) + ll_fn(x_mm)) / (4 * h * h)
                hess[j, i] = hess[i, j]
        return hess

    def _get_hessian_inverse(self) -> Optional[np.ndarray]:
        """Get the inverse of the negative Hessian (covariance matrix).

        Uses the following priority:
        1. Cached inverse Hessian (if already computed)
        2. Inverse of HVP-based negative Hessian (exact, via JAX autodiff)
        3. Solver's approximate inverse Hessian (e.g. L-BFGS-B hess_inv)
        4. Diagonal approximation from standard errors

        Returns
        -------
        np.ndarray or None
            Inverse of the negative Hessian (covariance matrix), or None.
        """
        if self._hessian_inverse is not None:
            return self._hessian_inverse

        # Try HVP-based Hessian first (exact, via JAX autodiff)
        if self._objective is not None and self._result is not None:
            try:
                hess = self._compute_hessian(self._result.coefficients.values)
                # hess is the Hessian of the log-likelihood (negative definite).
                # The covariance matrix is inv(-hess).
                neg_hess = -hess
                self._hessian_inverse = cho_solve(cho_factor(neg_hess), np.eye(neg_hess.shape[0]))
                return self._hessian_inverse
            except np.linalg.LinAlgError:
                # Not PD — fall back to general inverse
                try:
                    self._hessian_inverse = np.linalg.inv(-hess)
                    return self._hessian_inverse
                except Exception:
                    pass
            except Exception:
                pass

        # Fallback: solver's approximate inverse Hessian (e.g. L-BFGS-B)
        if (
            self._result is not None
            and self._result.solver_result
            and "scipy_result" in self._result.solver_result
        ):
            scipy_result = self._result.solver_result["scipy_result"]
            if hasattr(scipy_result, "hess_inv"):
                try:
                    self._hessian_inverse = np.asarray(
                        scipy_result.hess_inv.todense()
                        if hasattr(scipy_result.hess_inv, "todense")
                        else scipy_result.hess_inv
                    )
                    return self._hessian_inverse
                except Exception:
                    pass

        # Last resort: diagonal approximation from standard errors
        if (
            self._result is not None
            and self._result.std_errors is not None
            and not self._result.std_errors.isna().all()
        ):
            variances = self._result.std_errors.values**2
            self._hessian_inverse = np.diag(variances)
            return self._hessian_inverse

        return None

    # ------------------------------------------------------------------
    # Aggregated marginal effects / elasticities
    # ------------------------------------------------------------------

    def average_marginal_effect(
        self,
        variable: str,
        data=None,
        by: str = "alt",
    ):
        """Average direct marginal effect (AME) of ``variable``.

        Aggregates the per-observation marginal effects returned by
        :meth:`marginal_effect` over the sample.

        Parameters
        ----------
        variable : str
            Variable name.
        data : ChoiceTable, optional
            Data to evaluate on. Defaults to the estimation sample.
        by : {"alt", "obs", "overall"}, default ``"alt"``
            Aggregation level.  ``"alt"`` returns one value per
            alternative (the usual AME); ``"obs"`` returns one value per
            observation (averaged over its alternatives); ``"overall"``
            returns a single scalar.

        Returns
        -------
        pd.Series or float
        """
        me = self.marginal_effect(data=data, variable=variable)
        return _aggregate_per_obs_alt(me, by, name=f"ame_{variable}")

    def average_cross_marginal_effect(
        self,
        variable: str,
        data=None,
        by: str = "alt",
    ):
        """Average cross marginal effect of ``variable``.

        See :meth:`average_marginal_effect` for the ``by`` argument.
        """
        cme = self.cross_marginal_effect(data=data, variable=variable)
        return _aggregate_per_obs_alt(cme, by, name=f"acme_{variable}")

    def average_elasticity(
        self,
        variable: str,
        data=None,
        by: str = "alt",
    ):
        """Average direct elasticity of ``variable``.

        See :meth:`average_marginal_effect` for the ``by`` argument.
        """
        el = self.elasticity(data=data, variable=variable)
        return _aggregate_per_obs_alt(el, by, name=f"ae_{variable}")


# ---------------------------------------------------------------------------
# Spatial mixin
# ---------------------------------------------------------------------------


class SpatialMixin:
    """Mixin for models that require a spatial adjacency graph.

    Provides ``_resolve_spatial_graph()`` and ``_pre_fit()`` hooks
    that handle graph resolution, allocation computation, and
    ``EdgeStructure`` construction.  Subclasses that use this mixin
    must set ``self._graph_input`` before calling ``fit()``.

    This mixin eliminates the duplicated graph-resolution boilerplate
    that was previously copy-pasted across SCL, MSCL, NestedSCL, and
    MNSCL.
    """

    _graph_input: Any
    _omega: Optional[np.ndarray]
    _allocation: Optional[np.ndarray]
    _edge_list: Optional[list]
    _n_alts_graph: Optional[int]
    _edge_struct: Optional[Any]  # EdgeStructure from scl.py

    def _resolve_spatial_graph(self) -> tuple[np.ndarray, list, int]:
        """Resolve the spatial graph and store allocation/edge data.

        Delegates to :func:`locpick.models._spatial._resolve_spatial_graph`
        and caches the results on ``self``.

        Returns
        -------
        omega : np.ndarray
            Adjacency matrix with preserved weights.
        allocation : np.ndarray
            Row-standardised allocation parameters.
        edge_list : list of (int, int)
            Paired-nest edges.
        n_alts : int
            Number of alternatives (dimension of the graph).
        """
        from locpick.models._spatial import EdgeStructure, _resolve_spatial_graph

        omega, allocation, edge_list, n_alts = _resolve_spatial_graph(self._graph_input)
        self._omega = omega
        self._allocation = allocation
        self._edge_list = edge_list
        self._n_alts_graph = n_alts

        # Precompute edge structure (consumed by the JAX builders)
        self._edge_struct = EdgeStructure(edge_list, n_alts, allocation)

        return omega, allocation, edge_list, n_alts

    def _validate_graph_size(self, arrays: ChoiceArrays) -> None:
        """Validate that the spatial graph matches the choice data.

        Raises
        ------
        ValueError
            If the graph has a different number of nodes than the
            choice data has alternatives.
        """
        if self._n_alts_graph is not None and self._n_alts_graph != arrays.n_alts:
            raise ValueError(
                f"Spatial graph has {self._n_alts_graph} nodes but the choice "
                f"data has {arrays.n_alts} alternatives.  The graph must "
                f"cover exactly the same alternatives as the choice data."
            )
