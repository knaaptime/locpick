"""Multinomial logit model for location choice estimation.

This module provides the ``MultinomialLogit`` class, which is the main entry
point for estimating discrete choice models. It supports JAX-accelerated
maximum-likelihood estimation with formulaic-based model specification.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from locpick.data.arrays import ChoiceArrays
from locpick.data.problem import EstimationProblem
from locpick.models.base import BaseChoiceModel
from locpick.results.fit_result import FitResult
from locpick._solvers import Solver, SolverResult, get_solver
from locpick.spec import ModelSpec


class MultinomialLogit(BaseChoiceModel):
    r"""Multinomial logit model for location choice estimation.

    The MNL model assumes that the unobserved utility components are
    independently and identically distributed (i.i.d.) Gumbel across
    alternatives, yielding the familiar closed-form choice probability:

    .. math::

        P_{ij} = \frac{\exp(V_{ij})}{\sum_{k=1}^{J} \exp(V_{ik})}

    where :math:`V_{ij} = \mathbf{x}_{ij}'\boldsymbol{\beta}` is the
    systematic utility of alternative :math:`j` for decision-maker
    :math:`n`, :math:`\mathbf{x}_{ij}` is a vector of observed
    attributes, and :math:`\boldsymbol{\beta}` are coefficients to be
    estimated.

    The log-likelihood is:

    .. math::

        \mathcal{L} = \sum_{n=1}^{N} w_n \sum_{j=1}^{J}
            d_{nj} \log P_{nj}

    where :math:`d_{nj} = 1` if decision-maker :math:`n` chose alternative
    :math:`j` and 0 otherwise, and :math:`w_n` is an optional observation
    weight.

    This is the primary model class for estimating discrete choice models.
    It supports:

    - Formulaic-based model specification (via formula string)
    - ModelSpec-based formula/scoped-term specification
    - EstimationProblem-based specification (via problem parameter)
    - JAX-accelerated log-likelihood and gradient computation
    - Multiple solver backends (L-BFGS-B, Optimistix)
    - Weighted estimation
    - Alternative availability constraints

    Parameters
    ----------
    data : ChoiceTable or EstimationProblem
        The choice data to estimate on. If an ``EstimationProblem`` is
        provided, ``formula``, ``spec``, ``weights``, and ``availability``
        are ignored — the problem carries all configuration.
    formula : str, optional
        A formulaic formula string (e.g., ``"choice ~ cost + time"``).
        Mutually exclusive with ``spec``.
    spec : ModelSpec, optional
        A ModelSpec object defining formula/scoped-term model structure.
        Mutually exclusive with ``formula``.
    problem : EstimationProblem, optional
        A fully-specified estimation problem. If provided, ``formula``,
        ``spec``, ``weights``, and ``availability`` are ignored.
    weights : str or array-like, optional
        Observation weights. If a string, refers to a column in the data.
        Must be length ``n_obs`` (one weight per observation). Each
        observation's contribution to the log-likelihood is multiplied
        by its weight.
    availability : str or array-like, optional
        Alternative availability. If a string, refers to a column.
    solver : str or Solver, optional
        Solver name or instance. Default ``"lbfgs"``.
    solver_options : dict, optional
        Additional options passed to the solver constructor.

    Examples
    --------
    >>> from locpick import ChoiceTable, FitDiagnostics, MultinomialLogit
    >>> ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=10)
    >>> model = MultinomialLogit(ct, formula="chosen ~ cost + time")
    >>> result = model.fit()
    >>> print(FitDiagnostics.summary(result))
    """

    def __init__(
        self,
        data,
        formula: Optional[str] = None,
        spec: Optional[ModelSpec] = None,
        problem: Optional[EstimationProblem] = None,
        weights: Optional[Union[str, np.ndarray]] = None,
        availability: Optional[Union[str, np.ndarray]] = None,
        solver: Union[str, Solver] = "lbfgs",
        solver_options: Optional[dict] = None,
        backend: Optional[str] = None,
    ):
        self._solver_options = solver_options or {}
        self._backend = backend

        # Resolve solver
        if isinstance(solver, str):
            self._solver = get_solver(solver, **self._solver_options)
        else:
            self._solver = solver

        self._data = data
        self._problem = problem
        self._formula = None
        self._spec = None
        self._weights = None
        self._availability = None

        if self._problem is None:
            # Build from ChoiceTable + formula/spec
            if formula is None and spec is None:
                raise ValueError("Either 'formula', 'spec', or 'problem' must be provided.")
            if formula is not None and spec is not None:
                raise ValueError("Provide 'formula' or 'spec', not both.")

            self._formula = formula
            self._weights = weights
            self._availability = availability

            # Build ModelSpec from formula if needed
            if formula is not None:
                self._spec = ModelSpec(formula=formula)
            else:
                self._spec = spec

            # Lazy-initialized estimation arrays (for non-problem path)
            self._arrays: Optional[ChoiceArrays] = None
        else:
            # Canonical estimation config lives on EstimationProblem.
            self._arrays = self._problem.arrays

        self._result: Optional[FitResult] = None

        # Caches (cleared on re-estimation)
        self._hessian_inverse: Optional[np.ndarray] = None
        self._observation_scores_cache: dict = {}
        self._probabilities_cache: Optional[np.ndarray] = None
        self._utilities_cache: Optional[np.ndarray] = None
        self._covariance_bhhh_cache: Optional[np.ndarray] = None
        self._covariance_robust_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self):
        """The ChoiceTable data."""
        return self._data

    @property
    def spec(self) -> ModelSpec:
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

    def fit(self) -> FitResult:
        """Estimate the model and return results.

        Returns
        -------
        FitResult
            Complete estimation results including coefficients, standard
            errors, fit statistics, and prediction methods.
        """
        return super().fit()

    def probabilities(self, data=None, beta=None):
        """Compute choice probabilities.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to predict on.  If ``None``, uses estimation data.
        beta : np.ndarray or None
            Utility coefficients.  If ``None``, uses estimated values.

        Returns
        -------
        np.ndarray, shape (n_obs, n_alts)
            Choice probabilities for each observation and alternative.
        """
        from locpick._kernels.mnl_numpy import mnl_probs_numpy

        if self._arrays is None:
            raise RuntimeError("Model must be estimated before prediction.")

        arrays = self._arrays
        if data is not None:
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )

        if beta is None:
            beta = np.asarray(self._result.coefficients.values, dtype=np.float64)

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        # Systematic utility with sampling correction
        utilities = (dm @ beta).reshape(n_obs, n_alts)
        from locpick._sampling.correction import apply_sampling_correction
        utilities = apply_sampling_correction(utilities, arrays)

        # Availability
        if arrays.available is not None:
            available = np.asarray(arrays.available, dtype=np.float64).reshape(
                n_obs, n_alts
            )
        else:
            available = np.ones((n_obs, n_alts), dtype=np.float64)

        return mnl_probs_numpy(utilities, available, inclusion_probs=None)

    def utilities(self, data=None, beta=None):
        """Compute deterministic utilities.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to predict on.  If ``None``, uses estimation data.
        beta : np.ndarray or None
            Utility coefficients.  If ``None``, uses estimated values.

        Returns
        -------
        np.ndarray, shape (n_obs, n_alts)
            Deterministic utilities for each observation and alternative.
            Includes sampling correction (log of inclusion probability)
            when the data has sampling metadata.
        """
        if self._arrays is None:
            raise RuntimeError("Model must be estimated before prediction.")

        arrays = self._arrays
        if data is not None:
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )

        if beta is None:
            beta = np.asarray(self._result.coefficients.values, dtype=np.float64)

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        # Systematic utility
        V = (dm @ beta).reshape(n_obs, n_alts)

        # Add sampling correction if present
        from locpick._sampling.correction import apply_sampling_correction

        V = apply_sampling_correction(V, arrays)

        return V

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, data=None, n_draws: int = 1, seed: Optional[int] = None) -> pd.DataFrame:
        """Simulate choices from the estimated model.

        Draws random choices according to the model's predicted
        probabilities.  Useful for forecasting, validation, and Monte
        Carlo analysis.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to simulate on.  If ``None``, uses estimation data.
        n_draws : int, optional
            Number of simulation draws per observation.  Default 1.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Simulated choices with columns ``draw``, ``obs_id``,
            ``alt_id``, and ``probability``.
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated before simulation.")

        arrays = self._arrays
        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )
            ct = data

        rng = np.random.default_rng(seed)
        probs = self.probabilities(data=data)
        # Normalize probabilities to sum to 1 (numerical precision)
        probs = probs / probs.sum(axis=1, keepdims=True)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        # Get alternative IDs from the data
        df = ct.to_frame()
        alt_ids = df[ct.alt_id_col].values.reshape(n_obs, n_alts)
        obs_ids = df[ct.obs_id_col].values.reshape(n_obs, n_alts)[:, 0]

        # Vectorised simulation: draw all choices at once
        # cumulative_probs[i, j] = sum_{k=0}^j probs[i, k]
        cumulative_probs = np.cumsum(probs, axis=1)
        # Draw one uniform per observation per draw
        uniform_draws = rng.random((n_draws, n_obs))  # (n_draws, n_obs)
        # Find first index where cumulative_prob > uniform
        chosen_indices = np.argmax(cumulative_probs[None, :, :] > uniform_draws[:, :, None], axis=2)
        # Handle edge case where uniform == 1.0 (shouldn't happen with random(), but safe)
        chosen_indices = np.clip(chosen_indices, 0, n_alts - 1)

        chosen_alts = alt_ids[np.arange(n_obs), chosen_indices]  # (n_draws, n_obs)
        chosen_probs = probs[np.arange(n_obs), chosen_indices]   # (n_draws, n_obs)

        # Build results DataFrame
        results = []
        for draw in range(n_draws):
            for i in range(n_obs):
                results.append(
                    {
                        "draw": draw,
                        ct.obs_id_col: obs_ids[i],
                        ct.alt_id_col: chosen_alts[draw, i],
                        "probability": chosen_probs[draw, i],
                    }
                )

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Marginal Effects
    # ------------------------------------------------------------------

    def marginal_effect(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute direct marginal effects for a variable.

        For MNL, the direct marginal effect of variable *x* in alternative *i*
        for observation *q* is: :math:`(1 - P_{qi}) \\beta_x`.

        This measures the change in probability (in percentage points) for a
        one-unit increase in the variable.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute marginal effects on.  If ``None``, uses
            estimation data.
        variable : str
            Name of the variable to compute marginal effect for.

        Returns
        -------
        pd.Series
            Direct marginal effects, indexed by (obs_id, alt_id).
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated before computing marginal effects.")

        arrays = self._arrays
        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )
            ct = data

        probs = self.probabilities(data=data if data is not None else None)
        beta = self._result.coefficients.get(variable, 0.0)
        df = ct.to_frame()

        me = (1 - probs.ravel()) * beta

        index = pd.MultiIndex.from_arrays(
            [df[ct.obs_id_col].values, df[ct.alt_id_col].values],
            names=[ct.obs_id_col, ct.alt_id_col],
        )
        return pd.Series(me, index=index, name=f"marginal_effect_{variable}")

    def cross_marginal_effect(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute cross-marginal effects for a variable.

        For MNL, the cross-marginal effect of variable *x* in alternative *j*
        with respect to a change in alternative *i* is: :math:`-P_i \\beta_x`.

        This measures the change in probability for alternative *j* (in
        percentage points) from a one-unit increase in the variable for
        alternative *i*.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute cross-marginal effects on.  If ``None``, uses
            estimation data.
        variable : str
            Name of the variable.

        Returns
        -------
        pd.Series
            Cross-marginal effects, indexed by (obs_id, alt_id).
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated before computing marginal effects.")

        arrays = self._arrays
        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )
            ct = data

        probs = self.probabilities(data=data if data is not None else None)
        beta = self._result.coefficients.get(variable, 0.0)
        df = ct.to_frame()

        cross_me = -probs.ravel() * beta

        index = pd.MultiIndex.from_arrays(
            [df[ct.obs_id_col].values, df[ct.alt_id_col].values],
            names=[ct.obs_id_col, ct.alt_id_col],
        )
        return pd.Series(cross_me, index=index, name=f"cross_marginal_effect_{variable}")

    # ------------------------------------------------------------------
    # Elasticities
    # ------------------------------------------------------------------

    def elasticity(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute direct elasticities for a variable.

        For MNL, the direct elasticity of variable *x* in alternative *i*
        for observation *q* is: :math:`(1 - P_{qi}) \\beta_x x_{qi}`.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute elasticities on.  If ``None``, uses
            estimation data.
        variable : str
            Name of the variable to compute elasticity for.

        Returns
        -------
        pd.Series
            Direct elasticities, indexed by (obs_id, alt_id).
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated before computing elasticities.")

        arrays = self._arrays
        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )
            ct = data

        probs = self.probabilities(data=data if data is not None else None)
        df = ct.to_frame()
        x = df[variable].values
        beta = self._result.coefficients.get(variable, 0.0)

        elasticities = (1 - probs.ravel()) * beta * x

        index = pd.MultiIndex.from_arrays(
            [df[ct.obs_id_col].values, df[ct.alt_id_col].values],
            names=[ct.obs_id_col, ct.alt_id_col],
        )
        return pd.Series(elasticities, index=index, name=f"elasticity_{variable}")

    def cross_elasticity(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute cross-elasticities for a variable.

        For MNL, the cross-elasticity of variable *x* in alternative *j*
        with respect to a change in alternative *i* is:
        :math:`-P_i \\beta_x x_{ij}`.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute cross-elasticities on.  If ``None``, uses
            estimation data.
        variable : str
            Name of the variable.

        Returns
        -------
        pd.Series
            Cross-elasticities, indexed by (obs_id, alt_id).
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated before computing elasticities.")

        arrays = self._arrays
        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )
            ct = data

        probs = self.probabilities(data=data if data is not None else None)
        beta = self._result.coefficients.get(variable, 0.0)
        df = ct.to_frame()
        x = df[variable].values

        cross_elast = -probs.ravel() * beta * x

        index = pd.MultiIndex.from_arrays(
            [df[ct.obs_id_col].values, df[ct.alt_id_col].values],
            names=[ct.obs_id_col, ct.alt_id_col],
        )
        return pd.Series(cross_elast, index=index, name=f"cross_elasticity_{variable}")

    # ------------------------------------------------------------------
    # Covariance estimation
    # ------------------------------------------------------------------

    def covariance_bhhh(self, data=None) -> np.ndarray:
        """Compute the BHHH covariance matrix.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute covariance on.  If ``None``, uses
            estimation data.

        Returns
        -------
        np.ndarray, shape (n_parameters, n_parameters)
            BHHH covariance matrix.
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated first.")

        arrays = self._arrays
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )

        scores = self._observation_scores(arrays)
        bhhh = scores.T @ scores
        try:
            return np.linalg.inv(bhhh)
        except np.linalg.LinAlgError:
            return np.full_like(bhhh, np.nan)

    def covariance_robust(self, data=None) -> np.ndarray:
        """Compute the sandwich (Huber-White) robust covariance matrix.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute covariance on.  If ``None``, uses
            estimation data.

        Returns
        -------
        np.ndarray, shape (n_parameters, n_parameters)
            Sandwich (robust) covariance matrix.
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated first.")

        arrays = self._arrays
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )

        scores = self._observation_scores(arrays)
        B = scores.T @ scores
        H_inv = self._get_hessian_inverse(arrays)

        if H_inv is None:
            try:
                return np.linalg.inv(B)
            except np.linalg.LinAlgError:
                return np.full_like(B, np.nan)

        return H_inv @ B @ H_inv

    def covariance_clustered(self, data=None, groups=None) -> np.ndarray:
        """Compute cluster-robust (Rogers) covariance matrix.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute covariance on.  If ``None``, uses
            estimation data.
        groups : array-like, shape (n_obs,)
            Cluster/group identifiers for each observation.

        Returns
        -------
        np.ndarray, shape (n_parameters, n_parameters)
            Cluster-robust covariance matrix.
        """
        from locpick.data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated first.")

        arrays = self._arrays
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )

        if groups is None:
            raise ValueError("groups must be provided for cluster-robust covariance.")

        scores = self._observation_scores(arrays)
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        n_params = scores.shape[1]

        B_clustered = np.zeros((n_params, n_params))
        for g in unique_groups:
            mask = groups == g
            g_c = scores[mask].sum(axis=0)
            B_clustered += np.outer(g_c, g_c)

        H_inv = self._get_hessian_inverse(arrays)

        if H_inv is None:
            try:
                return np.linalg.inv(B_clustered)
            except np.linalg.LinAlgError:
                return np.full_like(B_clustered, np.nan)

        return H_inv @ B_clustered @ H_inv

    def std_errors_bhhh(self, data=None) -> pd.Series:
        """Compute BHHH standard errors.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute standard errors on.

        Returns
        -------
        pd.Series
            BHHH standard errors, indexed by parameter name.
        """
        cov = self.covariance_bhhh(data=data)
        se = np.sqrt(np.diag(cov))
        return pd.Series(se, index=self._result.coefficients.index, name="std_error_bhhh")

    def std_errors_robust(self, data=None) -> pd.Series:
        """Compute sandwich (Huber-White) robust standard errors.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute standard errors on.

        Returns
        -------
        pd.Series
            Robust standard errors, indexed by parameter name.
        """
        cov = self.covariance_robust(data=data)
        se = np.sqrt(np.abs(np.diag(cov)))
        return pd.Series(se, index=self._result.coefficients.index, name="std_error_robust")

    def std_errors_clustered(self, data=None, groups=None) -> pd.Series:
        """Compute cluster-robust standard errors.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute standard errors on.
        groups : array-like, shape (n_obs,)
            Cluster/group identifiers.

        Returns
        -------
        pd.Series
            Cluster-robust standard errors, indexed by parameter name.
        """
        cov = self.covariance_clustered(data=data, groups=groups)
        se = np.sqrt(np.abs(np.diag(cov)))
        return pd.Series(se, index=self._result.coefficients.index, name="std_error_clustered")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observation_scores(self, arrays) -> np.ndarray:
        """Compute observation-level score (gradient) vectors.

        Parameters
        ----------
        arrays : ChoiceArrays
            The estimation data arrays.

        Returns
        -------
        np.ndarray, shape (n_obs, n_params)
            Score vector for each observation.
        """
        cache_key = id(arrays)
        if cache_key in self._observation_scores_cache:
            return self._observation_scores_cache[cache_key]

        from locpick._kernels.mnl_numpy import mnl_observation_scores_numpy

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        chosen = np.asarray(arrays.chosen, dtype=np.float64).reshape(arrays.n_obs, arrays.n_alts)
        beta = np.asarray(self._result.coefficients.values, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        if arrays.available is not None:
            available = np.asarray(arrays.available, dtype=np.float64).reshape(n_obs, n_alts)
        else:
            available = np.ones((n_obs, n_alts), dtype=np.float64)

        from locpick._sampling.correction import get_sampling_correction

        inclusion_probs = get_sampling_correction(arrays)

        weights = None
        if arrays.weights is not None:
            weights = np.asarray(arrays.weights, dtype=np.float64)

        scores = mnl_observation_scores_numpy(
            beta=beta,
            design_matrix=dm,
            chosen=chosen,
            available=available,
            n_obs=n_obs,
            n_alts=n_alts,
            weights=weights,
            inclusion_probs=inclusion_probs,
        )
        self._observation_scores_cache[cache_key] = scores
        return scores

    def _clear_caches(self):
        """Clear all cached computation results."""
        self._hessian_inverse = None
        self._observation_scores_cache = {}
        self._probabilities_cache = None
        self._utilities_cache = None
        self._covariance_bhhh_cache = None
        self._covariance_robust_cache = None

    def _get_hessian_inverse(self, arrays) -> np.ndarray | None:
        """Get the inverse Hessian from the solver result.

        Returns
        -------
        np.ndarray or None
            Inverse Hessian matrix, or None if not available.
        """
        if self._hessian_inverse is not None:
            return self._hessian_inverse

        if self._result is not None and self._result.solver_result and "scipy_result" in self._result.solver_result:
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

        if self._result is not None and self._result.std_errors is not None and not self._result.std_errors.isna().all():
            variances = self._result.std_errors.values**2
            self._hessian_inverse = np.diag(variances)
            return self._hessian_inverse

        return None

    def _build_arrays(self) -> ChoiceArrays:
        """Build ChoiceArrays from the data and spec."""
        if self._problem is None:
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

    def _build_objective(self, arrays: ChoiceArrays):
        """Build log-likelihood and gradient functions.

        Uses JAX when available for JIT-compiled computation, falling back
        to NumPy otherwise.
        """
        import os

        backend = (self._backend or os.environ.get("CHOICEMODELS_MNL_BACKEND", "")).lower()
        if backend == "numpy":
            return self._build_objective_numpy(arrays)
        try:
            from locpick._jax.builders import build_mnl_objective

            objective = build_mnl_objective(arrays)
            return objective
        except ImportError:
            return self._build_objective_numpy(arrays)

    def _build_objective_numpy(self, arrays: ChoiceArrays):
        """Build NumPy log-likelihood and gradient (fallback).

        Delegates to the canonical MNL kernels in
        :mod:`locpick._kernels.mnl_numpy` to avoid code duplication.
        """
        from locpick._kernels.mnl_numpy import (
            mnl_gradient_numpy,
            mnl_log_likelihood_numpy,
        )

        dm = arrays.design_matrix.astype(np.float64)
        dm_sparse = getattr(arrays, "design_matrix_sparse", None)
        chosen = arrays.chosen.astype(np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        weights = None
        if arrays.weights is not None:
            weights = arrays.weights.astype(np.float64)

        # Availability: default to all available if not provided
        if arrays.available is not None:
            available = arrays.available.astype(np.float64)
        else:
            available = np.ones((n_obs, n_alts), dtype=np.float64)

        from locpick._sampling.correction import get_sampling_correction

        inclusion_probs = get_sampling_correction(arrays)

        def log_likelihood(beta: np.ndarray) -> float:
            return mnl_log_likelihood_numpy(
                beta=beta,
                design_matrix=dm,
                chosen=chosen,
                available=available,
                n_obs=n_obs,
                n_alts=n_alts,
                weights=weights,
                inclusion_probs=inclusion_probs,
                design_matrix_sparse=dm_sparse,
            )

        def gradient(beta: np.ndarray) -> np.ndarray:
            return mnl_gradient_numpy(
                beta=beta,
                design_matrix=dm,
                chosen=chosen,
                available=available,
                n_obs=n_obs,
                n_alts=n_alts,
                weights=weights,
                inclusion_probs=inclusion_probs,
                design_matrix_sparse=dm_sparse,
            )

        from locpick._jax.objective import Objective
        return Objective(fn=log_likelihood, grad=gradient)

    def _build_fit_result(self, solver_result: SolverResult, arrays: ChoiceArrays) -> FitResult:
        """Build a FitResult from solver output."""
        beta = solver_result.coefficients
        param_names = list(arrays.param_names)
        n_params = len(param_names)

        # Standard errors from inverse Hessian
        if solver_result.hessian is not None:
            try:
                hess = solver_result.hessian
                # If fixed parameters were used, the Hessian only covers
                # free parameters. Expand to full parameter space.
                if hess.shape[0] < n_params and self._problem is not None:
                    fixed_mask = self._problem.fixed_mask
                    if fixed_mask is not None:
                        free_mask = ~fixed_mask
                        int(free_mask.sum())
                        full_hess = np.zeros((n_params, n_params))
                        # Place the free-parameter block into the full matrix
                        free_idx = np.where(free_mask)[0]
                        for i, fi in enumerate(free_idx):
                            for j, fj in enumerate(free_idx):
                                full_hess[fi, fj] = hess[i, j]
                        # Fixed parameters get zero variance (they're constants)
                        hess = full_hess
                std_errors = np.sqrt(np.diag(hess))
            except Exception:
                std_errors = np.full(len(beta), np.nan)
        else:
            # Compute numerically if not available from solver
            std_errors = self._compute_std_errors(arrays, beta)

        # T-values and p-values
        # Fixed parameters have zero std error → set t/p to NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = np.where(std_errors > 0, beta / std_errors, np.nan)
        from scipy import stats

        p_values = 2 * (1 - stats.norm.cdf(np.abs(np.nan_to_num(t_values))))

        # Confidence intervals (95%)
        z_crit = stats.norm.ppf(0.975)
        conf_lower = beta - z_crit * std_errors
        conf_upper = beta + z_crit * std_errors

        # Log-likelihood
        ll = solver_result.log_likelihood

        # Null log-likelihood: equal probability among available alternatives
        # per observation. When availability varies, this accounts for
        # different choice set sizes across observations.
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts
        if arrays.available is not None:
            avail = np.asarray(arrays.available, dtype=np.float64)
            n_avail = avail.reshape(n_obs, -1).sum(axis=1)
            ll_null = -np.sum(np.log(n_avail))
        else:
            ll_null = -n_obs * np.log(n_alts)

        # Number of parameters
        k = len(beta)

        # Fit statistics
        aic = 2 * k - 2 * ll
        bic = k * np.log(n_obs) - 2 * ll
        rho_squared = 1 - ll / ll_null
        rho_bar_squared = 1 - (ll - k) / ll_null

        # Build pandas objects
        coefficients = pd.Series(beta, index=param_names, name="coefficient")
        std_err_series = pd.Series(std_errors, index=param_names, name="std_error")
        t_series = pd.Series(t_values, index=param_names, name="t_value")
        p_series = pd.Series(p_values, index=param_names, name="p_value")
        conf_int = pd.DataFrame(
            {"lower": conf_lower, "upper": conf_upper},
            index=param_names,
        )

        return FitResult(
            coefficients=coefficients,
            std_errors=std_err_series,
            t_values=t_series,
            p_values=p_series,
            conf_int=conf_int,
            log_likelihood=ll,
            log_likelihood_null=ll_null,
            n_observations=n_obs,
            n_parameters=k,
            n_alts=n_alts,
            aic=aic,
            bic=bic,
            rho_squared=rho_squared,
            rho_bar_squared=rho_bar_squared,
            spec=self._spec,
            model_type="Multinomial Logit",
            solver_name=solver_result.solver_name,
            solver_result=solver_result.raw,
        )

    def _compute_std_errors(self, arrays: ChoiceArrays, beta: np.ndarray) -> np.ndarray:
        """Compute standard errors numerically via the Hessian.

        This is used as a fallback when the solver doesn't provide an
        inverse Hessian approximation.
        """
        from scipy.optimize import approx_fprime

        objective = self._build_objective(arrays)
        ll_fn = objective.fn

        def neg_ll(x):
            return -ll_fn(x)

        # Numerical Hessian via finite differences
        n = len(beta)
        h = 1e-5
        hessian = np.zeros((n, n))

        for i in range(n):

            def neg_ll_i(x):
                return neg_ll(x)

            grad_plus = approx_fprime(beta.copy(), neg_ll_i, h)
            beta_minus = beta.copy()
            beta_minus[i] -= h
            grad_minus = approx_fprime(beta_minus, neg_ll_i, h)
            hessian[:, i] = (grad_plus - grad_minus) / (2 * h)

        try:
            inv_hessian = np.linalg.inv(hessian)
            return np.sqrt(np.diag(np.abs(inv_hessian)))
        except np.linalg.LinAlgError:
            return np.full(n, np.nan)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "estimated" if self._result is not None else "not estimated"
        formula_str = self._formula or "custom spec"
        return f"MultinomialLogit(formula='{formula_str}', {status})"
