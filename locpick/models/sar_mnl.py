"""Spatial Autoregressive Multinomial Logit (SAR-MNL) model.

Implements the pseudo maximum likelihood (PML) estimator from
Smirnov (2010): a spatial autoregressive lag in the systematic
utility of alternatives (spatial locations), with variance
normalisation by ``diag((I - ρW)^{-1})`` for consistency.

The model specifies:

.. math::

    V_j = \\rho \\sum_k w_{jk} V_k + Z_j \\beta + X_{ij} \\gamma

yielding reduced-form utilities :math:`V^* = (I - \\rho W)^{-1}
(Z\\beta + X\\gamma)`, normalised by :math:`D = \\text{diag}((I -
\\rho W)^{-1})`, with standard MNL choice probabilities.

Estimation is via JAX autodiff through the spatial solve and
variance normalisation.  No log-determinant Jacobian is needed
(this is pseudo-ML, not full ML).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from locpick._solvers import Solver, SolverResult
from locpick.data.arrays import ChoiceArrays
from locpick.models._spatial_weights import resolve_spatial_weights
from locpick.models.base import (
    BaseChoiceModel,
    _compute_fit_statistics,
    _compute_null_ll,
)
from locpick.results.fit_result import FitResult


class SARMNL(BaseChoiceModel):
    r"""Spatial Autoregressive Multinomial Logit (SAR-MNL).

    Specifies a spatial autoregressive lag in the systematic utility
    of alternatives (spatial locations):

    .. math::

        V_j = \rho \sum_k w_{jk} V_k + Z_j \beta + X_{ij} \gamma

    yielding reduced-form utilities :math:`V^* = (I - \rho W)^{-1}
    (Z\beta + X\gamma)`, normalised by :math:`D = \text{diag}((I -
    \rho W)^{-1})`, with standard MNL choice probabilities.

    Estimation is via pseudo maximum likelihood (PML, Smirnov 2010)
    with JAX autodiff through the spatial solve and variance
    normalisation.  No log-determinant Jacobian is needed (this is
    pseudo-ML, not full ML).

    Parameters
    ----------
    data : ChoiceTable or EstimationProblem
        The choice data.
    formula : str, optional
        Formulaic formula string.
    spec : ModelSpec, optional
        ModelSpec object.
    W : libpysal.graph.Graph, scipy.sparse, or np.ndarray
        J×J spatial weights matrix connecting alternatives (locations).
        Row-standardised internally. Zero diagonal. A ``libpysal.graph.Graph``
        is the preferred input type (matching bayespecon). ``scipy.sparse``
        and dense ``np.ndarray`` are also accepted and converted internally.
    weights : str or array-like, optional
        Observation weights.
    availability : str or array-like, optional
        Alternative availability.
    solver : str or Solver, optional
        Solver for PML optimisation. Default "lbfgs".
    solver_options : dict, optional
    backend : str, optional

    Examples
    --------
    >>> from locpick import ChoiceTable, SARMNL
    >>> from libpysal.graph import Graph
    >>> ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=10)
    >>> W = Graph.build_knn(gdf, k=7).transform("r")
    >>> model = SARMNL(ct, formula="chosen ~ cost + time", W=W)
    >>> result = model.fit()
    >>> print(result.summary())
    """

    def __init__(
        self,
        data,
        formula: Optional[str] = None,
        spec=None,
        W=None,
        weights: Optional[Union[str, np.ndarray]] = None,
        availability: Optional[Union[str, np.ndarray]] = None,
        solver: Union[str, Solver] = "lbfgs",
        solver_options: Optional[dict] = None,
        backend: Optional[str] = None,
    ):
        super().__init__(
            data=data,
            formula=formula,
            spec=spec,
            solver=solver,
            solver_options=solver_options,
            backend=backend,
            weights=weights,
            availability=availability,
        )
        if W is None:
            raise ValueError("W (spatial weights matrix) is required for SARMNL.")
        self._W_input = W
        self._W_sparse = None  # resolved at fit time

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def W(self):
        """The spatial weights matrix (libpysal.graph.Graph)."""
        return self._W_input

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def _pre_fit(self, arrays: ChoiceArrays) -> None:
        """Resolve the spatial weights matrix."""
        self._W_sparse = resolve_spatial_weights(
            self._W_input, arrays.n_alts, row_standardize=True
        )[1]  # get the CSR sparse

    def _build_objective(self, arrays: ChoiceArrays):
        """Build the PML objective using JAX."""
        from locpick._jax.sar_kernels import build_sar_mnl_objective

        return build_sar_mnl_objective(arrays, self._W_sparse)

    def _get_solver_inputs(self, arrays: ChoiceArrays):
        """Get initial values, param names, bounds, fixed mask.

        Appends an unconstrained ``alpha_rho`` (initial 0.0) so the
        spatial autoregressive parameter is estimated alongside the
        utility coefficients.  ``rho = tanh(alpha_rho) ∈ (-1, 1)``.
        """
        x0, names, bounds, fixed_mask = super()._get_solver_inputs(arrays)
        x0 = np.concatenate([x0, np.zeros(1)])
        names = list(names) + ["alpha_rho"]
        return x0, names, bounds, fixed_mask

    def _build_fit_result(
        self, solver_result: SolverResult, arrays: ChoiceArrays
    ) -> FitResult:
        """Build a FitResult from solver output."""
        all_params = solver_result.coefficients
        utility_param_names = list(arrays.param_names)
        k = len(utility_param_names)

        # Layout: [beta_1..k, alpha_rho]
        beta = all_params[:k]
        alpha_rho = all_params[k]
        rho = np.tanh(alpha_rho)  # ρ ∈ (-1, 1)

        display_values = np.concatenate([beta, [rho]])
        display_names = utility_param_names + ["rho"]
        model_type = "Spatial Autoregressive Multinomial Logit"
        n_params = len(display_values)

        # Standard errors via Hessian
        std_errors = np.full(n_params, np.nan)
        try:
            hess = self._compute_hessian(all_params)
            se_unconstrained = self._compute_std_errors_from_hessian(hess)
            # Delta method: SE(rho) = (1 - rho^2) * SE(alpha_rho)
            se_rho = (1.0 - rho**2) * se_unconstrained[k]
            std_errors = np.concatenate([se_unconstrained[:k], [se_rho]])
        except Exception:
            if solver_result.hessian is not None:
                try:
                    se = np.sqrt(np.maximum(np.diag(solver_result.hessian), 0))
                    se[se == 0] = np.nan
                    se_rho = (1.0 - rho**2) * se[k]
                    std_errors = np.concatenate([se[:k], [se_rho]])
                except Exception:
                    pass

        coefficients = pd.Series(
            display_values, index=display_names, name="coefficient"
        )
        std_err_series = pd.Series(
            std_errors, index=display_names, name="std_error"
        )
        ll = solver_result.log_likelihood
        ll_null = _compute_null_ll(arrays)

        stats = _compute_fit_statistics(
            ll=ll,
            ll_null=ll_null,
            n_obs=arrays.n_obs,
            n_params=n_params,
            n_alts=arrays.n_alts,
            coefficients=coefficients,
            std_errors=std_err_series,
            model_type=model_type,
            solver_name=solver_result.solver_name,
            solver_result_raw=solver_result.raw,
        )

        return FitResult(spec=self._spec, **stats)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def probabilities(self, data=None, beta=None, rho=None):
        """Compute choice probabilities under the SAR-MNL model.

        Uses the full PML model: spatially-filtered + variance-normalised
        utilities, then standard MNL softmax.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to predict on. If None, uses estimation data.
        beta : np.ndarray or None
            Utility coefficients. If None, uses estimated values.
        rho : float or None
            Spatial autoregressive parameter. If None, uses estimated value.

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

        n_obs = arrays.n_obs
        n_alts = arrays.n_alts
        k = arrays.design_matrix.shape[1]

        if beta is None:
            coef_vals = np.asarray(
                self._result.coefficients.values, dtype=np.float64
            )
            beta = coef_vals[:k]
        if rho is None:
            rho = float(self._result.coefficients.values[k])

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        W_dense = np.asarray(self._W_sparse.toarray(), dtype=np.float64)

        # Base utilities
        V_base = (dm @ beta).reshape(n_obs, n_alts)

        # Sampling correction
        from locpick._sampling.correction import apply_sampling_correction

        V_base = apply_sampling_correction(V_base, arrays)

        # Spatial filter + variance normalisation
        A = np.eye(n_alts) - rho * W_dense
        V_filtered = np.linalg.solve(A, V_base.T).T
        D = np.diag(np.linalg.inv(A))
        V_star = V_filtered / D[None, :]

        # Availability
        if arrays.available is not None:
            available = np.asarray(
                arrays.available, dtype=np.float64
            ).reshape(n_obs, n_alts)
        else:
            available = np.ones((n_obs, n_alts), dtype=np.float64)

        return mnl_probs_numpy(V_star, available, inclusion_probs=None)

    def utilities(self, data=None, beta=None, rho=None):
        """Compute spatially-filtered + variance-normalised utilities.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute utilities on. If None, uses estimation data.
        beta : np.ndarray or None
            Utility coefficients. If None, uses estimated values.
        rho : float or None
            Spatial autoregressive parameter. If None, uses estimated value.

        Returns
        -------
        np.ndarray, shape (n_obs, n_alts)
            Spatially-filtered and variance-normalised utilities.
        """
        if self._arrays is None:
            raise RuntimeError("Model must be estimated before prediction.")

        arrays = self._arrays
        if data is not None:
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )

        n_obs = arrays.n_obs
        n_alts = arrays.n_alts
        k = arrays.design_matrix.shape[1]

        if beta is None:
            beta = np.asarray(
                self._result.coefficients.values[:k], dtype=np.float64
            )
        if rho is None:
            rho = float(self._result.coefficients.values[k])

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        W_dense = np.asarray(self._W_sparse.toarray(), dtype=np.float64)

        V_base = (dm @ beta).reshape(n_obs, n_alts)

        from locpick._sampling.correction import apply_sampling_correction

        V_base = apply_sampling_correction(V_base, arrays)

        A = np.eye(n_alts) - rho * W_dense
        V_filtered = np.linalg.solve(A, V_base.T).T
        D = np.diag(np.linalg.inv(A))
        V_star = V_filtered / D[None, :]

        return V_star