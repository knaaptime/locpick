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
    estimator : str, optional
        "auto" (default), "pml", "pml_cg", or "linearized_gmm".
        ``auto`` selects ``pml`` (dense solve) for n_alts ≤ 2000 and
        ``pml_cg`` (conjugate gradient) for larger alternative sets.

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
        estimator: str = "auto",
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
        self._estimator = estimator

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

    def fit(self, **kwargs) -> FitResult:
        """Estimate the model and return results.

        Dispatches to the PML estimator (JAX autodiff) or the
        linearized GMM estimator based on the ``estimator`` setting.
        """
        if self._estimator == "linearized_gmm":
            return self._fit_linearized_gmm()
        return super().fit(**kwargs)

    def _fit_linearized_gmm(self) -> FitResult:
        """Two-step linearized GMM estimation (Carrión-Flores et al. 2018)."""
        arrays = self._get_arrays()
        self._arrays = arrays
        self._pre_fit(arrays)

        from locpick._kernels.sar_mnl_numpy import fit_linearized_gmm

        result_dict = fit_linearized_gmm(arrays, self._W_sparse)

        beta = result_dict["beta"]
        rho = result_dict["rho"]
        se = result_dict["se"]
        ll = result_dict["log_likelihood"]

        utility_param_names = list(arrays.param_names)
        k = len(utility_param_names)
        display_values = np.concatenate([beta, [rho]])
        display_names = utility_param_names + ["rho"]
        model_type = "SAR-MNL (Linearized GMM)"
        n_params = len(display_values)

        # SEs: first k are beta SEs, last is rho SE
        std_errors = se[: k + 1]

        coefficients = pd.Series(display_values, index=display_names, name="coefficient")
        std_err_series = pd.Series(std_errors, index=display_names, name="std_error")
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
            solver_name="linearized_gmm",
            solver_result_raw=result_dict,
        )

        self._result = FitResult(spec=self._spec, **stats)
        self._clear_caches()
        return self._result

    def _build_objective(self, arrays: ChoiceArrays):
        """Build the PML objective using JAX.

        Auto-selects dense solve (n_alts ≤ 2000) or conjugate gradient
        (n_alts > 2000) based on the estimator setting.
        """
        from locpick._jax.sar_kernels import build_sar_mnl_objective

        # Auto-select estimator
        if self._estimator == "auto":
            if arrays.n_alts <= 2000:
                self._estimator = "pml"
            else:
                self._estimator = "pml_cg"

        use_cg = self._estimator == "pml_cg"
        return build_sar_mnl_objective(arrays, self._W_sparse, use_cg=use_cg)

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

    def _build_fit_result(self, solver_result: SolverResult, arrays: ChoiceArrays) -> FitResult:
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

        coefficients = pd.Series(display_values, index=display_names, name="coefficient")
        std_err_series = pd.Series(std_errors, index=display_names, name="std_error")
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
            coef_vals = np.asarray(self._result.coefficients.values, dtype=np.float64)
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
            available = np.asarray(arrays.available, dtype=np.float64).reshape(n_obs, n_alts)
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
            beta = np.asarray(self._result.coefficients.values[:k], dtype=np.float64)
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

    # ------------------------------------------------------------------
    # Marginal effects (LeSage & Pace 2009)
    # ------------------------------------------------------------------

    def marginal_effects(self, data=None, variable: Optional[str] = None):
        """Compute average direct, indirect, and total marginal effects.

        In the SAR-MNL model, a change in an attribute of alternative
        *j* affects not only *j*'s utility but also neighbouring
        alternatives through the spatial multiplier
        :math:`(I - \\rho W)^{-1}`.

        Following LeSage & Pace (2009), the marginal effect of variable
        *r* on the probability of choosing alternative *k* is an
        :math:`J \\times J` matrix.  Summary measures are:

        - **Direct effect**: average of diagonal elements (impact on
          own alternative).
        - **Indirect effect**: average of off-diagonal row sums
          (spillover to neighbouring alternatives).
        - **Total effect**: direct + indirect.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute marginal effects on.  If None, uses
            estimation data.
        variable : str
            Name of the variable to compute marginal effects for.

        Returns
        -------
        dict
            Dictionary with keys ``"direct"``, ``"indirect"``,
            ``"total"``, each mapping to a ``pd.Series`` indexed by
            alternative ID.
        """
        if self._arrays is None:
            raise RuntimeError("Model must be estimated before computing marginal effects.")

        from locpick.data.choicetable import ChoiceTable

        ct = self._data
        arrays = self._arrays
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )
            ct = data

        n_obs = arrays.n_obs
        n_alts = arrays.n_alts
        k = arrays.design_matrix.shape[1]

        # Get estimated parameters
        coef_vals = np.asarray(self._result.coefficients.values, dtype=np.float64)
        beta = coef_vals[:k]
        rho = float(coef_vals[k])

        # Get the beta for the requested variable
        param_names = list(arrays.param_names)
        if variable not in param_names:
            raise ValueError(f"Variable '{variable}' not found in parameters: {param_names}")
        beta_r = beta[param_names.index(variable)]

        # Compute probabilities
        probs = self.probabilities(data=data)

        # Spatial multiplier (I - rho*W)^{-1}
        W_dense = np.asarray(self._W_sparse.toarray(), dtype=np.float64)
        A = np.eye(n_alts) - rho * W_dense
        Z_mat = np.linalg.inv(A)  # (n_alts, n_alts)

        # Marginal effect matrix for variable r, alternative k:
        # ME_{k,r} = P_k * (beta_r * Z_{kk} - sum_l P_l * beta_r * Z_{lk})
        #         = beta_r * P_k * (Z_{kk} - sum_l P_l * Z_{lk})
        # But P varies across choosers.  For the average marginal effect,
        # we average over choosers:
        # AME_{k,r} = beta_r * avg(P_k) * (Z_{kk} - sum_l avg(P_l) * Z_{lk})

        # Compute the J×J marginal effect matrix (averaged over choosers)
        # ME[j, k] = beta_r * avg_probs[k] * (Z[k, j] - sum_l avg_probs[l] * Z[l, j])
        # But the standard LeSage-Pace formulation for MNL is:
        # dP_k/dX_j = P_k * (delta_{kj} - P_j) * beta_r * Z[j, ...]
        # This is complex — we use the simpler average approach:
        #
        # For each alternative k, the direct effect is:
        #   dP_k/dX_k = beta_r * Z[k,k] * P_k * (1 - P_k)
        # The indirect (spillover) effect from j to k (j != k) is:
        #   dP_k/dX_j = -beta_r * Z[k,j] * P_k * P_j
        # But with the spatial multiplier, Z replaces the identity.

        # Direct effects: average over choosers of
        #   beta_r * Z[k,k] * P_ik * (1 - P_ik)
        direct = np.zeros(n_alts)
        indirect = np.zeros(n_alts)
        for k_alt in range(n_alts):
            # Direct: own-alternative effect
            direct[k_alt] = (
                beta_r * Z_mat[k_alt, k_alt] * np.mean(probs[:, k_alt] * (1 - probs[:, k_alt]))
            )
            # Indirect: spillover from neighbours
            # Sum over j != k of dP_k/dX_j = -beta_r * sum_{j!=k} Z[k,j] * P_k * P_j
            for j_alt in range(n_alts):
                if j_alt != k_alt:
                    indirect[k_alt] += (
                        -beta_r * Z_mat[k_alt, j_alt] * np.mean(probs[:, k_alt] * probs[:, j_alt])
                    )

        total = direct + indirect

        # Get alternative IDs from the data
        df = ct.to_frame()
        alt_ids = df[ct.alt_id_col].values.reshape(n_obs, n_alts)[0]

        return {
            "direct": pd.Series(direct, index=alt_ids, name=f"direct_{variable}"),
            "indirect": pd.Series(indirect, index=alt_ids, name=f"indirect_{variable}"),
            "total": pd.Series(total, index=alt_ids, name=f"total_{variable}"),
        }
