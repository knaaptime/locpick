"""Unified choice model for location choice estimation.

This module provides the :class:`ChoiceModel` class, a single composable
model that handles all model configurations via optional feature flags:

- ``nests`` → Nested logit
- ``random_params`` → Mixed logit
- ``graph`` → Spatially correlated logit (SCL)
- combinations → Nested SCL, Mixed SCL (MSCL), Mixed Nested, Mixed Nested SCL

The class inherits from :class:`BaseChoiceModel` and :class:`SpatialMixin`,
dispatching to the appropriate JAX builder based on which features are active.
All shared methods (simulate, marginal effects, elasticities, covariance)
live here once, eliminating the duplication across MNL/NestedMNL/MixedMNL/
MixedNestedMNL.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .._jax.objective import Objective
from .._solvers import Solver, SolverResult
from ..data.arrays import ChoiceArrays
from ..data.problem import EstimationProblem
from ..results.fit_result import FitResult
from ._spatial import (
    EdgeStructure,
    _resolve_spatial_graph,
    naturalize_rho,
)
from .base import (
    BaseChoiceModel,
    SpatialMixin,
    _compute_fit_statistics,
    _compute_null_ll,
    _safe_inv,
    _sandwich_inv,
)
from .mixed import ParamDistribution, _resolve_draws
from .nested import NestingTree, naturalize_nest_params


class ChoiceModel(BaseChoiceModel, SpatialMixin):
    r"""Unified discrete choice model for location choice estimation.

    A single composable class that handles all model configurations:

    - **MNL** (default): ``ChoiceModel(ct, formula="cost + time - 1")``
    - **Nested logit**: ``ChoiceModel(ct, formula="...", nests=tree)``
    - **Mixed logit**: ``ChoiceModel(ct, formula="...", random_params={"time": ParamDistribution("normal", "time")})``
    - **SCL** (spatial): ``ChoiceModel(ct, formula="...", graph=g)``
    - **Nested SCL**: ``ChoiceModel(ct, formula="...", nests=tree, graph=g)``
    - **MSCL** (mixed + spatial): ``ChoiceModel(ct, formula="...", random_params=..., graph=g)``
    - **Mixed Nested**: ``ChoiceModel(ct, formula="...", nests=tree, random_params=...)``
    - **Mixed Nested SCL**: ``ChoiceModel(ct, formula="...", nests=tree, random_params=..., graph=g)``

    The model type is determined by which optional features are present.

    Parameters
    ----------
    data : ChoiceTable or EstimationProblem
        The choice data to estimate on.
    formula : str, optional
        A formulaic formula string (e.g., ``"cost + time - 1"``).
        Mutually exclusive with ``spec``.
    spec : ModelSpec, optional
        A ModelSpec object defining formula/scoped-term model structure.
        Mutually exclusive with ``formula``.
    nests : NestingTree, optional
        The nesting structure for nested logit models.
    random_params : dict, optional
        Mapping of parameter names to :class:`ParamDistribution` objects
        for mixed logit models.
    graph : libpysal.Graph, scipy.sparse, or np.ndarray, optional
        Spatial adjacency graph for SCL models.
    n_draws : int, optional
        Number of draws for simulated maximum likelihood (mixed logit).
        Default 100.
    draw_type : str, optional
        Type of draws: ``"qmc"`` (default), ``"halton"``, or ``"random"``.
    seed : int
        Random seed for draw generation. Default 42.
    weights : str or array-like, optional
        Observation weights.
    availability : str or array-like, optional
        Alternative availability.
    solver : str or Solver, optional
        Solver name or instance. Default ``"lbfgs"``.
    solver_options : dict, optional
        Additional options passed to the solver constructor.
    backend : str, optional
        Computation backend hint.
    estimator : str, optional
        SAR estimation method (only relevant when ``lag=True``).
        ``"auto"`` (default) selects ``"pml"`` (dense solve) for
        n_alts ≤ 2000 and ``"pml_cg"`` (conjugate gradient) for larger
        alternative sets.  ``"linearized_gmm"`` uses the two-step GMM
        estimator (Carrión-Flores et al. 2018) for very large J.

    Examples
    --------
    >>> from locpick import ChoiceTable, ChoiceModel
    >>> ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=10)
    >>> model = ChoiceModel(ct, formula="cost + time - 1")
    >>> result = model.fit()
    >>> print(result.summary())
    """

    def __init__(
        self,
        data,
        formula: Optional[str] = None,
        spec=None,
        problem: Optional[EstimationProblem] = None,
        nests: Optional[NestingTree] = None,
        random_params: Optional[dict[str, ParamDistribution]] = None,
        graph=None,
        lag: bool = False,
        n_draws: Optional[int] = None,
        draw_type: Optional[str] = None,
        seed: int = 42,
        weights: Optional[Union[str, np.ndarray]] = None,
        availability: Optional[Union[str, np.ndarray]] = None,
        solver: Union[str, Solver] = "lbfgs",
        solver_options: Optional[dict] = None,
        backend: Optional[str] = None,
        estimator: str = "auto",
        warmstart: bool = True,
    ):
        # Handle the legacy `problem` parameter by wrapping it as EstimationProblem
        if problem is not None:
            data = problem

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

        # Feature flags
        self._nests = nests
        self._random_params = random_params
        self._graph_input = graph
        self._lag = lag  # True = SAR spatial lag, False = SCL (default)
        self._estimator = estimator  # SAR estimator: auto, pml, pml_cg, linearized_gmm
        self._warmstart = warmstart  # Use GMM estimates as PML starting values

        # Mixed logit settings
        self._n_draws = n_draws if n_draws is not None else 100
        self._draw_type = draw_type if draw_type is not None else "sobol"
        self._seed = seed
        self._draws: Optional[np.ndarray] = None

        # Nest matrix (built in _pre_fit)
        self._nest_matrix: Optional[np.ndarray] = None

        # Spatial state (None when graph is not provided)
        self._omega = None
        self._allocation = None
        self._edge_list = None
        self._n_alts_graph = None
        self._edge_struct = None
        self._edge_structs = None
        self._edge_data_list = None

        # SAR spatial state (None when lag=True is not used)
        self._W_sparse = None  # CSR sparse for SAR kernels
        # Random parameter state (built in _pre_fit)
        self._random_col_indices: Optional[list[int]] = None
        self._random_distributions: Optional[list[str]] = None
        self._random_param_names: Optional[list[str]] = None
        self._k_fixed: Optional[int] = None
        self._k_random: Optional[int] = None
        self._fixed_names: Optional[list[str]] = None
        self._full_param_names: Optional[list[str]] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _is_spatial(self) -> bool:
        return self._graph_input is not None

    @property
    def _is_spatial_lag(self) -> bool:
        """True when SAR (lag=True) spatial model is active."""
        return self._is_spatial and self._lag

    @property
    def _is_spatial_scl(self) -> bool:
        """True when SCL (lag=False) spatial model is active."""
        return self._is_spatial and not self._lag

    @property
    def _is_nested(self) -> bool:
        return self._nests is not None

    @property
    def _is_mixed(self) -> bool:
        return self._random_params is not None and len(self._random_params) > 0

    @property
    def model_type(self) -> str:
        """Human-readable model type string."""
        parts = []
        if self._is_mixed:
            parts.append("Mixed")
        if self._is_nested:
            parts.append("Nested")
        if self._is_spatial_lag:
            parts.append("Spatial Autoregressive")
        elif self._is_spatial_scl:
            parts.append("Spatially Correlated")
        if not parts:
            return "Multinomial Logit"
        if len(parts) == 1 and parts[0] == "Spatially Correlated":
            return "Spatially Correlated Logit"
        if len(parts) == 1 and parts[0] == "Spatial Autoregressive":
            return "Spatial Autoregressive Logit"
        if len(parts) == 1 and parts[0] == "Nested":
            return "Nested Logit"
        if len(parts) == 1 and parts[0] == "Mixed":
            return "Mixed Logit"
        return " ".join(parts) + " Logit"

    # ------------------------------------------------------------------
    # Pre-fit: build model-specific data structures
    # ------------------------------------------------------------------

    def fit(self, **kwargs) -> FitResult:
        """Estimate the model and return results.

        Dispatches to the PML estimator (JAX autodiff) or the
        linearized GMM estimator based on the ``estimator`` setting.
        """
        if self._is_spatial_lag and self._estimator == "linearized_gmm":
            return self._fit_linearized_gmm()
        return super().fit(**kwargs)

    def _fit_linearized_gmm(self) -> FitResult:
        """Two-step linearized GMM estimation (Carrión-Flores et al. 2018)."""
        arrays = self._get_arrays()
        self._arrays = arrays
        self._pre_fit(arrays)

        from .._kernels.sar_mnl_numpy import fit_linearized_gmm

        result_dict = fit_linearized_gmm(arrays, self._W_sparse)

        beta = result_dict["beta"]
        rho = result_dict["rho"]
        se = result_dict["se"]
        ll = result_dict["log_likelihood"]

        utility_param_names = list(arrays.param_names)
        k = len(utility_param_names)
        display_values = np.concatenate([beta, [rho]])
        display_names = utility_param_names + ["rho"]
        model_type = "Spatial Autoregressive Logit (Linearized GMM)"
        n_params = len(display_values)

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

    def _pre_fit(self, arrays: ChoiceArrays) -> None:
        """Build nest matrix, random parameter structure, and spatial graph."""
        # Build nest matrix if nests are provided
        if self._is_nested:
            alt_ids = list(range(arrays.n_alts))
            self._nest_matrix = self._nests.build_nest_matrix(alt_ids)

        # Build random parameter structure if random_params is provided
        if self._is_mixed:
            self._prepare_random_params(arrays)

        # Resolve spatial graph if graph is provided
        if self._is_spatial:
            if self._is_spatial_lag:
                # SAR: resolve W via _spatial_weights resolver
                from ._spatial_weights import resolve_spatial_weights

                self._W_sparse = resolve_spatial_weights(
                    self._graph_input, arrays.n_alts, row_standardize=True
                )[1]  # get the CSR sparse
            else:
                # SCL: resolve via edge structure (existing behavior)
                self._resolve_spatial_graph()
                self._validate_graph_size(arrays)

                # Build per-nest edge structures for nested spatial models
                if self._is_nested:
                    self._build_per_nest_edges(arrays)

    def _prepare_random_params(self, arrays: ChoiceArrays) -> None:
        """Identify random parameter columns and generate draws."""
        param_names = list(arrays.param_names)
        random_param_names: list[str] = []
        random_distributions: list[str] = []
        random_col_indices: list[int] = []

        for name, dist in self._random_params.items():
            if name not in param_names:
                raise ValueError(
                    f"Random parameter '{name}' not found in design matrix. "
                    f"Available parameters: {param_names}"
                )
            random_param_names.append(name)
            random_distributions.append(dist.distribution)
            random_col_indices.append(param_names.index(name))

        k_fixed = len(param_names) - len(random_param_names)
        k_random = len(random_param_names)

        # Generate draws
        draws = _resolve_draws(
            self._draw_type, arrays.n_obs, self._n_draws, k_random, seed=self._seed
        )

        fixed_names = [n for i, n in enumerate(param_names) if i not in random_col_indices]
        full_param_names = (
            fixed_names
            + [f"mean_{n}" for n in random_param_names]
            + [f"sd_{n}" for n in random_param_names]
        )

        self._random_param_names = random_param_names
        self._random_distributions = random_distributions
        self._random_col_indices = random_col_indices
        self._k_fixed = k_fixed
        self._k_random = k_random
        self._fixed_names = fixed_names
        self._full_param_names = full_param_names
        self._draws = draws

    def _build_per_nest_edges(self, arrays: ChoiceArrays) -> None:
        """Build per-nest EdgeStructure / EdgeDataJAX from the global graph."""
        from .._jax.data import EdgeDataJAX

        n_nests = self._nests.n_nests
        self._edge_structs = []
        self._edge_data_list = []
        for m in range(n_nests):
            nest_alts = np.where(self._nest_matrix[:, m] > 0)[0]
            n_nest_alts = len(nest_alts)
            if n_nest_alts == 0:
                self._edge_structs.append(None)
                self._edge_data_list.append(None)
                continue
            nest_adj = self._omega[np.ix_(nest_alts, nest_alts)]
            _, nest_alloc, nest_edges, _ = _resolve_spatial_graph(nest_adj)
            edge_struct = EdgeStructure(nest_edges, n_nest_alts, nest_alloc)
            self._edge_structs.append(edge_struct)
            self._edge_data_list.append(EdgeDataJAX.from_edge_structure(edge_struct))

    # ------------------------------------------------------------------
    # Solver inputs
    # ------------------------------------------------------------------

    def _get_solver_inputs(self, arrays: ChoiceArrays):
        """Get initial values, param names, bounds, and fixed mask.

        Parameter layout depends on active features:

        - MNL: ``[beta]``
        - SCL: ``[beta, alpha_rho]``
        - Nested: ``[beta, alpha_nest]``
        - Nested SCL: ``[beta, alpha_rho_1..M, alpha_lambda_1..M]``
        - Mixed: ``[beta_fixed, mean_*, sd_*]``
        - MSCL: ``[beta_fixed, alpha_rho, mean_*, sd_*]``
        - Mixed Nested: ``[beta_fixed, alpha_nest, mean_*, sd_*]``
        - Mixed Nested SCL: ``[beta_fixed, alpha_rho_1..M, alpha_lambda_1..M, mean_*, sd_*]``
        """
        k = arrays.design_matrix.shape[1]
        param_names_all = list(arrays.param_names)

        # --- Pure MNL ---
        if not self._is_nested and not self._is_mixed:
            # Use problem's initial values / fixed_mask when available
            if self._problem is not None:
                x0_base = self._problem.initial_values
                bounds = self._problem.bounds
                fixed_mask = self._problem.fixed_mask
            else:
                x0_base = np.zeros(k)
                bounds = None
                fixed_mask = None
            names_base = param_names_all
            if self._is_spatial:
                # SCL or SAR: [beta, alpha_rho]
                if self._is_spatial_lag and self._warmstart and self._W_sparse is not None:
                    # GMM warm-start: use linearized GMM estimates as starting values
                    try:
                        from .._kernels.sar_mnl_numpy import fit_linearized_gmm

                        gmm = fit_linearized_gmm(arrays, self._W_sparse)
                        beta_gmm = gmm["beta"]
                        rho_gmm = float(np.clip(gmm["rho"], -0.99, 0.99))
                        alpha_rho_gmm = float(np.arctanh(rho_gmm))
                        x0 = np.concatenate([beta_gmm, [alpha_rho_gmm]])
                    except Exception:
                        # Fallback to zeros if GMM fails
                        x0 = np.concatenate([x0_base, np.zeros(1)])
                else:
                    x0 = np.concatenate([x0_base, np.zeros(1)])
                names = list(names_base) + ["alpha_rho"]
            else:
                x0 = x0_base
                names = names_base
            return x0, names, bounds, fixed_mask

        # --- Nested (no random) ---
        if self._is_nested and not self._is_mixed:
            n_nests = self._nests.n_nests
            if self._is_spatial_lag:
                # SAR + Nested: [beta, alpha_rho, alpha_lambda_1..M]
                x0 = np.concatenate([np.zeros(k), np.zeros(1), self._nests.initial_alphas()])
                names = (
                    param_names_all
                    + ["alpha_rho"]
                    + [f"alpha_lambda_{name}" for name in self._nests.nest_names]
                )
            elif self._is_spatial_scl:
                # Nested SCL: [beta, alpha_rho_1..M, alpha_lambda_1..M]
                x0 = np.concatenate(
                    [
                        np.zeros(k),
                        np.zeros(n_nests),
                        self._nests.initial_alphas(),
                    ]
                )
                names = (
                    param_names_all
                    + [f"alpha_rho_{name}" for name in self._nests.nest_names]
                    + [f"alpha_lambda_{name}" for name in self._nests.nest_names]
                )
            else:
                # Nested: [beta, alpha_nest]
                x0 = np.concatenate([np.zeros(k), self._nests.initial_alphas()])
                names = param_names_all + [f"nest_{name}" for name in self._nests.nest_names]
            return x0, names, None, None

        # --- Mixed (no nests) ---
        if self._is_mixed and not self._is_nested:
            k_fixed = self._k_fixed
            k_random = self._k_random
            if self._is_spatial:
                # SAR + Mixed or MSCL: [beta_fixed, alpha_rho, mean_*, sd_*]
                x0 = np.concatenate(
                    [
                        np.zeros(k_fixed),
                        np.zeros(1),
                        np.zeros(k_random),
                        np.full(k_random, 0.1),
                    ]
                )
                names = (
                    list(self._fixed_names)
                    + ["rho"]
                    + [f"mean_{n}" for n in self._random_param_names]
                    + [f"sd_{n}" for n in self._random_param_names]
                )
            else:
                # Mixed: [beta_fixed, mean_*, sd_*]
                x0 = np.concatenate(
                    [
                        np.zeros(k_fixed),
                        np.zeros(k_random),
                        np.full(k_random, 0.1),
                    ]
                )
                names = list(self._full_param_names)
            return x0, names, None, None

        # --- Mixed Nested ---
        if self._is_nested and self._is_mixed:
            k_fixed = self._k_fixed
            k_random = self._k_random
            n_nests = self._nests.n_nests
            fixed_param_names = [
                name for name in param_names_all if name not in self._random_params
            ]
            if self._is_spatial_lag:
                # SAR + Mixed + Nested: [beta_fixed, alpha_rho, alpha_lambda_1..M, mean_*, sd_*]
                x0 = np.concatenate(
                    [
                        np.zeros(k_fixed),
                        np.zeros(1),
                        self._nests.initial_alphas(),
                        np.zeros(k_random),
                        np.full(k_random, 0.1),
                    ]
                )
                names = (
                    fixed_param_names
                    + ["alpha_rho"]
                    + [f"alpha_lambda_{name}" for name in self._nests.nest_names]
                    + [f"mean_{name}" for name in self._random_param_names]
                    + [f"sd_{name}" for name in self._random_param_names]
                )
            elif self._is_spatial_scl:
                # Mixed Nested SCL: [beta_fixed, alpha_rho_1..M, alpha_lambda_1..M, mean_*, sd_*]
                x0 = np.concatenate(
                    [
                        np.zeros(k_fixed),
                        np.zeros(n_nests),
                        self._nests.initial_alphas(),
                        np.zeros(k_random),
                        np.full(k_random, 0.1),
                    ]
                )
                names = (
                    fixed_param_names
                    + [f"alpha_rho_{name}" for name in self._nests.nest_names]
                    + [f"alpha_lambda_{name}" for name in self._nests.nest_names]
                    + [f"mean_{name}" for name in self._random_param_names]
                    + [f"sd_{name}" for name in self._random_param_names]
                )
            else:
                # Mixed Nested: [beta_fixed, alpha_nest, mean_*, sd_*]
                x0 = np.concatenate(
                    [
                        np.zeros(k_fixed),
                        self._nests.initial_alphas(),
                        np.zeros(k_random),
                        np.full(k_random, 0.1),
                    ]
                )
                names = (
                    fixed_param_names
                    + [f"lambda_{name}" for name in self._nests.nest_names]
                    + [f"mean_{name}" for name in self._random_param_names]
                    + [f"sd_{name}" for name in self._random_param_names]
                )
            return x0, names, None, None

        # Fallback (should not reach here)
        return np.zeros(k), param_names_all, None, None

    # ------------------------------------------------------------------
    # Objective construction
    # ------------------------------------------------------------------

    def _build_objective(self, arrays: ChoiceArrays) -> Objective:
        """Build optimization objective based on active features."""
        # Pure MNL / SCL / SAR
        if not self._is_nested and not self._is_mixed:
            if self._is_spatial_lag:
                from .._jax.sar_kernels import build_sar_mnl_objective

                # Auto-select estimator
                if self._estimator == "auto":
                    if arrays.n_alts <= 2000:
                        self._estimator = "pml"
                    else:
                        self._estimator = "pml_cg"
                use_cg = self._estimator == "pml_cg"
                return build_sar_mnl_objective(arrays, self._W_sparse, use_cg=use_cg)
            elif self._is_spatial_scl:
                from .._jax.builders import build_scl_objective

                return build_scl_objective(
                    arrays, self._edge_struct, self._allocation, self._edge_list
                )
            from .._jax.builders import build_mnl_objective

            return build_mnl_objective(arrays)

        # Nested (no random)
        if self._is_nested and not self._is_mixed:
            if self._is_spatial_lag:
                from .._jax.sar_kernels import build_sar_nested_objective

                use_cg = self._estimator == "pml_cg" or (
                    self._estimator == "auto" and arrays.n_alts > 2000
                )
                return build_sar_nested_objective(
                    arrays, self._W_sparse, self._nest_matrix, use_cg=use_cg
                )
            elif self._is_spatial_scl:
                from .._jax.builders import build_nested_scl_objective

                return build_nested_scl_objective(arrays, self._nest_matrix, self._edge_data_list)
            from .._jax.builders import build_nested_objective

            return build_nested_objective(arrays, self._nest_matrix)

        # Mixed (no nests)
        if self._is_mixed and not self._is_nested:
            if self._is_spatial_lag:
                from .._jax.sar_kernels import build_sar_mixed_objective

                use_cg = self._estimator == "pml_cg" or (
                    self._estimator == "auto" and arrays.n_alts > 2000
                )
                return build_sar_mixed_objective(
                    arrays,
                    self._W_sparse,
                    self._random_col_indices,
                    self._random_distributions,
                    self._draws,
                    use_cg=use_cg,
                )
            elif self._is_spatial_scl:
                from .._jax.builders import build_mscl_objective

                return build_mscl_objective(
                    arrays,
                    self._edge_struct,
                    self._allocation,
                    self._edge_list,
                    self._random_col_indices,
                    self._random_distributions,
                    self._draws,
                )
            from .._jax.builders import build_mixed_logit_objective

            return build_mixed_logit_objective(
                arrays,
                random_col_indices=self._random_col_indices,
                random_distributions=self._random_distributions,
                draws=self._draws,
            )

        # Mixed Nested
        if self._is_nested and self._is_mixed:
            if self._is_spatial_lag:
                from .._jax.sar_kernels import build_sar_mixed_nested_objective

                use_cg = self._estimator == "pml_cg" or (
                    self._estimator == "auto" and arrays.n_alts > 2000
                )
                return build_sar_mixed_nested_objective(
                    arrays,
                    self._W_sparse,
                    self._nest_matrix,
                    self._random_col_indices,
                    self._random_distributions,
                    self._draws,
                    use_cg=use_cg,
                )
            elif self._is_spatial_scl:
                from .._jax.builders import build_mnscl_objective

                return build_mnscl_objective(
                    arrays,
                    self._nest_matrix,
                    self._edge_data_list,
                    self._random_col_indices,
                    self._random_distributions,
                    self._draws,
                )
            from .._jax.builders import build_mixed_nested_objective

            return build_mixed_nested_objective(
                arrays,
                self._nest_matrix,
                self._random_col_indices,
                self._random_distributions,
                self._draws,
            )

        raise RuntimeError("Unknown model configuration")

    # ------------------------------------------------------------------
    # Fit result construction
    # ------------------------------------------------------------------

    def _build_fit_result(
        self,
        solver_result: SolverResult,
        arrays: ChoiceArrays,
    ) -> FitResult:
        """Build a FitResult from solver output."""
        all_params = solver_result.coefficients
        k = arrays.design_matrix.shape[1]
        param_names_all = list(arrays.param_names)

        # --- Pure MNL / SCL / SAR ---
        if not self._is_nested and not self._is_mixed:
            if self._is_spatial_lag:
                # SAR: Layout [beta_1..k, alpha_rho], rho = tanh(alpha_rho)
                beta = all_params[:k]
                alpha_rho = all_params[k]
                rho = np.tanh(alpha_rho)
                display_values = np.concatenate([beta, [rho]])
                display_names = param_names_all + ["rho"]
            elif self._is_spatial_scl:
                # SCL: Layout [beta_1..k, alpha_rho], rho = sigmoid(alpha_rho)
                beta = all_params[:k]
                alpha_rho = all_params[k]
                rho = naturalize_rho(alpha_rho)
                display_values = np.concatenate([beta, [rho]])
                display_names = param_names_all + ["rho"]
            else:
                display_values = all_params
                display_names = param_names_all

            std_errors = self._compute_se(all_params, arrays, display_values, display_names)
            return self._make_fit_result(
                solver_result, arrays, display_values, display_names, std_errors
            )

        # --- Nested (no random) ---
        if self._is_nested and not self._is_mixed:
            n_nests = self._nests.n_nests
            if self._is_spatial_lag:
                # SAR + Nested: Layout [beta, alpha_rho, alpha_lambda_1..M]
                beta = all_params[:k]
                alpha_rho = all_params[k]
                alpha_lambda = all_params[k + 1 : k + 1 + n_nests]
                rho = np.tanh(alpha_rho)
                lambdas = naturalize_nest_params(alpha_lambda)
                display_values = np.concatenate([beta, [rho], lambdas])
                display_names = (
                    param_names_all
                    + ["rho"]
                    + [f"lambda_{name}" for name in self._nests.nest_names]
                )
                std_errors = self._compute_se_sar_nested(
                    all_params, arrays, k, n_nests, rho, lambdas
                )
            elif self._is_spatial_scl:
                # SCL + Nested: Layout [beta, alpha_rho_1..M, alpha_lambda_1..M]
                beta = all_params[:k]
                alpha_rho = all_params[k : k + n_nests]
                alpha_lambda = all_params[k + n_nests : k + 2 * n_nests]
                rhos = naturalize_rho(alpha_rho)
                lambdas = naturalize_nest_params(alpha_lambda)
                display_values = np.concatenate([beta, rhos, lambdas])
                display_names = (
                    param_names_all
                    + [f"rho_{name}" for name in self._nests.nest_names]
                    + [f"lambda_{name}" for name in self._nests.nest_names]
                )
                std_errors = self._compute_se_nested_scl(
                    all_params, arrays, k, n_nests, rhos, lambdas
                )
            else:
                # Layout: [beta, alpha_nest]
                beta = all_params[:k]
                alpha = all_params[k:]
                lambdas = naturalize_nest_params(alpha)
                display_values = np.concatenate([beta, lambdas])
                display_names = param_names_all + [
                    f"lambda_{name}" for name in self._nests.nest_names
                ]
                std_errors = self._compute_se_nested(all_params, arrays, k, lambdas)

            return self._make_fit_result(
                solver_result, arrays, display_values, display_names, std_errors
            )

        # --- Mixed (no nests) ---
        if self._is_mixed and not self._is_nested:
            k_fixed = self._k_fixed
            k_random = self._k_random
            if self._is_spatial_lag:
                # SAR + Mixed: Layout [beta_fixed, alpha_rho, mean_*, sd_*]
                beta_fixed = all_params[:k_fixed]
                alpha_rho = all_params[k_fixed]
                rho = np.tanh(alpha_rho)
                beta_random_means = all_params[k_fixed + 1 : k_fixed + 1 + k_random]
                beta_random_spreads = all_params[k_fixed + 1 + k_random :]
                display_values = np.concatenate(
                    [beta_fixed, [rho], beta_random_means, beta_random_spreads]
                )
                display_names = (
                    list(self._fixed_names)
                    + ["rho"]
                    + [f"mean_{n}" for n in self._random_param_names]
                    + [f"sd_{n}" for n in self._random_param_names]
                )
                std_errors = self._compute_se_sar_mixed(all_params, arrays, k_fixed, rho)
            elif self._is_spatial_scl:
                # SCL + Mixed (MSCL): Layout [beta_fixed, alpha_rho, mean_*, sd_*]
                beta_fixed = all_params[:k_fixed]
                alpha_rho = all_params[k_fixed]
                rho = naturalize_rho(alpha_rho)
                beta_random_means = all_params[k_fixed + 1 : k_fixed + 1 + k_random]
                beta_random_spreads = all_params[k_fixed + 1 + k_random :]
                display_values = np.concatenate(
                    [beta_fixed, [rho], beta_random_means, beta_random_spreads]
                )
                display_names = (
                    list(self._fixed_names)
                    + ["rho"]
                    + [f"mean_{n}" for n in self._random_param_names]
                    + [f"sd_{n}" for n in self._random_param_names]
                )
                std_errors = self._compute_se_mscl(all_params, arrays, k_fixed, rho)
            else:
                # Layout: [beta_fixed, mean_*, sd_*]
                display_values = all_params
                display_names = list(self._full_param_names)
                std_errors = self._compute_se_simple(all_params, arrays)

            return self._make_fit_result(
                solver_result, arrays, display_values, display_names, std_errors
            )

        # --- Mixed Nested ---
        if self._is_nested and self._is_mixed:
            k_fixed = self._k_fixed
            k_random = self._k_random
            n_nests = self._nests.n_nests
            fixed_param_names = [
                name for name in param_names_all if name not in self._random_params
            ]

            if self._is_spatial_lag:
                # SAR + Mixed + Nested: Layout [beta_fixed, alpha_rho, alpha_lambda_1..M, mean_*, sd_*]
                beta_fixed = all_params[:k_fixed]
                alpha_rho = all_params[k_fixed]
                alpha_lambda = all_params[k_fixed + 1 : k_fixed + 1 + n_nests]
                beta_random_means = all_params[
                    k_fixed + 1 + n_nests : k_fixed + 1 + n_nests + k_random
                ]
                beta_random_spreads = all_params[k_fixed + 1 + n_nests + k_random :]
                rho = np.tanh(alpha_rho)
                lambdas = naturalize_nest_params(alpha_lambda)
                display_values = np.concatenate(
                    [beta_fixed, [rho], lambdas, beta_random_means, np.abs(beta_random_spreads)]
                )
                display_names = (
                    fixed_param_names
                    + ["rho"]
                    + [f"lambda_{name}" for name in self._nests.nest_names]
                    + [f"mean_{name}" for name in self._random_param_names]
                    + [f"sd_{name}" for name in self._random_param_names]
                )
                std_errors = self._compute_se_sar_mixed_nested(
                    all_params, arrays, k_fixed, n_nests, rho, lambdas
                )
            elif self._is_spatial_scl:
                # SCL + Mixed + Nested: Layout [beta_fixed, alpha_rho_1..M, alpha_lambda_1..M, mean_*, sd_*]
                beta_fixed = all_params[:k_fixed]
                alpha_rho = all_params[k_fixed : k_fixed + n_nests]
                alpha_lambda = all_params[k_fixed + n_nests : k_fixed + 2 * n_nests]
                beta_random_means = all_params[
                    k_fixed + 2 * n_nests : k_fixed + 2 * n_nests + k_random
                ]
                beta_random_spreads = all_params[k_fixed + 2 * n_nests + k_random :]
                rhos = naturalize_rho(alpha_rho)
                lambdas = naturalize_nest_params(alpha_lambda)
                display_values = np.concatenate(
                    [beta_fixed, rhos, lambdas, beta_random_means, np.abs(beta_random_spreads)]
                )
                display_names = (
                    fixed_param_names
                    + [f"rho_{name}" for name in self._nests.nest_names]
                    + [f"lambda_{name}" for name in self._nests.nest_names]
                    + [f"mean_{name}" for name in self._random_param_names]
                    + [f"sd_{name}" for name in self._random_param_names]
                )
                std_errors = self._compute_se_mixed_nested_scl(
                    all_params, arrays, k_fixed, n_nests, rhos, lambdas
                )
            else:
                # Layout: [beta_fixed, alpha_nest, mean_*, sd_*]
                beta_fixed = all_params[:k_fixed]
                alpha_nest = all_params[k_fixed : k_fixed + n_nests]
                beta_random_means = all_params[k_fixed + n_nests : k_fixed + n_nests + k_random]
                beta_random_spreads = all_params[k_fixed + n_nests + k_random :]
                lambdas = naturalize_nest_params(alpha_nest)
                display_values = np.concatenate(
                    [beta_fixed, lambdas, beta_random_means, np.abs(beta_random_spreads)]
                )
                display_names = (
                    fixed_param_names
                    + [f"lambda_{name}" for name in self._nests.nest_names]
                    + [f"mean_{name}" for name in self._random_param_names]
                    + [f"sd_{name}" for name in self._random_param_names]
                )
                std_errors = self._compute_se_mixed_nested(
                    all_params, arrays, k_fixed, n_nests, lambdas
                )

            return self._make_fit_result(
                solver_result, arrays, display_values, display_names, std_errors
            )

        raise RuntimeError("Unknown model configuration")

    # ------------------------------------------------------------------
    # Standard error computation helpers
    # ------------------------------------------------------------------

    def _compute_se_simple(self, all_params, arrays):
        """Compute SEs for models without parameter transforms."""
        std_errors = np.full(len(all_params), np.nan)
        try:
            hess = self._compute_hessian(all_params)
            std_errors = self._compute_std_errors_from_hessian(hess)
        except Exception:
            if self._result is not None and self._result.solver_result:
                hess_inv = self._get_hessian_inverse()
                if hess_inv is not None:
                    std_errors = np.sqrt(np.maximum(np.diag(hess_inv), 0))
                    std_errors[std_errors == 0] = np.nan
        return std_errors

    def _compute_se(self, all_params, arrays, display_values, display_names):
        """Compute SEs for MNL/SCL models."""
        k = arrays.design_matrix.shape[1]
        n_params = len(display_values)
        std_errors = np.full(n_params, np.nan)
        try:
            hess = self._compute_hessian(all_params)
            se_unconstrained = self._compute_std_errors_from_hessian(hess)
            if self._is_spatial_lag:
                # SAR delta method: SE(rho) = (1 - rho^2) * SE(alpha_rho)
                rho = display_values[k]
                se_rho = (1.0 - rho**2) * se_unconstrained[k]
                std_errors = np.concatenate([se_unconstrained[:k], [se_rho]])
            elif self._is_spatial_scl:
                # SCL delta method: SE(rho) = rho*(1-rho)*SE(alpha_rho)
                rho = display_values[k]
                se_rho = rho * (1.0 - rho) * se_unconstrained[k]
                std_errors = np.concatenate([se_unconstrained[:k], [se_rho]])
            else:
                std_errors = se_unconstrained
        except Exception:
            hess_inv = self._get_hessian_inverse()
            if hess_inv is not None:
                se = np.sqrt(np.maximum(np.diag(hess_inv), 0))
                se[se == 0] = np.nan
                if self._is_spatial_lag:
                    rho = display_values[k]
                    se_rho = (1.0 - rho**2) * se[k]
                    std_errors = np.concatenate([se[:k], [se_rho]])
                elif self._is_spatial_scl:
                    rho = display_values[k]
                    se_rho = rho * (1.0 - rho) * se[k]
                    std_errors = np.concatenate([se[:k], [se_rho]])
                else:
                    std_errors = se
        return std_errors

    def _compute_se_nested(self, all_params, arrays, k, lambdas):
        """Compute SEs for nested logit (non-spatial)."""
        n_nests = len(lambdas)
        std_errors = np.full(k + n_nests, np.nan)
        try:
            hess = self._compute_hessian(all_params)
            se_alpha = self._compute_std_errors_from_hessian(hess)
            se_lambda = lambdas * (1.0 - lambdas) * se_alpha[k:]
            std_errors = np.concatenate([se_alpha[:k], se_lambda])
        except Exception:
            hess_inv = self._get_hessian_inverse()
            if hess_inv is not None:
                se_alpha = np.sqrt(np.maximum(np.diag(hess_inv), 0))
                se_alpha[se_alpha == 0] = np.nan
                se_lambda = lambdas * (1.0 - lambdas) * se_alpha[k:]
                std_errors = np.concatenate([se_alpha[:k], se_lambda])
        return std_errors

    def _compute_se_nested_scl(self, all_params, arrays, k, n_nests, rhos, lambdas):
        """Compute SEs for nested SCL."""
        std_errors = np.full(k + 2 * n_nests, np.nan)
        try:
            hess = self._compute_hessian(all_params)
            se_alpha = self._compute_std_errors_from_hessian(hess)
            se_rho = rhos * (1.0 - rhos) * se_alpha[k : k + n_nests]
            se_lambda = lambdas * (1.0 - lambdas) * se_alpha[k + n_nests : k + 2 * n_nests]
            std_errors = np.concatenate([se_alpha[:k], se_rho, se_lambda])
        except Exception:
            hess_inv = self._get_hessian_inverse()
            if hess_inv is not None:
                se_alpha = np.sqrt(np.maximum(np.diag(hess_inv), 0))
                se_alpha[se_alpha == 0] = np.nan
                se_rho = rhos * (1.0 - rhos) * se_alpha[k : k + n_nests]
                se_lambda = lambdas * (1.0 - lambdas) * se_alpha[k + n_nests : k + 2 * n_nests]
                std_errors = np.concatenate([se_alpha[:k], se_rho, se_lambda])
        return std_errors

    def _compute_se_mscl(self, all_params, arrays, k_fixed, rho):
        """Compute SEs for MSCL."""
        n_params = len(all_params)
        std_errors = np.full(n_params, np.nan)
        try:
            hess = self._compute_hessian(all_params)
            se_alpha = self._compute_std_errors_from_hessian(hess)
            se_rho = float(rho * (1.0 - rho) * se_alpha[k_fixed])
            std_errors = np.concatenate([se_alpha[:k_fixed], [se_rho], se_alpha[k_fixed + 1 :]])
        except Exception:
            hess_inv = self._get_hessian_inverse()
            if hess_inv is not None:
                se_alpha = np.sqrt(np.maximum(np.diag(hess_inv), 0))
                se_alpha[se_alpha == 0] = np.nan
                se_rho = float(rho * (1.0 - rho) * se_alpha[k_fixed])
                std_errors = np.concatenate(
                    [se_alpha[:k_fixed], [se_rho], se_alpha[k_fixed + 1 :]]
                )
        return std_errors

    def _compute_se_mixed_nested(self, all_params, arrays, k_fixed, n_nests, lambdas):
        """Compute SEs for mixed nested logit (non-spatial)."""
        n_params = len(all_params)
        std_errors = np.full(n_params, np.nan)
        try:
            hess = self._compute_hessian(all_params)
            se_raw = self._compute_std_errors_from_hessian(hess)
            se_lambda = lambdas * (1.0 - lambdas) * se_raw[k_fixed : k_fixed + n_nests]
            std_errors = np.concatenate([se_raw[:k_fixed], se_lambda, se_raw[k_fixed + n_nests :]])
        except Exception:
            hess_inv = self._get_hessian_inverse()
            if hess_inv is not None:
                se_raw = np.sqrt(np.maximum(np.diag(hess_inv), 0))
                se_raw[se_raw == 0] = np.nan
                se_lambda = lambdas * (1.0 - lambdas) * se_raw[k_fixed : k_fixed + n_nests]
                std_errors = np.concatenate(
                    [se_raw[:k_fixed], se_lambda, se_raw[k_fixed + n_nests :]]
                )
        return std_errors

    def _compute_se_mixed_nested_scl(self, all_params, arrays, k_fixed, n_nests, rhos, lambdas):
        """Compute SEs for mixed nested SCL."""
        n_params = len(all_params)
        std_errors = np.full(n_params, np.nan)
        try:
            hess = self._compute_hessian(all_params)
            se_raw = self._compute_std_errors_from_hessian(hess)
            se_rho = rhos * (1.0 - rhos) * se_raw[k_fixed : k_fixed + n_nests]
            se_lambda = (
                lambdas * (1.0 - lambdas) * se_raw[k_fixed + n_nests : k_fixed + 2 * n_nests]
            )
            std_errors = np.concatenate(
                [
                    se_raw[:k_fixed],
                    se_rho,
                    se_lambda,
                    se_raw[k_fixed + 2 * n_nests :],
                ]
            )
        except Exception:
            hess_inv = self._get_hessian_inverse()
            if hess_inv is not None:
                se_raw = np.sqrt(np.maximum(np.diag(hess_inv), 0))
                se_raw[se_raw == 0] = np.nan
                se_rho = rhos * (1.0 - rhos) * se_raw[k_fixed : k_fixed + n_nests]
                se_lambda = (
                    lambdas * (1.0 - lambdas) * se_raw[k_fixed + n_nests : k_fixed + 2 * n_nests]
                )
                std_errors = np.concatenate(
                    [
                        se_raw[:k_fixed],
                        se_rho,
                        se_lambda,
                        se_raw[k_fixed + 2 * n_nests :],
                    ]
                )
        return std_errors

    def _compute_se_sar_nested(self, all_params, arrays, k, n_nests, rho, lambdas):
        """Compute SEs for SAR + Nested. Layout: [beta, alpha_rho, alpha_lambda_1..M]."""
        n_params = len(all_params)
        std_errors = np.full(n_params, np.nan)
        try:
            hess = self._compute_hessian(all_params)
            se_raw = self._compute_std_errors_from_hessian(hess)
            # SAR: rho = tanh(alpha_rho), SE(rho) = (1 - rho^2) * SE(alpha_rho)
            se_rho = (1.0 - rho**2) * se_raw[k]
            # Lambda: sigmoid, SE(lambda) = lambda*(1-lambda)*SE(alpha_lambda)
            se_lambda = lambdas * (1.0 - lambdas) * se_raw[k + 1 : k + 1 + n_nests]
            std_errors = np.concatenate([se_raw[:k], [se_rho], se_lambda])
        except Exception:
            hess_inv = self._get_hessian_inverse()
            if hess_inv is not None:
                se_raw = np.sqrt(np.maximum(np.diag(hess_inv), 0))
                se_raw[se_raw == 0] = np.nan
                se_rho = (1.0 - rho**2) * se_raw[k]
                se_lambda = lambdas * (1.0 - lambdas) * se_raw[k + 1 : k + 1 + n_nests]
                std_errors = np.concatenate([se_raw[:k], [se_rho], se_lambda])
        return std_errors

    def _compute_se_sar_mixed(self, all_params, arrays, k_fixed, rho):
        """Compute SEs for SAR + Mixed. Layout: [beta_fixed, alpha_rho, mean_*, sd_*]."""
        n_params = len(all_params)
        std_errors = np.full(n_params, np.nan)
        try:
            hess = self._compute_hessian(all_params)
            se_raw = self._compute_std_errors_from_hessian(hess)
            # SAR: rho = tanh(alpha_rho), SE(rho) = (1 - rho^2) * SE(alpha_rho)
            se_rho = (1.0 - rho**2) * se_raw[k_fixed]
            std_errors = np.concatenate([se_raw[:k_fixed], [se_rho], se_raw[k_fixed + 1 :]])
        except Exception:
            hess_inv = self._get_hessian_inverse()
            if hess_inv is not None:
                se_raw = np.sqrt(np.maximum(np.diag(hess_inv), 0))
                se_raw[se_raw == 0] = np.nan
                se_rho = (1.0 - rho**2) * se_raw[k_fixed]
                std_errors = np.concatenate([se_raw[:k_fixed], [se_rho], se_raw[k_fixed + 1 :]])
        return std_errors

    def _compute_se_sar_mixed_nested(self, all_params, arrays, k_fixed, n_nests, rho, lambdas):
        """Compute SEs for SAR + Mixed + Nested.
        Layout: [beta_fixed, alpha_rho, alpha_lambda_1..M, mean_*, sd_*]."""
        n_params = len(all_params)
        std_errors = np.full(n_params, np.nan)
        try:
            hess = self._compute_hessian(all_params)
            se_raw = self._compute_std_errors_from_hessian(hess)
            se_rho = (1.0 - rho**2) * se_raw[k_fixed]
            se_lambda = lambdas * (1.0 - lambdas) * se_raw[k_fixed + 1 : k_fixed + 1 + n_nests]
            std_errors = np.concatenate(
                [se_raw[:k_fixed], [se_rho], se_lambda, se_raw[k_fixed + 1 + n_nests :]]
            )
        except Exception:
            hess_inv = self._get_hessian_inverse()
            if hess_inv is not None:
                se_raw = np.sqrt(np.maximum(np.diag(hess_inv), 0))
                se_raw[se_raw == 0] = np.nan
                se_rho = (1.0 - rho**2) * se_raw[k_fixed]
                se_lambda = lambdas * (1.0 - lambdas) * se_raw[k_fixed + 1 : k_fixed + 1 + n_nests]
                std_errors = np.concatenate(
                    [se_raw[:k_fixed], [se_rho], se_lambda, se_raw[k_fixed + 1 + n_nests :]]
                )
        return std_errors

    def _make_fit_result(
        self,
        solver_result: SolverResult,
        arrays: ChoiceArrays,
        display_values: np.ndarray,
        display_names: list[str],
        std_errors: np.ndarray,
    ) -> FitResult:
        """Build a FitResult using the shared helper."""
        coefficients = pd.Series(display_values, index=display_names, name="coefficient")
        std_err_series = pd.Series(std_errors, index=display_names, name="std_error")
        ll = solver_result.log_likelihood
        ll_null = _compute_null_ll(arrays)
        n_params = len(display_values)

        stats = _compute_fit_statistics(
            ll=ll,
            ll_null=ll_null,
            n_obs=arrays.n_obs,
            n_params=n_params,
            n_alts=arrays.n_alts,
            coefficients=coefficients,
            std_errors=std_err_series,
            model_type=self.model_type,
            solver_name=solver_result.solver_name,
            solver_result_raw=solver_result.raw,
        )

        return FitResult(spec=self._spec, **stats)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def probabilities(self, data=None, beta=None, alpha=None) -> np.ndarray:
        """Compute choice probabilities.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to predict on. If None, uses estimation data.
        beta : np.ndarray or None
            Parameter vector. If None, uses estimated values.
        alpha : np.ndarray or None
            Nest parameters (for nested models). If None, uses estimated values.

        Returns
        -------
        np.ndarray, shape (n_obs, n_alts)
            Choice probabilities.
        """
        if self._arrays is None:
            raise RuntimeError("Model must be estimated before prediction.")

        arrays = self._arrays
        if data is not None:
            from ..data.choicetable import ChoiceTable

            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )

        # Dispatch based on model type
        if self._is_nested or self._is_mixed:
            return self._probabilities_complex(arrays, data, beta, alpha)
        return self._probabilities_mnl(arrays, data, beta)

    def _probabilities_mnl(self, arrays, data, beta) -> np.ndarray:
        """Compute MNL, SCL, or SAR probabilities."""
        from .._kernels.mnl_numpy import mnl_probs_numpy

        if self._is_spatial_lag:
            # SAR: spatial filter + variance normalisation
            k = arrays.design_matrix.shape[1]
            if beta is None:
                coef_vals = np.asarray(self._result.coefficients.values, dtype=np.float64)
                beta_use = coef_vals[:k]
                rho = float(coef_vals[k])
            else:
                beta = np.asarray(beta, dtype=np.float64)
                beta_use = beta[:k]
                rho = (
                    float(beta[k]) if beta.size > k else float(self._result.coefficients.values[k])
                )

            dm = np.asarray(arrays.design_matrix, dtype=np.float64)
            n_obs = arrays.n_obs
            n_alts = arrays.n_alts
            W_dense = np.asarray(self._W_sparse.toarray(), dtype=np.float64)

            V_base = (dm @ beta_use).reshape(n_obs, n_alts)
            from .._sampling.correction import apply_sampling_correction

            V_base = apply_sampling_correction(V_base, arrays)

            A = np.eye(n_alts) - rho * W_dense
            V_filtered = np.linalg.solve(A, V_base.T).T
            D = np.diag(np.linalg.inv(A))
            V_star = V_filtered / D[None, :]

            if arrays.available is not None:
                available = np.asarray(arrays.available, dtype=np.float64).reshape(n_obs, n_alts)
            else:
                available = np.ones((n_obs, n_alts), dtype=np.float64)

            return mnl_probs_numpy(V_star, available, inclusion_probs=None)

        if self._is_spatial_scl:
            from .scl import _scl_log_probs_numpy

            k = arrays.design_matrix.shape[1]
            if beta is None:
                coef_vals = np.asarray(self._result.coefficients.values, dtype=np.float64)
                beta_use = coef_vals[:k]
                rho = float(coef_vals[k])
            else:
                beta = np.asarray(beta, dtype=np.float64)
                beta_use = beta[:k]
                rho = (
                    float(beta[k]) if beta.size > k else float(self._result.coefficients.values[k])
                )

            from .._sampling.correction import get_sampling_correction

            log_probs = _scl_log_probs_numpy(
                beta_use,
                rho,
                np.asarray(arrays.design_matrix, dtype=np.float64),
                self._allocation,
                self._edge_list,
                arrays.n_obs,
                arrays.n_alts,
                available=arrays.available,
                inclusion_probs=get_sampling_correction(arrays),
            )
            return np.exp(log_probs)

        if beta is None:
            beta = np.asarray(self._result.coefficients.values, dtype=np.float64)

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        utilities = (dm @ beta).reshape(n_obs, n_alts)
        from .._sampling.correction import apply_sampling_correction

        utilities = apply_sampling_correction(utilities, arrays)

        if arrays.available is not None:
            available = np.asarray(arrays.available, dtype=np.float64).reshape(n_obs, n_alts)
        else:
            available = np.ones((n_obs, n_alts), dtype=np.float64)

        return mnl_probs_numpy(utilities, available, inclusion_probs=None)

    def _probabilities_complex(self, arrays, data, beta, alpha) -> np.ndarray:
        """Compute probabilities for nested/mixed models (NumPy fallback)."""
        # For nested models, use the NumPy kernel
        if self._is_nested and not self._is_mixed:
            from .nested import _nested_logit_probs_numpy

            k = arrays.design_matrix.shape[1]
            n_nests = self._nests.n_nests
            if beta is None:
                beta = np.asarray(self._result.coefficients.values[:k], dtype=np.float64)
                if alpha is None:
                    lambda_vals = self._result.coefficients.values[k : k + n_nests]
                    lambda_vals = np.clip(lambda_vals, 1e-10, 1.0 - 1e-10)
                    alpha = np.log(lambda_vals / (1.0 - lambda_vals))
            elif beta.size > k:
                # Full parameter vector passed — split into beta and alpha
                alpha = beta[k : k + n_nests]
                beta = beta[:k]
            if alpha is None:
                alpha = np.zeros(n_nests)

            from .._sampling.correction import get_sampling_correction

            return _nested_logit_probs_numpy(
                np.asarray(beta, dtype=np.float64),
                np.asarray(alpha, dtype=np.float64),
                np.asarray(arrays.design_matrix, dtype=np.float64),
                self._nest_matrix,
                arrays.n_obs,
                arrays.n_alts,
                available=arrays.available,
                inclusion_probs=get_sampling_correction(arrays),
            )

        # For mixed models, use the NumPy kernel
        if self._is_mixed and not self._is_nested:
            from .mixed import _mixed_logit_probs_numpy

            k_fixed = self._k_fixed
            k_random = self._k_random

            if beta is None:
                beta_fixed = self._result.coefficients.values[:k_fixed]
                beta_random_means = self._result.coefficients.values[k_fixed : k_fixed + k_random]
                beta_random_spreads = self._result.coefficients.values[k_fixed + k_random :]
            else:
                beta_fixed = beta[:k_fixed]
                beta_random_means = beta[k_fixed : k_fixed + k_random]
                beta_random_spreads = beta[k_fixed + k_random :]

            from .._sampling.correction import get_sampling_correction

            return _mixed_logit_probs_numpy(
                beta_fixed,
                beta_random_means,
                beta_random_spreads,
                self._random_distributions,
                self._draws,
                np.asarray(arrays.design_matrix, dtype=np.float64),
                self._random_col_indices,
                arrays.n_obs,
                arrays.n_alts,
                available=arrays.available,
                inclusion_probs=get_sampling_correction(arrays),
            )

        # Mixed nested — use NumPy fallback
        if self._is_nested and self._is_mixed:
            return self._probabilities_mixed_nested_numpy(arrays, beta, alpha)

        raise RuntimeError("Unknown model configuration for prediction")

    def _probabilities_mixed_nested_numpy(self, arrays, beta, alpha) -> np.ndarray:
        """Compute mixed nested logit probabilities (NumPy fallback)."""
        from .._sampling.correction import get_sampling_correction
        from .nested import _nested_logit_probs_numpy

        k_total = arrays.design_matrix.shape[1]
        k_fixed = self._k_fixed
        k_random = self._k_random
        n_nests = self._nests.n_nests

        if beta is None:
            beta_fixed = self._result.coefficients.values[:k_fixed]
        else:
            beta_fixed = beta[:k_fixed]

        if alpha is None:
            lambda_vals = self._result.coefficients.values[k_fixed : k_fixed + n_nests]
            lambda_vals = np.clip(lambda_vals, 1e-10, 1.0 - 1e-10)
            alpha = np.log(lambda_vals / (1.0 - lambda_vals))

        beta_random_means = self._result.coefficients.values[
            k_fixed + n_nests : k_fixed + n_nests + k_random
        ]
        beta_random_spreads = self._result.coefficients.values[k_fixed + n_nests + k_random :]

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts
        available = arrays.available
        inclusion_probs = get_sampling_correction(arrays)

        dm_fixed = dm[:, [i for i in range(k_total) if i not in self._random_col_indices]]
        dm_random = dm[:, self._random_col_indices]

        if dm_fixed.shape[1] > 0 and len(beta_fixed) > 0:
            v_fixed = (dm_fixed @ beta_fixed).reshape(n_obs, n_alts)
        else:
            v_fixed = np.zeros((n_obs, n_alts))

        if inclusion_probs is not None:
            sr = np.asarray(inclusion_probs, dtype=np.float64).reshape(n_obs, n_alts)
            v_fixed = v_fixed + np.log(np.maximum(sr, 1e-30))

        if available is not None:
            avail = np.asarray(available, dtype=np.float64).reshape(n_obs, n_alts)
        else:
            avail = np.ones((n_obs, n_alts), dtype=np.float64)

        n_draws = self._draws.shape[1]
        probs_sum = np.zeros((n_obs, n_alts), dtype=np.float64)

        for r in range(n_draws):
            beta_random_r = np.zeros((n_obs, k_random))
            for p in range(k_random):
                z_p = self._draws[:, r, p]
                mean_p = beta_random_means[p]
                spread_p = abs(beta_random_spreads[p])
                dist = self._random_distributions[p]

                if dist == "normal":
                    beta_random_r[:, p] = mean_p + spread_p * z_p
                elif dist == "lognormal":
                    exponent = mean_p + spread_p * z_p
                    beta_random_r[:, p] = np.exp(np.clip(exponent, -50, 50))
                elif dist == "triangular":
                    from scipy.stats import norm as norm_dist

                    u = norm_dist.cdf(z_p)
                    mask = u <= 0.5
                    beta_random_r[:, p] = np.where(
                        mask,
                        mean_p + spread_p * (np.sqrt(2 * u) - 1),
                        mean_p + spread_p * (1 - np.sqrt(2 * (1 - u))),
                    )
                elif dist == "uniform":
                    from scipy.stats import norm as norm_dist

                    u = norm_dist.cdf(z_p)
                    beta_random_r[:, p] = mean_p + spread_p * (2 * u - 1)

            v_random = np.sum(
                dm_random.reshape(n_obs, n_alts, k_random) * beta_random_r[:, None, :],
                axis=2,
            )
            V = v_fixed + v_random

            probs_r = _nested_logit_probs_numpy(
                np.zeros(k_total),  # beta not used — V is precomputed
                np.asarray(alpha, dtype=np.float64),
                V.reshape(-1, 1) if V.size > 0 else np.zeros((n_obs * n_alts, 1)),
                self._nest_matrix,
                n_obs,
                n_alts,
                available=avail,
                inclusion_probs=None,
            )
            probs_sum += probs_r

        return probs_sum / n_draws

    def utilities(self, data=None, beta=None) -> np.ndarray:
        """Compute deterministic utilities.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to predict on. If None, uses estimation data.
        beta : np.ndarray or None
            Utility coefficients. If None, uses estimated values.

        Returns
        -------
        np.ndarray, shape (n_obs, n_alts)
            Deterministic utilities.
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
            k = arrays.design_matrix.shape[1]
            beta = np.asarray(self._result.coefficients.values[:k], dtype=np.float64)

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        V = (dm @ beta).reshape(n_obs, n_alts)
        from .._sampling.correction import apply_sampling_correction

        V = apply_sampling_correction(V, arrays)
        return V

    # ------------------------------------------------------------------
    # Simulation (vectorized)
    # ------------------------------------------------------------------

    def simulate(self, data=None, n_draws: int = 1, seed: Optional[int] = None) -> pd.DataFrame:
        """Simulate choices from the estimated model.

        Uses vectorized inverse-CDF sampling — no Python loops over
        observations.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to simulate on. If None, uses estimation data.
        n_draws : int, optional
            Number of simulation draws per observation. Default 1.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Simulated choices with columns ``draw``, ``obs_id``,
            ``alt_id``, and ``probability``.
        """
        from ..data.choicetable import ChoiceTable

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
        probs = probs / probs.sum(axis=1, keepdims=True)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        df = ct.to_frame()
        alt_ids = df[ct.alt_id_col].values.reshape(n_obs, n_alts)
        obs_ids = df[ct.obs_id_col].values.reshape(n_obs, n_alts)[:, 0]

        # Vectorized simulation: draw all choices at once
        cumulative_probs = np.cumsum(probs, axis=1)
        uniform_draws = rng.random((n_draws, n_obs))
        chosen_indices = np.argmax(
            cumulative_probs[None, :, :] > uniform_draws[:, :, None], axis=2
        )
        chosen_indices = np.clip(chosen_indices, 0, n_alts - 1)

        chosen_alts = alt_ids[np.arange(n_obs), chosen_indices]
        chosen_probs = probs[np.arange(n_obs), chosen_indices]

        # Build results DataFrame (vectorized)
        results = pd.DataFrame(
            {
                "draw": np.repeat(np.arange(n_draws), n_obs),
                ct.obs_id_col: np.tile(obs_ids, n_draws),
                ct.alt_id_col: chosen_alts.T.ravel(),
                "probability": chosen_probs.T.ravel(),
            }
        )
        return results

    # ------------------------------------------------------------------
    # Marginal Effects
    # ------------------------------------------------------------------

    def _resolve_me_data(self, data=None):
        """Resolve data for marginal effects computation."""
        from ..data.choicetable import ChoiceTable

        if self._arrays is None:
            raise RuntimeError("Model must be estimated before computing marginal effects.")

        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            ct = data

        probs = self.probabilities(data=data)
        df = ct.to_frame()
        index = pd.MultiIndex.from_arrays(
            [df[ct.obs_id_col].values, df[ct.alt_id_col].values],
            names=[ct.obs_id_col, ct.alt_id_col],
        )
        return ct, probs, df, index

    def marginal_effects(self, data=None, variable: Optional[str] = None):
        """Compute average direct, indirect, and total marginal effects.

        In the SAR-MNL model (``lag=True``), a change in an attribute of
        alternative *j* affects not only *j*'s utility but also neighbouring
        alternatives through the spatial multiplier
        :math:`(I - \\rho W)^{-1}`.

        Following LeSage & Pace (2009):

        - **Direct effect**: impact on own alternative.
        - **Indirect effect**: spillover to neighbouring alternatives.
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
        if not self._is_spatial_lag:
            raise ValueError(
                "marginal_effects() is only available for SAR models (lag=True). "
                "Use marginal_effect() for non-spatial models."
            )
        if self._arrays is None:
            raise RuntimeError("Model must be estimated before computing marginal effects.")

        from ..data.choicetable import ChoiceTable

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

        coef_vals = np.asarray(self._result.coefficients.values, dtype=np.float64)
        beta = coef_vals[:k]
        rho = float(coef_vals[k])

        param_names = list(arrays.param_names)
        if variable not in param_names:
            raise ValueError(f"Variable '{variable}' not found in parameters: {param_names}")
        beta_r = beta[param_names.index(variable)]

        probs = self.probabilities(data=data)
        W_dense = np.asarray(self._W_sparse.toarray(), dtype=np.float64)
        A = np.eye(n_alts) - rho * W_dense
        Z_mat = np.linalg.inv(A)

        direct = np.zeros(n_alts)
        indirect = np.zeros(n_alts)
        for k_alt in range(n_alts):
            direct[k_alt] = (
                beta_r * Z_mat[k_alt, k_alt] * np.mean(probs[:, k_alt] * (1 - probs[:, k_alt]))
            )
            for j_alt in range(n_alts):
                if j_alt != k_alt:
                    indirect[k_alt] += (
                        -beta_r * Z_mat[k_alt, j_alt] * np.mean(probs[:, k_alt] * probs[:, j_alt])
                    )

        total = direct + indirect

        df = ct.to_frame()
        alt_ids = df[ct.alt_id_col].values.reshape(n_obs, n_alts)[0]

        return {
            "direct": pd.Series(direct, index=alt_ids, name=f"direct_{variable}"),
            "indirect": pd.Series(indirect, index=alt_ids, name=f"indirect_{variable}"),
            "total": pd.Series(total, index=alt_ids, name=f"total_{variable}"),
        }

    def marginal_effect(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute direct marginal effects for a variable.

        For MNL: :math:`(1 - P_{qi}) \\beta_x`.

        For nested logit: :math:`P_i (1 - \\lambda_m P_{i|m}) \\beta_x`
        where :math:`P_{i|m}` is the conditional probability within nest m.

        For mixed logit: :math:`E_z[(1 - P_i(z)) \\beta_x]` via simulation
        over draws.

        For SCL: raises ``NotImplementedError`` (derivation pending).

        Parameters
        ----------
        data : ChoiceTable or None
            Data to compute marginal effects on.
        variable : str
            Name of the variable.

        Returns
        -------
        pd.Series
            Direct marginal effects, indexed by (obs_id, alt_id).
        """
        ct, probs, df, index = self._resolve_me_data(data)
        beta = self._result.coefficients.get(variable, 0.0)

        if self._is_nested:
            # Nested logit: P_i * (1 - lambda_m * P_{i|m}) * beta
            n_obs = probs.shape[0]
            n_alts = probs.shape[1]
            n_nests = self._nests.n_nests

            # Get lambda values from estimated coefficients
            k = self._arrays.design_matrix.shape[1] if self._arrays is not None else None
            if k is None and data is not None:
                k = data.to_arrays(
                    formula=self._spec.formula,
                    spec=self._spec if self._spec.formula is None else None,
                ).design_matrix.shape[1]

            # Extract lambda values (naturalized)
            if self._is_spatial:
                # Nested SCL: [beta, rho_1..M, lambda_1..M]
                lambda_vals = self._result.coefficients.values[k + n_nests : k + 2 * n_nests]
            else:
                # Nested: [beta, lambda_1..M]
                lambda_vals = self._result.coefficients.values[k : k + n_nests]

            # Compute conditional probabilities P_{i|m} = P_i / P_m
            # P_m = sum of P_i for alts in nest m
            nest_matrix = self._nest_matrix  # (n_alts, n_nests)
            P_nest = probs @ nest_matrix  # (n_obs, n_nests)
            # P_{i|m} = P_i / P_m (avoid division by zero)
            # For each alt, find which nest it belongs to
            alt_in_nest = nest_matrix.sum(axis=1) > 0  # (n_alts,) bool

            # lambda for each alternative (from its nest)
            long_lambda = np.ones(n_alts)
            for m in range(n_nests):
                mask = nest_matrix[:, m] > 0
                long_lambda[mask] = lambda_vals[m]

            # P_{i|m} for each (obs, alt)
            P_i_given_m = np.zeros_like(probs)
            for m in range(n_nests):
                mask = nest_matrix[:, m] > 0
                if not mask.any():
                    continue
                P_m = P_nest[:, m : m + 1]  # (n_obs, 1)
                P_i_given_m[:, mask] = probs[:, mask] / np.maximum(P_m, 1e-30)

            # Marginal effect: P_i * (1 - lambda_m * P_{i|m}) * beta
            me = probs * (1 - long_lambda[None, :] * P_i_given_m) * beta
            me = me.ravel()

            # For root nest alternatives (not in any nest), use MNL formula
            if not alt_in_nest.all():
                root_mask = ~alt_in_nest
                me_2d = me.reshape(n_obs, n_alts)
                me_2d[:, root_mask] = (1 - probs[:, root_mask]) * beta
                me = me_2d.ravel()

        elif self._is_mixed:
            # Mixed logit: E_z[(1 - P_i(z)) * beta] via simulation
            # For now, use the MNL approximation with mean coefficients
            # (proper implementation requires per-draw probability computation)
            me = (1 - probs.ravel()) * beta

        elif self._is_spatial:
            # SCL: derivation pending
            raise NotImplementedError(
                "Marginal effects for SCL models are not yet implemented. "
                "The MNL approximation is incorrect for spatially correlated logit."
            )

        else:
            # MNL: (1 - P_i) * beta
            me = (1 - probs.ravel()) * beta

        return pd.Series(me, index=index, name=f"marginal_effect_{variable}")

    def cross_marginal_effect(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute cross-marginal effects for a variable.

        For MNL: :math:`-P_i \\beta_x`.

        For nested logit: :math:`-P_i \\lambda_m P_{i|m} \\beta_x`.

        For SCL: raises ``NotImplementedError``.

        Parameters
        ----------
        data : ChoiceTable or None
        variable : str

        Returns
        -------
        pd.Series
            Cross-marginal effects, indexed by (obs_id, alt_id).
        """
        ct, probs, df, index = self._resolve_me_data(data)
        beta = self._result.coefficients.get(variable, 0.0)

        if self._is_nested:
            # Nested logit cross-ME: -P_i * lambda_m * P_{i|m} * beta
            n_obs = probs.shape[0]
            n_alts = probs.shape[1]
            n_nests = self._nests.n_nests

            k = self._arrays.design_matrix.shape[1] if self._arrays is not None else None
            if k is None and data is not None:
                k = data.to_arrays(
                    formula=self._spec.formula,
                    spec=self._spec if self._spec.formula is None else None,
                ).design_matrix.shape[1]

            if self._is_spatial:
                lambda_vals = self._result.coefficients.values[k + n_nests : k + 2 * n_nests]
            else:
                lambda_vals = self._result.coefficients.values[k : k + n_nests]

            nest_matrix = self._nest_matrix
            P_nest = probs @ nest_matrix
            alt_in_nest = nest_matrix.sum(axis=1) > 0

            long_lambda = np.ones(n_alts)
            for m in range(n_nests):
                mask = nest_matrix[:, m] > 0
                long_lambda[mask] = lambda_vals[m]

            P_i_given_m = np.zeros_like(probs)
            for m in range(n_nests):
                mask = nest_matrix[:, m] > 0
                if not mask.any():
                    continue
                P_m = P_nest[:, m : m + 1]
                P_i_given_m[:, mask] = probs[:, mask] / np.maximum(P_m, 1e-30)

            cross_me = -probs * long_lambda[None, :] * P_i_given_m * beta
            cross_me = cross_me.ravel()

            if not alt_in_nest.all():
                root_mask = ~alt_in_nest
                cross_me_2d = cross_me.reshape(n_obs, n_alts)
                cross_me_2d[:, root_mask] = -probs[:, root_mask] * beta
                cross_me = cross_me_2d.ravel()

        elif self._is_spatial:
            raise NotImplementedError(
                "Cross-marginal effects for SCL models are not yet implemented."
            )

        else:
            # MNL and mixed (approximation): -P_i * beta
            cross_me = -probs.ravel() * beta

        return pd.Series(cross_me, index=index, name=f"cross_marginal_effect_{variable}")

    def elasticity(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute direct elasticities for a variable.

        For MNL: :math:`(1 - P_{qi}) \\beta_x x_{qi}`.

        For nested logit: :math:`P_i (1 - \\lambda_m P_{i|m}) \\beta_x x_{qi}`.

        For SCL: raises ``NotImplementedError``.

        Parameters
        ----------
        data : ChoiceTable or None
        variable : str

        Returns
        -------
        pd.Series
            Direct elasticities, indexed by (obs_id, alt_id).
        """
        ct, probs, df, index = self._resolve_me_data(data)
        x = df[variable].values

        if self._is_spatial and not self._is_nested:
            raise NotImplementedError("Elasticities for SCL models are not yet implemented.")

        # For MNL, nested, and mixed: elasticity = marginal_effect * x
        me = self.marginal_effect(data=data, variable=variable)
        elasticities = me.values * x

        return pd.Series(elasticities, index=index, name=f"elasticity_{variable}")

    def cross_elasticity(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute cross-elasticities for a variable.

        For MNL: :math:`-P_i \\beta_x x_{ij}`.

        For SCL: raises ``NotImplementedError``.

        Parameters
        ----------
        data : ChoiceTable or None
        variable : str

        Returns
        -------
        pd.Series
            Cross-elasticities, indexed by (obs_id, alt_id).
        """
        ct, probs, df, index = self._resolve_me_data(data)
        x = df[variable].values

        if self._is_spatial and not self._is_nested:
            raise NotImplementedError("Cross-elasticities for SCL models are not yet implemented.")

        # cross_elasticity = cross_marginal_effect * x
        cme = self.cross_marginal_effect(data=data, variable=variable)
        cross_elast = cme.values * x

        return pd.Series(cross_elast, index=index, name=f"cross_elasticity_{variable}")

    # ------------------------------------------------------------------
    # Covariance estimation
    # ------------------------------------------------------------------

    def covariance_robust(self, data=None) -> np.ndarray:
        """Compute the sandwich (Huber-White) robust covariance matrix.

        Parameters
        ----------
        data : ChoiceTable or None

        Returns
        -------
        np.ndarray, shape (n_parameters, n_parameters)
            Sandwich (robust) covariance matrix.
        """
        from ..data.choicetable import ChoiceTable

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
        H_inv = self._get_hessian_inverse()

        if H_inv is None:
            return _safe_inv(B)

        return _sandwich_inv(H_inv, B)

    def covariance_clustered(self, data=None, groups=None) -> np.ndarray:
        """Compute cluster-robust (Rogers) covariance matrix.

        Parameters
        ----------
        data : ChoiceTable or None
        groups : array-like, shape (n_obs,)
            Cluster/group identifiers.

        Returns
        -------
        np.ndarray, shape (n_parameters, n_parameters)
            Cluster-robust covariance matrix.
        """
        from ..data.choicetable import ChoiceTable

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

        H_inv = self._get_hessian_inverse()

        if H_inv is None:
            return _safe_inv(B_clustered)

        return _sandwich_inv(H_inv, B_clustered)

    def std_errors_robust(self, data=None) -> pd.Series:
        """Compute sandwich (Huber-White) robust standard errors."""
        cov = self.covariance_robust(data=data)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        se[se == 0] = np.nan
        return pd.Series(se, index=self._result.coefficients.index, name="std_error_robust")

    def std_errors_clustered(self, data=None, groups=None) -> pd.Series:
        """Compute cluster-robust standard errors."""
        cov = self.covariance_clustered(data=data, groups=groups)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        se[se == 0] = np.nan
        return pd.Series(se, index=self._result.coefficients.index, name="std_error_clustered")

    # ------------------------------------------------------------------
    # Observation scores
    # ------------------------------------------------------------------

    def _observation_scores(self, arrays) -> np.ndarray:
        """Compute observation-level score (gradient) vectors.

        Uses analytical MNL scores for MNL models, JAX jacrev for models
        with a JAX objective (spatial/nested/mixed), and finite differences
        as a last-resort fallback.
        """
        cache_key = id(arrays)
        if cache_key in self._observation_scores_cache:
            return self._observation_scores_cache[cache_key]

        if not self._is_spatial and not self._is_nested and not self._is_mixed:
            scores = self._mnl_observation_scores(arrays)
        elif self._objective is not None and self._objective.jax_fn is not None:
            scores = self._jax_observation_scores(arrays)
        else:
            scores = self._finite_diff_observation_scores(arrays)

        self._observation_scores_cache[cache_key] = scores
        return scores

    def _mnl_observation_scores(self, arrays) -> np.ndarray:
        """Compute MNL observation scores analytically."""
        from .._kernels.mnl_numpy import mnl_observation_scores_numpy

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        chosen = np.asarray(arrays.chosen, dtype=np.float64).reshape(arrays.n_obs, arrays.n_alts)
        beta = np.asarray(self._result.coefficients.values, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        if arrays.available is not None:
            available = np.asarray(arrays.available, dtype=np.float64).reshape(n_obs, n_alts)
        else:
            available = np.ones((n_obs, n_alts), dtype=np.float64)

        from .._sampling.correction import get_sampling_correction

        inclusion_probs = get_sampling_correction(arrays)

        weights = None
        if arrays.weights is not None:
            weights = np.asarray(arrays.weights, dtype=np.float64)

        return mnl_observation_scores_numpy(
            beta=beta,
            design_matrix=dm,
            chosen=chosen,
            available=available,
            n_obs=n_obs,
            n_alts=n_alts,
            weights=weights,
            inclusion_probs=inclusion_probs,
        )

    def _finite_diff_observation_scores(self, arrays) -> np.ndarray:
        """Compute observation scores via finite differences (fallback)."""
        eps = 1e-5
        n_params = len(self._result.coefficients)
        n_obs = arrays.n_obs

        full_params = np.asarray(self._result.coefficients.values, dtype=np.float64).copy()
        chosen = np.asarray(arrays.chosen, dtype=np.float64).reshape(n_obs, arrays.n_alts)
        scores = np.zeros((n_obs, n_params))

        for j in range(n_params):
            p_plus = full_params.copy()
            p_plus[j] += eps
            p_minus = full_params.copy()
            p_minus[j] -= eps

            probs_plus = self.probabilities(data=None, beta=p_plus)
            probs_minus = self.probabilities(data=None, beta=p_minus)

            ll_plus = np.log(np.maximum(np.sum(probs_plus * chosen, axis=1), 1e-30))
            ll_minus = np.log(np.maximum(np.sum(probs_minus * chosen, axis=1), 1e-30))
            scores[:, j] = (ll_plus - ll_minus) / (2 * eps)

        return scores

    def _jax_observation_scores(self, arrays) -> np.ndarray:
        """Compute observation scores via JAX jacrev on per-obs LL contributions.

        This uses ``jax.jacrev`` on the objective's per-observation log-likelihood
        contributions, giving exact gradients in a single backward pass — O(1)
        instead of O(n_params) probability evaluations.
        """
        import jax.numpy as jnp

        # Get per-observation LL contribution function from the objective
        if self._objective is None or self._objective.jax_fn is None:
            return self._finite_diff_observation_scores(arrays)

        # Try score_contribs (requires loglike_contribs_jax to be set)
        try:
            contribs_fn = self._objective.score_contribs
            beta = jnp.asarray(self._result.coefficients.values, dtype=jnp.float64)
            scores = np.asarray(contribs_fn(beta))
            return scores
        except (ValueError, AttributeError, TypeError):
            pass

        # Fall back to finite differences
        return self._finite_diff_observation_scores(arrays)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "estimated" if self._result is not None else "not estimated"
        formula_str = self._formula or "custom spec"
        return f"ChoiceModel(formula='{formula_str}', {status})"
