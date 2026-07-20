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

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from .._jax.objective import Objective
from .._kernels.constants import SAR_DENSE_CUTOFF
from .._solvers import Solver, SolverResult
from ..data.arrays import ChoiceArrays
from ..results.fit_result import FitResult
from ._spatial import (
    EdgeStructure,
    _resolve_spatial_graph,
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
from .nested import NestingTree


def _lambda_to_alpha(lambda_vals) -> np.ndarray:
    """Map display-scale nest lambdas in (0, 1) to unconstrained alphas.

    The nested kernels parameterise ``lambda = sigmoid(alpha)``; this is the
    inverse used whenever display-scale coefficients are handed back to them.
    """
    clipped = np.clip(np.asarray(lambda_vals, dtype=np.float64), 1e-10, 1.0 - 1e-10)
    return np.log(clipped / (1.0 - clipped))


@dataclass
class ParamLayout:
    """Describes how to extract display-scale parameters from the unconstrained vector.

    Encapsulates the parameter layout for a specific model variant,
    replacing the 8+ branches in ``_build_fit_result`` with a single
    data-driven extraction.

    Attributes
    ----------
    display_names : list[str]
        Names of display-scale parameters.
    transforms : list[tuple[int, str, float | None]]
        One per display parameter: (raw_idx, transform_type, value).
        transform_type is "identity", "sigmoid", "tanh", or "abs".
        value is the natural-scale value (for delta-method SE); None for identity.

    Notes
    -----
    The ``"abs"`` transform applies to mixed-logit spread parameters.  The
    kernels use the spread directly (``beta = mean + spread * z``) and every
    supported mixing distribution draws ``z`` symmetrically about zero, so the
    likelihood is even in the spread and its sign is not identified.  Reporting
    ``|raw|`` is therefore the natural scale, and since ``|d|x|/dx| = 1`` the
    delta-method factor is exactly one.
    """

    display_names: list[str]
    transforms: list[tuple[int, str, float | None]]

    def extract(self, all_params: np.ndarray) -> tuple[np.ndarray, list[str], list[dict]]:
        """Extract display-scale values, names, and transform_spec from unconstrained params.

        Parameters
        ----------
        all_params : np.ndarray
            Full parameter vector in unconstrained (optimizer) space.

        Returns
        -------
        display_values : np.ndarray
        display_names : list[str]
        transform_spec : list[dict]
            For ``_compute_se_generic``.
        """
        display_values = np.empty(len(self.transforms), dtype=np.float64)
        transform_spec = []
        for i, (raw_idx, ttype, _) in enumerate(self.transforms):
            raw_val = float(all_params[raw_idx])
            if ttype == "identity":
                display_values[i] = raw_val
            elif ttype == "sigmoid":
                display_values[i] = 1.0 / (1.0 + np.exp(-raw_val))
            elif ttype == "tanh":
                display_values[i] = np.tanh(raw_val)
            elif ttype == "abs":
                display_values[i] = abs(raw_val)
            else:
                display_values[i] = raw_val
            transform_spec.append(
                {
                    "raw_idx": raw_idx,
                    "type": ttype,
                    "value": float(display_values[i]) if ttype != "identity" else None,
                }
            )
        return display_values, self.display_names, transform_spec


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
        Type of draws: ``"sobol"`` (default, also spelled ``"qmc"``),
        ``"halton"``, ``"scrambled_halton"``, or ``"random"``.
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
        self._diag_precompute = None  # Precomputed diagonal for variance norm
        self._sparse_solve_ctx = None  # Sparse solve context (large n_alts)
        self._sparse_solve_fn = None  # Custom VJP sparse solve function
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
                self._setup_sar_filter(arrays)
            else:
                # SCL: resolve via edge structure (existing behavior)
                self._resolve_spatial_graph()
                self._validate_graph_size(arrays)

                # Build per-nest edge structures for nested spatial models
                if self._is_nested:
                    self._build_per_nest_edges(arrays)

    def _setup_sar_filter(self, arrays: ChoiceArrays) -> None:
        """Resolve W and set up the SAR spatial-filter solve + variance diagonal.

        The filter ``(I - ρW)^{-1} V_base`` and the normalisation diagonal
        ``diag((I - ρW)^{-1})`` are the per-iteration spatial cost.  When a
        pure-JAX sparse backend is available (cholgraph for symmetrizable W,
        klujax otherwise) the solve becomes a sparse, differentiable, JIT
        kernel; the diagonal is interpolated from exact nodes (cholgraph's
        selected inverse when symmetrizable).  Otherwise both fall back to
        the dense path.
        """
        from ._spatial_weights import resolve_spatial_weights

        self._W_sparse = resolve_spatial_weights(
            self._graph_input, arrays.n_alts, row_standardize=True
        )[1]

        # Prefer a pure-JAX sparse solve (cholgraph for symmetrizable W, klujax
        # otherwise): differentiable and JIT-native.  When neither is installed,
        # fall back to the scipy sparse custom-VJP path (CHOLMOD / KLU) — still
        # sparse, never dense.  The dense solve is used only below the threshold.
        self._sparse_solve_ctx = None
        self._sparse_solve_fn = None
        self._sparse_backend = None
        diag_at_nodes = None
        if arrays.n_alts > 500:
            from .._jax.sparse_backends import cholgraph_node_diagonals, make_sparse_solve_fn

            solve_fn, backend = make_sparse_solve_fn(self._W_sparse)
            # solve_fn is None only when neither cholgraph nor klujax is
            # installed; the estimation kernel is JIT-compiled, so the
            # host-side scipy factorisation cannot run inside it and the dense
            # solve is used instead.  (The numpy prediction path still uses the
            # scipy sparse factorisation — see ``_sar_sparse_filter``.)
            self._sparse_solve_fn = solve_fn
            self._sparse_backend = backend
            if backend == "cholgraph":
                diag_at_nodes = lambda nodes: cholgraph_node_diagonals(  # noqa: E731
                    self._W_sparse, nodes
                )

        # Precompute the differentiable, JIT-compatible variance diagonal
        # interpolant; D(ρ) depends only on ρ.
        if arrays.n_alts > 50:
            from .._jax.diag_precompute import precompute_diagonal

            self._diag_precompute = precompute_diagonal(
                self._W_sparse, diag_at_nodes=diag_at_nodes
            )
        else:
            self._diag_precompute = None

    def _sar_sparse_filter(self, rho: float, V_base: np.ndarray):
        """Apply the SAR filter in numpy via a sparse factorisation.

        Returns ``(V_filtered, D)`` where ``V_filtered = (I - ρW)^{-1} V_base``
        and ``D = diag((I - ρW)^{-1})``.  Uses CHOLMOD (symmetrizable W) or KLU
        via :func:`create_factorization` rather than densifying ``W`` — this is
        the numpy prediction/scoring counterpart to the JIT sparse solve.
        """
        from .._jax.sparse_solve import create_factorization

        fact = create_factorization(self._W_sparse, float(rho))
        V_filtered = fact.solve(V_base.T).T
        D = fact.diagonal_inverse()
        return V_filtered, D

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

    def _get_param_roles(self, arrays: ChoiceArrays) -> list[dict]:
        """Describe the parameter layout as a list of role dicts.

        Each dict has:
        - ``role``: "beta", "beta_fixed", "rho", "rho_per_nest", "lambda", "mean", "sd"
        - ``count``: number of parameters in this role
        - ``transform``: "identity", "tanh", "sigmoid", "abs"
        - ``init``: initial value(s) — scalar or array
        - ``solver_names``: list of solver-space names
        - ``display_names``: list of display-space names

        This is the single source of truth for the parameter layout.
        ``_get_solver_inputs`` and ``_build_param_layout`` both consume it,
        eliminating the duplicated 4×3 branching.
        """
        k = arrays.design_matrix.shape[1]
        param_names_all = list(arrays.param_names)
        n_nests = self._nests.n_nests if self._is_nested else 0
        nest_names = self._nests.nest_names if self._is_nested else []
        k_fixed = self._k_fixed if self._is_mixed else k
        k_random = self._k_random if self._is_mixed else 0
        random_names = self._random_param_names if self._is_mixed else []
        fixed_names = self._fixed_names if self._is_mixed else param_names_all

        roles: list[dict] = []

        # --- Beta block ---
        if self._is_mixed:
            roles.append(
                {
                    "role": "beta_fixed",
                    "count": k_fixed,
                    "transform": "identity",
                    "init": np.zeros(k_fixed),
                    "solver_names": list(fixed_names),
                    "display_names": list(fixed_names),
                }
            )
        else:
            roles.append(
                {
                    "role": "beta",
                    "count": k,
                    "transform": "identity",
                    "init": "problem_or_zeros",  # handled by _get_solver_inputs
                    "solver_names": param_names_all,
                    "display_names": param_names_all,
                }
            )

        # --- Rho block ---
        if self._is_spatial_lag:
            roles.append(
                {
                    "role": "rho",
                    "count": 1,
                    "transform": "tanh",
                    "init": "warmstart_or_zero",  # handled by _get_solver_inputs
                    "solver_names": ["alpha_rho"],
                    "display_names": ["rho"],
                }
            )
        elif self._is_spatial_scl:
            if self._is_nested:
                # Per-nest rho
                roles.append(
                    {
                        "role": "rho_per_nest",
                        "count": n_nests,
                        "transform": "sigmoid",
                        "init": np.zeros(n_nests),
                        "solver_names": [f"alpha_rho_{n}" for n in nest_names],
                        "display_names": [f"rho_{n}" for n in nest_names],
                    }
                )
            else:
                roles.append(
                    {
                        "role": "rho",
                        "count": 1,
                        "transform": "sigmoid",
                        "init": np.zeros(1),
                        "solver_names": ["alpha_rho"],
                        "display_names": ["rho"],
                    }
                )

        # --- Lambda block (nested) ---
        if self._is_nested:
            roles.append(
                {
                    "role": "lambda",
                    "count": n_nests,
                    "transform": "sigmoid",
                    "init": "nest_alphas",  # handled by _get_solver_inputs
                    "solver_names": (
                        [f"alpha_lambda_{n}" for n in nest_names]
                        if self._is_spatial_lag
                        else [
                            f"alpha_rho_{n}" for n in nest_names
                        ]  # SCL uses alpha_rho for lambdas too
                        if self._is_spatial_scl
                        else [f"nest_{n}" for n in nest_names]
                    ),
                    "display_names": [f"lambda_{n}" for n in nest_names],
                }
            )

        # --- Mean/SD blocks (mixed) ---
        if self._is_mixed:
            roles.append(
                {
                    "role": "mean",
                    "count": k_random,
                    "transform": "identity",
                    "init": np.zeros(k_random),
                    "solver_names": [f"mean_{n}" for n in random_names],
                    "display_names": [f"mean_{n}" for n in random_names],
                }
            )
            roles.append(
                {
                    "role": "sd",
                    "count": k_random,
                    "transform": "abs",
                    "init": np.full(k_random, 0.1),
                    "solver_names": [f"sd_{n}" for n in random_names],
                    "display_names": [f"sd_{n}" for n in random_names],
                }
            )

        return roles

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
        roles = self._get_param_roles(arrays)

        # Build x0, solver_names from roles
        x0_parts = []
        solver_names = []
        bounds = None
        fixed_mask = None

        for role in roles:
            init = role["init"]
            if isinstance(init, str) and init == "problem_or_zeros":
                # Pure MNL/SCL/SAR beta: use problem initial values when available
                if self._problem is not None:
                    x0_base = self._problem.initial_values
                    bounds = self._problem.bounds
                    fixed_mask = self._problem.fixed_mask
                else:
                    x0_base = np.zeros(role["count"])
                x0_parts.append(x0_base)
            elif isinstance(init, str) and init == "warmstart_or_zero":
                # SAR rho: GMM warm-start when available
                if self._is_spatial_lag and self._warmstart and self._W_sparse is not None:
                    try:
                        from .._kernels.sar_mnl_numpy import fit_linearized_gmm

                        gmm = fit_linearized_gmm(arrays, self._W_sparse)
                        rho_gmm = float(np.clip(gmm["rho"], -0.99, 0.99))
                        x0_parts.append(np.array([np.arctanh(rho_gmm)]))
                    except Exception:
                        x0_parts.append(np.zeros(1))
                else:
                    x0_parts.append(np.zeros(1))
            elif isinstance(init, str) and init == "nest_alphas":
                x0_parts.append(self._nests.initial_alphas())
            else:
                x0_parts.append(np.asarray(init))

            solver_names.extend(role["solver_names"])

        x0 = np.concatenate(x0_parts) if x0_parts else np.zeros(0)
        return x0, solver_names, bounds, fixed_mask

    # ------------------------------------------------------------------
    # Objective construction
    # ------------------------------------------------------------------

    def _resolve_use_cg(self, arrays: ChoiceArrays) -> bool:
        """Whether the SAR objective should use the conjugate-gradient solve.

        ``estimator="auto"`` picks CG once the alternative set is too large
        for a dense factorisation.  Resolved fresh on every fit so the choice
        tracks the data rather than a previous run.
        """
        if self._estimator == "pml_cg":
            return True
        if self._estimator == "auto":
            return arrays.n_alts > SAR_DENSE_CUTOFF
        return False

    def _build_objective(self, arrays: ChoiceArrays) -> Objective:
        """Build optimization objective based on active features."""
        # Pure MNL / SCL / SAR
        if not self._is_nested and not self._is_mixed:
            if self._is_spatial_lag:
                from .._jax.sar_kernels import build_sar_mnl_objective

                # Resolve "auto" locally: overwriting self._estimator would
                # make a second fit() see the previous run's choice rather
                # than re-deciding from the current data.
                use_cg = self._resolve_use_cg(arrays)
                return build_sar_mnl_objective(
                    arrays,
                    self._W_sparse,
                    use_cg=use_cg,
                    diag_precompute=self._diag_precompute,
                    sparse_solve_fn=self._sparse_solve_fn,
                )
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

                use_cg = self._resolve_use_cg(arrays)
                return build_sar_nested_objective(
                    arrays,
                    self._W_sparse,
                    self._nest_matrix,
                    use_cg=use_cg,
                    diag_precompute=self._diag_precompute,
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

                use_cg = self._resolve_use_cg(arrays)
                return build_sar_mixed_objective(
                    arrays,
                    self._W_sparse,
                    self._random_col_indices,
                    self._random_distributions,
                    self._draws,
                    use_cg=use_cg,
                    diag_precompute=self._diag_precompute,
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

                use_cg = self._resolve_use_cg(arrays)
                return build_sar_mixed_nested_objective(
                    arrays,
                    self._W_sparse,
                    self._nest_matrix,
                    self._random_col_indices,
                    self._random_distributions,
                    self._draws,
                    use_cg=use_cg,
                    diag_precompute=self._diag_precompute,
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

    def _build_param_layout(self, arrays: ChoiceArrays) -> ParamLayout:
        """Build the parameter layout for the current model variant.

        Consumes ``_get_param_roles`` — the single source of truth for the
        parameter layout — and converts it to a ``ParamLayout``.
        """
        roles = self._get_param_roles(arrays)

        display_names: list[str] = []
        transforms: list[tuple[int, str, float | None]] = []
        raw_idx = 0

        for role in roles:
            count = role["count"]
            ttype = role["transform"]
            for j in range(count):
                display_names.append(role["display_names"][j])
                transforms.append((raw_idx + j, ttype, None))
            raw_idx += count

        return ParamLayout(display_names=display_names, transforms=transforms)

    def _build_fit_result(
        self,
        solver_result: SolverResult,
        arrays: ChoiceArrays,
    ) -> FitResult:
        """Build a FitResult from solver output."""
        all_params = solver_result.coefficients
        layout = self._build_param_layout(arrays)
        display_values, display_names, transform_spec = layout.extract(all_params)

        # Retain the unconstrained solution: the objective (and therefore every
        # Hessian and score evaluation) is a function of this vector, not of
        # the display-scale coefficients.
        self._raw_params = np.asarray(all_params, dtype=np.float64)
        self._transform_spec = transform_spec

        std_errors = self._compute_se_generic(all_params, transform_spec)
        return self._make_fit_result(
            solver_result, arrays, display_values, display_names, std_errors
        )

    # ------------------------------------------------------------------
    # Standard error computation helpers
    # ------------------------------------------------------------------

    def _compute_se_generic(self, all_params, transform_spec):
        """Compute standard errors with delta-method transforms, data-driven.

        Replaces the 10+ ``_compute_se_*`` methods with a single approach.
        Each parameter's SE is transformed from unconstrained to natural
        scale via the delta method: ``SE_natural = SE_raw * |d(natural)/d(raw)|``.

        Parameters
        ----------
        all_params : np.ndarray
            Full parameter vector in unconstrained (optimizer) space.
        transform_spec : list of dict
            One entry per **display** parameter.  Each dict has:
            - ``"raw_idx"``: index into the unconstrained parameter vector
            - ``"type"``: ``"identity"``, ``"sigmoid"``, ``"tanh"``, or ``"abs"``
            - ``"value"``: natural-scale value (for sigmoid/tanh delta method)

        Returns
        -------
        np.ndarray
            Standard errors in natural (display) scale.
        """
        n_display = len(transform_spec)
        std_errors = np.full(n_display, np.nan)

        # Get raw SEs (unconstrained scale)
        se_raw = None
        try:
            hess = self._compute_hessian(all_params)
            se_raw = self._compute_std_errors_from_hessian(hess)
        except Exception:
            hess_inv = self._get_hessian_inverse()
            if hess_inv is not None:
                se_raw = np.sqrt(np.maximum(np.diag(hess_inv), 0))
                se_raw[se_raw == 0] = np.nan

        if se_raw is None:
            return std_errors

        # Apply delta method per parameter
        for i, spec in enumerate(transform_spec):
            raw_idx = spec["raw_idx"]
            transform_type = spec["type"]
            se_r = se_raw[raw_idx]

            if transform_type == "identity":
                std_errors[i] = se_r
            elif transform_type == "sigmoid":
                # natural = sigmoid(raw), d(natural)/d(raw) = natural * (1 - natural)
                val = spec["value"]
                std_errors[i] = val * (1.0 - val) * se_r
            elif transform_type == "tanh":
                # natural = tanh(raw), d(natural)/d(raw) = 1 - natural^2
                val = spec["value"]
                std_errors[i] = (1.0 - val**2) * se_r
            elif transform_type == "abs":
                # natural = |raw|, so |d(natural)/d(raw)| = 1 exactly.
                std_errors[i] = se_r
            else:
                std_errors[i] = se_r

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

        # Delta-method covariance on the display scale, so that consumers
        # (WTP, Wald tests) get correlations rather than a diagonal.
        cov_display = None
        hess_inv = self._get_hessian_inverse()
        if hess_inv is not None:
            try:
                cov_display = self._to_display_covariance(hess_inv)
            except Exception:
                cov_display = None

        return FitResult(spec=self._spec, covariance_matrix=cov_display, **stats)

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

            V_base = (dm @ beta_use).reshape(n_obs, n_alts)
            from .._sampling.correction import apply_sampling_correction

            V_base = apply_sampling_correction(V_base, arrays)

            V_filtered, D = self._sar_sparse_filter(rho, V_base)
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
                    alpha = _lambda_to_alpha(self._result.coefficients.values[k : k + n_nests])
            elif beta.size > k:
                # Full parameter vector passed.  It is display scale (the
                # same layout as ``result.coefficients``), so the nest
                # entries are lambdas and need mapping to the kernel's
                # unconstrained alpha.
                alpha = _lambda_to_alpha(beta[k : k + n_nests])
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

        # Display-scale layout: [fixed betas, lambdas, random means, spreads].
        # A caller-supplied vector must drive every block, otherwise
        # perturbing it (as the finite-difference scores do) silently has no
        # effect on most parameters.
        params = self._result.coefficients.values if beta is None else np.asarray(beta)
        if params.size < k_fixed + n_nests + 2 * k_random:
            params = self._result.coefficients.values

        beta_fixed = params[:k_fixed]
        if alpha is None:
            alpha = _lambda_to_alpha(params[k_fixed : k_fixed + n_nests])
        beta_random_means = params[k_fixed + n_nests : k_fixed + n_nests + k_random]
        beta_random_spreads = params[k_fixed + n_nests + k_random :]

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
            me = self._marginal_effect_mixed(data, variable, probs)

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

    def _marginal_effect_mixed(self, data, variable: str, probs: np.ndarray) -> np.ndarray:
        """Simulated direct marginal effect for mixed logit.

        Mirrors the MNL convention (``d log P_i / d x_i``), but the
        expectation runs over the mixing distribution rather than being
        evaluated at mean coefficients::

            d log P_i / d x_i = E_r[P_i(r) (1 - P_i(r)) b_i(r)] / E_r[P_i(r)]

        where ``b_i(r)`` is the coefficient realised for draw ``r``.  It
        collapses to ``(1 - P_i) * beta`` when the spread goes to zero.
        """
        if self._is_spatial:
            raise NotImplementedError(
                "Marginal effects for spatially correlated mixed logit (MSCL) "
                "are not yet implemented."
            )

        from .._sampling.correction import get_sampling_correction
        from .mixed import _mixed_logit_per_draw_log_probs_numpy

        arrays = self._arrays
        if data is not None:
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )

        k_fixed = self._k_fixed
        k_random = self._k_random
        values = self._result.coefficients.values

        log_probs_draws, beta_random_draws, _ = _mixed_logit_per_draw_log_probs_numpy(
            values[:k_fixed],
            values[k_fixed : k_fixed + k_random],
            values[k_fixed + k_random :],
            self._random_distributions,
            self._draws,
            np.asarray(arrays.design_matrix, dtype=np.float64),
            self._random_col_indices,
            arrays.n_obs,
            arrays.n_alts,
            available=arrays.available,
            inclusion_probs=get_sampling_correction(arrays),
        )
        probs_draws = np.exp(log_probs_draws)  # (n_obs, n_draws, n_alts)

        # Coefficient on `variable` for each draw: random parameters vary by
        # draw and observation, fixed ones are constant.
        if variable in self._random_param_names:
            p = self._random_param_names.index(variable)
            beta_draws = beta_random_draws[:, :, p][:, :, None]  # (n_obs, n_draws, 1)
        else:
            beta_draws = float(self._result.coefficients.get(variable, 0.0))

        numerator = np.mean(probs_draws * (1.0 - probs_draws) * beta_draws, axis=1)
        denominator = np.maximum(np.mean(probs_draws, axis=1), 1e-30)
        return (numerator / denominator).ravel()

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

        # Scores and Hessian are both raw-space; convert the sandwich to
        # display scale so it lines up with the reported coefficient names.
        cov_raw = _safe_inv(B) if H_inv is None else _sandwich_inv(H_inv, B)
        return self._to_display_covariance(cov_raw)

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

        cov_raw = _safe_inv(B_clustered) if H_inv is None else _sandwich_inv(H_inv, B_clustered)
        return self._to_display_covariance(cov_raw)

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
        # Key on identity, but retain the arrays object alongside the result:
        # without a strong reference the id could be recycled by a later
        # allocation and silently match a different dataset.
        cache_key = id(arrays)
        cached = self._observation_scores_cache.get(cache_key)
        if cached is not None and cached[0] is arrays:
            return cached[1]

        if not self._is_spatial and not self._is_nested and not self._is_mixed:
            scores = self._mnl_observation_scores(arrays)
        elif self._objective is not None and self._objective.jax_fn is not None:
            scores = self._jax_observation_scores(arrays)
        else:
            scores = self._finite_diff_observation_scores(arrays)

        self._observation_scores_cache[cache_key] = (arrays, scores)
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
        """Compute observation scores via finite differences (fallback).

        Perturbs display-scale coefficients (that is what
        :meth:`probabilities` accepts), then maps the resulting scores into
        raw space with the delta-method Jacobian so every score path shares
        the same coordinate system.
        """
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

        jac = self._delta_jacobian()
        if jac is not None and jac.shape[0] == scores.shape[1]:
            scores = scores @ jac

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

        # Try score_contribs (requires loglike_contribs_jax to be set).
        # The contribution kernels take the *unconstrained* vector — they
        # apply tanh/sigmoid to rho/lambda internally — so differentiate at
        # the raw solution, giving raw-space scores.
        try:
            contribs_fn = self._objective.score_contribs
            raw = self._raw_params
            if raw is None:
                raise ValueError("raw parameters unavailable")
            scores = np.asarray(contribs_fn(jnp.asarray(raw, dtype=jnp.float64)))
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
