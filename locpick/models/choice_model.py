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

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from .._jax.objective import Objective
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
    _sigmoid,
)
from .mixed import ParamDistribution, _resolve_draws
from .nested import NestingTree

#: Largest |rho| from the linearised GMM that is still trusted as a PML
#: warm start.  Beyond this the ``rho = 0`` expansion has broken down and the
#: estimate is discarded rather than clipped onto the boundary.
_WARMSTART_RHO_MAX: float = 0.95


def _lambda_to_alpha(lambda_vals) -> np.ndarray:
    """Map display-scale nest lambdas in (0, 1) to unconstrained alphas.

    The nested kernels parameterise ``lambda = sigmoid(alpha)``; this is the
    inverse used whenever display-scale coefficients are handed back to them.
    """
    clipped = np.clip(np.asarray(lambda_vals, dtype=np.float64), 1e-10, 1.0 - 1e-10)
    return np.log(clipped / (1.0 - clipped))


def _draw_rows(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF draw of one column per row of a row-stochastic matrix."""
    u = rng.random(probs.shape[0])
    idx = np.argmax(np.cumsum(probs, axis=1) > u[:, None], axis=1)
    return np.minimum(idx, probs.shape[1] - 1)


def _lottery_accept(code: np.ndarray, room: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Which of this round's picks an alternative accepts.

    ``code[i]`` is chooser ``i``'s pick as an index into ``room`` (the last
    slot, ``-1``, is unconstrained).  Each alternative accepts a uniformly
    random subset of the choosers who picked it, up to its remaining room.
    """
    order = np.lexsort((rng.random(code.size), code))  # by alternative, random within
    sorted_code = code[order]
    starts = np.flatnonzero(np.r_[True, sorted_code[1:] != sorted_code[:-1]])
    group_sizes = np.diff(np.r_[starts, code.size])
    rank = np.arange(code.size) - np.repeat(starts, group_sizes)
    accept_sorted = rank < room[sorted_code]
    accept = np.empty(code.size, dtype=bool)
    accept[order] = accept_sorted
    return accept


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
        transform_type is "identity", "sigmoid", "tanh", or "softplus".
        value is the natural-scale value (for delta-method SE); None for identity.

    Notes
    -----
    The ``"softplus"`` transform applies to mixed-logit spread parameters.
    Every mixed kernel maps the unconstrained spread through
    ``softplus(raw) = log(1 + exp(raw))`` before realising coefficients
    (``beta = mean + softplus(raw) * z``), so the natural-scale spread is
    ``softplus(raw)`` and its delta-method factor is ``sigmoid(raw)``.
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
            elif ttype == "softplus":
                display_values[i] = np.logaddexp(0.0, raw_val)
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

        ``"auto"`` (default) and ``"pml"`` use Smirnov (2010) pseudo maximum
        likelihood: utilities are spatially filtered *and* divided by
        ``diag((I - ρW)^{-1})`` to standardise the heteroskedasticity induced
        by filtering the disturbances.  Because it is a pseudo-likelihood, the
        information matrix equality does not hold — prefer
        :meth:`std_errors_robust` over the default Hessian-based errors, and
        treat likelihood-ratio comparisons with care.

        ``"reduced"`` fits the reduced form ``ψ = (I - ρW)^{-1} Xβ`` with no
        normalisation.  This is the model in which only *systematic* utility
        reaches a spatial equilibrium while the errors stay iid EV1, so the
        softmax is the **exact** likelihood: default standard errors and LR
        tests are valid, and the diagonal interpolant precompute is skipped
        entirely.  Both specifications share the same score at ``ρ = 0``.

        ``"linearized_gmm"`` uses the two-step GMM estimator
        (Carrión-Flores et al. 2018) for very large J.
    warmstart : bool, default True
        SAR models only: start ``rho`` at the linearised GMM estimate when it
        lies safely inside the stationary region.
    panel : str or array-like, optional
        Mixed logit only: the decision-maker of each choice situation, as a
        chooser column name or one id per observation.  A person's taste draw
        is then shared across all their choices, and the simulated likelihood
        is ``prod_n mean_r prod_t P(y_nt | beta_nr)``.  Robust and clustered
        standard errors treat the person as the independent unit.
    correlated : bool, default False
        Mixed logit only: estimate a full covariance for the random
        coefficients, ``beta = mean + L z`` with ``L`` lower triangular
        (reported as ``chol_<a>_<b>``; see :meth:`random_parameter_covariance`).
        Requires normal or lognormal coefficients.

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
        panel=None,
        correlated: bool = False,
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
        self._estimator = estimator  # SAR estimator: auto, pml, reduced, linearized_gmm
        if estimator not in ("auto", "pml", "reduced", "linearized_gmm"):
            raise ValueError(
                f"Unknown estimator {estimator!r}; expected one of "
                "'auto', 'pml', 'reduced', 'linearized_gmm'."
            )
        # "reduced" drops the variance normalisation, making the softmax over
        # (I - ρW)^{-1} Xβ an exact likelihood rather than a pseudo-likelihood.
        self._sar_normalize = estimator != "reduced"
        self._warmstart = warmstart  # Use GMM estimates as PML starting values

        # Mixed logit settings
        self._panel = panel
        self._correlated = bool(correlated)
        self._correlated_active = False  # set in _prepare_random_params (needs k_random > 1)
        self._panel_structure: Optional[tuple[np.ndarray, int]] = None
        if panel is not None and not self._is_mixed:
            raise ValueError(
                "panel applies to mixed logit (random_params), where a person's taste "
                "draw is shared across their choices.  For other models choices are "
                "independent given the covariates; to account for repeated choices "
                "use covariance_clustered(groups=...)."
            )
        if correlated and not self._is_mixed:
            raise ValueError("correlated applies to mixed logit (random_params).")
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

        # GMM has no likelihood objective; drop state a previous PML fit left
        # behind so post-estimation methods cannot silently reuse it.
        self._objective = None
        self._raw_params = None
        self._transform_spec = None
        self._solver_result = None
        self._fixed_mask = None

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
        pure-JAX sparse backend is available (sparsax for symmetrizable W,
        KLU otherwise) the solve becomes a sparse, differentiable, JIT
        kernel; the diagonal is interpolated from exact nodes (sparsax's
        selected inverse when symmetrizable).  Otherwise both fall back to
        the dense path.
        """
        from ._spatial_weights import resolve_spatial_weights

        self._W_sparse = resolve_spatial_weights(
            self._graph_input, arrays.n_alts, row_standardize=True
        )[1]

        # Prefer a pure-JAX sparse solve (sparsax: CHOLMOD for symmetrizable W,
        # KLU otherwise): differentiable and JIT-native.  When it is not
        # installed, fall back to the scipy sparse custom-VJP path (CHOLMOD /
        # KLU) — still sparse, never dense.  The dense solve is used only below
        # the threshold.
        self._sparse_solve_ctx = None
        self._sparse_solve_fn = None
        self._sparse_backend = None
        diag_at_nodes = None
        if arrays.n_alts > 500:
            from .._jax.sparse_backends import make_sparse_solve_fn, sparsax_node_diagonals

            solve_fn, backend = make_sparse_solve_fn(self._W_sparse)
            # solve_fn is None only when sparsax is not installed; the
            # estimation kernel is JIT-compiled, so the host-side scipy
            # factorisation cannot run inside it and the dense solve is used
            # instead.  (The numpy prediction path still uses the scipy sparse
            # factorisation — see ``_sar_sparse_filter``.)
            self._sparse_solve_fn = solve_fn
            self._sparse_backend = backend
            if backend == "sparsax":
                diag_at_nodes = lambda nodes: sparsax_node_diagonals(  # noqa: E731
                    self._W_sparse, nodes
                )

        # Precompute the differentiable, JIT-compatible variance diagonal
        # interpolant; D(ρ) depends only on ρ.  The reduced form never divides
        # by D, so it skips this entirely — the precompute is the dominant
        # setup cost (≈9 s at n_alts = 3000, and it scales worse than linearly
        # because it fits one rational approximant per alternative).
        if self._sar_normalize and arrays.n_alts > 50:
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

        if self._correlated:
            bad = [
                n
                for n, d in zip(random_param_names, random_distributions)
                if d not in ("normal", "lognormal")
            ]
            if bad:
                raise ValueError(
                    "correlated random parameters must be normal or lognormal; "
                    f"got other distributions for {bad}."
                )
        # With one random coefficient there is nothing to correlate.
        self._correlated_active = self._correlated and k_random > 1

        # Draws: with panels, one set per decision-maker, shared by all of that
        # person's choice situations.
        if self._panel is not None:
            codes, n_panels = self._resolve_panel(arrays)
            person_draws = _resolve_draws(
                self._draw_type, n_panels, self._n_draws, k_random, seed=self._seed
            )
            draws = person_draws[codes]
            self._panel_structure = (codes, n_panels)
        else:
            draws = _resolve_draws(
                self._draw_type, arrays.n_obs, self._n_draws, k_random, seed=self._seed
            )
            self._panel_structure = None

        fixed_names = [n for i, n in enumerate(param_names) if i not in random_col_indices]
        full_param_names = (
            fixed_names
            + [f"mean_{n}" for n in random_param_names]
            + self._spread_names(random_param_names)
        )

        self._random_param_names = random_param_names
        self._random_distributions = random_distributions
        self._random_col_indices = random_col_indices
        self._k_fixed = k_fixed
        self._k_random = k_random
        self._fixed_names = fixed_names
        self._full_param_names = full_param_names
        self._draws = draws

    def _spread_names(self, random_names: list[str]) -> list[str]:
        """Names of the spread parameters: ``sd_<a>``, or packed ``chol_<a>_<b>``."""
        if not self._correlated_active:
            return [f"sd_{n}" for n in random_names]
        rows, cols = np.tril_indices(len(random_names))
        return [f"chol_{random_names[r]}_{random_names[c]}" for r, c in zip(rows, cols)]

    def _spread_from_display(self, values) -> np.ndarray:
        """Display-scale spread parameters as the kernels' scale: vector or Cholesky ``L``."""
        values = np.asarray(values, dtype=np.float64)
        if not self._correlated_active:
            return values
        k = self._k_random
        L = np.zeros((k, k))
        L[np.tril_indices(k)] = values
        return L

    def _resolve_panel(self, arrays: ChoiceArrays) -> tuple[np.ndarray, int]:
        """Decision-maker codes (``0..n_panels-1``) per choice situation."""
        panel = self._panel
        n_obs = arrays.n_obs
        if isinstance(panel, str):
            if self._data is None:
                raise ValueError("panel as a column name requires a ChoiceTable.")
            df = self._data.to_frame()
            if panel not in df.columns:
                raise KeyError(f"panel column {panel!r} not found in the choice table.")
            values = df[panel].to_numpy().reshape(n_obs, arrays.n_alts)
            if not (values == values[:, :1]).all():
                raise ValueError(f"panel column {panel!r} varies within a choice situation.")
            ids = values[:, 0]
        else:
            ids = np.asarray(panel)
            if ids.shape != (n_obs,):
                raise ValueError(
                    f"panel must have one id per observation ({n_obs}), got shape {ids.shape}."
                )
        codes, uniques = pd.factorize(ids)
        return codes.astype(np.int32), len(uniques)

    def random_parameter_covariance(self) -> pd.DataFrame:
        """Covariance of the random coefficients' mixing distribution, ``L L'``.

        For independent coefficients this is diagonal (squared spreads).  For
        lognormal coefficients it is the covariance of ``log(beta)``.
        """
        if self._result is None or not self._is_mixed:
            raise RuntimeError("Requires a fitted mixed logit model.")
        names = self._spread_names(self._random_param_names)
        spread = self._spread_from_display(self._result.coefficients[names].to_numpy())
        L = spread if spread.ndim == 2 else np.diag(spread)
        return pd.DataFrame(
            L @ L.T, index=self._random_param_names, columns=self._random_param_names
        )

    def _draws_for(self, arrays: ChoiceArrays) -> np.ndarray:
        """Simulation draws matching ``arrays``.

        The estimation draws are sized to the estimation sample.  Draws are
        deterministic given the draw type, sample size and seed, so data of
        any size gets its own draws from the same generator.
        """
        if not self._is_mixed:
            return None
        if self._same_data(arrays):
            return self._draws
        return _resolve_draws(
            self._draw_type, arrays.n_obs, self._n_draws, self._k_random, seed=self._seed
        )

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
        - ``transform``: "identity", "tanh", "sigmoid", "softplus"
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
                    **self._spread_role(k_random, random_names),
                }
            )

        return roles

    def _spread_role(self, k_random: int, random_names: list[str]) -> dict:
        """Layout of the spread block: ``k`` spreads, or a packed Cholesky factor."""
        names = self._spread_names(random_names)
        if not self._correlated_active:
            return {
                "count": k_random,
                "transform": "softplus",
                "init": np.full(k_random, 0.1),
                "solver_names": names,
                "display_names": names,
            }
        rows, cols = np.tril_indices(k_random)
        diag = rows == cols
        return {
            "count": len(names),
            # Diagonal through softplus (positive); off-diagonal unconstrained.
            "transform": ["softplus" if d else "identity" for d in diag],
            "init": np.where(diag, 0.1, 0.0),
            "solver_names": names,
            "display_names": names,
        }

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
                        rho_gmm = float(gmm["rho"])
                        # The linearised GMM expands around rho = 0, so under
                        # strong dependence it routinely returns |rho| > 1 —
                        # outside the stationary region and carrying no usable
                        # location.  Clipping such a value onto the boundary is
                        # actively harmful for the PML: as rho -> 1,
                        # D = diag((I - rho W)^-1) diverges, filtered utilities
                        # collapse toward zero and the likelihood approaches the
                        # uniform-choice value, which is a local optimum the
                        # optimiser cannot escape.  Discard instead of clipping.
                        if np.isfinite(rho_gmm) and abs(rho_gmm) < _WARMSTART_RHO_MAX:
                            x0_parts.append(np.array([np.arctanh(rho_gmm)]))
                        else:
                            x0_parts.append(np.zeros(1))
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

    def _build_objective(self, arrays: ChoiceArrays, draws=None) -> Objective:
        """Build optimization objective based on active features.

        ``draws`` defaults to the estimation draws; prediction on other data
        passes draws sized to that sample.
        """
        if draws is None:
            draws = self._draws
        # The panel describes the estimation sample only.
        panel = self._panel_structure if self._same_data(arrays) else None
        # Pure MNL / SCL / SAR
        if not self._is_nested and not self._is_mixed:
            if self._is_spatial_lag:
                from .._jax.sar_kernels import build_sar_mnl_objective

                # Resolve "auto" locally: overwriting self._estimator would
                # make a second fit() see the previous run's choice rather
                # than re-deciding from the current data.
                return build_sar_mnl_objective(
                    arrays,
                    self._W_sparse,
                    diag_precompute=self._diag_precompute,
                    sparse_solve_fn=self._sparse_solve_fn,
                    normalize=self._sar_normalize,
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

                return build_sar_nested_objective(
                    arrays,
                    self._W_sparse,
                    self._nest_matrix,
                    diag_precompute=self._diag_precompute,
                    sparse_solve_fn=self._sparse_solve_fn,
                    normalize=self._sar_normalize,
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

                return build_sar_mixed_objective(
                    arrays,
                    self._W_sparse,
                    self._random_col_indices,
                    self._random_distributions,
                    draws,
                    diag_precompute=self._diag_precompute,
                    sparse_solve_fn=self._sparse_solve_fn,
                    normalize=self._sar_normalize,
                    panel=panel,
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
                    draws,
                    panel=panel,
                )
            from .._jax.builders import build_mixed_logit_objective

            return build_mixed_logit_objective(
                arrays,
                random_col_indices=self._random_col_indices,
                random_distributions=self._random_distributions,
                draws=draws,
                panel=panel,
            )

        # Mixed Nested
        if self._is_nested and self._is_mixed:
            if self._is_spatial_lag:
                from .._jax.sar_kernels import build_sar_mixed_nested_objective

                return build_sar_mixed_nested_objective(
                    arrays,
                    self._W_sparse,
                    self._nest_matrix,
                    self._random_col_indices,
                    self._random_distributions,
                    draws,
                    diag_precompute=self._diag_precompute,
                    sparse_solve_fn=self._sparse_solve_fn,
                    normalize=self._sar_normalize,
                    panel=panel,
                )
            elif self._is_spatial_scl:
                from .._jax.builders import build_mnscl_objective

                return build_mnscl_objective(
                    arrays,
                    self._nest_matrix,
                    self._edge_data_list,
                    self._random_col_indices,
                    self._random_distributions,
                    draws,
                    panel=panel,
                )
            from .._jax.builders import build_mixed_nested_objective

            return build_mixed_nested_objective(
                arrays,
                self._nest_matrix,
                self._random_col_indices,
                self._random_distributions,
                draws,
                panel=panel,
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
            transform = role["transform"]
            for j in range(count):
                display_names.append(role["display_names"][j])
                ttype = transform[j] if isinstance(transform, list) else transform
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
            - ``"type"``: ``"identity"``, ``"sigmoid"``, ``"tanh"``, or ``"softplus"``
            - ``"value"``: natural-scale value (for sigmoid/tanh delta method)

        Returns
        -------
        np.ndarray
            Standard errors in natural (display) scale.
        """
        n_display = len(transform_spec)
        std_errors = np.full(n_display, np.nan)

        # Raw SEs (unconstrained scale).  The inverse is cached, so the full
        # covariance built afterwards reuses this Hessian rather than
        # computing a second one.  Held-fixed parameters have zero variance
        # and are reported as NaN.
        hess_inv = self._get_hessian_inverse()
        if hess_inv is None:
            return std_errors
        se_raw = np.sqrt(np.maximum(np.diag(hess_inv), 0))
        se_raw[se_raw == 0] = np.nan

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
            elif transform_type == "softplus":
                # natural = softplus(raw), d(natural)/d(raw) = sigmoid(raw)
                std_errors[i] = _sigmoid(all_params[raw_idx]) * se_r
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

        return FitResult(
            spec=self._spec,
            covariance_matrix=cov_display,
            converged=bool(solver_result.converged),
            n_iterations=int(solver_result.n_iterations or 0),
            message=str(solver_result.message or ""),
            **stats,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def probabilities(self, data=None, beta=None, alpha=None, available=None) -> np.ndarray:
        """Compute choice probabilities.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to predict on. If None, uses estimation data.  For nested and
            spatial models it must hold the same alternatives as the
            estimation data (the nesting and the spatial graph are defined
            over them).
        beta : np.ndarray or None
            Parameter vector. If None, uses estimated values.
        alpha : np.ndarray or None
            Nest parameters (for nested models). If None, uses estimated values.
        available : array-like or None, shape (n_obs, n_alts)
            Restrict each observation's choice set: False (or 0) removes an
            alternative, on top of the data's own availability.  The model
            recomputes probabilities over the remaining alternatives (for SAR,
            removed alternatives still pass utility to their neighbours).

        Returns
        -------
        np.ndarray, shape (n_obs, n_alts)
            Choice probabilities.
        """
        if self._arrays is None:
            raise RuntimeError("Model must be estimated before prediction.")

        arrays = self._prediction_arrays(data)
        if available is not None:
            arrays = self._with_availability(arrays, available)

        if self._uses_kernel_probs:
            return self._probabilities_kernel(arrays, beta, alpha)

        # Dispatch based on model type
        if self._is_nested or self._is_mixed:
            return self._probabilities_complex(arrays, data, beta, alpha)
        return self._probabilities_mnl(arrays, data, beta)

    def _prediction_arrays(self, data) -> ChoiceArrays:
        """Arrays for ``data``: the estimation arrays, or ``data``'s own (cached).

        The last prediction table's arrays are kept, keyed on the table object,
        so repeated calls (simulation rounds, marginal effects) build them once.
        """
        if data is None:
            return self._arrays
        from ..data.choicetable import ChoiceTable

        if not isinstance(data, ChoiceTable):
            raise TypeError("data must be a ChoiceTable")
        cached = getattr(self, "_prediction_cache", None)
        if cached is not None and cached["data"] is data:
            return cached["arrays"]
        arrays = data.to_arrays(
            formula=self._spec.formula,
            spec=self._spec if self._spec.formula is None else None,
        )
        if (self._is_nested or self._is_spatial) and arrays.n_alts != self._arrays.n_alts:
            raise ValueError(
                f"{self.model_type} models predict over the estimation alternatives: "
                f"data has {arrays.n_alts} alternatives per observation, the model "
                f"{self._arrays.n_alts}."
            )
        self._prediction_cache = {"data": data, "arrays": arrays, "objective": None}
        return arrays

    @staticmethod
    def _with_availability(arrays: ChoiceArrays, available) -> ChoiceArrays:
        """Copy of ``arrays`` whose availability also excludes ``available == 0``."""
        import dataclasses

        mask = np.asarray(available, dtype=np.float64)
        if mask.size != arrays.n_obs * arrays.n_alts:
            raise ValueError(
                f"available must have {arrays.n_obs} x {arrays.n_alts} entries, "
                f"got shape {mask.shape}."
            )
        mask = (mask.reshape(arrays.n_obs, arrays.n_alts) > 0).astype(np.float64)
        if arrays.available is not None:
            base = np.asarray(arrays.available, dtype=np.float64).reshape(mask.shape)
            mask = mask * (base > 0)
        return dataclasses.replace(arrays, available=mask)

    @property
    def _uses_kernel_probs(self) -> bool:
        """True for configurations without a dedicated NumPy probability path.

        Combining two or more of nesting, mixing and spatial structure is
        served by the estimation kernel itself (see
        :func:`~locpick._jax.objective.make_log_probs_fn`), so the
        probabilities cannot drift from the likelihood that was maximised.
        """
        n_features = sum([self._is_nested, self._is_mixed, self._is_spatial])
        return n_features >= 2

    def _display_to_raw(self, values: np.ndarray) -> np.ndarray:
        """Invert the display transforms, mapping coefficients to solver space."""
        values = np.asarray(values, dtype=np.float64)
        raw = np.array(self._raw_params, dtype=np.float64, copy=True)
        for i, spec in enumerate(self._transform_spec):
            v = values[i]
            ttype = spec["type"]
            if ttype == "sigmoid":
                v = np.clip(v, 1e-12, 1.0 - 1e-12)
                raw[spec["raw_idx"]] = np.log(v / (1.0 - v))
            elif ttype == "tanh":
                raw[spec["raw_idx"]] = np.arctanh(np.clip(v, -1.0 + 1e-12, 1.0 - 1e-12))
            elif ttype == "softplus":
                # softplus^{-1}(v) = log(expm1(v)), written stably for large v
                v = max(v, 1e-12)
                raw[spec["raw_idx"]] = v + np.log(-np.expm1(-v))
            else:
                raw[spec["raw_idx"]] = v
        return raw

    def _probabilities_kernel(self, arrays, beta, alpha) -> np.ndarray:
        """Probabilities from the estimation kernel, for combined configurations.

        On the estimation data this reuses the fitted objective; on other data
        it builds (and caches) an objective over the new arrays with the same
        nesting, spatial structure and parameters, and draws sized to the new
        sample.  Availability is passed through, so restricted choice sets are
        exact under the model rather than renormalised.
        """
        if alpha is not None:
            raise ValueError(
                f"alpha is not accepted for {self.model_type} models; pass the full "
                "display-scale coefficient vector as beta instead."
            )
        if self._objective is None or self._objective.log_probs_jax is None:
            raise RuntimeError("Model must be estimated before prediction.")
        import jax.numpy as jnp

        objective = self._objective_for(arrays)
        raw = self._raw_params if beta is None else self._display_to_raw(beta)
        available = None
        if arrays.available is not None:
            available = jnp.asarray(
                np.asarray(arrays.available, dtype=np.float64).reshape(arrays.n_obs, arrays.n_alts)
            )
        log_probs = np.asarray(objective.log_probs_jax(jnp.asarray(raw), available))
        return np.exp(log_probs)

    def _objective_for(self, arrays: ChoiceArrays) -> Objective:
        """The fitted objective when ``arrays`` holds the estimation data, else one built for it."""
        if self._same_data(arrays):
            return self._objective
        cached = getattr(self, "_prediction_cache", None)
        if (
            cached is not None
            and cached["arrays"] is not None
            and self._same_data(arrays, cached["arrays"])
        ):
            if cached["objective"] is None:
                cached["objective"] = self._build_objective(
                    cached["arrays"], draws=self._draws_for(cached["arrays"])
                )
            return cached["objective"]
        return self._build_objective(arrays, draws=self._draws_for(arrays))

    def _same_data(self, arrays: ChoiceArrays, reference: Optional[ChoiceArrays] = None) -> bool:
        """Whether ``arrays`` carries ``reference``'s design (availability may differ)."""
        reference = self._arrays if reference is None else reference
        return arrays.design_matrix is reference.design_matrix

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
            # Must mirror the estimation kernel: the reduced form never divides
            # by D, so normalising here would score a different model than the
            # one that was fitted.
            V_star = V_filtered / D[None, :] if self._sar_normalize else V_filtered

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
            beta_random_spreads = self._spread_from_display(beta_random_spreads)

            from .._sampling.correction import get_sampling_correction

            return _mixed_logit_probs_numpy(
                beta_fixed,
                beta_random_means,
                beta_random_spreads,
                self._random_distributions,
                self._draws_for(arrays),
                np.asarray(arrays.design_matrix, dtype=np.float64),
                self._random_col_indices,
                arrays.n_obs,
                arrays.n_alts,
                available=arrays.available,
                inclusion_probs=get_sampling_correction(arrays),
            )

        raise RuntimeError("Unknown model configuration for prediction")

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

    def simulate(
        self,
        data=None,
        n_draws: int = 1,
        seed: Optional[int] = None,
        capacity=None,
        max_rounds: int = 100,
    ) -> pd.DataFrame:
        """Simulate choices from the estimated model.

        Without ``capacity`` every chooser draws independently from its choice
        probabilities (vectorized inverse-CDF sampling).

        With ``capacity`` alternatives can fill up, and choices are allocated
        by an iterative lottery (the scheme used by UrbanSim's choicemodels):
        in each round every unplaced chooser draws from the model's
        probabilities over the alternatives that still have room; an
        alternative drawn by more choosers than it has room for accepts a
        random subset of them; the rest draw again next round.  The
        probabilities are recomputed under the model for each restricted
        choice set (see ``available`` in :meth:`probabilities`), so for SAR
        models full alternatives still pass utility to their neighbours.

        For MNL this matches choosers taking their best remaining alternative,
        since by IIA a restricted choice set only renormalises the
        probabilities.  For nested, mixed and spatial models a rejected chooser
        re-draws from the model's prediction for the restricted set, without
        conditioning on the alternative they first drew -- the standard
        approximation in capacity-constrained location-choice simulation.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to simulate on. If None, uses estimation data.
        n_draws : int, optional
            Number of independent simulations. Default 1.
        seed : int or None, optional
            Random seed for reproducibility.
        capacity : pd.Series, dict or None, optional
            Number of choosers each alternative can take, indexed by
            alternative id; non-integer values are floored.  Alternatives
            absent from ``capacity`` (or with ``np.inf``) are unconstrained.
        max_rounds : int, optional
            Maximum lottery rounds per simulation.  Default 100.

        Returns
        -------
        pd.DataFrame
            One row per draw and observation with columns ``draw``, the
            observation id, the alternative id, and ``probability`` (of the
            chosen alternative, in the round it was drawn).  With ``capacity``
            there is also ``round`` (1-based); choosers left unplaced, because
            every alternative in their choice set filled up or the round limit
            was reached, have a missing alternative id and ``round`` 0.
        """
        if self._arrays is None:
            raise RuntimeError("Model must be estimated before simulation.")

        arrays = self._prediction_arrays(data)
        ct = self._data if data is None else data
        rng = np.random.default_rng(seed)
        n_obs, n_alts = arrays.n_obs, arrays.n_alts

        df = ct.to_frame()
        alt_ids = df[ct.alt_id_col].to_numpy().reshape(n_obs, n_alts)
        obs_ids = df[ct.obs_id_col].to_numpy().reshape(n_obs, n_alts)[:, 0]

        if capacity is None:
            probs = self.probabilities(data=data)
            probs = probs / probs.sum(axis=1, keepdims=True)
            chosen = np.stack([_draw_rows(probs, rng) for _ in range(n_draws)])
            rows = np.arange(n_obs)
            return pd.DataFrame(
                {
                    "draw": np.repeat(np.arange(n_draws), n_obs),
                    ct.obs_id_col: np.tile(obs_ids, n_draws),
                    ct.alt_id_col: alt_ids[rows, chosen].ravel(),
                    "probability": probs[rows, chosen].ravel(),
                }
            )

        cap = pd.Series(capacity, dtype=np.float64)
        if (cap < 0).any():
            raise ValueError("capacity must be non-negative.")
        cap_index = pd.Index(cap.index)
        # Column code of each (obs, alt) cell in the capacity vector; -1 means
        # the alternative is unconstrained.
        cell_code = cap_index.get_indexer(alt_ids.ravel()).reshape(n_obs, n_alts)
        base_cap = np.floor(cap.to_numpy())

        frames = []
        for d in range(n_draws):
            chosen_alt, chosen_prob, placed_round = self._capacity_lottery(
                data, alt_ids, cell_code, base_cap.copy(), rng, max_rounds
            )
            frames.append(
                pd.DataFrame(
                    {
                        "draw": d,
                        ct.obs_id_col: obs_ids,
                        ct.alt_id_col: chosen_alt,
                        "probability": chosen_prob,
                        "round": placed_round,
                    }
                )
            )
        results = pd.concat(frames, ignore_index=True)
        n_unplaced = int((results["round"] == 0).sum())
        if n_unplaced:
            warnings.warn(
                f"{n_unplaced} of {len(results)} simulated choices could not be placed: "
                "every alternative in their choice set filled up, or max_rounds was "
                "reached.",
                RuntimeWarning,
                stacklevel=2,
            )
        return results

    def _capacity_lottery(self, data, alt_ids, cell_code, remaining, rng, max_rounds):
        """One iterative-lottery allocation; see :meth:`simulate`."""
        n_obs, n_alts = alt_ids.shape
        chosen_alt = np.full(n_obs, None, dtype=object)
        chosen_prob = np.full(n_obs, np.nan)
        placed_round = np.zeros(n_obs, dtype=np.int64)
        unplaced = np.ones(n_obs, dtype=bool)
        # cell_code == -1 marks unconstrained cells; give them infinite room.
        room = np.append(remaining, np.inf)
        arrays = self._prediction_arrays(data)
        in_data = (
            np.ones((n_obs, n_alts), dtype=bool)
            if arrays.available is None
            else np.asarray(arrays.available).reshape(n_obs, n_alts) > 0
        )

        for rnd in range(1, max_rounds + 1):
            open_cells = room[cell_code] > 0  # (n_obs, n_alts); -1 indexes the inf slot
            stuck = unplaced & ~(open_cells & in_data).any(axis=1)
            unplaced &= ~stuck
            if not unplaced.any():
                break
            probs = self.probabilities(data=data, available=open_cells)
            idx = np.flatnonzero(unplaced)
            p = probs[idx]
            p = p / p.sum(axis=1, keepdims=True)
            col = _draw_rows(p, rng)
            code = cell_code[idx, col]
            accept = _lottery_accept(code, room, rng)
            got = idx[accept]
            chosen_alt[got] = alt_ids[got, col[accept]]
            chosen_prob[got] = p[accept, col[accept]]
            placed_round[got] = rnd
            unplaced[got] = False
            taken = code[accept]
            taken = taken[taken >= 0]
            room[: len(remaining)] -= np.bincount(taken, minlength=len(remaining))
        # Nullable ids: unplaced choosers get <NA> (Int64 for integer ids).
        return pd.array(list(chosen_alt)), chosen_prob, placed_round

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

    def spatial_impacts(
        self,
        variable: str,
        data=None,
        *,
        max_order: int = 4,
        n_sources: Optional[int] = None,
        seed: int = 0,
    ):
        """LeSage-style spatial impacts for a SAR choice model (``lag=True``).

        Decomposes how a change to ``variable`` at one alternative propagates to
        the choice probabilities of *all* alternatives through the spatial
        multiplier.  With ``T = diag(D)^{-1}(I - ρW)^{-1}`` (``D ≡ 1`` for the
        reduced form), the effect on alternative ``j`` of perturbing source
        alternative ``l`` is

        .. math::
            \\frac{\\partial P_{ij}}{\\partial x_{il}}
              = \\beta_k\\, P_{ij}\\,\\big(T_{jl} - (P_i'T)_l\\big).

        Interpretation differs from the linear SAR case in one essential way:
        because :math:`\\sum_j P_{ij} = 1`, the column sums are identically
        zero.  A perturbation cannot raise total probability, only
        **redistribute** it, so the LeSage "total impact" is structurally 0 and
        ``indirect = -direct`` by construction rather than by estimation.  The
        informative quantity is therefore not the total but the *spatial
        profile* of the redistribution — how much of the own-alternative effect
        is absorbed by 1-hop, 2-hop, … neighbours.  That profile is what
        separates a global SAR spillover from a first-order SLX one, which puts
        everything at order 1 and nothing beyond.

        Parameters
        ----------
        variable : str
            Name of the coefficient whose perturbation is propagated.
        data : ChoiceTable or None
            Data to evaluate at; defaults to the estimation sample.
        max_order : int, default 4
            Highest neighbour order reported in the decay profile.  Alternatives
            farther than this (or unreachable) are pooled into ``">max_order"``.
        n_sources : int or None
            Number of source alternatives to average over.  ``None`` uses all of
            them when ``n_alts <= 500``, else a random sample of 200 — each
            source costs one sparse solve, so this bounds the work at large J.
        seed : int, default 0
            Seed for source sampling.

        Returns
        -------
        dict
            ``direct``, ``indirect``, ``total``, and ``profile`` (a
            ``pd.Series`` of mean absolute redistribution by neighbour order,
            plus ``share_beyond_order_1``).

        Raises
        ------
        RuntimeError
            If the model is not a fitted SAR (``lag=True``) model.
        """
        import scipy.sparse as sp_

        self._check_sar_mnl("Spatial impacts", exc=RuntimeError)
        if variable not in self._result.coefficients:
            raise KeyError(f"Unknown variable {variable!r}.")

        _, probs, _, _ = self._resolve_me_data(data)
        probs = np.asarray(probs, dtype=np.float64)
        n_obs, n_alts = probs.shape
        beta_k = float(self._result.coefficients[variable])
        rho = float(self._result.coefficients["rho"])

        from scipy.sparse.csgraph import shortest_path

        from .._jax.sparse_solve import create_factorization

        W = sp_.csr_matrix(self._W_sparse, dtype=np.float64)
        fact = create_factorization(W, rho)

        rng = np.random.default_rng(seed)
        if n_sources is None:
            n_sources = n_alts if n_alts <= 500 else 200
        sources = (
            np.arange(n_alts)
            if n_sources >= n_alts
            else np.sort(rng.choice(n_alts, size=n_sources, replace=False))
        )

        # Hop distances only from the sampled sources: (n_sources, n_alts),
        # so the profile never materialises an all-pairs J x J matrix.
        hops = shortest_path(
            (W != 0).astype(np.int8), method="D", unweighted=True, indices=sources
        )

        # D depends only on rho, so compute it once rather than per source.
        D = fact.diagonal_inverse() if self._sar_normalize else None

        Pbar = probs.mean(axis=0)  # average over choosers
        direct = 0.0
        by_order = {o: 0.0 for o in range(max_order + 1)}
        beyond = 0.0

        for s_idx, src in enumerate(sources):
            e = np.zeros(n_alts)
            e[src] = 1.0
            t_col = fact.solve(e[:, None])[:, 0]  # column `src` of (I - rho W)^-1
            if D is not None:
                t_col = t_col / D  # -> column of T = diag(D)^-1 (I - rho W)^-1
            # dP_ij/dx_i,src = beta P_ij (T_j,src - P_i . T_src), averaged over
            # choosers.  The second term is the mean of a product, so it is
            # taken over choosers rather than evaluated at mean probabilities.
            c = probs @ t_col  # (n_obs,)
            eff = beta_k * (Pbar * t_col - (probs.T @ c) / n_obs)

            direct += eff[src]
            d = hops[s_idx]
            mask = np.ones(n_alts, dtype=bool)
            mask[src] = False
            orders = d[mask]
            mags = np.abs(eff[mask])
            near = np.isfinite(orders) & (orders <= max_order)
            for o in range(1, max_order + 1):
                by_order[o] += float(mags[near & (orders == o)].sum())
            beyond += float(mags[~near].sum())

        ns = len(sources)
        direct /= ns
        prof = {f"order_{o}": by_order[o] / ns for o in range(1, max_order + 1)}
        prof[f">order_{max_order}"] = beyond / ns
        tot_indirect_abs = sum(prof.values())
        prof["share_beyond_order_1"] = (
            (tot_indirect_abs - prof["order_1"]) / tot_indirect_abs
            if tot_indirect_abs > 0
            else 0.0
        )

        return {
            "direct": direct,
            "indirect": -direct,  # exact: column sums of dP/dx vanish
            "total": 0.0,  # structural, not estimated
            "profile": pd.Series(prof),
        }

    def _check_sar_mnl(self, what: str, exc: type = ValueError) -> None:
        """Require a fitted SAR-MNL: the impact formulas assume MNL on filtered utilities."""
        if not self._is_spatial_lag:
            raise exc(f"{what} requires a SAR model (lag=True).")
        if self._is_nested or self._is_mixed:
            raise NotImplementedError(f"{what} is not yet implemented for {self.model_type}.")
        if self._result is None or self._arrays is None:
            raise RuntimeError(f"Model must be estimated before computing {what.lower()}.")

    def marginal_effects(self, data=None, variable: Optional[str] = None):
        """Average direct, indirect and total effects per alternative (SAR-MNL).

        In the SAR-MNL model (``lag=True``) a change in an attribute of
        alternative ``l`` moves every alternative's utility through
        :math:`T = \\mathrm{diag}(D)^{-1}(I - \\rho W)^{-1}` (``D ≡ 1`` for the
        reduced form), so

        .. math::
            \\frac{\\partial P_{ij}}{\\partial x_{il}}
              = \\beta\\, P_{ij}\\,\\big(T_{jl} - (P_i'T)_l\\big).

        Following LeSage & Pace (2009), for each source alternative ``l``:

        - **Direct**: the effect on ``l`` itself, averaged over choosers.
        - **Indirect**: the summed effect on every other alternative.  Because
          probabilities sum to one this is exactly ``-direct``.
        - **Total**: direct + indirect, identically zero.

        Choice probabilities can only be redistributed, never created, so the
        informative output is the direct effect and, via
        :meth:`spatial_impacts`, how far the redistribution reaches.

        The computation uses one sparse factorisation of ``I - ρW'`` and never
        forms the dense inverse.

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
        from .._jax.sparse_solve import create_factorization

        self._check_sar_mnl("Marginal effects")
        if variable not in self._result.coefficients:
            raise ValueError(
                f"Variable '{variable}' not found in parameters: "
                f"{list(self._result.coefficients.index)}"
            )

        ct, probs, df, _ = self._resolve_me_data(data)
        probs = np.asarray(probs, dtype=np.float64)
        n_obs, n_alts = probs.shape
        beta_r = float(self._result.coefficients[variable])
        rho = float(self._result.coefficients["rho"])

        # diag((I - rho W)^-1); under PML this is D itself, so diag(T) == 1.
        diag_inv = create_factorization(self._W_sparse, rho).diagonal_inverse()
        D = diag_inv if self._sar_normalize else np.ones(n_alts)
        diag_T = diag_inv / D

        # Rows of P T: (P_i T)' = (I - rho W')^-1 (P_i / D)'
        fact_T = create_factorization(self._W_sparse.T, rho)
        PT = fact_T.solve((probs / D[None, :]).T).T

        direct = beta_r * np.mean(probs * (diag_T[None, :] - PT), axis=0)

        alt_ids = df[ct.alt_id_col].values.reshape(n_obs, n_alts)[0]
        return {
            "direct": pd.Series(direct, index=alt_ids, name=f"direct_{variable}"),
            "indirect": pd.Series(-direct, index=alt_ids, name=f"indirect_{variable}"),
            "total": pd.Series(np.zeros(n_alts), index=alt_ids, name=f"total_{variable}"),
        }

    def _check_me_supported(self, what: str) -> None:
        """Raise for configurations whose effects have no implemented derivation."""
        if self._is_spatial_lag:
            raise NotImplementedError(
                f"{what} is not defined per alternative for SAR models (lag=True); "
                "use spatial_impacts(), which propagates the change through the "
                "spatial multiplier."
            )
        if self._is_spatial:
            label = "MSCL" if self._is_mixed else "SCL"
            raise NotImplementedError(f"{what} is not yet implemented for {label} models.")
        if self._is_nested and self._is_mixed:
            raise NotImplementedError(f"{what} is not yet implemented for mixed nested models.")

    def _nest_terms(self, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-alternative nest lambdas and within-nest conditional probabilities.

        Returns ``(long_lambda, P_cond)``: ``long_lambda[j]`` is the
        dissimilarity of the nest holding ``j`` (1 for root alternatives) and
        ``P_cond[n, j] = P_nj / P_n,m(j)`` (``P_nj`` itself for root
        alternatives, where the formulas below then reduce to MNL).
        """
        names = [f"lambda_{n}" for n in self._nests.nest_names]
        lambda_vals = self._result.coefficients[names].to_numpy(dtype=np.float64)
        nest_matrix = self._nest_matrix
        long_lambda = np.ones(probs.shape[1])
        P_cond = probs.copy()
        P_nest = probs @ nest_matrix
        for m in range(nest_matrix.shape[1]):
            mask = nest_matrix[:, m] > 0
            if not mask.any():
                continue
            long_lambda[mask] = lambda_vals[m]
            P_cond[:, mask] = probs[:, mask] / np.maximum(P_nest[:, m : m + 1], 1e-30)
        return long_lambda, P_cond

    def marginal_effect(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute direct marginal effects for a variable.

        Effects are semi-elasticities, :math:`\\partial \\ln P_{qi} / \\partial x_{qi}`.

        For MNL: :math:`(1 - P_{qi}) \\beta_x`.

        For nested logit, with :math:`\\lambda_m` the dissimilarity of the nest
        holding ``i`` and :math:`P_{i|m}` the within-nest probability:
        :math:`\\beta_x [1/\\lambda_m - ((1 - \\lambda_m)/\\lambda_m) P_{i|m} - P_i]`,
        which reduces to the MNL value at :math:`\\lambda_m = 1`.

        For mixed logit: :math:`E_r[P_i(r)(1 - P_i(r)) b_x(r)] / E_r[P_i(r)]`,
        simulated over the draws.

        Spatial and mixed nested models raise ``NotImplementedError``; for SAR
        models use :meth:`spatial_impacts`.

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
        self._check_me_supported("Marginal effects")
        ct, probs, df, index = self._resolve_me_data(data)
        beta = self._result.coefficients.get(variable, 0.0)

        if self._is_nested:
            lam, P_cond = self._nest_terms(probs)
            me = beta * (1.0 / lam - ((1.0 - lam) / lam) * P_cond - probs)
            me = me.ravel()
        elif self._is_mixed:
            numerator, p_bar = self._mixed_own_derivative(data, variable)
            me = (numerator / np.maximum(p_bar, 1e-30)).ravel()
        else:
            # MNL: (1 - P_i) * beta
            me = (1 - probs.ravel()) * beta

        return pd.Series(me, index=index, name=f"marginal_effect_{variable}")

    def _mixed_own_derivative(self, data, variable: str) -> tuple[np.ndarray, np.ndarray]:
        """Simulated own-derivative pieces for mixed logit.

        Returns ``(dP, P)`` with ``dP[n, i] = E_r[P_ni(r) (1 - P_ni(r)) b_n(r)]``
        (that is, :math:`\\partial P_{ni} / \\partial x_{ni}`) and
        ``P[n, i] = E_r[P_ni(r)]``, where ``b_n(r)`` is the coefficient on
        ``variable`` realised for draw ``r``.
        """
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
            self._spread_from_display(values[k_fixed + k_random :]),
            self._random_distributions,
            self._draws_for(arrays),
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

        dP = np.mean(probs_draws * (1.0 - probs_draws) * beta_draws, axis=1)
        return dP, np.mean(probs_draws, axis=1)

    def cross_marginal_effect(
        self, data=None, variable: Optional[str] = None, within_nest: bool = True
    ) -> pd.Series:
        """Compute cross-marginal effects for a variable.

        The effect of :math:`x_{qi}` on the log-probability of another
        alternative, :math:`\\partial \\ln P_{qj} / \\partial x_{qi}`, reported
        per ``(obs, i)``.

        For MNL: :math:`-P_i \\beta_x`, the same for every ``j``.

        For nested logit it depends on whether ``j`` shares ``i``'s nest:
        :math:`-\\beta_x [((1 - \\lambda_m)/\\lambda_m) P_{i|m} + P_i]` within
        the nest (``within_nest=True``, the default) and :math:`-P_i \\beta_x`
        across nests.

        For mixed logit the effect varies with ``j``; the value reported is
        its probability-weighted average over the other alternatives,
        :math:`-(\\partial P_i / \\partial x_i) / (1 - P_i)`, which is exact
        (probabilities sum to one) and equals :math:`-P_i \\beta_x` when the
        spreads are zero.

        Spatial and mixed nested models raise ``NotImplementedError``.

        Parameters
        ----------
        data : ChoiceTable or None
        variable : str
        within_nest : bool, default True
            Nested logit only: report the effect on alternatives in the same
            nest (True) or in other nests (False).

        Returns
        -------
        pd.Series
            Cross-marginal effects, indexed by (obs_id, alt_id).
        """
        self._check_me_supported("Cross-marginal effects")
        ct, probs, df, index = self._resolve_me_data(data)
        beta = self._result.coefficients.get(variable, 0.0)

        if self._is_nested and within_nest:
            lam, P_cond = self._nest_terms(probs)
            cross_me = (-beta * (((1.0 - lam) / lam) * P_cond + probs)).ravel()
        elif self._is_mixed:
            dP, p_bar = self._mixed_own_derivative(data, variable)
            cross_me = (-dP / np.maximum(1.0 - p_bar, 1e-30)).ravel()
        else:
            # MNL, and nested logit across nests: -P_i * beta
            cross_me = -probs.ravel() * beta

        return pd.Series(cross_me, index=index, name=f"cross_marginal_effect_{variable}")

    def elasticity(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute direct elasticities for a variable.

        The marginal effect of :meth:`marginal_effect` scaled by
        :math:`x_{qi}`, giving :math:`\\partial \\ln P_{qi} / \\partial \\ln x_{qi}`.

        Parameters
        ----------
        data : ChoiceTable or None
        variable : str

        Returns
        -------
        pd.Series
            Direct elasticities, indexed by (obs_id, alt_id).
        """
        me = self.marginal_effect(data=data, variable=variable)
        x = (data if data is not None else self._data).to_frame()[variable].values
        return pd.Series(me.values * x, index=me.index, name=f"elasticity_{variable}")

    def cross_elasticity(
        self, data=None, variable: Optional[str] = None, within_nest: bool = True
    ) -> pd.Series:
        """Compute cross-elasticities for a variable.

        The cross-marginal effect of :meth:`cross_marginal_effect` scaled by
        :math:`x_{qi}`.

        Parameters
        ----------
        data : ChoiceTable or None
        variable : str
        within_nest : bool, default True
            Nested logit only; see :meth:`cross_marginal_effect`.

        Returns
        -------
        pd.Series
            Cross-elasticities, indexed by (obs_id, alt_id).
        """
        cme = self.cross_marginal_effect(data=data, variable=variable, within_nest=within_nest)
        x = (data if data is not None else self._data).to_frame()[variable].values
        return pd.Series(cme.values * x, index=cme.index, name=f"cross_elasticity_{variable}")

    # ------------------------------------------------------------------
    # Covariance estimation
    # ------------------------------------------------------------------

    def _require_estimation_sample(self, data, what: str) -> None:
        """Reject ``data``: sandwich estimators are defined on the estimation sample."""
        if self._arrays is None:
            raise RuntimeError("Model must be estimated first.")
        if data is not None and data is not self._data:
            raise ValueError(
                f"{what} is defined only on the estimation sample, whose scores and "
                "Hessian it combines; call it without data."
            )

    def covariance_robust(self, data=None) -> np.ndarray:
        """Compute the sandwich (Huber-White) robust covariance matrix.

        Parameters
        ----------
        data : ChoiceTable or None
            Must be None or the estimation data: the sandwich is defined on
            the estimation sample only.

        Returns
        -------
        np.ndarray, shape (n_parameters, n_parameters)
            Sandwich (robust) covariance matrix.
        """
        self._require_estimation_sample(data, "Robust covariance")

        scores = self._observation_scores(self._arrays)
        B = scores.T @ scores
        H_inv = self._get_hessian_inverse()

        # Scores and Hessian are both raw-space; convert the sandwich to
        # display scale so it lines up with the reported coefficient names.
        cov_raw = _safe_inv(B) if H_inv is None else _sandwich_inv(H_inv, B)
        return self._to_display_covariance(cov_raw)

    def covariance_clustered(self, data=None, groups=None, correction: bool = True) -> np.ndarray:
        """Compute cluster-robust (Rogers) covariance matrix.

        Parameters
        ----------
        data : ChoiceTable or None
            Must be None or the estimation data.
        groups : array-like, shape (n_obs,)
            Cluster/group identifiers, one per observation.
        correction : bool, default True
            Scale by ``G / (G - 1)`` for ``G`` clusters, the usual
            small-sample adjustment (it matters when clusters are few).

        Returns
        -------
        np.ndarray, shape (n_parameters, n_parameters)
            Cluster-robust covariance matrix.
        """
        self._require_estimation_sample(data, "Cluster-robust covariance")
        if groups is None:
            raise ValueError("groups must be provided for cluster-robust covariance.")

        scores = self._observation_scores(self._arrays)
        groups = np.asarray(groups)
        if (
            self._panel_structure is not None
            and groups.shape[0] == self._arrays.n_obs != scores.shape[0]
        ):
            # Panel models score per person; clusters must contain whole people.
            codes, _ = self._panel_structure
            per_person = pd.Series(groups).groupby(codes).agg(["first", "nunique"])
            if (per_person["nunique"] > 1).any():
                raise ValueError("clusters must contain all of a decision-maker's choices.")
            groups = per_person["first"].to_numpy()
        if groups.shape[0] != scores.shape[0]:
            raise ValueError(
                f"groups has {groups.shape[0]} entries but the model has "
                f"{scores.shape[0]} observations."
            )

        # Sum scores within each cluster in one pass.
        codes, _ = pd.factorize(groups)
        n_groups = int(codes.max()) + 1
        cluster_scores = np.zeros((n_groups, scores.shape[1]))
        np.add.at(cluster_scores, codes, scores)
        B_clustered = cluster_scores.T @ cluster_scores
        if correction and n_groups > 1:
            B_clustered *= n_groups / (n_groups - 1)

        H_inv = self._get_hessian_inverse()

        cov_raw = _safe_inv(B_clustered) if H_inv is None else _sandwich_inv(H_inv, B_clustered)
        return self._to_display_covariance(cov_raw)

    def std_errors_robust(self, data=None) -> pd.Series:
        """Compute sandwich (Huber-White) robust standard errors."""
        cov = self.covariance_robust(data=data)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        se[se == 0] = np.nan
        return pd.Series(se, index=self._result.coefficients.index, name="std_error_robust")

    def std_errors_clustered(self, data=None, groups=None, correction: bool = True) -> pd.Series:
        """Compute cluster-robust standard errors."""
        cov = self.covariance_clustered(data=data, groups=groups, correction=correction)
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
        except (ValueError, AttributeError, TypeError) as exc:
            # Every JAX builder supplies ``loglike_contribs_jax``, so reaching
            # here means the exact-score path is genuinely unavailable (or
            # broken).  Finite differences are materially less accurate, and
            # these scores feed robust/clustered standard errors — so say so
            # rather than degrading the inference silently.
            warnings.warn(
                "Exact observation scores unavailable "
                f"({type(exc).__name__}: {exc}); falling back to finite "
                "differences.  Robust and clustered standard errors will be "
                "less accurate.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Fall back to finite differences
        return self._finite_diff_observation_scores(arrays)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "estimated" if self._result is not None else "not estimated"
        formula_str = self._formula or "custom spec"
        return f"ChoiceModel(formula='{formula_str}', {status})"
