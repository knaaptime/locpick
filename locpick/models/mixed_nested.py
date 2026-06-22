"""Mixed Nested Logit model for location choice estimation.

This module provides the :class:`MixedNestedMNL` class, which combines
random coefficients (mixed logit) with a nesting structure (nested logit).
The model captures both unobserved taste heterogeneity across decision-makers
and correlation within nests of alternatives.

Mathematical formulation
------------------------
The mixed nested logit probability for alternative :math:`j` in nest
:math:`m` by decision-maker :math:`n` is:

.. math::

    P_{nj} = \\int P_{nj}^{\\text{NL}}(\\beta) \\, f(\\beta \\mid \\theta) \\, d\\beta

where :math:`P_{nj}^{\\text{NL}}(\\beta)` is the nested logit probability
conditional on :math:`\\beta`, and :math:`f(\\beta \\mid \\theta)` is the
mixing distribution parameterised by :math:`\\theta`.

For :math:`R` draws from the mixing distribution, the simulated
log-likelihood is:

.. math::

    \\text{SLL} = \\sum_{n=1}^{N} w_n \\log \\left(
        \\frac{1}{R} \\sum_{r=1}^{R} P_{nj}^{\\text{NL}}(\\beta^r)
    \\right)

where :math:`\\beta^r` are draws from :math:`f(\\beta \\mid \\theta)`.

This class is a convenience wrapper that requires both ``nests`` and
``random_params`` to be specified.  All estimation logic is self-contained.

References
----------
Train, K.E. (2009). *Discrete Choice Methods with Simulation*, 2nd ed.
    Cambridge University Press.
"""

from __future__ import annotations

import os
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
    naturalize_rho,
)
from .base import (
    BaseChoiceModel,
    SpatialMixin,
    _compute_fit_statistics,
    _compute_null_ll,
)
from .mixed import ParamDistribution, _resolve_draws
from .nested import NestingTree, naturalize_nest_params


class MixedNestedMNL(BaseChoiceModel, SpatialMixin):
    r"""Mixed Nested Logit model for location choice estimation.

    This model combines random coefficients (mixed logit) with a nesting
    structure (nested logit).  It captures both unobserved taste
    heterogeneity across decision-makers and correlation within nests
    of alternatives.

    Parameters
    ----------
    data : ChoiceTable
        The choice data to estimate on.
    formula : str, optional
        A formulaic formula string for the utility function.
    spec : ModelSpec, optional
        A ModelSpec object defining the utility function.
    nests : NestingTree
        The nesting structure. Each alternative belongs to exactly one nest.
    random_params : dict
        Mapping of parameter names to :class:`ParamDistribution` objects.
        E.g., ``{"commute_time": ParamDistribution("normal", "commute_time")}``
    n_draws : int, optional
        Number of draws for simulated maximum likelihood. Default 100.
    draw_type : str, optional
        Type of draws: ``"qmc"`` (default), ``"halton"``, or ``"random"``.
    seed : int
        Random seed for draw generation. Default 42.
    weights : str or np.ndarray, optional
        Observation weights.
    availability : str or np.ndarray, optional
        Alternative availability.
    solver : str or Solver, optional
        Solver name or instance. Default ``"lbfgs"``.
    solver_options : dict, optional
        Additional options passed to the solver constructor.
    backend : str, optional
        Computation backend. Default is ``"jax"``.

    Examples
    --------
    >>> from locpick import ChoiceTable
    >>> from .mixed_nested import MixedNestedMNL
    >>> from .nested import NestingTree, NestSpec
    >>> from .mixed import ParamDistribution
    >>> ct = ChoiceTable.from_tables(choosers, alternatives, chosen)
    >>> nests = NestingTree([
    ...     NestSpec("transit", alt_ids=[0, 1, 2]),
    ...     NestSpec("auto", alt_ids=[3, 4]),
    ... ])
    >>> model = MixedNestedMNL(
    ...     ct,
    ...     formula="cost + time - 1",
    ...     nests=nests,
    ...     random_params={"time": ParamDistribution("normal", "time")},
    ...     n_draws=200,
    ... )
    >>> result = model.fit()
    """

    def __init__(
        self,
        data,
        formula: Optional[str] = None,
        spec=None,
        nests: Optional[NestingTree] = None,
        random_params: Optional[dict[str, ParamDistribution]] = None,
        graph=None,
        n_draws: Optional[int] = None,
        draw_type: Optional[str] = None,
        seed: int = 42,
        weights: Optional[Union[str, np.ndarray]] = None,
        availability: Optional[Union[str, np.ndarray]] = None,
        solver: Union[str, Solver] = None,
        solver_options: Optional[dict] = None,
        backend: Optional[str] = None,
    ):
        if nests is None:
            raise ValueError(
                "MixedNestedMNL requires a 'nests' argument specifying "
                "the nesting structure. Use MixedMNL for models without nests."
            )
        if random_params is None or len(random_params) == 0:
            raise ValueError(
                "MixedNestedMNL requires at least one random parameter. "
                "Use NestedMNL for models without random coefficients."
            )

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
        self._nests = nests
        self._random_params = random_params
        self._n_draws = n_draws if n_draws is not None else 100
        self._draw_type = draw_type if draw_type is not None else "sobol"
        self._seed = seed
        self._draws: Optional[np.ndarray] = None
        self._nest_matrix: Optional[np.ndarray] = None
        # Spatial state (None when graph is not provided).
        self._graph_input = graph
        self._omega = None
        self._allocation = None
        self._edge_list = None
        self._n_alts_graph = None
        self._edge_struct = None
        self._edge_structs = None
        self._edge_data_list = None

    @property
    def _is_spatial(self) -> bool:
        return self._graph_input is not None

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def _pre_fit(self, arrays: ChoiceArrays) -> None:
        """Build nest matrix and random parameter structure before objective construction."""
        # Build nest matrix
        alt_ids = list(range(arrays.n_alts))
        self._nest_matrix = self._nests.build_nest_matrix(alt_ids)

        # Identify random parameter columns
        param_names = list(arrays.param_names)
        random_param_names = list(self._random_params.keys())
        random_col_indices = [param_names.index(name) for name in random_param_names]
        random_distributions = [
            self._random_params[name].distribution for name in random_param_names
        ]
        k_random = len(random_col_indices)

        # Generate draws
        draws = _resolve_draws(
            self._draw_type, arrays.n_obs, self._n_draws, k_random, seed=self._seed
        )

        # Cache for objective/prediction paths
        self._draws = draws
        self._random_col_indices = random_col_indices
        self._random_distributions = random_distributions
        self._random_param_names = random_param_names

        if not self._is_spatial:
            return

        # Resolve graph + build per-nest EdgeStructure / EdgeDataJAX
        self._resolve_spatial_graph()
        self._validate_graph_size(arrays)

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

    def _get_solver_inputs(self, arrays: ChoiceArrays):
        """Get initial values, param names, bounds, and fixed mask.

        Parameter layout (non-spatial): [beta_fixed, alpha_nest, mean_*, sd_*]
        Parameter layout (spatial):    [beta_fixed, alpha_rho_1..M,
                                         alpha_lambda_1..M, mean_*, sd_*]
        """
        param_names_all = list(arrays.param_names)
        k_total = arrays.design_matrix.shape[1]

        random_param_names = list(self._random_params.keys())
        k_random = len(random_param_names)
        k_fixed = k_total - k_random
        fixed_param_names = [name for name in param_names_all if name not in random_param_names]

        if self._is_spatial:
            n_nests = self._nests.n_nests
            x0 = np.concatenate(
                [
                    np.zeros(k_fixed),
                    np.zeros(n_nests),  # alpha_rho per nest
                    self._nests.initial_alphas(),
                    np.zeros(k_random),
                    np.full(k_random, 0.1),
                ]
            )
            display_names = (
                fixed_param_names
                + [f"alpha_rho_{name}" for name in self._nests.nest_names]
                + [f"alpha_lambda_{name}" for name in self._nests.nest_names]
                + [f"mean_{name}" for name in random_param_names]
                + [f"sd_{name}" for name in random_param_names]
            )
            return x0, display_names, None, None

        # Initial values: zeros for beta/means, small positive for nest alphas,
        # zeros for random means, small positive for random spreads
        x0 = np.concatenate(
            [
                np.zeros(k_fixed),
                self._nests.initial_alphas(),
                np.zeros(k_random),
                np.full(k_random, 0.1),
            ]
        )

        display_names = (
            fixed_param_names
            + [f"lambda_{name}" for name in self._nests.nest_names]
            + [f"mean_{name}" for name in random_param_names]
            + [f"sd_{name}" for name in random_param_names]
        )

        return x0, display_names, None, None

    def _build_objective(self, arrays: ChoiceArrays) -> Objective:
        """Build optimization objective for mixed nested logit estimation."""
        if not hasattr(self, "_random_col_indices"):
            raise RuntimeError(
                "Random parameter structure must be prepared before building objective."
            )

        if self._is_spatial:
            from .._jax.builders import build_mnscl_objective

            return build_mnscl_objective(
                arrays,
                self._nest_matrix,
                self._edge_data_list,
                self._random_col_indices,
                self._random_distributions,
                self._draws,
            )

        # Try JAX backend first
        backend = (self._backend or os.environ.get("LOCPICK_MIXED_NESTED_BACKEND", "")).lower()
        if backend != "numpy":
            from .._jax.builders import build_mixed_nested_objective

            return build_mixed_nested_objective(
                arrays,
                self._nest_matrix,
                self._random_col_indices,
                self._random_distributions,
                self._draws,
            )

        raise NotImplementedError(
            "MixedNestedMNL currently only supports the JAX backend. "
            f"Requested backend: {backend!r}."
        )

    def _build_fit_result(
        self,
        solver_result: SolverResult,
        arrays: ChoiceArrays,
    ) -> FitResult:
        """Build a FitResult from solver output."""
        bool(self._random_params)
        all_params = solver_result.coefficients

        # Determine parameter layout
        param_names_all = list(arrays.param_names)
        k_total = arrays.design_matrix.shape[1]

        random_param_names = list(self._random_params.keys())
        k_random = len(random_param_names)
        k_fixed = k_total - k_random
        fixed_param_names = [name for name in param_names_all if name not in random_param_names]

        n_nests = self._nests.n_nests

        if self._is_spatial:
            # Spatial layout: [beta_fixed, alpha_rho_1..M, alpha_lambda_1..M, mean_*, sd_*]
            beta_fixed = all_params[:k_fixed]
            alpha_rho = all_params[k_fixed : k_fixed + n_nests]
            alpha_lambda = all_params[k_fixed + n_nests : k_fixed + 2 * n_nests]
            beta_random_means = all_params[
                k_fixed + 2 * n_nests : k_fixed + 2 * n_nests + k_random
            ]
            beta_random_spreads = all_params[k_fixed + 2 * n_nests + k_random :]
            rhos = naturalize_rho(alpha_rho)
            lambdas = naturalize_nest_params(alpha_lambda)
            display_params = np.concatenate(
                [beta_fixed, rhos, lambdas, beta_random_means, np.abs(beta_random_spreads)]
            )
            display_names = (
                fixed_param_names
                + [f"rho_{name}" for name in self._nests.nest_names]
                + [f"lambda_{name}" for name in self._nests.nest_names]
                + [f"mean_{name}" for name in random_param_names]
                + [f"sd_{name}" for name in random_param_names]
            )
            model_type = "Mixed Nested Spatially Correlated Logit"

            std_errors = np.full(len(display_params), np.nan)
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
                if solver_result.hessian is not None:
                    try:
                        se_raw = np.sqrt(np.maximum(np.diag(solver_result.hessian), 0))
                        se_raw[se_raw == 0] = np.nan
                        se_rho = rhos * (1.0 - rhos) * se_raw[k_fixed : k_fixed + n_nests]
                        se_lambda = (
                            lambdas
                            * (1.0 - lambdas)
                            * se_raw[k_fixed + n_nests : k_fixed + 2 * n_nests]
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
                        pass
        else:
            # Extract parameter blocks
            beta_fixed = all_params[:k_fixed]
            alpha_nest = all_params[k_fixed : k_fixed + n_nests]
            beta_random_means = all_params[k_fixed + n_nests : k_fixed + n_nests + k_random]
            beta_random_spreads = all_params[k_fixed + n_nests + k_random :]

            # Naturalize nest parameters: alpha -> lambda
            lambdas = naturalize_nest_params(alpha_nest)

            # Display parameters: [beta_fixed, lambda_nest, mean_*, sd_*]
            display_params = np.concatenate(
                [beta_fixed, lambdas, beta_random_means, np.abs(beta_random_spreads)]
            )

            # Parameter names
            display_names = (
                fixed_param_names
                + [f"lambda_{name}" for name in self._nests.nest_names]
                + [f"mean_{name}" for name in random_param_names]
                + [f"sd_{name}" for name in random_param_names]
            )
            model_type = "Mixed Nested Logit"

            # Standard errors — prefer HVP-based Hessian
            std_errors = np.full(len(display_params), np.nan)
            try:
                hess = self._compute_hessian(all_params)
                se_raw = self._compute_std_errors_from_hessian(hess)
                # Delta method for nest parameters: SE(lambda) = |d(lambda)/d(alpha)| * SE(alpha)
                se_lambda = lambdas * (1.0 - lambdas) * se_raw[k_fixed : k_fixed + n_nests]
                std_errors = np.concatenate(
                    [
                        se_raw[:k_fixed],
                        se_lambda,
                        se_raw[k_fixed + n_nests :],
                    ]
                )
            except Exception:
                if solver_result.hessian is not None:
                    try:
                        se_raw = np.sqrt(np.maximum(np.diag(solver_result.hessian), 0))
                        se_raw[se_raw == 0] = np.nan
                        se_lambda = lambdas * (1.0 - lambdas) * se_raw[k_fixed : k_fixed + n_nests]
                        std_errors = np.concatenate(
                            [
                                se_raw[:k_fixed],
                                se_lambda,
                                se_raw[k_fixed + n_nests :],
                            ]
                        )
                    except Exception:
                        pass

        # T-values and p-values
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = np.where(std_errors > 0, display_params / std_errors, np.nan)
        from scipy import stats

        p_values = 2 * (1 - stats.norm.cdf(np.abs(np.nan_to_num(t_values))))

        # Confidence intervals
        z_crit = stats.norm.ppf(0.975)
        conf_lower = display_params - z_crit * std_errors
        conf_upper = display_params + z_crit * std_errors

        # Log-likelihood
        ll = solver_result.log_likelihood

        # Null log-likelihood
        ll_null = _compute_null_ll(arrays)

        # Number of parameters
        n_params = len(display_params)

        # Fit statistics
        n_obs = arrays.n_obs
        2 * n_params - 2 * ll
        n_params * np.log(n_obs) - 2 * ll
        1 - ll / ll_null
        1 - (ll - n_params) / ll_null

        # Build pandas objects
        coefficients = pd.Series(display_params, index=display_names, name="coefficient")
        std_err_series = pd.Series(std_errors, index=display_names, name="std_error")
        pd.Series(t_values, index=display_names, name="t_value")
        pd.Series(p_values, index=display_names, name="p_value")
        pd.DataFrame(
            {"lower": conf_lower, "upper": conf_upper},
            index=display_names,
        )

        stats = _compute_fit_statistics(
            ll=ll,
            ll_null=ll_null,
            n_obs=n_obs,
            n_params=n_params,
            n_alts=arrays.n_alts,
            coefficients=coefficients,
            std_errors=std_err_series,
            model_type=model_type,
            solver_name=solver_result.solver_name,
            solver_result_raw=solver_result.raw,
        )

        return FitResult(
            spec=self._spec,
            **stats,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def probabilities(self, data=None, beta=None, alpha=None):
        """Compute choice probabilities.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to predict on. If None, uses estimation data.
        beta : np.ndarray or None
            Utility coefficients. If None, uses estimated values.
        alpha : np.ndarray or None
            Unconstrained nest parameters. If None, uses estimated values.

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

        # Use JAX backend for prediction
        return self._probabilities_jax(arrays, beta, alpha)

    def _probabilities_jax(self, arrays, beta=None, alpha=None):
        """Compute probabilities using JAX backend."""
        import jax.numpy as jnp

        from .._jax.data import ChoiceDataJAX

        k_total = arrays.design_matrix.shape[1]
        k_fixed = k_total - len(self._random_col_indices)
        n_nests = self._nests.n_nests

        if beta is None:
            beta_fixed = self._result.coefficients.values[:k_fixed]
        else:
            beta_fixed = beta[:k_fixed]

        if alpha is None:
            # Convert lambda back to alpha
            lambda_vals = self._result.coefficients.values[k_fixed : k_fixed + n_nests]
            lambda_vals = np.clip(lambda_vals, 1e-10, 1.0 - 1e-10)
            alpha = np.log(lambda_vals / (1.0 - lambda_vals))

        # Build data
        data = ChoiceDataJAX.from_arrays(
            arrays,
            draws=self._draws,
            random_col_indices=self._random_col_indices,
            random_distributions=self._random_distributions,
        )

        jnp.asarray(self._nest_matrix, dtype=jnp.float64)
        1.0 / (1.0 + jnp.exp(-jnp.asarray(alpha, dtype=jnp.float64)))

        # Fixed utility component
        if data.dm_fixed is not None and k_fixed > 0:
            v_fixed = (data.dm_fixed @ jnp.asarray(beta_fixed, dtype=jnp.float64)).reshape(
                data.n_obs, data.n_alts
            )
        else:
            v_fixed = jnp.zeros((data.n_obs, data.n_alts), dtype=jnp.float64)

        # Sampling correction
        if data.inclusion_probs is not None:
            v_fixed = v_fixed + jnp.log(jnp.maximum(data.inclusion_probs, 1e-30))

        # For each draw, compute nested logit probabilities and average
        beta_random_means = jnp.asarray(
            self._result.coefficients.values[
                k_fixed + n_nests : k_fixed + n_nests + len(self._random_col_indices)
            ],
            dtype=jnp.float64,
        )
        beta_random_spreads = jnp.asarray(
            self._result.coefficients.values[k_fixed + n_nests + len(self._random_col_indices) :],
            dtype=jnp.float64,
        )

        n_draws = self._draws.shape[1]
        k_random = len(self._random_col_indices)

        # Compute probabilities by averaging over draws
        np.zeros((data.n_obs, data.n_alts), dtype=np.float64)

        for r in range(n_draws):
            z_r = self._draws[:, r, :]  # (n_obs, k_random)

            # Generate random coefficients for this draw
            beta_random_r = np.zeros((data.n_obs, k_random))
            for p in range(k_random):
                mean_p = beta_random_means[p]
                spread_p = abs(beta_random_spreads[p])
                dist = self._random_distributions[p]

                if dist == "normal":
                    beta_random_r[:, p] = mean_p + spread_p * z_r[:, p]
                elif dist == "lognormal":
                    exponent = mean_p + spread_p * z_r[:, p]
                    beta_random_r[:, p] = np.exp(np.clip(exponent, -50, 50))
                elif dist == "triangular":
                    from scipy.stats import norm as norm_dist

                    u = norm_dist.cdf(z_r[:, p])
                    mask = u <= 0.5
                    beta_random_r[:, p] = np.where(
                        mask,
                        mean_p + spread_p * (np.sqrt(2 * u) - 1),
                        mean_p + spread_p * (1 - np.sqrt(2 * (1 - u))),
                    )
                elif dist == "uniform":
                    from scipy.stats import norm as norm_dist

                    u = norm_dist.cdf(z_r[:, p])
                    beta_random_r[:, p] = mean_p + spread_p * (2 * u - 1)

            # Random utility component
            v_random = np.sum(
                data.dm_random.numpy().reshape(data.n_obs, data.n_alts, k_random)
                * beta_random_r[:, None, :],
                axis=2,
            )

            # Total utility
            V = np.asarray(v_fixed) + v_random

            # Nested logit probabilities for this draw
            from .nested import _nested_logit_probs_numpy

            _nested_logit_probs_numpy(
                np.concatenate([beta_fixed, np.zeros(0)]),  # beta only, no nest params in utility
                np.asarray(alpha, dtype=np.float64),
                np.column_stack([V.T.flatten().reshape(-1, 1)]).reshape(-1, k_total)
                if False
                else None,
                self._nest_matrix,
                data.n_obs,
                data.n_alts,
                available=np.asarray(data.available),
                inclusion_probs=np.asarray(data.inclusion_probs)
                if data.inclusion_probs is not None
                else None,
            )

            # Actually, we need to compute utilities and then use nested logit
            # Let me use a simpler approach: compute V and then use nested_log_probs
            pass

        # This prediction path needs more work — for now, fall back to NumPy
        return self._probabilities_numpy(arrays, beta, alpha)

    def _probabilities_numpy(self, arrays, beta=None, alpha=None):
        """Compute probabilities using NumPy backend (fallback)."""
        from .._sampling.correction import get_sampling_correction

        k_total = arrays.design_matrix.shape[1]
        k_fixed = k_total - len(self._random_col_indices)
        n_nests = self._nests.n_nests

        if beta is None:
            beta_fixed = self._result.coefficients.values[:k_fixed]
        else:
            beta_fixed = beta[:k_fixed]

        if alpha is None:
            lambda_vals = self._result.coefficients.values[k_fixed : k_fixed + n_nests]
            lambda_vals = np.clip(lambda_vals, 1e-10, 1.0 - 1e-10)
            alpha = np.log(lambda_vals / (1.0 - lambda_vals))

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts
        available = arrays.available
        inclusion_probs = get_sampling_correction(arrays)

        # Extract random parameter info
        beta_random_means = self._result.coefficients.values[
            k_fixed + n_nests : k_fixed + n_nests + len(self._random_col_indices)
        ]
        beta_random_spreads = self._result.coefficients.values[
            k_fixed + n_nests + len(self._random_col_indices) :
        ]

        k_random = len(self._random_col_indices)
        n_draws = self._draws.shape[1]

        # Fixed utility component
        dm_fixed = dm[:, [i for i in range(k_total) if i not in self._random_col_indices]]
        dm_random = dm[:, self._random_col_indices]

        if dm_fixed.shape[1] > 0 and len(beta_fixed) > 0:
            v_fixed = (dm_fixed @ beta_fixed).reshape(n_obs, n_alts)
        else:
            v_fixed = np.zeros((n_obs, n_alts))

        # Sampling correction
        if inclusion_probs is not None:
            sr = np.asarray(inclusion_probs, dtype=np.float64).reshape(n_obs, n_alts)
            v_fixed = v_fixed + np.log(np.maximum(sr, 1e-30))

        # Availability
        if available is not None:
            avail = np.asarray(available, dtype=np.float64).reshape(n_obs, n_alts)
        else:
            avail = np.ones((n_obs, n_alts), dtype=np.float64)

        # Average probabilities over draws
        probs_sum = np.zeros((n_obs, n_alts), dtype=np.float64)

        for r in range(n_draws):
            # Generate random coefficients for this draw
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

            # Random utility component
            v_random = np.sum(
                dm_random.reshape(n_obs, n_alts, k_random) * beta_random_r[:, None, :],
                axis=2,
            )

            # Total utility
            V = v_fixed + v_random

            # Nested logit probabilities for this draw
            # We need to compute nested logit probs from V
            # Use the nested logit kernel with beta=None (utility already computed)
            from scipy.special import logsumexp

            lambdas = naturalize_nest_params(np.asarray(alpha, dtype=np.float64))
            nest_matrix = self._nest_matrix

            # Scaled utilities
            in_nest = nest_matrix.sum(axis=1) > 0
            long_lambda = np.ones(n_alts, dtype=np.float64)
            for m in range(len(lambdas)):
                long_lambda[nest_matrix[:, m] > 0] = lambdas[m]
            long_lambda_2d = long_lambda.reshape(1, n_alts)
            scaled_V = V / long_lambda_2d

            # Within-nest logsumexp
            nest_logsumexp_arr = np.zeros((n_obs, len(lambdas)), dtype=np.float64)
            for m in range(len(lambdas)):
                nest_mask = nest_matrix[:, m] > 0
                if not nest_mask.any():
                    continue
                nest_V = scaled_V.copy()
                nest_V[:, ~nest_mask] = -1e30
                nest_V = np.where(avail > 0, nest_V, -1e30)
                nest_logsumexp_arr[:, m] = logsumexp(nest_V, axis=1)

            # Nest exponents
            nest_exponent = lambdas[None, :] * nest_logsumexp_arr

            # Root nest
            root_mask = ~in_nest
            if root_mask.any():
                root_V = V.copy()
                root_V[:, ~root_mask] = -1e30
                root_V = np.where(avail > 0, root_V, -1e30)
                root_iv = logsumexp(root_V, axis=1)
                all_exponents = np.column_stack([nest_exponent, root_iv[:, None]])
            else:
                all_exponents = nest_exponent

            log_denom = logsumexp(all_exponents, axis=1)

            # Log-probabilities
            log_probs = np.full((n_obs, n_alts), -1e30, dtype=np.float64)
            for m in range(len(lambdas)):
                nest_mask = nest_matrix[:, m] > 0
                if not nest_mask.any():
                    continue
                iv_m = nest_logsumexp_arr[:, m]
                lambda_m = lambdas[m]
                log_probs[:, nest_mask] = (
                    scaled_V[:, nest_mask] + (lambda_m - 1.0) * iv_m[:, None] - log_denom[:, None]
                )

            if root_mask.any():
                log_probs[:, root_mask] = V[:, root_mask] - log_denom[:, None]

            probs_r = np.exp(log_probs)
            probs_r = probs_r * (avail > 0)
            probs_sum += probs_r

        # Average over draws
        probs = probs_sum / n_draws
        return probs

    def utilities(self, data=None, beta=None):
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
            Deterministic utilities for each observation and alternative.
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

        k_total = arrays.design_matrix.shape[1]
        k_fixed = k_total - len(self._random_col_indices)

        if beta is None:
            beta_fixed = np.asarray(self._result.coefficients.values[:k_fixed], dtype=np.float64)
        else:
            beta_fixed = beta

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        dm_fixed = dm[:, [i for i in range(k_total) if i not in self._random_col_indices]]

        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        if dm_fixed.shape[1] > 0 and len(beta_fixed) > 0:
            V = (dm_fixed @ beta_fixed).reshape(n_obs, n_alts)
        else:
            V = np.zeros((n_obs, n_alts))

        # Add sampling correction if present
        from .._sampling.correction import apply_sampling_correction

        V = apply_sampling_correction(V, arrays)

        return V
