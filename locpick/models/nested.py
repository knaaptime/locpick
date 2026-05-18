"""Nested logit model for location choice estimation.

This module provides the ``NestedLogit`` class and supporting functions for
estimating nested logit models. The nested logit generalises the MNL by
grouping alternatives into nests, allowing correlated error terms within
each nest.

Mathematical formulation
------------------------
For alternative :math:`j` in nest :math:`m`, the choice probability is:

.. math::

    P_{ij} = \\frac{\\exp(V_{ij}/\\lambda_m)
             \\left(\\sum_{k \\in C_m} \\exp(V_{ik}/\\lambda_m)\\right)^{\\lambda_m - 1}}
            {\\sum_{m'} \\left(\\sum_{k \\in C_{m'}} \\exp(V_{ik}/\\lambda_{m'})\\right)^{\\lambda_{m'}}}

where :math:`\\lambda_m \\in (0, 1]` is the dissimilarity (nest) parameter
for nest :math:`m`. When :math:`\\lambda_m = 1` for all nests, the model
reduces to MNL.

Nest parameters are estimated via a logistic transform to enforce the
constraint :math:`\\lambda_m \\in (0, 1]`:

.. math::

    \\lambda_m = \\frac{1}{1 + \\exp(-\\alpha_m)}

where :math:`\\alpha_m` is the unconstrained parameter estimated by the
optimizer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from locpick._jax.objective import Objective
from locpick._solvers import Solver, SolverResult
from locpick.data.arrays import ChoiceArrays
from locpick.data.problem import EstimationProblem
from locpick.models.base import BaseChoiceModel, _compute_fit_statistics, _compute_null_ll
from locpick.results.fit_result import FitResult

# ---------------------------------------------------------------------------
# Nest specification
# ---------------------------------------------------------------------------


@dataclass
class NestSpec:
    """Specification of a nest in a nested logit model.

    Parameters
    ----------
    name : str
        Human-readable name for the nest (e.g., ``"transit"``, ``"auto"``).
    alt_ids : list of int
        Alternative IDs belonging to this nest.
    alpha : float or None
        Initial value for the unconstrained nest parameter
        (logit of the dissimilarity parameter). None means use 0.0
        (which corresponds to :math:`\\lambda = 0.5`).
    """

    name: str
    alt_ids: list[int]
    alpha: Optional[float] = None


@dataclass
class NestingTree:
    """Hierarchical nesting structure for nested logit models.

    A two-level nesting tree where each alternative belongs to exactly
    one nest. Alternatives not assigned to any nest are placed in a
    "root" nest with :math:`\\lambda = 1` (i.e., MNL behavior).

    Parameters
    ----------
    nests : list of NestSpec
        The nests in the tree. Each nest has a name, a list of
        alternative IDs, and an optional initial value for the
        unconstrained nest parameter.
    """

    nests: list[NestSpec]

    def __post_init__(self) -> None:
        """Validate the nesting tree."""
        if not self.nests:
            raise ValueError("NestingTree must have at least one nest.")

        # Check for duplicate alternative IDs across nests
        all_alt_ids = []
        for nest in self.nests:
            all_alt_ids.extend(nest.alt_ids)
        if len(all_alt_ids) != len(set(all_alt_ids)):
            raise ValueError(
                "Each alternative can belong to at most one nest. "
                "Found duplicate alternative IDs across nests."
            )

    @property
    def n_nests(self) -> int:
        """Number of nests."""
        return len(self.nests)

    @property
    def nest_names(self) -> list[str]:
        """Names of the nests."""
        return [n.name for n in self.nests]

    @property
    def all_alt_ids(self) -> list[int]:
        """All alternative IDs assigned to nests."""
        ids = []
        for nest in self.nests:
            ids.extend(nest.alt_ids)
        return ids

    def build_nest_matrix(self, alt_ids: list[int]) -> np.ndarray:
        """Build the nest membership matrix.

        Parameters
        ----------
        alt_ids : list of int
            The full list of alternative IDs in the choice set,
            in the order they appear in the design matrix.

        Returns
        -------
        np.ndarray
            Matrix of shape ``(n_rows, n_nests)`` where ``n_rows`` is
            the total number of (obs, alt) rows. Each column corresponds
            to a nest, with 1.0 if the alternative belongs to that nest
            and 0.0 otherwise. Alternatives not in any nest get a row
            of zeros (they are in the implicit root nest).
        """
        n_alts = len(alt_ids)
        n_nests = self.n_nests

        # Build per-alternative nest membership
        alt_to_nest = np.zeros((n_alts, n_nests), dtype=np.float64)
        for j, nest in enumerate(self.nests):
            for alt_id in nest.alt_ids:
                if alt_id in alt_ids:
                    idx = alt_ids.index(alt_id)
                    alt_to_nest[idx, j] = 1.0

        return alt_to_nest

    def initial_alphas(self) -> np.ndarray:
        """Return initial values for the unconstrained nest parameters.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_nests,)`` with initial alpha values.
        """
        return np.array([nest.alpha if nest.alpha is not None else 0.0 for nest in self.nests])


# ---------------------------------------------------------------------------
# Parameter transformation
# ---------------------------------------------------------------------------


def naturalize_nest_params(alpha: np.ndarray) -> np.ndarray:
    """Transform unconstrained nest parameters to natural (0, 1] scale.

    Parameters
    ----------
    alpha : np.ndarray
        Unconstrained nest parameters (logit scale).

    Returns
    -------
    np.ndarray
        Natural nest parameters :math:`\\lambda \\in (0, 1]`.
        Computed as :math:`\\lambda = 1 / (1 + \\exp(-\\alpha))`.
    """
    return 1.0 / (1.0 + np.exp(-alpha))


def constrain_nest_params(alpha: np.ndarray) -> np.ndarray:
    """Apply logistic transform to enforce :math:`\\lambda \\in (0, 1]`.

    Alias for :func:`naturalize_nest_params`.
    """
    return naturalize_nest_params(alpha)


# ---------------------------------------------------------------------------
# Nested logit probability kernel (NumPy)
# ---------------------------------------------------------------------------


def _nested_logit_probs_numpy(
    beta: np.ndarray,
    alpha: np.ndarray,
    design_matrix: np.ndarray,
    nest_matrix: np.ndarray,
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute nested logit probabilities (NumPy backend).

    Parameters
    ----------
    beta : np.ndarray, shape (k,)
        Utility coefficients.
    alpha : np.ndarray, shape (n_nests,)
        Unconstrained nest parameters (logit scale).
    design_matrix : np.ndarray, shape (n_obs * n_alts, k)
        Design matrix.
    nest_matrix : np.ndarray, shape (n_alts, n_nests)
        Alternative-to-nest membership matrix.
    n_obs : int
        Number of observations.
    n_alts : int
        Number of alternatives per observation.
    available : np.ndarray or None, shape (n_obs, n_alts)
        Availability matrix. 1.0 if available, 0.0 if not.
    inclusion_probs : np.ndarray or None, shape (n_obs, n_alts)
        Sampling rates for correction.

    Returns
    -------
    np.ndarray, shape (n_obs, n_alts)
        Choice probabilities for each (obs, alt) pair.
    """
    from scipy.special import logsumexp

    # Natural nest parameters: lambda_m in (0, 1]
    lambdas = naturalize_nest_params(alpha)  # (n_nests,)

    # Step 1: systematic utility
    utilities = (design_matrix @ beta).reshape(n_obs, n_alts)

    # Step 2: sampling correction
    if inclusion_probs is not None:
        sr = np.asarray(inclusion_probs, dtype=np.float64).reshape(n_obs, n_alts)
        utilities = utilities + np.log(np.maximum(sr, 1e-30))

    # Step 3: availability masking
    if available is not None:
        avail = np.asarray(available, dtype=np.float64).reshape(n_obs, n_alts)
    else:
        avail = np.ones((n_obs, n_alts), dtype=np.float64)

    from locpick._kernels.constants import NEG_INF

    utilities = np.where(avail > 0, utilities, NEG_INF)

    # Step 4: scaled utilities V_ij / lambda_m
    in_nest = nest_matrix.sum(axis=1) > 0  # (n_alts,) bool
    long_lambda = np.ones(n_alts, dtype=np.float64)
    for m in range(len(lambdas)):
        long_lambda[nest_matrix[:, m] > 0] = lambdas[m]
    long_lambda = long_lambda.reshape(1, n_alts)
    scaled_utilities = utilities / long_lambda  # (n_obs, n_alts)

    # Step 5: within-nest logsumexp for each nest
    n_nests = len(lambdas)
    nest_logsumexp = np.zeros((n_obs, n_nests), dtype=np.float64)
    for m in range(n_nests):
        nest_mask = nest_matrix[:, m] > 0  # (n_alts,)
        if not nest_mask.any():
            continue
        nest_utils = scaled_utilities.copy()
        nest_utils[:, ~nest_mask] = NEG_INF
        nest_utils = np.where(avail > 0, nest_utils, NEG_INF)
        nest_logsumexp[:, m] = logsumexp(nest_utils, axis=1)

    # Step 6: nest-level exponents
    nest_exponent = lambdas[None, :] * nest_logsumexp  # (n_obs, n_nests)

    # Step 7: root nest inclusive value (if any root alternatives)
    root_mask = ~in_nest  # (n_alts,)
    if root_mask.any():
        root_utils = utilities.copy()
        root_utils[:, ~root_mask] = NEG_INF
        root_utils = np.where(avail > 0, root_utils, NEG_INF)
        root_iv = logsumexp(root_utils, axis=1)  # (n_obs,)
        root_exponent = root_iv  # lambda=1, so exponent = 1 * IV
        all_exponents = np.column_stack([nest_exponent, root_exponent[:, None]])
    else:
        all_exponents = nest_exponent

    log_denom = logsumexp(all_exponents, axis=1)  # (n_obs,)

    # Step 8: unconditional probabilities (vectorised)
    # P(j) = exp(V_ij / lambda_m + (lambda_m - 1) * IV_m - log_denom)
    # For root nest: lambda=1, IV=V, so P(j) = exp(V_ij - log_denom)
    log_probs = np.full((n_obs, n_alts), NEG_INF, dtype=np.float64)

    for m in range(n_nests):
        nest_mask = nest_matrix[:, m] > 0  # (n_alts,)
        if not nest_mask.any():
            continue
        iv_m = nest_logsumexp[:, m]  # (n_obs,)
        lambda_m = lambdas[m]
        # Vectorised assignment for all alts in this nest
        log_probs[:, nest_mask] = (
            scaled_utilities[:, nest_mask] + (lambda_m - 1.0) * iv_m[:, None] - log_denom[:, None]
        )

    if root_mask.any():
        log_probs[:, root_mask] = utilities[:, root_mask] - log_denom[:, None]

    probs = np.exp(log_probs)
    probs = probs * (avail > 0)

    return probs


def _nested_logit_ll_numpy(
    beta: np.ndarray,
    alpha: np.ndarray,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    nest_matrix: np.ndarray,
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute nested logit log-likelihood (NumPy backend).

    Parameters
    ----------
    beta : np.ndarray, shape (k,)
        Utility coefficients.
    alpha : np.ndarray, shape (n_nests,)
        Unconstrained nest parameters.
    design_matrix, chosen, nest_matrix, n_obs, n_alts, available, inclusion_probs, weights
        See :func:`_nested_logit_probs_numpy`.

    Returns
    -------
    float
        Log-likelihood value.
    """
    probs = _nested_logit_probs_numpy(
        beta,
        alpha,
        design_matrix,
        nest_matrix,
        n_obs,
        n_alts,
        available=available,
        inclusion_probs=inclusion_probs,
    )

    # Chosen probabilities
    chosen_probs = (probs * chosen.reshape(n_obs, n_alts)).sum(axis=1)

    # Avoid log(0)
    chosen_probs = np.maximum(chosen_probs, 1e-30)
    log_chosen = np.log(chosen_probs)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(n_obs)
        log_chosen = log_chosen * w

    return float(log_chosen.sum())


def _nested_logit_gradient_numpy(
    beta: np.ndarray,
    alpha: np.ndarray,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    nest_matrix: np.ndarray,
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute nested logit gradient via finite differences (NumPy backend).

    This is a fallback gradient that uses finite differences. A proper
    analytical gradient will be implemented in a future version.

    Parameters
    ----------
    beta, alpha, design_matrix, chosen, nest_matrix, n_obs, n_alts, available, inclusion_probs, weights
        See :func:`_nested_logit_ll_numpy`.

    Returns
    -------
    np.ndarray, shape (k + n_nests,)
        Gradient of the log-likelihood with respect to [beta, alpha].
    """
    params = np.concatenate([beta, alpha])
    eps = 1e-5
    n_params = len(params)
    grad = np.zeros(n_params)

    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps

        beta_plus, alpha_plus = params_plus[: len(beta)], params_plus[len(beta) :]
        beta_minus, alpha_minus = params_minus[: len(beta)], params_minus[len(beta) :]

        ll_plus = _nested_logit_ll_numpy(
            beta_plus,
            alpha_plus,
            design_matrix,
            chosen,
            nest_matrix,
            n_obs,
            n_alts,
            available=available,
            inclusion_probs=inclusion_probs,
            weights=weights,
        )
        ll_minus = _nested_logit_ll_numpy(
            beta_minus,
            alpha_minus,
            design_matrix,
            chosen,
            nest_matrix,
            n_obs,
            n_alts,
            available=available,
            inclusion_probs=inclusion_probs,
            weights=weights,
        )

        grad[i] = (ll_plus - ll_minus) / (2 * eps)

    return grad


# ---------------------------------------------------------------------------
# NestedLogit model class
# ---------------------------------------------------------------------------


class NestedMNL(BaseChoiceModel):
    """Nested logit model for location choice estimation.

    This model generalises the multinomial logit by grouping alternatives
    into nests, allowing correlated error terms within each nest.

    Parameters
    ----------
    data : ChoiceTable
        The choice data to estimate on.
    formula : str, optional
        A formulaic formula string for the utility function.
    spec : ModelSpec, optional
        A ModelSpec object defining the utility function.
    nests : NestingTree
        The nesting structure.
    solver : str or Solver, optional
        Solver name or instance. Default ``"lbfgs"``.
    solver_options : dict, optional
        Additional options passed to the solver constructor.

    Examples
    --------
    >>> from locpick import ChoiceTable, MultinomialLogit
    >>> from locpick.models.nested import NestedLogit, NestSpec, NestingTree
    >>> nests = NestingTree([
    ...     NestSpec("transit", alt_ids=[0, 1, 2]),
    ...     NestSpec("auto", alt_ids=[3, 4]),
    ... ])
    >>> model = NestedLogit(ct, formula="cost + time", nests=nests)
    >>> result = model.fit()
    """

    def __init__(
        self,
        data,
        formula: Optional[str] = None,
        spec=None,
        nests: Optional[NestingTree] = None,
        weights: Optional[Union[str, np.ndarray]] = None,
        availability: Optional[Union[str, np.ndarray]] = None,
        solver: Union[str, Solver] = "lbfgs",
        solver_options: Optional[dict] = None,
        backend: Optional[str] = None,
    ):
        if nests is None:
            raise ValueError("NestedLogit requires a 'nests' argument (NestingTree).")

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

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def _pre_fit(self, arrays: ChoiceArrays) -> None:
        """Build nest matrix before objective construction."""
        alt_ids = list(range(arrays.n_alts))
        self._nest_matrix = self._nests.build_nest_matrix(alt_ids)

    def _get_solver_inputs(self, arrays: ChoiceArrays):
        """Get initial values, param names, bounds, and fixed mask.

        Extends the base class to include nest parameters.
        """
        k = arrays.design_matrix.shape[1]
        n_nests = self._nests.n_nests
        x0 = np.concatenate([np.zeros(k), self._nests.initial_alphas()])
        param_names = list(arrays.param_names) + [
            f"nest_{name}" for name in self._nests.nest_names
        ]
        return x0, param_names, None, None

    def _build_objective(self, arrays: ChoiceArrays) -> Objective:
        """Build optimization objective for nested logit estimation."""
        nest_matrix = self._nest_matrix

        # Try JAX backend first (default when available)
        backend = (self._backend or os.environ.get("CHOICEMODELS_NESTED_BACKEND", "")).lower()
        if backend != "numpy":
            try:
                from locpick._jax.builders import build_nested_objective

                return build_nested_objective(arrays, nest_matrix)
            except ImportError:
                pass

        # NumPy backend
        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        chosen = np.asarray(arrays.chosen, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts
        available = arrays.available
        weights = arrays.weights

        from locpick._sampling.correction import get_sampling_correction

        inclusion_probs = get_sampling_correction(arrays)
        k = dm.shape[1]

        def ll_fn(params):
            beta = params[:k]
            alpha = params[k:]
            return _nested_logit_ll_numpy(
                beta,
                alpha,
                dm,
                chosen,
                nest_matrix,
                n_obs,
                n_alts,
                available=available,
                inclusion_probs=inclusion_probs,
                weights=weights,
            )

        def grad_fn(params):
            beta = params[:k]
            alpha = params[k:]
            return _nested_logit_gradient_numpy(
                beta,
                alpha,
                dm,
                chosen,
                nest_matrix,
                n_obs,
                n_alts,
                available=available,
                inclusion_probs=inclusion_probs,
                weights=weights,
            )

        return Objective.from_numpy(
            ll_fn=ll_fn,
            grad_fn=grad_fn,
            param_names=list(arrays.param_names)
            + [f"nest_{name}" for name in self._nests.nest_names],
        )

    def _build_fit_result(
        self,
        solver_result: SolverResult,
        arrays: ChoiceArrays,
    ) -> FitResult:
        """Build a FitResult from solver output."""
        k = arrays.design_matrix.shape[1]
        n_nests = self._nests.n_nests

        # Build a FitResult from solver output.
        # Store naturalized lambda values (not raw alpha) in coefficients.
        all_params = solver_result.coefficients
        beta = all_params[:k]
        alpha = all_params[k:]
        lambdas = naturalize_nest_params(alpha)

        # Parameter names
        param_names = list(arrays.param_names) + [
            f"lambda_{name}" for name in self._nests.nest_names
        ]

        # Store naturalized values: beta + lambda (not alpha)
        display_params = np.concatenate([beta, lambdas])

        # Standard errors from inverse Hessian
        # The Hessian is in the alpha (unconstrained) space, so we need
        # to transform the nest-parameter standard errors via the delta method:
        #   SE(lambda) = |d(lambda)/d(alpha)| * SE(alpha)
        #   d(lambda)/d(alpha) = lambda * (1 - lambda) for logistic transform
        if solver_result.hessian is not None:
            try:
                se_alpha = np.sqrt(np.diag(solver_result.hessian))
                # Delta method: transform SE for nest parameters
                se_lambda = lambdas * (1.0 - lambdas) * se_alpha[k:]
                std_errors = np.concatenate([se_alpha[:k], se_lambda])
            except Exception:
                std_errors = np.full(len(display_params), np.nan)
        else:
            # Compute Hessian lazily via objective if available
            if hasattr(self, "_objective") and self._objective is not None:
                try:
                    hess = self._objective.hessian(all_params)
                    se_alpha = np.sqrt(np.diag(hess))
                    se_lambda = lambdas * (1.0 - lambdas) * se_alpha[k:]
                    std_errors = np.concatenate([se_alpha[:k], se_lambda])
                except Exception:
                    std_errors = np.full(len(display_params), np.nan)
            else:
                std_errors = np.full(len(display_params), np.nan)

        # Build result using shared helper
        coefficients = pd.Series(display_params, index=param_names, name="coefficient")
        std_err_series = pd.Series(std_errors, index=param_names, name="std_error")
        ll = solver_result.log_likelihood
        ll_null = _compute_null_ll(arrays)

        stats = _compute_fit_statistics(
            ll=ll,
            ll_null=ll_null,
            n_obs=arrays.n_obs,
            n_params=len(display_params),
            n_alts=arrays.n_alts,
            coefficients=coefficients,
            std_errors=std_err_series,
            model_type="Nested Logit",
            solver_name=solver_result.solver_name,
            solver_result_raw=solver_result.raw,
        )

        return FitResult(
            spec=self._spec,
            **stats,
        )

    def _observation_scores(self, arrays) -> np.ndarray:
        """Compute observation-level score vectors via numerical differentiation.

        Parameters
        ----------
        arrays : ChoiceArrays
            The estimation data arrays.

        Returns
        -------
        np.ndarray, shape (n_obs, n_params)
            Score vector for each observation.
        """
        if self._result is None:
            raise RuntimeError("Model must be estimated first.")

        cache_key = id(arrays)
        if cache_key in self._observation_scores_cache:
            return self._observation_scores_cache[cache_key]

        eps = 1e-5
        k = arrays.design_matrix.shape[1]
        n_nests = self._nests.n_nests
        n_params = k + n_nests
        n_obs = arrays.n_obs

        # Full parameter vector: [beta, alpha]
        full_params = self._result.coefficients.values.copy()
        beta_hat = full_params[:k]
        alpha_hat = full_params[k : k + n_nests]

        # Chosen indicator
        chosen = np.asarray(arrays.chosen, dtype=np.float64).reshape(n_obs, arrays.n_alts)

        # Base probabilities and per-observation LL
        probs_base = self.probabilities(data=None, beta=beta_hat, alpha=alpha_hat)
        np.log(np.maximum(np.sum(probs_base * chosen, axis=1), 1e-30))

        scores = np.zeros((n_obs, n_params))

        for j in range(n_params):
            params_plus = full_params.copy()
            params_plus[j] += eps
            params_minus = full_params.copy()
            params_minus[j] -= eps

            # Split perturbed params into beta and alpha
            beta_plus = params_plus[:k]
            alpha_plus = params_plus[k : k + n_nests]
            beta_minus = params_minus[:k]
            alpha_minus = params_minus[k : k + n_nests]

            probs_plus = self.probabilities(data=None, beta=beta_plus, alpha=alpha_plus)
            probs_minus = self.probabilities(data=None, beta=beta_minus, alpha=alpha_minus)

            ll_plus = np.log(np.maximum(np.sum(probs_plus * chosen, axis=1), 1e-30))
            ll_minus = np.log(np.maximum(np.sum(probs_minus * chosen, axis=1), 1e-30))

            scores[:, j] = (ll_plus - ll_minus) / (2 * eps)

        self._observation_scores_cache[cache_key] = scores
        return scores

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
            k = arrays.design_matrix.shape[1]
            beta = np.asarray(self._result.coefficients.values[:k], dtype=np.float64)

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

        results = []
        for draw in range(n_draws):
            chosen_indices = np.array([rng.choice(n_alts, p=probs[i]) for i in range(n_obs)])
            chosen_alts = alt_ids[np.arange(n_obs), chosen_indices]
            chosen_probs = probs[np.arange(n_obs), chosen_indices]

            for i in range(n_obs):
                results.append(
                    {
                        "draw": draw,
                        ct.obs_id_col: obs_ids[i],
                        ct.alt_id_col: chosen_alts[i],
                        "probability": chosen_probs[i],
                    }
                )

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Marginal Effects
    # ------------------------------------------------------------------

    def marginal_effect(self, data=None, variable: Optional[str] = None) -> pd.Series:
        """Compute direct marginal effects for a variable.

        For Nested Logit, the direct marginal effect of variable *x* in
        alternative *i* is approximately :math:`(1 - P_{qi}) \\beta_x`.

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

        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            data.to_arrays(
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

        For Nested Logit, the cross-marginal effect of variable *x* in
        alternative *j* with respect to a change in alternative *i* is
        approximately :math:`-P_i \\beta_x`.

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

        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            data.to_arrays(
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

        The direct elasticity measures the percentage change in the
        probability of choosing an alternative with respect to a
        percentage change in a variable.

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

        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            data.to_arrays(
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

        The cross-elasticity measures the percentage change in the
        probability of choosing one alternative with respect to a
        percentage change in a variable of another alternative.

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

        ct = self._data
        if data is not None:
            if not isinstance(data, ChoiceTable):
                raise TypeError("data must be a ChoiceTable")
            data.to_arrays(
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
        H_inv = self._get_hessian_inverse()

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

        H_inv = self._get_hessian_inverse()

        if H_inv is None:
            try:
                return np.linalg.inv(B_clustered)
            except np.linalg.LinAlgError:
                return np.full_like(B_clustered, np.nan)

        return H_inv @ B_clustered @ H_inv

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
            arrays = data.to_arrays(
                formula=self._spec.formula,
                spec=self._spec if self._spec.formula is None else None,
            )

        if beta is None:
            beta = self._result.coefficients.values[: arrays.design_matrix.shape[1]]
        if alpha is None:
            k = arrays.design_matrix.shape[1]
            # Coefficients store lambda values; convert back to alpha
            # alpha = -log(1/lambda - 1) = log(lambda / (1 - lambda))
            lambda_vals = self._result.coefficients.values[k : k + self._nests.n_nests]
            # Clamp lambda to avoid log(0) or log(inf)
            lambda_vals = np.clip(lambda_vals, 1e-10, 1.0 - 1e-10)
            alpha = np.log(lambda_vals / (1.0 - lambda_vals))

        alt_ids = list(range(arrays.n_alts))
        nest_matrix = self._nests.build_nest_matrix(alt_ids)

        # Resolve canonical sampling correction tensor.
        from locpick._sampling.correction import get_sampling_correction

        sampling_correction = get_sampling_correction(arrays)

        return _nested_logit_probs_numpy(
            beta,
            alpha,
            np.asarray(arrays.design_matrix, dtype=np.float64),
            nest_matrix,
            arrays.n_obs,
            arrays.n_alts,
            available=arrays.available,
            inclusion_probs=sampling_correction,
        )
