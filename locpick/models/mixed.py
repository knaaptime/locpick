"""Mixed logit (random coefficients) model for location choice estimation.

This module provides the ``MixedLogit`` class and supporting functions for
estimating mixed logit models. The mixed logit generalises the MNL by
allowing some or all coefficients to follow random distributions, capturing
unobserved taste heterogeneity across decision-makers.

Mathematical formulation
------------------------
The mixed logit probability for alternative :math:`j` by decision-maker
:math:`n` is:

.. math::

    P_{nj} = \\int L_{nj}(\\beta) f(\\beta \\mid \\theta) \\, d\\beta

where :math:`L_{nj}(\\beta)` is the MNL probability conditional on
:math:`\\beta`, and :math:`f(\\beta \\mid \\theta)` is the mixing
distribution parameterised by :math:`\\theta`.

For :math:`R` draws from the mixing distribution, the simulated
log-likelihood is:

.. math::

    \\text{SLL} = \\sum_{n=1}^{N} w_n \\log \\left(
        \\frac{1}{R} \\sum_{r=1}^{R} L_{nj}(\\beta^r)
    \\right)

where :math:`\\beta^r` are draws from :math:`f(\\beta \\mid \\theta)`.

Supported distributions
----------------------
- **Normal**: :math:`\\beta \\sim N(\\mu, \\sigma^2)` — two parameters per
  random coefficient (mean and standard deviation).
- **Lognormal**: :math:`\\beta = \\exp(\\mu + \\sigma \\cdot z)` where
  :math:`z \\sim N(0, 1)` — ensures :math:`\\beta > 0`. Two parameters
  per random coefficient.
- **Triangular**: :math:`\\beta \\sim \\text{Triangular}(\\mu-\\sigma, \\mu+\\sigma)`
  with mode at :math:`\\mu` — bounded support, peaked at the mean.
  Two parameters per random coefficient.
- **Uniform**: :math:`\\beta \\sim \\text{Uniform}(\\mu - \\sigma, \\mu + \\sigma)`
  — bounded support, constant density. Two parameters per random coefficient.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from locpick.data.arrays import ChoiceArrays
from locpick.data.problem import EstimationProblem
from locpick.models.base import BaseChoiceModel
from locpick.results.fit_result import FitResult
from locpick._jax.objective import Objective
from locpick._solvers import Solver, SolverResult, get_solver
from locpick._kernels.constants import NEG_INF
from locpick.spec import ModelSpec, ParamRef

# ---------------------------------------------------------------------------
# Distribution specifications
# ---------------------------------------------------------------------------


@dataclass
class ParamDistribution:
    """Distribution specification for a random parameter (mixed logit).

    Parameters
    ----------
    distribution : str
        Distribution name: ``"normal"``, ``"lognormal"``,
        ``"triangular"``, or ``"uniform"``.
    param : ParamRef or str
        The parameter to assign a random distribution to. Can be a
        ``ParamRef`` or a string (parameter name).
    """

    distribution: str
    param: Union[ParamRef, str]

    def __post_init__(self) -> None:
        valid = {"normal", "lognormal", "triangular", "uniform"}
        if self.distribution not in valid:
            raise ValueError(
                f"Unknown distribution '{self.distribution}'. Must be one of {valid}."
            )

    @property
    def param_name(self) -> str:
        """Return the parameter name as a string."""
        if isinstance(self.param, ParamRef):
            return self.param.name
        return self.param

    @property
    def n_params(self) -> int:
        """Number of distribution parameters (always 2: mean and spread)."""
        return 2


# ---------------------------------------------------------------------------
# Draw manager
# ---------------------------------------------------------------------------


def _halton_sequence(n: int, base: int) -> np.ndarray:
    """Generate a Halton sequence of length ``n`` for the given prime base.

    Parameters
    ----------
    n : int
        Number of draws.
    base : int
        Prime base for the Halton sequence (e.g., 2, 3, 5, 7, ...).

    Returns
    -------
    np.ndarray, shape (n,)
        Halton sequence values in [0, 1).
    """
    result = np.zeros(n)
    for i in range(n):
        f, r = 1.0, 0.0
        idx = i + 1  # 1-indexed
        while idx > 0:
            idx, remainder = divmod(idx, base)
            f = f / base
            r = r + f * remainder
        result[i] = r
    return result


def generate_halton_draws(
    n_obs: int,
    n_draws: int,
    n_random_params: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate Halton draws for simulated maximum likelihood.

    Uses standard normal draws derived from Halton sequences with
    prime bases for each random parameter.

    Parameters
    ----------
    n_obs : int
        Number of observations (decision-makers).
    n_draws : int
        Number of draws per observation per random parameter.
    n_random_params : int
        Number of random parameters.
    seed : int
        Random seed for shuffling (to avoid correlation across parameters).

    Returns
    -------
    np.ndarray, shape (n_obs, n_draws, n_random_params)
        Standard normal draws for simulated integration.
    """
    # First n_random_params primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if n_random_params > len(primes):
        raise ValueError(
            f"Maximum {len(primes)} random parameters supported with "
            f"Halton draws. Got {n_random_params}."
        )

    rng = np.random.default_rng(seed)
    draws = np.zeros((n_obs, n_draws, n_random_params))

    for p in range(n_random_params):
        # Generate Halton sequence for this prime base
        halton = _halton_sequence(n_obs * n_draws, primes[p])
        # Shuffle to reduce correlation across observations
        rng.shuffle(halton)
        # Transform to standard normal via inverse CDF
        from scipy.stats import norm

        z = norm.ppf(halton)
        draws[:, :, p] = z.reshape(n_obs, n_draws)

    return draws


def generate_random_draws(
    n_obs: int,
    n_draws: int,
    n_random_params: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate pseudo-random standard normal draws.

    Parameters
    ----------
    n_obs : int
        Number of observations.
    n_draws : int
        Number of draws per observation per random parameter.
    n_random_params : int
        Number of random parameters.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray, shape (n_obs, n_draws, n_random_params)
        Standard normal draws.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_obs, n_draws, n_random_params))


# ---------------------------------------------------------------------------
# Mixed logit probability kernel (NumPy)
# ---------------------------------------------------------------------------


def _apply_distribution(
    z: np.ndarray,
    mean: np.ndarray,
    spread: np.ndarray,
    distribution: str,
) -> np.ndarray:
    """Apply a mixing distribution to standard normal draws.

    Parameters
    ----------
    z : np.ndarray, shape (n_obs, n_draws)
        Standard normal draws.
    mean : np.ndarray, shape (n_obs,) or scalar
        Mean of the random coefficient.
    spread : np.ndarray, shape (n_obs,) or scalar
        Spread (standard deviation) of the random coefficient.
    distribution : str
        Distribution name: "normal", "lognormal", "triangular", "uniform".

    Returns
    -------
    np.ndarray, shape (n_obs, n_draws)
        Realised coefficient values for each draw.
    """
    if distribution == "normal":
        return mean[:, None] + spread[:, None] * z

    elif distribution == "lognormal":
        # beta = exp(mu + sigma * z), so log(beta) ~ N(mu, sigma^2)
        # Clip exponent to avoid overflow
        exponent = mean[:, None] + spread[:, None] * z
        return np.exp(np.clip(exponent, -50, 50))

    elif distribution == "triangular":
        from scipy.stats import norm as norm_dist

        u = norm_dist.cdf(z)
        mask = u <= 0.5
        return np.where(
            mask,
            mean[:, None] + spread[:, None] * (np.sqrt(2 * u) - 1),
            mean[:, None] + spread[:, None] * (1 - np.sqrt(2 * (1 - u))),
        )

    elif distribution == "uniform":
        from scipy.stats import norm as norm_dist

        u = norm_dist.cdf(z)
        return mean[:, None] + spread[:, None] * (2 * u - 1)

    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def _mixed_logit_probs_numpy(
    beta_fixed: np.ndarray,
    beta_random_means: np.ndarray,
    beta_random_spreads: np.ndarray,
    random_distributions: list[str],
    draws: np.ndarray,
    design_matrix: np.ndarray,
    random_col_indices: list[int],
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute mixed logit choice probabilities (NumPy backend).

    Parameters
    ----------
    beta_fixed : np.ndarray, shape (k_fixed,)
        Coefficients for fixed (non-random) parameters.
    beta_random_means : np.ndarray, shape (k_random,)
        Mean coefficients for random parameters.
    beta_random_spreads : np.ndarray, shape (k_random,)
        Spread (standard deviation) coefficients for random parameters.
    random_distributions : list of str
        Distribution name for each random parameter.
    draws : np.ndarray, shape (n_obs, n_draws, k_random)
        Standard normal draws for simulated integration.
    design_matrix : np.ndarray, shape (n_obs * n_alts, k_total)
        Design matrix (k_total = k_fixed + k_random).
    random_col_indices : list of int
        Column indices in the design matrix for random parameters.
    n_obs : int
        Number of observations.
    n_alts : int
        Number of alternatives per observation.
    available : np.ndarray or None, shape (n_obs, n_alts)
        Availability matrix.
    inclusion_probs : np.ndarray or None, shape (n_obs, n_alts)
        Sampling rates for correction.

    Returns
    -------
    np.ndarray, shape (n_obs, n_alts)
        Mixed logit choice probabilities (simulated).
    """
    from scipy.special import logsumexp

    n_draws = draws.shape[1]
    k_random = len(random_col_indices)

    # Build the full design matrix columns
    # Split into fixed and random columns
    all_col_indices = list(range(design_matrix.shape[1]))
    fixed_col_indices = [i for i in all_col_indices if i not in random_col_indices]

    dm_fixed = design_matrix[:, fixed_col_indices] if fixed_col_indices else None
    dm_random = design_matrix[:, random_col_indices]

    # Fixed utility component: V_fixed = X_fixed @ beta_fixed
    if dm_fixed is not None and len(beta_fixed) > 0:
        v_fixed = (dm_fixed @ beta_fixed).reshape(n_obs, n_alts)
    else:
        v_fixed = np.zeros((n_obs, n_alts))

    # Availability mask
    if available is not None:
        avail = np.asarray(available, dtype=np.float64).reshape(n_obs, n_alts)
    else:
        avail = np.ones((n_obs, n_alts), dtype=np.float64)

    # Sampling correction
    if inclusion_probs is not None:
        sr = np.asarray(inclusion_probs, dtype=np.float64).reshape(n_obs, n_alts)
        v_fixed = v_fixed + np.log(np.maximum(sr, 1e-30))

    # For each draw, compute conditional MNL probabilities
    # Then average across draws (simulated integration)
    log_probs_draws = np.zeros((n_obs, n_draws, n_alts), dtype=np.float64)

    for r in range(n_draws):
        # Realise random coefficients for this draw
        beta_r = np.zeros(k_random)
        for p in range(k_random):
            z_p = draws[:, r, p]  # (n_obs,)
            mean_p = np.full(n_obs, beta_random_means[p])
            spread_p = np.full(n_obs, beta_random_spreads[p])
            beta_r[p] = _apply_distribution(
                z_p[:, None],
                mean_p[:, None],
                spread_p[:, None],
                random_distributions[p],
            ).ravel()[0]  # scalar for this draw

        # Actually, we need per-observation random coefficients
        # beta_r[n, p] = mean_p + spread_p * z[n, r, p]
        beta_random_r = np.zeros((n_obs, k_random))
        for p in range(k_random):
            z_p = draws[:, r, p]  # (n_obs,)
            mean_p = beta_random_means[p]
            spread_p = beta_random_spreads[p]

            if random_distributions[p] == "normal":
                beta_random_r[:, p] = mean_p + spread_p * z_p
            elif random_distributions[p] == "lognormal":
                # Clip exponent to avoid overflow
                exponent = mean_p + spread_p * z_p
                beta_random_r[:, p] = np.exp(np.clip(exponent, -50, 50))
            elif random_distributions[p] == "triangular":
                from scipy.stats import norm as norm_dist

                u = norm_dist.cdf(z_p)
                mask = u <= 0.5
                beta_random_r[:, p] = np.where(
                    mask,
                    mean_p + spread_p * (np.sqrt(2 * u) - 1),
                    mean_p + spread_p * (1 - np.sqrt(2 * (1 - u))),
                )

            elif random_distributions[p] == "uniform":
                from scipy.stats import norm as norm_dist

                u = norm_dist.cdf(z_p)
                beta_random_r[:, p] = mean_p + spread_p * (2 * u - 1)

        # Random utility component: V_random[n, j] = X_random[n,j,:] @ beta_random[n,:]
        # dm_random: (n_obs * n_alts, k_random)
        # beta_random_r: (n_obs, k_random)
        v_random = np.sum(
            dm_random.reshape(n_obs, n_alts, k_random) * beta_random_r[:, None, :],
            axis=2,
        )

        # Total utility
        utilities = v_fixed + v_random  # (n_obs, n_alts)

        # Mask unavailable
        utilities = np.where(avail > 0, utilities, NEG_INF)

        # Log-probabilities via stable logsumexp
        log_sum_exp = logsumexp(utilities, axis=1)  # (n_obs,)
        log_probs = utilities - log_sum_exp[:, None]  # (n_obs, n_alts)

        log_probs_draws[:, r, :] = log_probs

    # Simulated mixed logit probability:
    # P(j) = (1/R) * sum_r L(j | beta^r)
    # In log-space: log P(j) = logsumexp_r(log L(j | beta^r)) - log(R)
    log_probs_sim = logsumexp(log_probs_draws, axis=1) - np.log(n_draws)
    probs = np.exp(log_probs_sim)

    # Zero out unavailable alternatives
    probs = probs * (avail > 0)

    return probs


def _mixed_logit_ll_numpy(
    beta_fixed: np.ndarray,
    beta_random_means: np.ndarray,
    beta_random_spreads: np.ndarray,
    random_distributions: list[str],
    draws: np.ndarray,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    random_col_indices: list[int],
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute mixed logit simulated log-likelihood (NumPy backend).

    Parameters
    ----------
    beta_fixed, beta_random_means, beta_random_spreads, random_distributions, draws, design_matrix, chosen, random_col_indices, n_obs, n_alts, available, inclusion_probs, weights
        See :func:`_mixed_logit_probs_numpy`.

    Returns
    -------
    float
        Simulated log-likelihood value.
    """
    from scipy.special import logsumexp

    n_draws = draws.shape[1]
    k_random = len(random_col_indices)

    # Build column indices
    all_col_indices = list(range(design_matrix.shape[1]))
    fixed_col_indices = [i for i in all_col_indices if i not in random_col_indices]

    dm_fixed = design_matrix[:, fixed_col_indices] if fixed_col_indices else None
    dm_random = design_matrix[:, random_col_indices]

    # Fixed utility component
    if dm_fixed is not None and len(beta_fixed) > 0:
        v_fixed = (dm_fixed @ beta_fixed).reshape(n_obs, n_alts)
    else:
        v_fixed = np.zeros((n_obs, n_alts))

    # Availability mask
    if available is not None:
        avail = np.asarray(available, dtype=np.float64).reshape(n_obs, n_alts)
    else:
        avail = np.ones((n_obs, n_alts), dtype=np.float64)

    # Sampling correction
    if inclusion_probs is not None:
        sr = np.asarray(inclusion_probs, dtype=np.float64).reshape(n_obs, n_alts)
        v_fixed = v_fixed + np.log(np.maximum(sr, 1e-30))

    # Chosen indicator
    chosen_2d = np.asarray(chosen, dtype=np.float64).reshape(n_obs, n_alts)

    # For each draw, compute conditional log-likelihood
    # SLL = sum_n w_n * log( (1/R) * sum_r L_n(beta^r) )
    # = sum_n w_n * (logsumexp_r(log L_n(beta^r)) - log(R))

    log_L_draws = np.zeros((n_obs, n_draws), dtype=np.float64)

    for r in range(n_draws):
        # Realise random coefficients for this draw
        beta_random_r = np.zeros((n_obs, k_random))
        for p in range(k_random):
            z_p = draws[:, r, p]  # (n_obs,)
            mean_p = beta_random_means[p]
            spread_p = beta_random_spreads[p]

            if random_distributions[p] == "normal":
                beta_random_r[:, p] = mean_p + spread_p * z_p
            elif random_distributions[p] == "lognormal":
                # Clip exponent to avoid overflow
                exponent = mean_p + spread_p * z_p
                beta_random_r[:, p] = np.exp(np.clip(exponent, -50, 50))
            elif random_distributions[p] in ("triangular", "uniform"):
                from scipy.stats import norm as norm_dist

                u = norm_dist.cdf(z_p)
                beta_random_r[:, p] = mean_p + spread_p * (2 * u - 1)

        # Random utility component
        v_random = np.sum(
            dm_random.reshape(n_obs, n_alts, k_random) * beta_random_r[:, None, :],
            axis=2,
        )

        # Total utility
        utilities = v_fixed + v_random  # (n_obs, n_alts)

        # Mask unavailable
        utilities = np.where(avail > 0, utilities, NEG_INF)

        # Log-probabilities via stable logsumexp
        log_sum_exp = logsumexp(utilities, axis=1)  # (n_obs,)
        log_probs = utilities - log_sum_exp[:, None]  # (n_obs, n_alts)

        # Chosen log-probability for this draw
        log_L_draws[:, r] = (log_probs * chosen_2d).sum(axis=1)

    # Simulated log-likelihood
    # log P(j|n) = logsumexp_r(log L_n(beta^r)) - log(R)
    log_sim_probs = logsumexp(log_L_draws, axis=1) - np.log(n_draws)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(n_obs)
        log_sim_probs = log_sim_probs * w

    return float(np.sum(log_sim_probs))


def _mixed_logit_gradient_numpy(
    params: np.ndarray,
    random_col_indices: list[int],
    k_fixed: int,
    k_random: int,
    random_distributions: list[str],
    draws: np.ndarray,
    design_matrix: np.ndarray,
    chosen: np.ndarray,
    n_obs: int,
    n_alts: int,
    available: Optional[np.ndarray] = None,
    inclusion_probs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute mixed logit gradient via finite differences (NumPy backend).

    Parameters
    ----------
    params : np.ndarray, shape (k_fixed + 2 * k_random,)
        Full parameter vector: [beta_fixed, beta_random_means, beta_random_spreads].
    random_col_indices, k_fixed, k_random, random_distributions, draws, design_matrix, chosen, n_obs, n_alts, available, inclusion_probs, weights
        See :func:`_mixed_logit_ll_numpy`.

    Returns
    -------
    np.ndarray, shape (k_fixed + 2 * k_random,)
        Gradient of the simulated log-likelihood.
    """
    eps = 1e-5
    n_params = len(params)
    grad = np.zeros(n_params)

    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps

        def _unpack(p):
            bf = p[:k_fixed]
            rm = p[k_fixed : k_fixed + k_random]
            rs = p[k_fixed + k_random :]
            return bf, rm, rs

        bf_plus, rm_plus, rs_plus = _unpack(params_plus)
        bf_minus, rm_minus, rs_minus = _unpack(params_minus)

        ll_plus = _mixed_logit_ll_numpy(
            bf_plus,
            rm_plus,
            rs_plus,
            random_distributions,
            draws,
            design_matrix,
            chosen,
            random_col_indices,
            n_obs,
            n_alts,
            available=available,
            inclusion_probs=inclusion_probs,
            weights=weights,
        )
        ll_minus = _mixed_logit_ll_numpy(
            bf_minus,
            rm_minus,
            rs_minus,
            random_distributions,
            draws,
            design_matrix,
            chosen,
            random_col_indices,
            n_obs,
            n_alts,
            available=available,
            inclusion_probs=inclusion_probs,
            weights=weights,
        )

        grad[i] = (ll_plus - ll_minus) / (2 * eps)

    return grad


# ---------------------------------------------------------------------------
# MixedLogit model class
# ---------------------------------------------------------------------------


class MixedLogit(BaseChoiceModel):
    """Mixed logit (random coefficients) model for location choice estimation.

    This model generalises the multinomial logit by allowing some or all
    coefficients to follow random distributions, capturing unobserved taste
    heterogeneity across decision-makers.

    Parameters
    ----------
    data : ChoiceTable
        The choice data to estimate on.
    formula : str, optional
        A formulaic formula string for the utility function.
    spec : ModelSpec, optional
        A ModelSpec object defining the utility function.
    random_params : dict
        Mapping of parameter names to ``ParamDistribution`` objects.
        E.g., ``{"commute_time": ParamDistribution("normal", "commute_time")}``
    n_draws : int
        Number of draws for simulated maximum likelihood. Default 100.
    draw_type : str
        Type of draws: ``"halton"`` (default) or ``"random"``.
    seed : int
        Random seed for draw generation. Default 42.
    weights : str or np.ndarray, optional
        Observation weights.
    availability : str or np.ndarray, optional
        Alternative availability.
    solver : str or Solver
        Solver name or instance. Default ``"lbfgs"``.
    solver_options : dict, optional
        Additional solver options.

    Examples
    --------
    >>> from locpick import ChoiceTable
    >>> from locpick.models.mixed import MixedLogit, ParamDistribution
    >>> ct = ChoiceTable.from_tables(choosers, alternatives, chosen)
    >>> model = MixedLogit(
    ...     ct, formula="cost + time - 1",
    ...     random_params={"time": ParamDistribution("normal", "time")},
    ...     n_draws=200,
    ... )
    >>> result = model.fit()
    """

    def __init__(
        self,
        data,
        formula: Optional[str] = None,
        spec: Optional[ModelSpec] = None,
        random_params: Optional[dict] = None,
        n_draws: int = 100,
        draw_type: str = "halton",
        seed: int = 42,
        weights: Optional[Union[str, np.ndarray]] = None,
        availability: Optional[Union[str, np.ndarray]] = None,
        solver: Union[str, Solver] = "lbfgs",
        solver_options: Optional[dict] = None,
        backend: Optional[str] = None,
    ):
        if random_params is None or len(random_params) == 0:
            raise ValueError(
                "MixedLogit requires at least one random parameter. "
                "Use MultinomialLogit for models without random coefficients."
            )
        if formula is None and spec is None:
            raise ValueError("Either 'formula' or 'spec' must be provided.")
        if formula is not None and spec is not None:
            raise ValueError("Provide 'formula' or 'spec', not both.")

        self._data = data
        self._problem: Optional[EstimationProblem] = None
        self._formula = formula
        self._random_params = random_params
        self._n_draws = n_draws
        self._draw_type = draw_type
        self._seed = seed
        self._weights = weights
        self._availability = availability
        self._solver_options = solver_options or {}
        self._backend = backend

        # Build ModelSpec from formula if needed
        if formula is not None:
            self._spec = ModelSpec(formula=formula)
        else:
            self._spec = spec

        # Resolve solver
        if isinstance(solver, str):
            self._solver = get_solver(solver, **self._solver_options)
        else:
            self._solver = solver

        # Lazy-initialized
        self._arrays: Optional[ChoiceArrays] = None
        self._result: Optional[FitResult] = None

        # Caches (cleared on re-estimation)
        self._hessian_inverse: Optional[np.ndarray] = None
        self._observation_scores_cache: dict = {}
        self._probabilities_cache: Optional[np.ndarray] = None
        self._utilities_cache: Optional[np.ndarray] = None
        self._covariance_bhhh_cache: Optional[np.ndarray] = None
        self._covariance_robust_cache: Optional[np.ndarray] = None
        self._draws: Optional[np.ndarray] = None

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
        """Estimate the mixed logit model and return results.

        Returns
        -------
        FitResult
            Complete estimation results including fixed coefficients,
            random coefficient means and spreads, and fit statistics.
        """
        # Build estimation arrays
        arrays = self._build_arrays()
        self._arrays = arrays

        # Identify random parameter columns
        param_names = list(arrays.param_names)
        random_param_names = []
        random_distributions = []
        random_col_indices = []

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

        objective = self._build_objective(arrays)

        # Initial values: zeros for fixed/means, small positive spreads
        x0 = np.concatenate(
            [
                np.zeros(k_fixed),
                np.zeros(k_random),
                np.full(k_random, 0.1),
            ]
        )

        fixed_names = [n for i, n in enumerate(param_names) if i not in random_col_indices]
        full_param_names = (
            fixed_names
            + [f"mean_{n}" for n in random_param_names]
            + [f"sd_{n}" for n in random_param_names]
        )

        solver_result = self._solver.solve(
            objective=objective,
            x0=x0,
            param_names=full_param_names,
        )

        # Build FitResult
        self._result = self._build_fit_result(
            solver_result,
            arrays,
            k_fixed,
            k_random,
            random_param_names,
            random_distributions,
        )
        self._clear_caches()
        return self._result

    def _build_objective(self, arrays: ChoiceArrays) -> Objective:
        """Build optimization objective for mixed logit estimation."""
        param_names = list(arrays.param_names)
        random_param_names = []
        random_distributions = []
        random_col_indices = []

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

        if self._draw_type == "halton":
            self._draws = generate_halton_draws(
                arrays.n_obs,
                self._n_draws,
                k_random,
                seed=self._seed,
            )
        else:
            self._draws = generate_random_draws(
                arrays.n_obs,
                self._n_draws,
                k_random,
                seed=self._seed,
            )

        backend = (self._backend or os.environ.get("CHOICEMODELS_MIXED_BACKEND", "")).lower()
        if backend != "numpy":
            try:
                from locpick._jax.builders import build_mixed_logit_objective

                return build_mixed_logit_objective(
                    arrays,
                    random_col_indices=random_col_indices,
                    random_distributions=random_distributions,
                    draws=self._draws,
                )
            except ImportError:
                pass

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        chosen = np.asarray(arrays.chosen, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts
        available = arrays.available
        weights = arrays.weights

        from locpick._sampling.correction import get_sampling_correction

        inclusion_probs = get_sampling_correction(arrays)

        def ll_fn(params):
            beta_fixed = params[:k_fixed]
            beta_random_means = params[k_fixed : k_fixed + k_random]
            beta_random_spreads = params[k_fixed + k_random :]
            return _mixed_logit_ll_numpy(
                beta_fixed,
                beta_random_means,
                beta_random_spreads,
                random_distributions,
                self._draws,
                dm,
                chosen,
                random_col_indices,
                n_obs,
                n_alts,
                available=available,
                inclusion_probs=inclusion_probs,
                weights=weights,
            )

        def grad_fn(params):
            return _mixed_logit_gradient_numpy(
                params,
                random_col_indices,
                k_fixed,
                k_random,
                random_distributions,
                self._draws,
                dm,
                chosen,
                n_obs,
                n_alts,
                available=available,
                inclusion_probs=inclusion_probs,
                weights=weights,
            )

        fixed_names = [n for i, n in enumerate(param_names) if i not in random_col_indices]
        full_param_names = (
            fixed_names
            + [f"mean_{n}" for n in random_param_names]
            + [f"sd_{n}" for n in random_param_names]
        )
        return Objective.from_numpy(
            ll_fn=ll_fn,
            grad_fn=grad_fn,
            param_names=full_param_names,
        )

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

    def _build_fit_result(
        self,
        solver_result: SolverResult,
        arrays: ChoiceArrays,
        k_fixed: int,
        k_random: int,
        random_param_names: list[str],
        random_distributions: list[str],
    ) -> FitResult:
        """Build a FitResult from solver output."""
        all_params = solver_result.coefficients
        all_params[:k_fixed]
        beta_random_means = all_params[k_fixed : k_fixed + k_random]
        beta_random_spreads = all_params[k_fixed + k_random :]

        # Parameter names
        param_names = list(arrays.param_names)
        fixed_names = [
            n
            for i, n in enumerate(param_names)
            if i not in [param_names.index(rn) for rn in random_param_names]
        ]
        random_mean_names = [f"mean_{n}" for n in random_param_names]
        random_spread_names = [f"sd_{n}" for n in random_param_names]
        full_param_names = fixed_names + random_mean_names + random_spread_names

        # Standard errors
        if solver_result.hessian is not None:
            try:
                std_errors = np.sqrt(np.abs(np.diag(solver_result.hessian)))
            except Exception:
                std_errors = np.full(len(all_params), np.nan)
        else:
            std_errors = np.full(len(all_params), np.nan)

        # T-values and p-values
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = np.where(std_errors > 0, all_params / std_errors, np.nan)
        from scipy import stats

        p_values = 2 * (1 - stats.norm.cdf(np.abs(np.nan_to_num(t_values))))

        # Confidence intervals
        z_crit = stats.norm.ppf(0.975)
        conf_lower = all_params - z_crit * std_errors
        conf_upper = all_params + z_crit * std_errors

        # Log-likelihood
        ll = solver_result.log_likelihood

        # Null log-likelihood
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts
        if arrays.available is not None:
            avail = np.asarray(arrays.available, dtype=np.float64).reshape(n_obs, -1)
            n_avail = avail.sum(axis=1)
            ll_null = -np.sum(np.log(n_avail))
        else:
            ll_null = -n_obs * np.log(n_alts)

        # Number of parameters
        n_params = len(all_params)

        # Fit statistics
        aic = 2 * n_params - 2 * ll
        bic = n_params * np.log(n_obs) - 2 * ll
        rho_squared = 1 - ll / ll_null
        rho_bar_squared = 1 - (ll - n_params) / ll_null

        # Build pandas objects
        coefficients = pd.Series(all_params, index=full_param_names, name="coefficient")
        std_err_series = pd.Series(std_errors, index=full_param_names, name="std_error")
        t_series = pd.Series(t_values, index=full_param_names, name="t_value")
        p_series = pd.Series(p_values, index=full_param_names, name="p_value")
        conf_int = pd.DataFrame(
            {"lower": conf_lower, "upper": conf_upper},
            index=full_param_names,
        )

        # Random parameter summary
        pd.DataFrame(
            {
                "distribution": random_distributions,
                "mean": beta_random_means,
                "sd": np.abs(beta_random_spreads),
            },
            index=random_param_names,
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
            n_parameters=n_params,
            n_alts=n_alts,
            aic=aic,
            bic=bic,
            rho_squared=rho_squared,
            rho_bar_squared=rho_bar_squared,
            spec=self._spec,
            model_type="Mixed Logit",
            solver_name=solver_result.solver_name,
            solver_result=solver_result.raw,
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _clear_caches(self):
        """Clear all cached computation results."""
        self._hessian_inverse = None
        self._observation_scores_cache = {}
        self._probabilities_cache = None
        self._utilities_cache = None
        self._covariance_bhhh_cache = None
        self._covariance_robust_cache = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_hessian_inverse(self) -> np.ndarray | None:
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
        param_names = list(arrays.param_names)
        random_col_indices = []
        for name, dist in self._random_params.items():
            random_col_indices.append(param_names.index(name))

        k_fixed = len(param_names) - len(random_col_indices)
        k_random = len(random_col_indices)
        n_params = k_fixed + 2 * k_random
        n_obs = arrays.n_obs

        # Full parameter vector: [beta_fixed, beta_random_means, beta_random_spreads]
        full_params = self._result.coefficients.values.copy()
        beta_fixed_hat = full_params[:k_fixed]
        beta_random_means_hat = full_params[k_fixed : k_fixed + k_random]
        beta_random_spreads_hat = full_params[k_fixed + k_random :]

        # Chosen indicator
        chosen = np.asarray(arrays.chosen, dtype=np.float64).reshape(n_obs, arrays.n_alts)

        # Base probabilities and per-observation LL
        probs_base = self.probabilities(
            data=None,
            beta_fixed=beta_fixed_hat,
            beta_random_means=beta_random_means_hat,
            beta_random_spreads=beta_random_spreads_hat,
        )
        ll_base = np.log(np.maximum(np.sum(probs_base * chosen, axis=1), 1e-30))

        scores = np.zeros((n_obs, n_params))

        for j in range(n_params):
            params_plus = full_params.copy()
            params_plus[j] += eps
            params_minus = full_params.copy()
            params_minus[j] -= eps

            bf_plus = params_plus[:k_fixed]
            brm_plus = params_plus[k_fixed : k_fixed + k_random]
            brs_plus = params_plus[k_fixed + k_random :]
            bf_minus = params_minus[:k_fixed]
            brm_minus = params_minus[k_fixed : k_fixed + k_random]
            brs_minus = params_minus[k_fixed + k_random :]

            probs_plus = self.probabilities(
                data=None, beta_fixed=bf_plus, beta_random_means=brm_plus, beta_random_spreads=brs_plus
            )
            probs_minus = self.probabilities(
                data=None, beta_fixed=bf_minus, beta_random_means=brm_minus, beta_random_spreads=brs_minus
            )

            ll_plus = np.log(np.maximum(np.sum(probs_plus * chosen, axis=1), 1e-30))
            ll_minus = np.log(np.maximum(np.sum(probs_minus * chosen, axis=1), 1e-30))

            scores[:, j] = (ll_plus - ll_minus) / (2 * eps)

        self._observation_scores_cache[cache_key] = scores
        return scores

    def utilities(self, data=None, beta_fixed=None, beta_random_means=None):
        """Compute deterministic utilities (expected utility at population means).

        For mixed logit models, the deterministic utility is computed
        using the expected (mean) coefficients: fixed coefficients for
        non-random variables and mean coefficients for random variables.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to predict on.  If ``None``, uses estimation data.
        beta_fixed : np.ndarray or None
            Fixed utility coefficients.  If ``None``, uses estimated values.
        beta_random_means : np.ndarray or None
            Random coefficient means.  If ``None``, uses estimated values.

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

        # Extract parameters
        param_names = list(arrays.param_names)
        random_col_indices = []
        for name, dist in self._random_params.items():
            random_col_indices.append(param_names.index(name))
        k_fixed = len(param_names) - len(random_col_indices)
        k_random = len(random_col_indices)

        if beta_fixed is None:
            beta_fixed = np.asarray(self._result.coefficients.values[:k_fixed], dtype=np.float64)
        if beta_random_means is None:
            beta_random_means = np.asarray(
                self._result.coefficients.values[k_fixed : k_fixed + k_random], dtype=np.float64
            )

        dm = np.asarray(arrays.design_matrix, dtype=np.float64)
        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        # Build full coefficient vector: fixed + random means
        # Place fixed coefficients at their positions and random means at random positions
        beta_full = np.zeros(dm.shape[1], dtype=np.float64)
        fixed_mask = np.ones(dm.shape[1], dtype=bool)
        fixed_mask[random_col_indices] = False
        beta_full[fixed_mask] = beta_fixed
        beta_full[random_col_indices] = beta_random_means

        # Systematic utility
        V = (dm @ beta_full).reshape(n_obs, n_alts)

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
            chosen_indices = np.array(
                [rng.choice(n_alts, p=probs[i]) for i in range(n_obs)]
            )
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

        For Mixed Logit, the direct marginal effect uses the mean coefficient
        value (``mean_{variable}`` in the results).

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

        For Mixed Logit, the cross-marginal effect uses the mean coefficient
        value (``mean_{variable}`` in the results).

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

        The direct elasticity measures the percentage change in the
        probability of choosing an alternative with respect to a
        percentage change in a variable.

        For random parameters, the coefficient used is the mean value
        (``mean_{variable}`` in the results).

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
        # For random parameters, use the mean coefficient
        beta = self._result.coefficients.get(variable, self._result.coefficients.get(f"mean_{variable}", 0.0))

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

        For random parameters, the coefficient used is the mean value
        (``mean_{variable}`` in the results).

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
        beta = self._result.coefficients.get(variable, self._result.coefficients.get(f"mean_{variable}", 0.0))
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

    def probabilities(
        self, data=None, beta_fixed=None, beta_random_means=None, beta_random_spreads=None
    ):
        """Compute choice probabilities.

        Parameters
        ----------
        data : ChoiceTable or None
            Data to predict on. If None, uses estimation data.
        beta_fixed : np.ndarray or None
            Fixed coefficients. If None, uses estimated values.
        beta_random_means : np.ndarray or None
            Random coefficient means. If None, uses estimated values.
        beta_random_spreads : np.ndarray or None
            Random coefficient spreads. If None, uses estimated values.

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

        # Extract parameters from result
        param_names = list(arrays.param_names)
        random_col_indices = []
        random_distributions = []
        for name, dist in self._random_params.items():
            random_col_indices.append(param_names.index(name))
            random_distributions.append(dist.distribution)

        k_fixed = len(param_names) - len(random_col_indices)
        k_random = len(random_col_indices)

        if beta_fixed is None:
            beta_fixed = self._result.coefficients.values[:k_fixed]
        if beta_random_means is None:
            beta_random_means = self._result.coefficients.values[k_fixed : k_fixed + k_random]
        if beta_random_spreads is None:
            beta_random_spreads = self._result.coefficients.values[k_fixed + k_random :]

        # Regenerate draws for prediction data
        if self._draws is not None and self._draws.shape[0] != arrays.n_obs:
            if self._draw_type == "halton":
                draws = generate_halton_draws(
                    arrays.n_obs,
                    self._n_draws,
                    k_random,
                    seed=self._seed + 1,
                )
            else:
                draws = generate_random_draws(
                    arrays.n_obs,
                    self._n_draws,
                    k_random,
                    seed=self._seed + 1,
                )
        else:
            draws = self._draws

        # Resolve canonical sampling correction tensor.
        from locpick._sampling.correction import get_sampling_correction

        sampling_correction = get_sampling_correction(arrays)

        return _mixed_logit_probs_numpy(
            beta_fixed,
            beta_random_means,
            beta_random_spreads,
            random_distributions,
            draws,
            np.asarray(arrays.design_matrix, dtype=np.float64),
            random_col_indices,
            arrays.n_obs,
            arrays.n_alts,
            available=arrays.available,
            inclusion_probs=sampling_correction,
        )
