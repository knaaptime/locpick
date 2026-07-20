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

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .._kernels.constants import NEG_INF

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
    param : str
        The parameter name to assign a random distribution to.
    """

    distribution: str
    param: str

    def __post_init__(self) -> None:
        valid = {"normal", "lognormal", "triangular", "uniform"}
        if self.distribution not in valid:
            raise ValueError(
                f"Unknown distribution '{self.distribution}'. Must be one of {valid}."
            )

    @property
    def param_name(self) -> str:
        """Return the parameter name as a string."""
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


def _resolve_draws(
    draw_type: str,
    n_obs: int,
    n_draws: int,
    n_random_params: int,
    seed: int = 42,
) -> np.ndarray:
    """Dispatch draw generation by type string.

    Parameters
    ----------
    draw_type : str
        One of ``"halton"``, ``"random"``, ``"qmc"`` / ``"sobol"``,
        or ``"scrambled_halton"``.
    n_obs, n_draws, n_random_params, seed
        Forwarded to the underlying generator.

    Returns
    -------
    np.ndarray, shape (n_obs, n_draws, n_random_params)
    """
    draw_type = draw_type.lower().strip()
    if draw_type == "halton":
        return generate_halton_draws(n_obs, n_draws, n_random_params, seed)
    if draw_type == "random":
        return generate_random_draws(n_obs, n_draws, n_random_params, seed)
    if draw_type in ("qmc", "sobol"):
        return generate_qmc_draws(n_obs, n_draws, n_random_params, seed, engine="sobol")
    if draw_type == "scrambled_halton":
        return generate_qmc_draws(n_obs, n_draws, n_random_params, seed, engine="halton")
    raise ValueError(f"Unknown draw_type: {draw_type!r}")


def generate_qmc_draws(
    n_obs: int,
    n_draws: int,
    n_random_params: int,
    seed: int = 42,
    engine: str = "sobol",
) -> np.ndarray:
    """Generate scrambled QMC draws for simulated maximum likelihood.

    Uses ``scipy.stats.qmc`` engines (Sobol or Halton) with scrambling
    to produce low-discrepancy standard normal draws.

    Parameters
    ----------
    n_obs : int
        Number of observations (decision-makers).
    n_draws : int
        Number of draws per observation per random parameter.
    n_random_params : int
        Number of random parameters.
    seed : int
        Seed for the scrambling randomization.
    engine : {"sobol", "halton"}
        QMC engine name.

    Returns
    -------
    np.ndarray, shape (n_obs, n_draws, n_random_params)
        Standard normal draws for simulated integration.
    """
    engine = engine.lower().strip()
    if engine not in {"sobol", "halton"}:
        raise ValueError(f"Unknown QMC engine: {engine!r}")

    from scipy.stats import norm, qmc

    n_total = n_obs * n_draws
    dim = n_random_params

    if engine == "sobol":
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    else:
        sampler = qmc.Halton(d=dim, scramble=True, seed=seed)

    # Generate uniform samples in [0, 1)
    uniform = sampler.random(n=n_total)

    # Transform to standard normal via inverse CDF
    z = norm.ppf(uniform)

    # Reshape to (n_obs, n_draws, n_random_params)
    return z.reshape(n_obs, n_draws, n_random_params).astype(np.float64)


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


def realize_random_coefficients(
    draws: np.ndarray,
    beta_random_means: np.ndarray,
    beta_random_spreads: np.ndarray,
    random_distributions: list[str],
) -> np.ndarray:
    """Realise random coefficients for every draw.

    Parameters
    ----------
    draws : np.ndarray, shape (n_obs, n_draws, k_random)
        Standard normal draws.
    beta_random_means, beta_random_spreads : np.ndarray, shape (k_random,)
    random_distributions : list of str
        One distribution name per random parameter.

    Returns
    -------
    np.ndarray, shape (n_obs, n_draws, k_random)
        Realised coefficient values.
    """
    n_obs, n_draws, k_random = draws.shape
    out = np.zeros((n_obs, n_draws, k_random), dtype=np.float64)
    for p in range(k_random):
        mean_p = np.full(n_obs, beta_random_means[p], dtype=np.float64)
        spread_p = np.full(n_obs, beta_random_spreads[p], dtype=np.float64)
        out[:, :, p] = _apply_distribution(
            draws[:, :, p], mean_p, spread_p, random_distributions[p]
        )
    return out


def _mixed_logit_per_draw_log_probs_numpy(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conditional log-probabilities for each simulation draw.

    Returns
    -------
    log_probs_draws : np.ndarray, shape (n_obs, n_draws, n_alts)
        Conditional MNL log-probabilities given each draw's coefficients.
    beta_random_draws : np.ndarray, shape (n_obs, n_draws, k_random)
        The realised random coefficients behind those probabilities.
    avail : np.ndarray, shape (n_obs, n_alts)
        Resolved availability mask.
    """
    from scipy.special import logsumexp

    k_random = len(random_col_indices)
    all_col_indices = list(range(design_matrix.shape[1]))
    fixed_col_indices = [i for i in all_col_indices if i not in random_col_indices]

    dm_fixed = design_matrix[:, fixed_col_indices] if fixed_col_indices else None
    dm_random = design_matrix[:, random_col_indices].reshape(n_obs, n_alts, k_random)

    if dm_fixed is not None and len(beta_fixed) > 0:
        v_fixed = (dm_fixed @ beta_fixed).reshape(n_obs, n_alts)
    else:
        v_fixed = np.zeros((n_obs, n_alts))

    if available is not None:
        avail = np.asarray(available, dtype=np.float64).reshape(n_obs, n_alts)
    else:
        avail = np.ones((n_obs, n_alts), dtype=np.float64)

    if inclusion_probs is not None:
        sr = np.asarray(inclusion_probs, dtype=np.float64).reshape(n_obs, n_alts)
        v_fixed = v_fixed + np.log(np.maximum(sr, 1e-30))

    beta_random_draws = realize_random_coefficients(
        draws, beta_random_means, beta_random_spreads, random_distributions
    )

    # V_random[n, r, j] = sum_p X_random[n, j, p] * beta_random[n, r, p]
    v_random = np.einsum("njp,nrp->nrj", dm_random, beta_random_draws)

    utilities = v_fixed[:, None, :] + v_random
    utilities = np.where(avail[:, None, :] > 0, utilities, NEG_INF)
    log_probs_draws = utilities - logsumexp(utilities, axis=2)[:, :, None]

    return log_probs_draws, beta_random_draws, avail


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

    log_probs_draws, _, avail = _mixed_logit_per_draw_log_probs_numpy(
        beta_fixed,
        beta_random_means,
        beta_random_spreads,
        random_distributions,
        draws,
        design_matrix,
        random_col_indices,
        n_obs,
        n_alts,
        available=available,
        inclusion_probs=inclusion_probs,
    )

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

    # Shared realisation: this function previously inlined the distribution
    # branches and applied the uniform inverse-CDF to triangular draws.
    beta_random_draws = realize_random_coefficients(
        draws, beta_random_means, beta_random_spreads, random_distributions
    )

    for r in range(n_draws):
        beta_random_r = beta_random_draws[:, r, :]

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
