"""Likelihood ratio test and other statistical tests for choice models.

This module provides the ``LikelihoodRatioTest`` class for comparing nested
choice models, as well as convenience functions for common hypothesis tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from locpick.results.fit_result import FitResult


@dataclass
class LikelihoodRatioTest:
    """Likelihood ratio test for comparing nested models.

    Parameters
    ----------
    statistic : float
        The LR test statistic: 2 * (LL_unrestricted - LL_restricted).
    df : int
        Degrees of freedom (difference in number of parameters).
    restricted_model : FitResult, optional
        The restricted (null) model.
    unrestricted_model : FitResult, optional
        The unrestricted (alternative) model.
    """

    statistic: float
    df: int
    restricted_model: "FitResult | None" = None
    unrestricted_model: "FitResult | None" = None

    @property
    def p_value(self) -> float:
        """P-value from the chi-squared distribution."""
        return float(stats.chi2.sf(self.statistic, self.df))

    @property
    def critical_value_05(self) -> float:
        """Critical value at the 5% significance level."""
        return float(stats.chi2.ppf(0.95, self.df))

    @property
    def significant_at_05(self) -> bool:
        """Whether the test is significant at the 5% level."""
        return self.p_value < 0.05

    def summary(self) -> str:
        """Return a formatted summary of the test."""
        lines = [
            "Likelihood Ratio Test",
            "=" * 40,
            f"LR statistic:  {self.statistic:.4f}",
            f"DF:           {self.df}",
            f"P-value:       {self.p_value:.6f}",
            f"Critical (5%): {self.critical_value_05:.4f}",
            f"Significant:   {'Yes' if self.significant_at_05 else 'No'}",
        ]
        if self.restricted_model is not None:
            lines.append(f"Restricted LL:  {self.restricted_model.log_likelihood:.4f}")
        if self.unrestricted_model is not None:
            lines.append(f"Unrestricted LL: {self.unrestricted_model.log_likelihood:.4f}")
        lines.append("=" * 40)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LikelihoodRatioTest(statistic={self.statistic:.4f}, "
            f"df={self.df}, p_value={self.p_value:.6f})"
        )


def lr_test(restricted: "FitResult", unrestricted: "FitResult") -> LikelihoodRatioTest:
    """Perform a likelihood-ratio test between nested models."""
    return LikelihoodRatioTest(
        statistic=2 * (unrestricted.log_likelihood - restricted.log_likelihood),
        df=unrestricted.n_parameters - restricted.n_parameters,
        restricted_model=restricted,
        unrestricted_model=unrestricted,
    )


def wald_test(
    coefficients: np.ndarray,
    variance_covariance: np.ndarray,
    r_matrix: np.ndarray,
) -> "WaldTest":
    """Perform a Wald test for linear restrictions.

    Tests the null hypothesis Rβ = 0, where R is the restriction matrix.

    Parameters
    ----------
    coefficients : np.ndarray
        Estimated parameter vector.
    variance_covariance : np.ndarray
        Variance-covariance matrix of the estimates.
    r_matrix : np.ndarray
        Restriction matrix (q x k), where q is the number of restrictions
        and k is the number of parameters.

    Returns
    -------
    WaldTest
    """
    r_beta = r_matrix @ coefficients
    middle = r_matrix @ variance_covariance @ r_matrix.T
    statistic = float(r_beta.T @ np.linalg.inv(middle) @ r_beta)
    df = r_matrix.shape[0]
    p_value = float(stats.chi2.sf(statistic, df))
    return WaldTest(statistic=statistic, df=df, p_value=p_value)


@dataclass
class WaldTest:
    """Wald test result.

    Parameters
    ----------
    statistic : float
        The Wald test statistic.
    df : int
        Degrees of freedom.
    p_value : float
        P-value from the chi-squared distribution.
    """

    statistic: float
    df: int
    p_value: float

    @property
    def significant_at_05(self) -> bool:
        """Whether the test is significant at the 5% level."""
        return self.p_value < 0.05

    def summary(self) -> str:
        """Return a formatted summary of the test."""
        lines = [
            "Wald Test",
            "=" * 40,
            f"Statistic:  {self.statistic:.4f}",
            f"DF:         {self.df}",
            f"P-value:    {self.p_value:.6f}",
            f"Significant: {'Yes' if self.significant_at_05 else 'No'}",
            "=" * 40,
        ]
        return "\n".join(lines)
