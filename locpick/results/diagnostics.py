"""Likelihood ratio test and other statistical tests for choice models.

This module provides the ``LikelihoodRatioTest`` class for comparing nested
choice models, as well as convenience functions for common hypothesis tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
from scipy.linalg import cho_factor, cho_solve

if TYPE_CHECKING:
    from .fit_result import FitResult


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
    # middle is PSD (sandwich of PSD covariance), so use Cholesky
    try:
        statistic = float(r_beta @ cho_solve(cho_factor(middle), r_beta))
    except np.linalg.LinAlgError:
        # Fall back to general inverse if not PD
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


@dataclass
class HausmanTest:
    """Hausman specification test result.

    Tests the null hypothesis that the efficient estimator is consistent by
    comparing it against an estimator that is consistent under both the null
    and alternative.  Under the null,

    .. math::

        H = (\\hat\\beta_c - \\hat\\beta_e)'
            (V_c - V_e)^{+}
            (\\hat\\beta_c - \\hat\\beta_e) \\;\\sim\\; \\chi^2_q

    where :math:`V_c - V_e` is positive semidefinite under the null and
    :math:`q` is its rank (equal to the number of compared parameters in the
    well-identified case).

    Parameters
    ----------
    statistic : float
        The Hausman test statistic.
    df : int
        Degrees of freedom (rank of the covariance difference).
    p_value : float
        P-value from the chi-squared distribution.
    params : list of str
        Names of the parameters compared.
    coef_diff : np.ndarray
        Difference :math:`\\hat\\beta_c - \\hat\\beta_e` for the compared parameters.
    psd_warning : bool
        ``True`` if the covariance difference had negative eigenvalues
        (suggesting violations of Hausman's regularity conditions or finite
        sample noise); the statistic is computed via the Moore-Penrose
        pseudoinverse after clipping negative eigenvalues.
    """

    statistic: float
    df: int
    p_value: float
    params: list[str]
    coef_diff: np.ndarray
    psd_warning: bool = False

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
            "Hausman Test",
            "=" * 40,
            f"Statistic:    {self.statistic:.4f}",
            f"DF:           {self.df}",
            f"P-value:      {self.p_value:.6f}",
            f"Critical (5%): {self.critical_value_05:.4f}",
            f"Significant:  {'Yes' if self.significant_at_05 else 'No'}",
            f"Parameters:   {', '.join(self.params)}",
        ]
        if self.psd_warning:
            lines.append(
                "Warning:      V_c - V_e was not PSD; pseudoinverse used.",
            )
        lines.append("=" * 40)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"HausmanTest(statistic={self.statistic:.4f}, "
            f"df={self.df}, p_value={self.p_value:.6f})"
        )


def hausman_test(
    efficient: "FitResult",
    consistent: "FitResult",
    params: list[str] | None = None,
    tol: float = 1e-10,
) -> HausmanTest:
    """Perform a Hausman specification test between two estimators.

    Under the null hypothesis, ``efficient`` is consistent and asymptotically
    efficient while ``consistent`` is consistent but inefficient.  Under the
    alternative, only ``consistent`` is consistent.

    Parameters
    ----------
    efficient : FitResult
        The estimator that is efficient under the null hypothesis (e.g. MNL
        when testing the IIA assumption against a nested logit).
    consistent : FitResult
        The estimator that is consistent under both the null and the
        alternative (e.g. nested logit, or MNL fit on a restricted choice set).
    params : list of str, optional
        Names of parameters to compare.  Defaults to the intersection of the
        two coefficient indices.
    tol : float, optional
        Tolerance for treating eigenvalues of :math:`V_c - V_e` as zero when
        computing the pseudoinverse rank.  Default ``1e-10``.

    Returns
    -------
    HausmanTest
    """
    eff_coefs = efficient.coefficients
    con_coefs = consistent.coefficients

    if params is None:
        params = [name for name in eff_coefs.index if name in con_coefs.index]
    if not params:
        raise ValueError(
            "No common parameters between the two FitResults to compare.",
        )

    missing_eff = [p for p in params if p not in eff_coefs.index]
    missing_con = [p for p in params if p not in con_coefs.index]
    if missing_eff or missing_con:
        raise ValueError(
            f"Parameters missing: efficient={missing_eff}, consistent={missing_con}",
        )

    idx_eff = [eff_coefs.index.get_loc(p) for p in params]
    idx_con = [con_coefs.index.get_loc(p) for p in params]

    diff = np.asarray(con_coefs.values[idx_con] - eff_coefs.values[idx_eff], dtype=np.float64)

    cov_eff_full = efficient.covariance()
    cov_con_full = consistent.covariance()
    V_e = np.asarray(cov_eff_full, dtype=np.float64)[np.ix_(idx_eff, idx_eff)]
    V_c = np.asarray(cov_con_full, dtype=np.float64)[np.ix_(idx_con, idx_con)]

    delta = V_c - V_e
    # Symmetrize to suppress numerical asymmetry
    delta = 0.5 * (delta + delta.T)

    eigvals, eigvecs = np.linalg.eigh(delta)
    max_eig = float(eigvals.max()) if eigvals.size else 0.0
    psd_warning = bool(eigvals.min() < -max(tol, tol * abs(max_eig)))

    # Moore-Penrose pseudoinverse via eigendecomposition; clip negatives to 0.
    eig_tol = max(tol, tol * abs(max_eig))
    inv_eigs = np.where(eigvals > eig_tol, 1.0 / np.where(eigvals > 0, eigvals, 1.0), 0.0)
    delta_pinv = (eigvecs * inv_eigs) @ eigvecs.T

    statistic = float(diff @ delta_pinv @ diff)
    df = int(np.sum(eigvals > eig_tol))
    if df == 0:
        df = len(params)
    p_value = float(stats.chi2.sf(statistic, df))

    return HausmanTest(
        statistic=statistic,
        df=df,
        p_value=p_value,
        params=list(params),
        coef_diff=diff,
        psd_warning=psd_warning,
    )
