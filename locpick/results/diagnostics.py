"""Likelihood ratio test and other statistical tests for choice models.

This module provides the ``LikelihoodRatioTest`` class for comparing nested
choice models, as well as convenience functions for common hypothesis tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
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


class FitDiagnostics:
    """Post-estimation diagnostics and reporting helpers.

    This service keeps behavioral logic out of :class:`FitResult`, which is
    intended to be a pure estimation-output container.
    """

    @staticmethod
    def tidy(result: "FitResult") -> pd.DataFrame:
        """Return a tidy DataFrame of estimation results."""
        df = pd.DataFrame(
            {
                "coefficient": result.coefficients,
                "std_error": result.std_errors,
                "t_value": result.t_values,
                "p_value": result.p_values,
            }
        )
        if result.conf_int is not None and len(result.conf_int) > 0:
            df["conf_lower"] = result.conf_int.iloc[:, 0]
            df["conf_upper"] = result.conf_int.iloc[:, 1]
        return df

    @staticmethod
    def fit_statistics(result: "FitResult") -> pd.DataFrame:
        """Return a DataFrame of model fit statistics."""
        stats_map = {
            "Log-likelihood": result.log_likelihood,
            "Log-likelihood (null)": result.log_likelihood_null,
            "Observations": result.n_observations,
            "Alternatives": result.n_alts,
            "Parameters": result.n_parameters,
            "DF residual": result.n_observations - result.n_parameters,
            "AIC": result.aic,
            "BIC": result.bic,
            "Rho-squared": result.rho_squared,
            "Adjusted rho-squared": result.rho_bar_squared,
        }
        return pd.DataFrame({"statistic": list(stats_map.keys()), "value": list(stats_map.values())})

    @staticmethod
    def summary(result: "FitResult", format: str = "text") -> str:
        """Return a formatted summary string for an estimation result."""
        if format == "html":
            return FitDiagnostics._summary_html(result)
        if format == "latex":
            return FitDiagnostics._summary_latex(result)
        return FitDiagnostics._summary_text(result)

    @staticmethod
    def to_latex(result: "FitResult") -> str:
        """Return a LaTeX-formatted coefficient table."""
        return FitDiagnostics._summary_latex(result)

    @staticmethod
    def to_html(result: "FitResult") -> str:
        """Return an HTML-formatted coefficient table."""
        return FitDiagnostics._summary_html(result)

    @staticmethod
    def _summary_text(result: "FitResult") -> str:
        lines = []
        lines.append("=" * 60)
        lines.append(f"{result.model_type} Estimation Results")
        lines.append("=" * 60)
        lines.append(f"Observations:    {result.n_observations:>12}")
        lines.append(f"Alternatives:    {result.n_alts:>12}")
        lines.append(f"Parameters:     {result.n_parameters:>12}")
        lines.append(f"DF residual:     {result.n_observations - result.n_parameters:>12}")
        lines.append("-" * 60)
        lines.append(f"Log-likelihood: {result.log_likelihood:>12.4f}")
        lines.append(f"LL (null):       {result.log_likelihood_null:>12.4f}")
        lines.append(f"AIC:             {result.aic:>12.4f}")
        lines.append(f"BIC:             {result.bic:>12.4f}")
        lines.append(f"Rho-squared:     {result.rho_squared:>12.4f}")
        lines.append(f"Adj. rho-sq:     {result.rho_bar_squared:>12.4f}")
        lines.append("-" * 60)
        lines.append(f"{'Parameter':<20} {'Coef':>10} {'Std.Err':>10} {'t':>8} {'P>|t|':>8}")
        lines.append("-" * 60)
        for name in result.coefficients.index:
            coef = result.coefficients[name]
            se = result.std_errors.get(name, float("nan"))
            t_val = result.t_values.get(name, float("nan"))
            p_val = result.p_values.get(name, float("nan"))
            lines.append(f"{name:<20} {coef:>10.4f} {se:>10.4f} {t_val:>8.3f} {p_val:>8.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def _summary_html(result: "FitResult") -> str:
        tidy = FitDiagnostics.tidy(result)
        html = "<table>\n"
        html += "<tr><th>Parameter</th><th>Coef</th><th>Std.Err</th>"
        html += "<th>t</th><th>P>|t|</th></tr>\n"
        for name, row in tidy.iterrows():
            html += (
                f"<tr><td>{name}</td><td>{row['coefficient']:.4f}</td>"
                f"<td>{row['std_error']:.4f}</td>"
                f"<td>{row['t_value']:.3f}</td>"
                f"<td>{row['p_value']:.4f}</td></tr>\n"
            )
        html += "</table>"
        return html

    @staticmethod
    def _summary_latex(result: "FitResult") -> str:
        tidy = FitDiagnostics.tidy(result)
        lines = [
            "\\begin{tabular}{lcccc}",
            "\\hline",
            "Parameter & Coef & Std.Err & t & P>|t| \\\\",
            "\\hline",
        ]
        for name, row in tidy.iterrows():
            lines.append(
                f"{name} & {row['coefficient']:.4f} & {row['std_error']:.4f} "
                f"& {row['t_value']:.3f} & {row['p_value']:.4f} \\\\",
            )
        lines.extend(["\\hline", "\\end{tabular}"])
        return "\n".join(lines)

    @staticmethod
    def lr_test(restricted: "FitResult", unrestricted: "FitResult") -> "LikelihoodRatioTest":
        """Perform a likelihood-ratio test between nested models."""
        return LikelihoodRatioTest(
            statistic=2 * (unrestricted.log_likelihood - restricted.log_likelihood),
            df=unrestricted.n_parameters - restricted.n_parameters,
            restricted_model=restricted,
            unrestricted_model=unrestricted,
        )

    @staticmethod
    def covariance(result: "FitResult") -> np.ndarray:
        """Return Hessian-based covariance if available, else diagonal fallback."""
        if result.solver_result and "scipy_result" in result.solver_result:
            scipy_result = result.solver_result["scipy_result"]
            if hasattr(scipy_result, "hess_inv"):
                try:
                    return np.asarray(
                        scipy_result.hess_inv.todense()
                        if hasattr(scipy_result.hess_inv, "todense")
                        else scipy_result.hess_inv,
                    )
                except Exception:
                    pass

        if result.std_errors is not None and not result.std_errors.isna().all():
            variances = result.std_errors.values**2
            return np.diag(variances)

        raise ValueError(
            "Covariance matrix is not available. The model may not have been "
            "estimated, or the solver did not provide a Hessian.",
        )

    @staticmethod
    def wtp(result: "FitResult", numerator: str, denominator: str = "cost") -> pd.Series:
        """Compute willingness-to-pay with delta-method standard errors."""
        from scipy.stats import norm as norm_dist

        if denominator not in result.coefficients.index:
            raise ValueError(
                f"Denominator '{denominator}' not found in model coefficients. "
                f"Available: {list(result.coefficients.index)}",
            )
        if numerator not in result.coefficients.index:
            raise ValueError(
                f"Numerator '{numerator}' not found in model coefficients. "
                f"Available: {list(result.coefficients.index)}",
            )

        beta_num = result.coefficients[numerator]
        beta_den = result.coefficients[denominator]
        wtp_point = -beta_num / beta_den

        cov = FitDiagnostics.covariance(result)
        num_idx = result.coefficients.index.get_loc(numerator)
        den_idx = result.coefficients.index.get_loc(denominator)
        var_num = cov[num_idx, num_idx]
        var_den = cov[den_idx, den_idx]
        cov_num_den = cov[num_idx, den_idx]

        var_wtp = var_num / beta_den**2 + beta_num**2 * var_den / beta_den**4 - 2 * beta_num * cov_num_den / beta_den**3
        se_wtp = np.sqrt(max(var_wtp, 0.0))
        t_stat = wtp_point / se_wtp if se_wtp > 0 else np.inf
        p_value = 2 * (1 - norm_dist.cdf(abs(t_stat)))

        return pd.Series(
            {
                "wtp": wtp_point,
                "se": se_wtp,
                "t_stat": t_stat,
                "p_value": p_value,
            },
            name=f"wtp_{numerator}_per_{denominator}",
        )

    @staticmethod
    def vot(result: "FitResult", time_var: str = "time", cost_var: str = "cost") -> pd.Series:
        """Compute value-of-time as a WTP specialization."""
        return FitDiagnostics.wtp(result, numerator=time_var, denominator=cost_var)


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
