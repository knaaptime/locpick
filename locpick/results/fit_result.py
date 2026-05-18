"""Estimation results payload for location choice models.

`FitResult` stores immutable estimation outputs and provides post-estimation
reporting and diagnostic methods directly on the result object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class FitResult:
    """Immutable estimation outputs for a fitted choice model.

    Provides instance methods for common post-estimation tasks such as
    formatting summaries, computing WTP/VOT, and extracting covariance.
    """

    # Core parameter results
    coefficients: pd.Series
    std_errors: pd.Series
    t_values: pd.Series
    p_values: pd.Series
    conf_int: pd.DataFrame

    # Model fit statistics
    log_likelihood: float
    log_likelihood_null: float
    n_observations: int
    n_parameters: int
    n_alts: int = 0
    aic: float = 0.0
    bic: float = 0.0
    rho_squared: float = 0.0
    rho_bar_squared: float = 0.0

    # Estimation metadata
    spec: object = None
    model_type: str = "Multinomial Logit"
    solver_name: str = ""
    solver_result: dict = field(default_factory=dict)
    data_hash: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # ------------------------------------------------------------------
    # Reporting / formatting
    # ------------------------------------------------------------------

    def tidy(self) -> pd.DataFrame:
        """Return a tidy DataFrame of estimation results."""
        df = pd.DataFrame(
            {
                "coefficient": self.coefficients,
                "std_error": self.std_errors,
                "t_value": self.t_values,
                "p_value": self.p_values,
            }
        )
        if self.conf_int is not None and len(self.conf_int) > 0:
            df["conf_lower"] = self.conf_int.iloc[:, 0]
            df["conf_upper"] = self.conf_int.iloc[:, 1]
        return df

    def fit_statistics(self) -> pd.DataFrame:
        """Return a DataFrame of model fit statistics."""
        stats_map = {
            "Log-likelihood": self.log_likelihood,
            "Log-likelihood (null)": self.log_likelihood_null,
            "Observations": self.n_observations,
            "Alternatives": self.n_alts,
            "Parameters": self.n_parameters,
            "DF residual": self.n_observations - self.n_parameters,
            "AIC": self.aic,
            "BIC": self.bic,
            "Rho-squared": self.rho_squared,
            "Adjusted rho-squared": self.rho_bar_squared,
        }
        return pd.DataFrame(
            {"statistic": list(stats_map.keys()), "value": list(stats_map.values())}
        )

    def summary(self, format: str = "text") -> str:
        """Return a formatted summary string for an estimation result."""
        if format == "html":
            return self._summary_html()
        if format == "latex":
            return self._summary_latex()
        return self._summary_text()

    def to_latex(self) -> str:
        """Return a LaTeX-formatted coefficient table."""
        return self._summary_latex()

    def to_html(self) -> str:
        """Return an HTML-formatted coefficient table."""
        return self._summary_html()

    def _summary_text(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append(f"{self.model_type} Estimation Results")
        lines.append("=" * 60)
        lines.append(f"Observations:    {self.n_observations:>12}")
        lines.append(f"Alternatives:    {self.n_alts:>12}")
        lines.append(f"Parameters:     {self.n_parameters:>12}")
        lines.append(f"DF residual:     {self.n_observations - self.n_parameters:>12}")
        lines.append("-" * 60)
        lines.append(f"Log-likelihood: {self.log_likelihood:>12.4f}")
        lines.append(f"LL (null):       {self.log_likelihood_null:>12.4f}")
        lines.append(f"AIC:             {self.aic:>12.4f}")
        lines.append(f"BIC:             {self.bic:>12.4f}")
        lines.append(f"Rho-squared:     {self.rho_squared:>12.4f}")
        lines.append(f"Adj. rho-sq:     {self.rho_bar_squared:>12.4f}")
        lines.append("-" * 60)
        lines.append(f"{'Parameter':<20} {'Coef':>10} {'Std.Err':>10} {'t':>8} {'P>|t|':>8}")
        lines.append("-" * 60)
        for name in self.coefficients.index:
            coef = self.coefficients[name]
            se = self.std_errors.get(name, float("nan"))
            t_val = self.t_values.get(name, float("nan"))
            p_val = self.p_values.get(name, float("nan"))
            lines.append(f"{name:<20} {coef:>10.4f} {se:>10.4f} {t_val:>8.3f} {p_val:>8.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _summary_html(self) -> str:
        tidy = self.tidy()
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

    def _summary_latex(self) -> str:
        tidy = self.tidy()
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

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def covariance(self) -> np.ndarray:
        """Return Hessian-based covariance if available, else diagonal fallback."""
        if self.solver_result and "scipy_result" in self.solver_result:
            scipy_result = self.solver_result["scipy_result"]
            if hasattr(scipy_result, "hess_inv"):
                try:
                    return np.asarray(
                        scipy_result.hess_inv.todense()
                        if hasattr(scipy_result.hess_inv, "todense")
                        else scipy_result.hess_inv,
                    )
                except Exception:
                    pass

        if self.std_errors is not None and not self.std_errors.isna().all():
            variances = self.std_errors.values**2
            return np.diag(variances)

        raise ValueError(
            "Covariance matrix is not available. The model may not have been "
            "estimated, or the solver did not provide a Hessian.",
        )

    def wtp(self, numerator: str, denominator: str = "cost") -> pd.Series:
        """Compute willingness-to-pay with delta-method standard errors."""
        if denominator not in self.coefficients.index:
            raise ValueError(
                f"Denominator '{denominator}' not found in model coefficients. "
                f"Available: {list(self.coefficients.index)}",
            )
        if numerator not in self.coefficients.index:
            raise ValueError(
                f"Numerator '{numerator}' not found in model coefficients. "
                f"Available: {list(self.coefficients.index)}",
            )

        beta_num = self.coefficients[numerator]
        beta_den = self.coefficients[denominator]
        wtp_point = -beta_num / beta_den

        cov = self.covariance()
        num_idx = self.coefficients.index.get_loc(numerator)
        den_idx = self.coefficients.index.get_loc(denominator)
        var_num = cov[num_idx, num_idx]
        var_den = cov[den_idx, den_idx]
        cov_num_den = cov[num_idx, den_idx]

        var_wtp = (
            var_num / beta_den**2
            + beta_num**2 * var_den / beta_den**4
            - 2 * beta_num * cov_num_den / beta_den**3
        )
        se_wtp = np.sqrt(max(var_wtp, 0.0))
        t_stat = wtp_point / se_wtp if se_wtp > 0 else np.inf
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return pd.Series(
            {
                "wtp": wtp_point,
                "se": se_wtp,
                "t_stat": t_stat,
                "p_value": p_value,
            },
            name=f"wtp_{numerator}_per_{denominator}",
        )

    def vot(self, time_var: str = "time", cost_var: str = "cost") -> pd.Series:
        """Compute value-of-time as a WTP specialization."""
        return self.wtp(numerator=time_var, denominator=cost_var)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the fit result payload to a dictionary."""
        return {
            "coefficients": self.coefficients.to_dict(),
            "std_errors": self.std_errors.to_dict(),
            "t_values": self.t_values.to_dict(),
            "p_values": self.p_values.to_dict(),
            "log_likelihood": self.log_likelihood,
            "log_likelihood_null": self.log_likelihood_null,
            "n_observations": self.n_observations,
            "n_parameters": self.n_parameters,
            "n_alts": self.n_alts,
            "aic": self.aic,
            "bic": self.bic,
            "rho_squared": self.rho_squared,
            "rho_bar_squared": self.rho_bar_squared,
            "solver_name": self.solver_name,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"FitResult(n_obs={self.n_observations}, n_params={self.n_parameters}, "
            f"LL={self.log_likelihood:.3f}, rho_sq={self.rho_squared:.4f})"
        )
