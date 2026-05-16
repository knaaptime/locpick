"""Estimation results payload for location choice models.

`FitResult` stores immutable estimation outputs and metadata only.
Behavioral diagnostics/reporting utilities live in `locpick.results.diagnostics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class FitResult:
    """Immutable estimation outputs for a fitted choice model."""

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
