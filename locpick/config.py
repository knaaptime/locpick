"""Global configuration for locpick.

Defaults can be overridden via environment variables with the ``LOCPICK_``
prefix (e.g. ``LOCPICK_DEFAULT_SOLVER=optimistix``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Config:
    """Runtime configuration for locpick.

    Parameters
    ----------
    jax_enable_x64 : bool
        Enable 64-bit floating point in JAX. Default ``True``.
    numba_parallel : bool
        Use ``prange`` in Numba kernels when available. Default ``True``.
    default_solver : str
        Default solver name passed to ``model.fit()``. Default ``"lbfgs"``.
    default_n_draws_mixed : int
        Default number of QMC draws for mixed logit / MSCL / MNSCL.
        Default ``100``.
    default_qmc_engine : str
        Default QMC engine for mixed models. Default ``"sobol"``.
    """

    jax_enable_x64: bool = True
    numba_parallel: bool = True
    default_solver: str = "lbfgs"
    default_n_draws_mixed: int = 100
    default_qmc_engine: str = "sobol"

    @classmethod
    def from_env(cls) -> "Config":
        """Build a ``Config`` from environment variables.

        Recognised variables (all prefixed with ``LOCPICK_``):

        - ``LOCPICK_JAX_ENABLE_X64`` — ``"1"``/``"true"``/``"yes"`` → ``True``
        - ``LOCPICK_NUMBA_PARALLEL`` — same boolean logic
        - ``LOCPICK_DEFAULT_SOLVER`` — solver name string
        - ``LOCPICK_DEFAULT_N_DRAWS_MIXED`` — integer
        - ``LOCPICK_DEFAULT_QMC_ENGINE`` — engine name string
        """
        kwargs: dict[str, Any] = {}

        if (val := os.environ.get("LOCPICK_JAX_ENABLE_X64")) is not None:
            kwargs["jax_enable_x64"] = _str_to_bool(val)
        if (val := os.environ.get("LOCPICK_NUMBA_PARALLEL")) is not None:
            kwargs["numba_parallel"] = _str_to_bool(val)
        if (val := os.environ.get("LOCPICK_DEFAULT_SOLVER")) is not None:
            kwargs["default_solver"] = val
        if (val := os.environ.get("LOCPICK_DEFAULT_N_DRAWS_MIXED")) is not None:
            kwargs["default_n_draws_mixed"] = int(val)
        if (val := os.environ.get("LOCPICK_DEFAULT_QMC_ENGINE")) is not None:
            kwargs["default_qmc_engine"] = val

        return cls(**kwargs)


def _str_to_bool(val: str) -> bool:
    """Parse a string as a boolean."""
    return val.lower() in {"1", "true", "yes", "on"}


# Singleton instance — created once at import time.
config = Config.from_env()
