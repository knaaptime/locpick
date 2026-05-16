"""Backend compatibility layer.

Centralizes availability checks for optional dependencies (JAX, Numba)
to avoid repeated try/except blocks across the codebase.
"""

import os

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

try:
    import numba

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

# Conditional parallel flag for Numba-accelerated models
_NUMBA_PARALLEL = _NUMBA_AVAILABLE and os.environ.get("NUMBA_PARALLEL", "1") != "0"

__all__ = ["_JAX_AVAILABLE", "_NUMBA_AVAILABLE", "_NUMBA_PARALLEL"]
