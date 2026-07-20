"""Shared constants for choice model kernels.

This module provides numerical constants used across all model
implementations (MNL, nested, mixed, SCL, MSCL) and both NumPy
and JAX backends.  Centralising them avoids duplication and makes
it easy to adjust numerical tolerances in one place.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Large negative value used to mask unavailable alternatives in
#: utility arrays.  Must be negative enough that
#: ``exp(NEG_INF) ≈ 0`` in double precision, but not so negative
#: that it causes overflow in intermediate calculations.
#:
#: The value ``-1e30`` is chosen because:
#: * ``exp(-1e30)`` underflows to exactly 0.0 in float64.
#: * It is well above the float64 minimum (≈ ``-1.8e308``), so
#:   addition with finite utilities does not overflow.
#: * It is far enough from 0 that ``softmax``-style computations
#:   treat masked entries as having zero probability.
NEG_INF: float = -1e30

#: NumPy scalar version for use in array operations.
NEG_INF_NP: np.floating = np.float64(-1e30)

# ---------------------------------------------------------------------------
# Solver selection thresholds
# ---------------------------------------------------------------------------

#: Number of alternatives above which the SAR spatial filter switches from a
#: dense factorisation to a conjugate-gradient solve.  Lives here rather than
#: in the JAX kernels so the model layer can consult it without importing JAX.
SAR_DENSE_CUTOFF: int = 2000
