"""Probability and likelihood kernels for choice models.

This subpackage provides the canonical implementations of probability,
log-likelihood, and gradient kernels used across all model types.
Centralising these kernels eliminates code duplication and ensures
that bug fixes propagate automatically.

Submodules and attributes are loaded lazily following SPEC 1
(https://scientific-python.org/specs/spec-0001/).
"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
