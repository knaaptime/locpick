"""Estimation results, inference, and diagnostics for location choice models.

This package provides:

- :mod:`locpick.results.fit_result` — ``FitResult`` payload dataclass (estimation outputs and reporting)
- :mod:`locpick.results.diagnostics` — ``LikelihoodRatioTest``, ``WaldTest``, statistical tests

Submodules and attributes are loaded lazily following SPEC 1
(https://scientific-python.org/specs/spec-0001/).
"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
