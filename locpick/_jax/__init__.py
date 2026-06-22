"""JAX-accelerated backend for locpick choice models.

This package provides:

- :class:`ChoiceDataJAX` — JAX-ready estimation data container
- :class:`EdgeDataJAX` — JAX-ready spatial edge structure
- :class:`Objective` — unified optimization objective (LL + gradient + Hessian)
- :class:`ParamTransform` — parameter transformation utilities
- Pure JAX probability kernels (:func:`mnl_log_probs`, :func:`mnl_probs`, :func:`scl_log_probs`,
  :func:`nested_log_probs`, :func:`mixed_logit_ll`)
- Objective builders (:func:`build_mnl_objective`, :func:`build_scl_objective`,
  :func:`build_mscl_objective`, :func:`build_nested_objective`,
  :func:`build_mixed_logit_objective`)

The kernels are standalone, testable, composable JAX functions that can be
JIT-compiled and vmap'd independently of any model class.
"""

from .builders import (
    build_mixed_logit_objective,
    build_mnl_objective,
    build_mscl_objective,
    build_nested_objective,
    build_scl_objective,
)
from .data import ChoiceDataJAX, EdgeDataJAX
from .kernels import (
    mixed_logit_ll,
    mnl_log_probs,
    mnl_probs,
    nested_log_probs,
    scl_log_probs,
)
from .objective import Objective
from .transforms import Identity, ParamTransform, Sigmoid, SoftPlus

__all__ = [
    "ChoiceDataJAX",
    "EdgeDataJAX",
    "Objective",
    "ParamTransform",
    "Sigmoid",
    "SoftPlus",
    "Identity",
    "mnl_log_probs",
    "mnl_probs",
    "scl_log_probs",
    "nested_log_probs",
    "mixed_logit_ll",
    "build_mnl_objective",
    "build_scl_objective",
    "build_mscl_objective",
    "build_nested_objective",
    "build_mixed_logit_objective",
]
