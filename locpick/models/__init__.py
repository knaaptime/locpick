"""Choice model classes for location choice estimation.

This package provides:

- :mod:`locpick.models.base` — ``ChoiceModel`` protocol
- :mod:`locpick.models.mnl` — ``MultinomialLogit`` class
- :mod:`locpick.models.nested` — ``NestedLogit``, ``NestSpec``, ``NestingTree``
- :mod:`locpick.models.mixed` — ``MixedLogit``, ``ParamDistribution``
- :mod:`locpick.models.scl` — ``SpatiallyCorrelatedLogit``
- :mod:`locpick.models.mscl` — ``MixedSpatiallyCorrelatedLogit``

Submodules and attributes are loaded lazily following SPEC 1
(https://scientific-python.org/specs/spec-0001/).
"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
