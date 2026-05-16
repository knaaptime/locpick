"""Data assembly, containers, and utilities for location choice models.

This package provides the core data layer for locpick v2:

- :mod:`locpick.data.arrays` — ``ChoiceArrays`` dataclass (estimation-ready arrays)
- :mod:`locpick.data.choicetable` — ``ChoiceTable`` class (data assembly)
- :mod:`locpick.data.dataset` — xarray Dataset helpers for choice data
- :mod:`locpick.data.problem` — ``EstimationProblem`` dataclass
- :mod:`locpick.data.format` — formatting utilities for results
- :mod:`locpick.data.distance` — distance matrix and nearest-neighbor helpers

Submodules and attributes are loaded lazily following SPEC 1
(https://scientific-python.org/specs/spec-0001/).
"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)

__all__ = [
    # arrays
    "ChoiceArrays",
    # choicetable
    "ChoiceTable",
    # dataset
    "_resolve_interaction",
    "build_choice_dataset",
    "build_choice_dataset_from_long",
    "dataset_to_long_frame",
    "validate_choice_dataset",
    # problem
    "EstimationProblem",
    # format
    "format_coefficient_table",
    "format_fit_statistics",
    "format_side_by_side",
    # distance
    "distance_bands",
    "distance_matrix",
    "euclidean_distance_matrix",
    "great_circle_distance_matrix",
    "great_circle_vec",
    "nearest_neighbors",
    "pairwise_distance",
]