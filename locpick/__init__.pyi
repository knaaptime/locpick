"""Public API stub for locpick.

The top-level namespace exposes only user-facing symbols. Internal
subpackages (jax, kernels, sampling, solvers) remain importable directly
for advanced use but are not advertised as public API.
"""

from . import data as data
from . import dgp as dgp
from . import models as models
from . import results as results
from . import spec as spec

from .data import (
    ChoiceArrays as ChoiceArrays,
    ChoiceTable as ChoiceTable,
    EstimationProblem as EstimationProblem,
    distance_bands as distance_bands,
    distance_matrix as distance_matrix,
    euclidean_distance_matrix as euclidean_distance_matrix,
    format_coefficient_table as format_coefficient_table,
    format_fit_statistics as format_fit_statistics,
    format_side_by_side as format_side_by_side,
    great_circle_distance_matrix as great_circle_distance_matrix,
    nearest_neighbors as nearest_neighbors,
    pairwise_distance as pairwise_distance,
)
from .dgp import (
    MNLDataset as MNLDataset,
    MixedLogitDataset as MixedLogitDataset,
    MSCLDataset as MSCLDataset,
    NestedLogitDataset as NestedLogitDataset,
    SCLDataset as SCLDataset,
    simulate_mixed_logit as simulate_mixed_logit,
    simulate_mnl as simulate_mnl,
    simulate_mscl as simulate_mscl,
    simulate_nested_logit as simulate_nested_logit,
    simulate_scl as simulate_scl,
)
from .models import (
    ChoiceModel as ChoiceModel,
    MixedLogit as MixedLogit,
    MixedSpatiallyCorrelatedLogit as MixedSpatiallyCorrelatedLogit,
    MultinomialLogit as MultinomialLogit,
    NestedLogit as NestedLogit,
    NestingTree as NestingTree,
    NestSpec as NestSpec,
    ParamDistribution as ParamDistribution,
    SpatiallyCorrelatedLogit as SpatiallyCorrelatedLogit,
    naturalize_nest_params as naturalize_nest_params,
)
from .results import (
    FitDiagnostics as FitDiagnostics,
    FitResult as FitResult,
    LikelihoodRatioTest as LikelihoodRatioTest,
    WaldTest as WaldTest,
    wald_test as wald_test,
)
from ._sampling.kernels import sample_alternatives as sample_alternatives
from .spec import (
    InteractionTerm as InteractionTerm,
    LinearFunction as LinearFunction,
    LinearTerm as LinearTerm,
    ModelSpec as ModelSpec,
    P as P,
    ParamRef as ParamRef,
    ScopedTerm as ScopedTerm,
    X as X,
    interaction as interaction,
)

__version__: str
