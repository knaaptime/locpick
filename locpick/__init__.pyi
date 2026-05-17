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
from ._sampling.kernels import sample_alternatives as sample_alternatives
from .data import (
    ChoiceArrays as ChoiceArrays,
)
from .data import (
    ChoiceTable as ChoiceTable,
)
from .data import (
    EstimationProblem as EstimationProblem,
)
from .data import (
    distance_bands as distance_bands,
)
from .data import (
    distance_matrix as distance_matrix,
)
from .data import (
    euclidean_distance_matrix as euclidean_distance_matrix,
)
from .data import (
    format_coefficient_table as format_coefficient_table,
)
from .data import (
    format_fit_statistics as format_fit_statistics,
)
from .data import (
    format_side_by_side as format_side_by_side,
)
from .data import (
    great_circle_distance_matrix as great_circle_distance_matrix,
)
from .data import (
    nearest_neighbors as nearest_neighbors,
)
from .data import (
    pairwise_distance as pairwise_distance,
)
from .dgp import (
    MixedLogitDataset as MixedLogitDataset,
)
from .dgp import (
    MNLDataset as MNLDataset,
)
from .dgp import (
    MSCLDataset as MSCLDataset,
)
from .dgp import (
    NestedLogitDataset as NestedLogitDataset,
)
from .dgp import (
    SCLDataset as SCLDataset,
)
from .dgp import (
    simulate_mixed_logit as simulate_mixed_logit,
)
from .dgp import (
    simulate_mnl as simulate_mnl,
)
from .dgp import (
    simulate_mscl as simulate_mscl,
)
from .dgp import (
    simulate_nested_logit as simulate_nested_logit,
)
from .dgp import (
    simulate_scl as simulate_scl,
)
from .models import (
    ChoiceModel as ChoiceModel,
)
from .models import (
    MixedLogit as MixedLogit,
)
from .models import (
    MixedSpatiallyCorrelatedLogit as MixedSpatiallyCorrelatedLogit,
)
from .models import (
    MultinomialLogit as MultinomialLogit,
)
from .models import (
    NestedLogit as NestedLogit,
)
from .models import (
    NestingTree as NestingTree,
)
from .models import (
    NestSpec as NestSpec,
)
from .models import (
    ParamDistribution as ParamDistribution,
)
from .models import (
    SpatiallyCorrelatedLogit as SpatiallyCorrelatedLogit,
)
from .models import (
    naturalize_nest_params as naturalize_nest_params,
)
from .results import (
    FitDiagnostics as FitDiagnostics,
)
from .results import (
    FitResult as FitResult,
)
from .results import (
    LikelihoodRatioTest as LikelihoodRatioTest,
)
from .results import (
    WaldTest as WaldTest,
)
from .results import (
    wald_test as wald_test,
)
from .spec import (
    InteractionTerm as InteractionTerm,
)
from .spec import (
    LinearFunction as LinearFunction,
)
from .spec import (
    LinearTerm as LinearTerm,
)
from .spec import (
    ModelSpec as ModelSpec,
)
from .spec import (
    P as P,
)
from .spec import (
    ParamRef as ParamRef,
)
from .spec import (
    ScopedTerm as ScopedTerm,
)
from .spec import (
    X as X,
)
from .spec import (
    interaction as interaction,
)

__version__: str
