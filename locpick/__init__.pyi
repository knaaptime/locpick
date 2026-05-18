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
    MixedMNLDataset as MixedMNLDataset,
)
from .dgp import (
    MNLDataset as MNLDataset,
)
from .dgp import (
    MNSCLDataset as MNSCLDataset,
)
from .dgp import (
    MSCLDataset as MSCLDataset,
)
from .dgp import (
    NestedMNLDataset as NestedMNLDataset,
)
from .dgp import (
    NestedSCLDataset as NestedSCLDataset,
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
    simulate_mnscl as simulate_mnscl,
)
from .dgp import (
    simulate_mscl as simulate_mscl,
)
from .dgp import (
    simulate_nested_logit as simulate_nested_logit,
)
from .dgp import (
    simulate_nested_scl as simulate_nested_scl,
)
from .dgp import (
    simulate_scl as simulate_scl,
)
from .models import (
    MNL as MNL,
)
from .models import (
    SCL as SCL,
)
from .models import (
    ChoiceModel as ChoiceModel,
)
from .models import (
    MixedMNL as MixedMNL,
)
from .models import (
    MixedNestedSCL as MixedNestedSCL,
)
from .models import (
    MixedSCL as MixedSCL,
)
from .models import (
    NestedMNL as NestedMNL,
)
from .models import (
    NestedSCL as NestedSCL,
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
    naturalize_nest_params as naturalize_nest_params,
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
    lr_test as lr_test,
)
from .results import (
    wald_test as wald_test,
)
from .spec import (
    InteractionTerm as InteractionTerm,
)
from .spec import (
    ModelSpec as ModelSpec,
)
from .spec import (
    ScopedTerm as ScopedTerm,
)
from .spec import (
    interaction as interaction,
)

__version__: str
