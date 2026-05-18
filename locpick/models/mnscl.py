"""Mixed Nested Spatially Correlated Logit (MNSCL) model for location choice estimation.

This module provides the :class:`MixedNestedSpatiallyCorrelatedLogit` class,
which is a convenience alias for :class:`SpatiallyCorrelatedLogit` with
both ``nests`` and ``random_params`` specified.  All estimation logic lives
in :mod:`locpick.models.scl`.

References
----------
Al-Haideri et al. (2026). Cyclists' crossing behaviour at roundabouts:
    A Generalized Spatially Correlated Nested Logit model.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from locpick._solvers import Solver
from locpick.models.mixed import ParamDistribution
from locpick.models.nested import NestingTree
from locpick.models.scl import SCL


class MixedNestedSCL(SCL):
    r"""Mixed Nested Spatially Correlated Logit model for location choice estimation.

    This model combines three structures:

    1. **Nested logit upper level**: alternatives are grouped into nests,
       each with a nest dissimilarity parameter :math:`\lambda_m \in (0, 1]`.
    2. **SCL lower levels**: within each nest, spatial correlation between
       contiguous alternatives is captured via a paired GNL structure with
       nest-specific spatial dissimilarity :math:`\rho_m \in (0, 1]`.
    3. **Random coefficients**: unobserved heterogeneity is captured via
       mixed logit with simulated maximum likelihood.

    The choice probability for alternative :math:`i` in nest :math:`m` is:

    .. math::

        P_i = \frac{1}{R} \sum_{r=1}^{R} P_{\text{SCL}}(i \mid m, \beta^r)
              \times P_{\text{NL}}(m \mid \beta^r)

    where :math:`\beta^r` is the :math:`r`-th draw of the random coefficients.

    This class is a thin convenience wrapper around
    :class:`SpatiallyCorrelatedLogit` that requires both ``nests`` and
    ``random_params`` to be specified.

    Parameters
    ----------
    data : ChoiceTable
        The choice data to estimate on.
    formula : str, optional
        A formulaic formula string for the utility function.
    spec : ModelSpec, optional
        A ModelSpec object defining the utility function.
    nests : NestingTree
        The nesting structure. Each alternative belongs to exactly one nest.
    graph : libpysal.graph.Graph, scipy.sparse array, or np.ndarray
        Spatial adjacency structure encoding contiguity between alternatives.
    random_params : dict, optional
        Mapping of parameter names to :class:`ParamDistribution` for
        mixed logit.  E.g. ``{"commute_time": ParamDistribution("normal")}``.
    n_draws : int, optional
        Number of draws for simulated maximum likelihood.  Default 50.
    draw_type : str, optional
        Type of draws: ``"qmc"`` (default), ``"halton"``, or ``"random"``.
    weights : str or array-like, optional
        Observation weights.
    availability : str or array-like, optional
        Alternative availability.
    solver : str or Solver, optional
        Solver name or instance.  Default ``"lbfgs"``.
    solver_options : dict, optional
        Additional options passed to the solver constructor.
    backend : str, optional
        Computation backend.  Default is ``"jax"``.

    Examples
    --------
    >>> from locpick import ChoiceTable, MixedNestedSpatiallyCorrelatedLogit
    >>> from locpick.models.nested import NestingTree, NestSpec
    >>> from locpick.models.mixed import ParamDistribution
    >>> from libpysal import graph
    >>> ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=10)
    >>> nests = NestingTree([NestSpec("inner", [0, 1, 2]), NestSpec("outer", [3, 4, 5])])
    >>> g = graph.Graph.build_contiguity(tracts_gdf, rook=False)
    >>> model = MixedNestedSpatiallyCorrelatedLogit(
    ...     ct,
    ...     formula="cost + time",
    ...     nests=nests,
    ...     graph=g,
    ...     random_params={"time": ParamDistribution("normal")},
    ...     n_draws=100,
    ... )
    >>> result = model.fit()
    >>> print(result.summary())
    """

    def __init__(
        self,
        data,
        formula: Optional[str] = None,
        spec=None,
        nests: Optional[NestingTree] = None,
        graph: Any = None,
        random_params: Optional[dict[str, ParamDistribution]] = None,
        n_draws: Optional[int] = None,
        draw_type: Optional[str] = None,
        weights: Optional[Union[str, np.ndarray]] = None,
        availability: Optional[Union[str, np.ndarray]] = None,
        solver: Union[str, Solver] = None,
        solver_options: Optional[dict] = None,
        backend: Optional[str] = None,
    ):
        if nests is None:
            raise ValueError(
                "MixedNestedSpatiallyCorrelatedLogit requires a 'nests' argument "
                "specifying the nesting structure."
            )
        if graph is None:
            raise ValueError(
                "MixedNestedSpatiallyCorrelatedLogit requires a 'graph' argument "
                "specifying the spatial adjacency structure."
            )

        super().__init__(
            data=data,
            formula=formula,
            spec=spec,
            graph=graph,
            nests=nests,
            random_params=random_params,
            n_draws=n_draws,
            draw_type=draw_type,
            weights=weights,
            availability=availability,
            solver=solver,
            solver_options=solver_options,
            backend=backend,
        )
