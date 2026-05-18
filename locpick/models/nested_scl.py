"""Nested Spatially Correlated Logit (Nested SCL) model.

This module provides the :class:`NestedSpatiallyCorrelatedLogit` class,
which is a convenience alias for :class:`SpatiallyCorrelatedLogit` with
``nests`` specified.  All estimation logic lives in
:mod:`locpick.models.scl`.

References
----------
Al-Haideri et al. (2026). Cyclists' crossing behaviour at roundabouts:
    A Generalized Spatially Correlated Nested Logit model.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from locpick._solvers import Solver
from locpick.models.nested import NestingTree
from locpick.models.scl import SCL


class NestedSCL(SCL):
    r"""Nested Spatially Correlated Logit model for location choice estimation.

    This model combines a nested logit upper level with spatially correlated
    logit (SCL) lower levels. Each nest :math:`m` has:

    - A spatial dissimilarity parameter :math:`\rho_m \in (0, 1]` governing
      correlation between spatially adjacent alternatives within the nest.
    - A nest dissimilarity parameter :math:`\lambda_m \in (0, 1]` governing
      correlation between alternatives in the same nest.

    The choice probability for alternative :math:`i` in nest :math:`m` is:

    .. math::

        P_i = P_{\text{SCL}}(i \mid m) \times P_{\text{NL}}(m)

    where :math:`P_{\text{SCL}}(i \mid m)` is the SCL conditional probability
    within nest :math:`m` and :math:`P_{\text{NL}}(m)` is the nested logit
    probability of choosing nest :math:`m`.

    This class is a thin convenience wrapper around
    :class:`SpatiallyCorrelatedLogit` that requires ``nests`` to be specified.

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
        Can be a ``libpysal.graph.Graph`` (recommended), a ``scipy.sparse``
        array, or a dense NumPy array.
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
    >>> from locpick import ChoiceTable, NestedSpatiallyCorrelatedLogit
    >>> from locpick.models.nested import NestingTree, NestSpec
    >>> from libpysal import graph
    >>> ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=10)
    >>> nests = NestingTree([NestSpec("inner", [0, 1, 2]), NestSpec("outer", [3, 4, 5])])
    >>> g = graph.Graph.build_contiguity(tracts_gdf, rook=False)
    >>> model = NestedSpatiallyCorrelatedLogit(ct, formula="cost + time", nests=nests, graph=g)
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
        weights: Optional[Union[str, np.ndarray]] = None,
        availability: Optional[Union[str, np.ndarray]] = None,
        solver: Union[str, Solver] = None,
        solver_options: Optional[dict] = None,
        backend: Optional[str] = None,
    ):
        if nests is None:
            raise ValueError(
                "NestedSpatiallyCorrelatedLogit requires a 'nests' argument "
                "specifying the nesting structure."
            )
        if graph is None:
            raise ValueError(
                "NestedSpatiallyCorrelatedLogit requires a 'graph' argument "
                "specifying the spatial adjacency structure."
            )

        super().__init__(
            data=data,
            formula=formula,
            spec=spec,
            graph=graph,
            nests=nests,
            random_params=None,
            n_draws=None,
            draw_type=None,
            weights=weights,
            availability=availability,
            solver=solver,
            solver_options=solver_options,
            backend=backend,
        )
