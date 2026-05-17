"""Estimation problem container for location choice models.

This module provides the ``EstimationProblem`` dataclass, which binds
together data arrays, parameter metadata, and model configuration into
a single object that serves as the stable contract between data, spec,
and model kernels.

The flow is::

    ChoiceTable + ModelSpec → EstimationProblem → model kernel → FitResult

``ChoiceArrays`` remains the low-level numeric data container.
``EstimationProblem`` adds model-level structure on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from locpick.data.arrays import ChoiceArrays


@dataclass
class EstimationProblem:
    """Complete specification for a choice model estimation problem.

    Binds together data arrays, parameter metadata, and model configuration.
    This is the single object that model kernels receive — they never need
    to access ChoiceTable, ModelSpec, or solver configuration directly.

    Parameters
    ----------
    arrays : ChoiceArrays
        The numeric data container (design matrix, chosen, available, etc.).
    param_names : list of str
        Names of the estimated parameters, in design-matrix column order.
    param_bounds : list of (float, float) or None
        Per-parameter bounds for the optimizer. None means unbounded.
    param_fixed : list of bool or None
        Per-parameter fixed flags. True means the parameter is held at its
        initial value during estimation.
    param_initial : list of float or None
        Per-parameter initial values for the optimizer. None means 0.0.
    backend : {"numpy", "jax", "auto"}
        Computation backend preference. "auto" uses JAX if available.
    solver_name : str
        Name of the solver to use. Default "lbfgs".
    solver_options : dict or None
        Additional solver-specific options.
    model_type : str
        Model family identifier. Default "mnl". Reserved for future use
        with nested logit ("nl"), mixed logit ("mxl"), etc.
    sampling_design : dict or None
        Sampling provenance metadata (e.g., method, sample size, replace).
        This is estimation context and intentionally does not live on
        ``ChoiceArrays``.
    nest_ids : array-like or None
        Nest assignment for each (obs, alt) row. Reserved for nested logit.
    mixing_ids : array-like or None
        Group/panel IDs for mixed logit. Reserved for mixed logit.
    """

    arrays: ChoiceArrays
    param_names: list[str] = field(default_factory=list)
    param_bounds: Optional[list[tuple[float, float]]] = None
    param_fixed: Optional[list[bool]] = None
    param_initial: Optional[list[float]] = None
    backend: str = "auto"
    solver_name: str = "lbfgs"
    solver_options: Optional[dict] = None
    model_type: str = "mnl"
    sampling_design: Optional[dict[str, Any]] = None
    nest_ids: Optional[np.ndarray] = None
    mixing_ids: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate and infer defaults."""
        # Infer param_names from arrays if not provided
        if not self.param_names and self.arrays.param_names:
            self.param_names = list(self.arrays.param_names)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_choice_table(
        cls,
        choice_table,
        spec=None,
        formula: Optional[str] = None,
        weights: Optional[Union[str, np.ndarray]] = None,
        available: Optional[Union[str, np.ndarray]] = None,
        backend: str = "auto",
        solver_name: str = "lbfgs",
        solver_options: Optional[dict] = None,
    ) -> "EstimationProblem":
        """Create an EstimationProblem from a ChoiceTable and ModelSpec.

        Parameters
        ----------
        choice_table : ChoiceTable
            The choice data.
        spec : ModelSpec or None
            Model specification. Mutually exclusive with ``formula``.
        formula : str or None
            Formulaic formula string. Mutually exclusive with ``spec``.
        weights : str or array-like or None
            Observation weights.
        available : str or array-like or None
            Alternative availability.
        backend : {"numpy", "jax", "auto"}
            Computation backend preference.
        solver_name : str
            Solver name. Default "lbfgs".
        solver_options : dict or None
            Additional solver options.

        Returns
        -------
        EstimationProblem
        """
        from locpick.spec import ModelSpec

        # Resolve spec
        if spec is None and formula is not None:
            spec = ModelSpec(formula=formula)
        elif spec is not None and formula is not None:
            raise ValueError("Provide 'spec' or 'formula', not both.")

        # Build arrays via ChoiceTable
        arrays = choice_table.to_arrays(
            formula=formula,
            spec=spec,
            weights=weights,
            available=available,
        )

        # Extract param metadata from spec if available
        param_names = list(arrays.param_names) if arrays.param_names else []
        param_bounds = None
        param_fixed = None
        param_initial = None
        if spec is not None and hasattr(spec, "param_metadata"):
            meta = spec.param_metadata()
            if meta.get("bounds"):
                param_bounds = meta["bounds"]
            if meta.get("fixed"):
                param_fixed = meta["fixed"]
            if meta.get("initial"):
                param_initial = meta["initial"]

        # Sampling provenance is estimation context metadata, not array payload.
        sampling_design = choice_table.sampling_design

        return cls(
            arrays=arrays,
            param_names=param_names,
            param_bounds=param_bounds,
            param_fixed=param_fixed,
            param_initial=param_initial,
            backend=backend,
            solver_name=solver_name,
            solver_options=solver_options,
            sampling_design=sampling_design,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Number of estimated parameters."""
        return self.arrays.design_matrix.shape[1]

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self.arrays.n_obs

    @property
    def n_alts(self) -> int:
        """Number of alternatives per observation."""
        return self.arrays.n_alts

    @property
    def design_matrix(self) -> np.ndarray:
        """The design matrix, shape (n_obs * n_alts, n_params)."""
        return self.arrays.design_matrix

    @property
    def chosen(self) -> np.ndarray:
        """The chosen indicator matrix, shape (n_obs, n_alts)."""
        return self.arrays.chosen

    @property
    def available(self) -> Optional[np.ndarray]:
        """The availability matrix, shape (n_obs, n_alts), or None."""
        return self.arrays.available

    @property
    def weights(self) -> Optional[np.ndarray]:
        """Observation-level weights, shape (n_obs,), or None."""
        return self.arrays.weights

    @property
    def inclusion_probs(self) -> Optional[np.ndarray]:
        """Sampling inclusion probabilities, shape (n_obs, n_alts), or None."""
        return self.arrays.inclusion_probs

    @property
    def initial_values(self) -> np.ndarray:
        """Initial parameter values for the optimizer.

        Uses param_initial if provided, otherwise zeros.
        Fixed parameters are set to their specified values.
        """
        n = self.n_params
        x0 = np.zeros(n, dtype=np.float64)
        if self.param_initial is not None:
            for i, v in enumerate(self.param_initial):
                if i < n:
                    x0[i] = v
        return x0

    @property
    def bounds(self) -> Optional[list[tuple[float, float]]]:
        """Per-parameter bounds for the optimizer.

        Returns None if all bounds are the default (-1e6, 1e6).
        """
        if self.param_bounds is None:
            return None
        return self.param_bounds

    @property
    def fixed_mask(self) -> Optional[np.ndarray]:
        """Boolean mask of fixed parameters, shape (n_params,).

        True means the parameter is held fixed during estimation.
        Returns None if no parameters are fixed.
        """
        if self.param_fixed is None:
            return None
        return np.array(self.param_fixed, dtype=bool)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EstimationProblem("
            f"n_obs={self.n_obs}, n_alts={self.n_alts}, "
            f"n_params={self.n_params}, "
            f"model_type={self.model_type!r}, "
            f"backend={self.backend!r})"
        )
