"""Estimation-ready data containers for location choice models.

This module provides the ``ChoiceArrays`` dataclass, which holds JAX/NumPy
arrays ready for estimation. It is produced by ``ChoiceTable.to_arrays()``
and consumed by ``MultinomialLogit.fit()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp
import numpy as np

ArrayType = np.ndarray | jnp.ndarray  # type: ignore[name-defined]


@dataclass
class ChoiceArrays:
    """JAX/NumPy arrays ready for estimation.

    This is the estimation-ready representation of choice data. It is
    produced by ``ChoiceTable.to_arrays()`` and consumed by the solver
    in ``MultinomialLogit.fit()``.

    Parameters
    ----------
    design_matrix : array-like, shape (n_obs * n_alts, n_params)
        Design matrix of explanatory variables in long format. Each row
        corresponds to one (observation, alternative) pair.
    chosen : array-like, shape (n_obs, n_alts)
        Binary indicator matrix. ``chosen[i, j] = 1`` if observation *i*
        chose alternative *j*, 0 otherwise.
    available : array-like or None, shape (n_obs, n_alts)
        Binary availability matrix. ``available[i, j] = 1`` if alternative
        *j* is available to observation *i*. If None, all alternatives
        are assumed available.
    weights : array-like or None, shape (n_obs,)
        Observation-level weights. Each observation receives a single
        weight that scales its contribution to the log-likelihood.
        If None, unweighted estimation is performed (all observations
        weighted equally).
    n_obs : int
        Number of observations (choosers).
    n_alts : int
        Number of alternatives per observation (including sampled alternatives).
    param_names : list of str
        Names of the parameters (columns in the design matrix).
    obs_ids : array-like or None, shape (n_obs * n_alts,)
        Observation IDs corresponding to each row of the design matrix.
        If None, inferred from the shape.
    alt_ids : array-like or None, shape (n_obs * n_alts,)
        Alternative IDs corresponding to each row of the design matrix.
        If None, inferred from the shape.
    """

    design_matrix: ArrayType
    chosen: ArrayType
    available: Optional[ArrayType] = None
    weights: Optional[ArrayType] = None
    n_obs: int = 0
    n_alts: int = 0
    param_names: list[str] = field(default_factory=list)
    obs_ids: Optional[ArrayType] = None
    alt_ids: Optional[ArrayType] = None
    inclusion_probs: Optional[ArrayType] = None
    """Per-alternative inclusion probabilities for sampling correction.

    Shape ``(n_obs, n_alts)``. When alternatives are sampled from a
    larger choice set, ``inclusion_probs[i, j]`` gives the probability
    that alternative *j* was included in the choice set for observation *i*.
    The sampling correction term added to utilities is
    ``log(inclusion_probs)``.

    If None, no sampling correction is applied.
    """

    def __post_init__(self) -> None:
        """Infer and validate core array shapes."""
        if np.ndim(self.design_matrix) != 2:
            raise ValueError("design_matrix must be 2D with shape (n_obs * n_alts, n_params).")

        if self.chosen is not None and np.ndim(self.chosen) != 2:
            raise ValueError("chosen must be 2D with shape (n_obs, n_alts).")

        if self.n_obs == 0 and self.chosen is not None:
            shape = np.shape(self.chosen)
            if len(shape) == 2:
                self.n_obs, self.n_alts = shape

        if self.available is not None and np.shape(self.available) != (self.n_obs, self.n_alts):
            raise ValueError("available must have shape (n_obs, n_alts).")

        if self.inclusion_probs is not None and np.shape(self.inclusion_probs) != (
            self.n_obs,
            self.n_alts,
        ):
            raise ValueError("inclusion_probs must have shape (n_obs, n_alts).")

    @property
    def n_params(self) -> int:
        """Number of parameters (columns in the design matrix)."""
        return self.design_matrix.shape[1]

    def to_numpy(self) -> "ChoiceArrays":
        """Return a copy with all arrays converted to NumPy.

        This is useful for inspection and debugging when the arrays
        are stored as JAX arrays.
        """
        return ChoiceArrays(
            design_matrix=np.asarray(self.design_matrix),
            chosen=np.asarray(self.chosen),
            available=np.asarray(self.available) if self.available is not None else None,
            weights=np.asarray(self.weights) if self.weights is not None else None,
            n_obs=self.n_obs,
            n_alts=self.n_alts,
            param_names=list(self.param_names),
            obs_ids=np.asarray(self.obs_ids) if self.obs_ids is not None else None,
            alt_ids=np.asarray(self.alt_ids) if self.alt_ids is not None else None,
            inclusion_probs=np.asarray(self.inclusion_probs)
            if self.inclusion_probs is not None
            else None,
        )

    def __repr__(self) -> str:
        return f"ChoiceArrays(n_obs={self.n_obs}, n_alts={self.n_alts}, n_params={self.n_params})"
