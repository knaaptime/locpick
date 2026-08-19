"""Data generating process (DGP) utilities for synthetic choice data.

This module provides functions for generating synthetic choice data with
known ground-truth parameters for each model type.  Intended for parameter
recovery tests and simulation studies.

Supported DGPs
--------------
- MNL (multinomial logit)
- Nested logit
- SCL (spatially correlated logit)
- Nested SCL (spatially correlated logit with nesting)
- Mixed logit (random coefficients)
- MSCL (mixed spatially correlated logit)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp

# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MNLDataset:
    """Synthetic dataset drawn from a known MNL data generating process.

    Attributes
    ----------
    choosers : pd.DataFrame
        Obs-id-indexed DataFrame containing chooser-level attributes and a
        ``choice`` column (integer index of chosen alternative).
    alternatives : pd.DataFrame
        Alt-id-indexed DataFrame containing alternative-level attributes.
    interactions : dict[str, pd.Series]
        Named ``(obs_id, alt_id)``-indexed Series representing chooser ×
        alternative interaction terms.
    true_params : dict[str, float]
        Ground-truth parameter values used to generate the data.  Keys match
        column names in the long-format table so assertions can be written as::

            npt.assert_allclose(result.coefficients["alt_feature"],
                                dataset.true_params["alt_feature"], rtol=0.10)
    choice_table : object
        Assembled v2 ``ChoiceTable`` (no sampling).
    n_obs : int
    n_alts : int
    seed : int
    """

    choosers: pd.DataFrame
    alternatives: pd.DataFrame
    interactions: dict[str, pd.Series]
    true_params: dict[str, float]
    choice_table: Any  # ChoiceTable (lazy import avoids circular deps)
    n_obs: int
    n_alts: int
    seed: int


@dataclass
class NestedMNLDataset:
    """Synthetic dataset drawn from a known nested logit DGP.

    Attributes
    ----------
    choosers : pd.DataFrame
        Obs-id-indexed DataFrame with chooser attributes and ``choice`` column.
    alternatives : pd.DataFrame
        Alt-id-indexed DataFrame with alternative attributes.
    true_params : dict[str, float]
        Ground-truth beta coefficients.
    true_lambdas : dict[str, float]
        Ground-truth nest dissimilarity parameters (λ ∈ (0, 1]).
    nests : NestingTree
        The nesting structure used to generate the data.
    choice_table : object
        Assembled ``ChoiceTable``.
    n_obs : int
    n_alts : int
    seed : int
    """

    choosers: pd.DataFrame
    alternatives: pd.DataFrame
    true_params: dict[str, float]
    true_lambdas: dict[str, float]
    nests: Any  # NestingTree
    choice_table: Any
    n_obs: int
    n_alts: int
    seed: int


@dataclass
class SCLDataset:
    """Synthetic dataset drawn from a known SCL DGP.

    Attributes
    ----------
    choosers : pd.DataFrame
        Obs-id-indexed DataFrame with chooser attributes and ``choice`` column.
    alternatives : pd.DataFrame
        Alt-id-indexed DataFrame with alternative attributes.
    true_params : dict[str, float]
        Ground-truth beta coefficients.
    true_rho : float
        Ground-truth dissimilarity parameter ρ ∈ (0, 1].
    adjacency : np.ndarray
        Binary adjacency matrix used for the spatial graph.
    choice_table : object
        Assembled ``ChoiceTable``.
    n_obs : int
    n_alts : int
    seed : int
    """

    choosers: pd.DataFrame
    alternatives: pd.DataFrame
    true_params: dict[str, float]
    true_rho: float
    adjacency: np.ndarray
    choice_table: Any
    n_obs: int
    n_alts: int
    seed: int


@dataclass
class MixedMNLDataset:
    """Synthetic dataset drawn from a known mixed logit DGP.

    Attributes
    ----------
    choosers : pd.DataFrame
        Obs-id-indexed DataFrame with chooser attributes and ``choice`` column.
    alternatives : pd.DataFrame
        Alt-id-indexed DataFrame with alternative attributes.
    true_params : dict[str, float]
        Ground-truth parameter values.  Fixed coefficients use their column
        names as keys; random coefficients use ``"mean_<name>"`` and
        ``"sd_<name>"`` keys.
    random_params : dict[str, str]
        Mapping of random parameter name → distribution name.
    choice_table : object
        Assembled ``ChoiceTable``.
    n_obs : int
    n_alts : int
    seed : int
    """

    choosers: pd.DataFrame
    alternatives: pd.DataFrame
    true_params: dict[str, float]
    random_params: dict[str, str]
    choice_table: Any
    n_obs: int
    n_alts: int
    seed: int


@dataclass
class MixedNestedMNLDataset:
    """Synthetic dataset drawn from a known mixed nested logit DGP.

    Attributes
    ----------
    choosers : pd.DataFrame
        Obs-id-indexed DataFrame with chooser attributes and ``choice`` column.
    alternatives : pd.DataFrame
        Alt-id-indexed DataFrame with alternative attributes.
    true_params : dict[str, float]
        Ground-truth fixed beta coefficients.
    true_lambdas : dict[str, float]
        Ground-truth nest dissimilarity parameters λ_m ∈ (0, 1].
    true_random_means : dict[str, float]
        Ground-truth random coefficient means.
    true_random_spreads : dict[str, float]
        Ground-truth random coefficient spreads (std devs).
    random_params : dict[str, str]
        Mapping of random parameter name → distribution name.
    nests : NestingTree
        The nesting structure used to generate the data.
    choice_table : object
        Assembled ``ChoiceTable``.
    n_obs : int
    n_alts : int
    seed : int
    """

    choosers: pd.DataFrame
    alternatives: pd.DataFrame
    true_params: dict[str, float]
    true_lambdas: dict[str, float]
    true_random_means: dict[str, float]
    true_random_spreads: dict[str, float]
    random_params: dict[str, str]
    nests: Any
    choice_table: Any
    n_obs: int
    n_alts: int
    seed: int


@dataclass
class MSCLDataset:
    """Synthetic dataset drawn from a known MSCL DGP.

    Attributes
    ----------
    choosers : pd.DataFrame
        Obs-id-indexed DataFrame with chooser attributes and ``choice`` column.
    alternatives : pd.DataFrame
        Alt-id-indexed DataFrame with alternative attributes.
    true_params : dict[str, float]
        Ground-truth parameter values.  Fixed coefficients use their column
        names; random coefficients use ``"mean_<name>"`` and ``"sd_<name>"``.
    true_rho : float
        Ground-truth dissimilarity parameter ρ ∈ (0, 1].
    adjacency : np.ndarray
        Binary adjacency matrix used for the spatial graph.
    random_params : dict[str, str]
        Mapping of random parameter name → distribution name.
    choice_table : object
        Assembled ``ChoiceTable``.
    n_obs : int
    n_alts : int
    seed : int
    """

    choosers: pd.DataFrame
    alternatives: pd.DataFrame
    true_params: dict[str, float]
    true_rho: float
    adjacency: np.ndarray
    random_params: dict[str, str]
    choice_table: Any
    n_obs: int
    n_alts: int
    seed: int


@dataclass
class NestedSCLDataset:
    """Synthetic dataset drawn from a known Nested SCL DGP.

    Attributes
    ----------
    choosers : pd.DataFrame
        Obs-id-indexed DataFrame with chooser attributes and ``choice`` column.
    alternatives : pd.DataFrame
        Alt-id-indexed DataFrame with alternative attributes.
    true_params : dict[str, float]
        Ground-truth beta coefficients.
    true_rhos : dict[str, float]
        Ground-truth spatial dissimilarity parameters ρ_m ∈ (0, 1] per nest.
    true_lambdas : dict[str, float]
        Ground-truth nest dissimilarity parameters λ_m ∈ (0, 1] per nest.
    nests : NestingTree
        The nesting structure used to generate the data.
    adjacency : np.ndarray
        Binary adjacency matrix used for the spatial graph.
    choice_table : object
        Assembled ``ChoiceTable``.
    n_obs : int
    n_alts : int
    seed : int
    """

    choosers: pd.DataFrame
    alternatives: pd.DataFrame
    true_params: dict[str, float]
    true_rhos: dict[str, float]
    true_lambdas: dict[str, float]
    nests: Any
    adjacency: np.ndarray
    choice_table: Any
    n_obs: int
    n_alts: int
    seed: int


@dataclass
class MNSCLDataset:
    """Synthetic dataset drawn from a known MNSCL DGP.

    Attributes
    ----------
    choosers : pd.DataFrame
        Obs-id-indexed DataFrame with chooser attributes and ``choice`` column.
    alternatives : pd.DataFrame
        Alt-id-indexed DataFrame with alternative attributes.
    true_params : dict[str, float]
        Ground-truth fixed beta coefficients.
    true_rhos : dict[str, float]
        Ground-truth spatial dissimilarity parameters ρ_m ∈ (0, 1] per nest.
    true_lambdas : dict[str, float]
        Ground-truth nest dissimilarity parameters λ_m ∈ (0, 1] per nest.
    true_random_means : dict[str, float]
        Ground-truth random coefficient means.
    true_random_spreads : dict[str, float]
        Ground-truth random coefficient spreads (std devs).
    random_params : dict[str, str]
        Mapping of random parameter name → distribution name.
    nests : NestingTree
        The nesting structure used to generate the data.
    adjacency : np.ndarray
        Binary adjacency matrix used for the spatial graph.
    choice_table : object
        Assembled ``ChoiceTable``.
    n_obs : int
    n_alts : int
    seed : int
    """

    choosers: pd.DataFrame
    alternatives: pd.DataFrame
    true_params: dict[str, float]
    true_rhos: dict[str, float]
    true_lambdas: dict[str, float]
    true_random_means: dict[str, float]
    true_random_spreads: dict[str, float]
    random_params: dict[str, str]
    nests: Any
    adjacency: np.ndarray
    choice_table: Any
    n_obs: int
    n_alts: int
    seed: int


# ---------------------------------------------------------------------------
# Helper: build a ChoiceTable from choosers / alternatives / choices
# ---------------------------------------------------------------------------


def _build_choice_table(choosers, alternatives, choices, matrix_data=None):
    """Build a ChoiceTable from component DataFrames."""
    from .data.choicetable import ChoiceTable

    return ChoiceTable.from_tables(
        choosers=choosers.drop(columns="choice", errors="ignore"),
        alternatives=alternatives,
        chosen_alternatives=choices,
        matrix_data=matrix_data,
    )


# ---------------------------------------------------------------------------
# Shared DGP helpers
# ---------------------------------------------------------------------------


def _build_choosers(n_obs, seed, feature_name="obs_feature"):
    """Build choosers DataFrame with obs_id index and a random feature."""
    rng = np.random.default_rng(seed)
    obs_ids = pd.Index(np.arange(n_obs), name="oid")
    choosers = pd.DataFrame({feature_name: rng.standard_normal(n_obs)}, index=obs_ids)
    return choosers, rng, obs_ids


def _build_alternatives(n_alts, rng, alt_features):
    """Build alternatives DataFrame with alt_id index.

    Parameters
    ----------
    n_alts : int
    rng : np.random.Generator
    alt_features : dict[str, tuple]
        Mapping of column name → (low, high) for uniform draw.
    """
    alt_ids = pd.Index(np.arange(n_alts), name="aid")
    data = {}
    for col, (low, high) in alt_features.items():
        data[col] = rng.uniform(low, high, n_alts)
    alternatives = pd.DataFrame(data, index=alt_ids)
    return alternatives, alt_ids


def _simulate_choices_from_probs(probs, rng, n_obs, n_alts):
    """Vectorized choice simulation from probability matrix.

    Replaces the Python loop ``np.array([rng.choice(n_alts, p=probs[i]) for i in range(n_obs)])``
    with vectorized inverse-CDF sampling.
    """
    # Normalize to sum to 1 (numerical safety)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(probs, axis=1)
    uniform = rng.random(n_obs)
    choices = np.argmax(cumulative > uniform[:, None], axis=1)
    return np.clip(choices, 0, n_alts - 1)


def _build_circular_adjacency(n_alts):
    """Build a circular adjacency matrix where zone i is adjacent to i±1."""
    adjacency = np.zeros((n_alts, n_alts), dtype=np.float64)
    for i in range(n_alts):
        adjacency[i, (i - 1) % n_alts] = 1.0
        adjacency[i, (i + 1) % n_alts] = 1.0
    return adjacency


def _generate_alternatives(alt_params, n_alts, rng):
    """Generate alternatives DataFrame with columns from alt_params keys."""
    alt_ids = pd.Index(np.arange(n_alts), name="aid")
    alt_data = {col: rng.uniform(1, 10, n_alts) for col in alt_params}
    return pd.DataFrame(alt_data, index=alt_ids), alt_ids


def _generate_interactions(interaction_params, choosers, alternatives, obs_ids, alt_ids, rng):
    """Generate interaction Series from interaction_params keys.

    Interaction names follow the convention ``"{chooser_col}_x_{alt_col}"``.
    The chooser column must exist in ``choosers``, and the alt column must
    exist in ``alternatives``.
    """
    n_obs = len(obs_ids)
    n_alts = len(alt_ids)
    interaction_index = pd.MultiIndex.from_product([obs_ids, alt_ids], names=["oid", "aid"])
    interactions = {}
    for name in interaction_params:
        parts = name.split("_x_", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Interaction name '{name}' must follow '{{chooser}}_x_{{alt}}' convention"
            )
        chooser_col, alt_col = parts
        if chooser_col not in choosers.columns:
            raise KeyError(
                f"Interaction '{name}' references chooser column '{chooser_col}' "
                f"not in choosers. Available: {list(choosers.columns)}"
            )
        if alt_col not in alternatives.columns:
            raise KeyError(
                f"Interaction '{name}' references alt column '{alt_col}' "
                f"not in alternatives. Available: {list(alternatives.columns)}"
            )
        chooser_tiled = np.repeat(choosers[chooser_col].to_numpy(), n_alts)
        alt_tiled = np.tile(alternatives[alt_col].to_numpy(), n_obs)
        interactions[name] = pd.Series(
            chooser_tiled * alt_tiled, index=interaction_index, name=name
        )
    return interactions


def _ensure_random_param_columns(alt_params, random_params, n_alts, rng):
    """Ensure ``alt_params`` includes all columns referenced by ``random_params``.

    Random-coefficient columns must exist in the generated ``alternatives``
    frame (``_generate_alternatives`` builds columns from ``alt_params`` keys),
    but they must NOT carry a *fixed* coefficient — their effect comes entirely
    from the random draw.  A missing column is therefore added with a fixed
    coefficient of ``0.0`` so the deterministic-utility loop contributes nothing
    for it (a non-zero value would inject a spurious alternative-specific term).
    """
    alt_params = dict(alt_params)
    for col in random_params:
        if col not in alt_params:
            alt_params[col] = 0.0
    return alt_params


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MNL DGP
# ---------------------------------------------------------------------------


def simulate_mnl(
    n_obs: int = 3000,
    n_alts: int = 6,
    alt_params: dict[str, float] | None = None,
    interaction_params: dict[str, float] | None = None,
    seed: int = 1234,
) -> MNLDataset:
    """Generate synthetic MNL choice data with known parameters.

    Default DGP
    -----------
    * ``alt_feature`` — linearly spaced from -1 to 1 across ``n_alts``
    * ``obs_feature`` — i.i.d. Normal(0, 1) across ``n_obs``
    * Interaction terms are created only when ``interaction_params`` is given.
      Each key follows the ``{chooser}_x_{alt}`` convention (e.g.
      ``"obs_feature_x_alt_feature"``) and is the element-wise product of the
      named chooser and alternative columns, broadcast over the full
      ``(n_obs × n_alts)`` grid.
    * Deterministic utility::

          U_det = alt_params["alt_feature"] * alt_feature
                + sum(interaction_params[name] * interaction[name])

    * Stochastic utility: ``U_det + Gumbel(0, 1)``
    * Chosen alternative: ``argmax`` of stochastic utility per chooser

    Parameters
    ----------
    n_obs : int, default 3000
    n_alts : int, default 6
    alt_params : dict, optional
        Mapping of alternative-level column name → true coefficient.
        Default: ``{"alt_feature": -0.65}``.
    interaction_params : dict, optional
        Mapping of interaction column name (``{chooser}_x_{alt}``) → true
        coefficient.  Default: ``{}`` (no interaction terms).  A formula that
        references an interaction column requires that column to be listed
        here; interactions are never created implicitly.
    seed : int, default 1234

    Returns
    -------
    MNLDataset
    """
    if alt_params is None:
        alt_params = {"alt_feature": -0.65}
    if interaction_params is None:
        interaction_params = {}

    rng = np.random.default_rng(seed)

    # --- Choosers -------------------------------------------------------
    obs_ids = pd.Index(np.arange(n_obs), name="oid")
    obs_feature = rng.standard_normal(n_obs)
    choosers = pd.DataFrame({"obs_feature": obs_feature}, index=obs_ids)

    # --- Alternatives (columns from alt_params keys) -------------------
    alt_ids = pd.Index(np.arange(n_alts), name="aid")
    alt_data = {}
    for col in alt_params:
        alt_data[col] = (
            np.linspace(-1.0, 1.0, n_alts) if col == "alt_feature" else rng.uniform(1, 10, n_alts)
        )
    alternatives = pd.DataFrame(alt_data, index=alt_ids)

    # --- Interactions (from interaction_params keys) ------------------
    interaction_index = pd.MultiIndex.from_product([obs_ids, alt_ids], names=["oid", "aid"])
    obs_feat_tiled = np.repeat(obs_feature, n_alts)
    interactions: dict[str, pd.Series] = {}
    for name in interaction_params:
        parts = name.split("_x_", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Interaction name '{name}' must follow '{{chooser}}_x_{{alt}}' convention"
            )
        chooser_col, alt_col = parts
        if alt_col not in alternatives.columns:
            raise KeyError(
                f"Interaction '{name}' references alt column '{alt_col}' not in alternatives. Available: {list(alternatives.columns)}"
            )
        alt_tiled = np.tile(alternatives[alt_col].to_numpy(), n_obs)
        interactions[name] = pd.Series(
            obs_feat_tiled * alt_tiled, index=interaction_index, name=name
        )

    # --- Deterministic utility ------------------------------------------
    det_utility = np.zeros((n_obs, n_alts))
    for col, coef in alt_params.items():
        alt_vals = alternatives[col].to_numpy()
        det_utility += coef * np.tile(alt_vals, n_obs).reshape(n_obs, n_alts)
    for name, coef in interaction_params.items():
        det_utility += coef * interactions[name].to_numpy().reshape(n_obs, n_alts)

    # --- Simulate choices -----------------------------------------------
    gumbel_noise = rng.gumbel(size=(n_obs, n_alts))
    sim_utility = det_utility + gumbel_noise
    choice_idx = sim_utility.argmax(axis=1)  # integer index of chosen alt
    choosers = choosers.copy()
    choosers["choice"] = choice_idx

    # --- True params ----------------------------------------------------
    true_params: dict[str, float] = {}
    true_params.update(alt_params)
    true_params.update(interaction_params)

    # --- v2 ChoiceTable -------------------------------------------------
    choice_table = _build_choice_table(
        choosers, alternatives, choosers["choice"], matrix_data=interactions
    )

    return MNLDataset(
        choosers=choosers,
        alternatives=alternatives,
        interactions=interactions,
        true_params=true_params,
        choice_table=choice_table,
        n_obs=n_obs,
        n_alts=n_alts,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Nested logit DGP
# ---------------------------------------------------------------------------


def simulate_nested_logit(
    n_obs: int = 10000,
    n_alts: int = 4,
    alt_params: dict[str, float] | None = None,
    nest_lambdas: dict[str, float] | None = None,
    interaction_params: dict[str, float] | None = None,
    seed: int = 1234,
) -> NestedMNLDataset:
    """Generate synthetic nested logit choice data with known parameters.

    The default DGP creates two nests with 2 alternatives each, includes
    both alternative-level and chooser×alternative interaction terms,
    and simulates choices using the nested logit probability formula.

    Parameters
    ----------
    n_obs : int, default 10000
        Number of observations (decision-makers).
    n_alts : int, default 4
        Number of alternatives.  Must be evenly divisible by the number of
        nests in ``nest_lambdas``.
    alt_params : dict, optional
        Mapping of alternative-level column name → true coefficient.
        Default: ``{"cost": -0.5, "time": -0.1}``.
    nest_lambdas : dict, optional
        Mapping of nest name → true dissimilarity parameter λ ∈ (0, 1].
        Default: ``{"transit": 0.7, "auto": 0.8}``.
    seed : int, default 1234
        Random seed for reproducibility.

    Returns
    -------
    NestedLogitDataset
    """
    from .models.nested import (
        NestingTree,
        NestSpec,
        _nested_logit_probs_numpy,
    )

    if alt_params is None:
        alt_params = {"cost": -0.5, "time": -0.1}
    if nest_lambdas is None:
        nest_lambdas = {"transit": 0.7, "auto": 0.8}
    if interaction_params is None:
        interaction_params = {}

    rng = np.random.default_rng(seed)

    # --- Build nesting structure ----------------------------------------
    nest_names = list(nest_lambdas.keys())
    alts_per_nest = n_alts // len(nest_names)
    if n_alts % len(nest_names) != 0:
        raise ValueError(
            f"n_alts ({n_alts}) must be evenly divisible by the number of "
            f"nests ({len(nest_names)})."
        )

    nest_specs = []
    alt_id = 0
    for name in nest_names:
        nest_specs.append(NestSpec(name, alt_ids=list(range(alt_id, alt_id + alts_per_nest))))
        alt_id += alts_per_nest
    nests = NestingTree(nest_specs)

    # --- Choosers and alternatives --------------------------------------
    obs_ids = pd.Index(np.arange(n_obs), name="oid")
    income = rng.standard_normal(n_obs)
    choosers = pd.DataFrame({"income": income}, index=obs_ids)

    alt_ids = pd.Index(np.arange(n_alts), name="aid")
    alt_data = {col: rng.uniform(1, 10, n_alts) for col in alt_params}
    alternatives = pd.DataFrame(alt_data, index=alt_ids)

    # --- Interactions (from interaction_params keys) ------------------
    interaction_index = pd.MultiIndex.from_product([obs_ids, alt_ids], names=["oid", "aid"])
    income_tiled = np.repeat(income, n_alts)
    interactions = {}
    for name in interaction_params:
        parts = name.split("_x_", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Interaction name '{name}' must follow '{{chooser}}_x_{{alt}}' convention"
            )
        chooser_col, alt_col = parts
        if alt_col not in alternatives.columns:
            raise KeyError(
                f"Interaction '{name}' references alt column '{alt_col}' not in alternatives. Available: {list(alternatives.columns)}"
            )
        alt_tiled = np.tile(alternatives[alt_col].to_numpy(), n_obs)
        interactions[name] = pd.Series(
            income_tiled * alt_tiled, index=interaction_index, name=name
        )

    # --- Deterministic utility ------------------------------------------
    det_utility = np.zeros((n_obs, n_alts))
    for col, coef in alt_params.items():
        alt_vals = alternatives[col].to_numpy()
        det_utility += coef * np.tile(alt_vals, n_obs).reshape(n_obs, n_alts)
    for name, coef in interaction_params.items():
        det_utility += coef * interactions[name].to_numpy().reshape(n_obs, n_alts)

    # --- Build nest matrix and simulate choices -------------------------
    alt_id_list = list(range(n_alts))
    nest_matrix = nests.build_nest_matrix(alt_id_list)

    # Convert lambdas to alpha (unconstrained) for the probability kernel
    lambda_values = np.array([nest_lambdas[name] for name in nest_names])
    # alpha = log(lambda / (1 - lambda))  (inverse logistic)
    alpha_values = np.log(lambda_values / (1.0 - lambda_values + 1e-30))

    # Compute nested logit probabilities
    # Build design matrix including interaction terms so all utility
    # components are captured in the probability calculation.
    beta = np.array([alt_params[col] for col in alternatives.columns])
    design_matrix = np.tile(alternatives.to_numpy(), (n_obs, 1))
    # Append interaction columns
    for name in interactions:
        interaction_vals = interactions[name].to_numpy().reshape(n_obs, n_alts)
        design_matrix = np.column_stack([design_matrix, interaction_vals.ravel()])
    beta = np.append(beta, [interaction_params[name] for name in interactions])

    probs = _nested_logit_probs_numpy(
        beta,
        alpha_values,
        design_matrix,
        nest_matrix,
        n_obs,
        n_alts,
    )

    # Simulate choices from probabilities
    choices = _simulate_choices_from_probs(probs, rng, n_obs, n_alts)
    choosers = choosers.copy()
    choosers["choice"] = choices

    # --- Build ChoiceTable -----------------------------------------------
    choice_table = _build_choice_table(
        choosers, alternatives, choosers["choice"], matrix_data=interactions
    )

    # Include interaction params in true_params
    true_params = dict(alt_params)
    true_params.update(interaction_params)

    return NestedMNLDataset(
        choosers=choosers,
        alternatives=alternatives,
        true_params=true_params,
        true_lambdas=dict(nest_lambdas),
        nests=nests,
        choice_table=choice_table,
        n_obs=n_obs,
        n_alts=n_alts,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# SCL DGP
# ---------------------------------------------------------------------------


def simulate_scl(
    n_obs: int = 10000,
    n_alts: int = 12,
    alt_params: dict[str, float] | None = None,
    rho: float = 0.7,
    adjacency: np.ndarray | None = None,
    interaction_params: dict[str, float] | None = None,
    seed: int = 1234,
) -> SCLDataset:
    """Generate synthetic SCL choice data with known parameters.

    The default DGP creates a circular adjacency graph with 12 zones,
    includes both alternative-level and chooser×alternative interaction
    terms, and simulates choices using the SCL probability formula.

    Parameters
    ----------
    n_obs : int, default 10000
        Number of observations (decision-makers).
    n_alts : int, default 12
        Number of alternatives (zones).
    alt_params : dict, optional
        Mapping of alternative-level column name → true coefficient.
        Default: ``{"cost": -0.5, "time": -0.1}``.
    rho : float, default 0.7
        True dissimilarity parameter ρ ∈ (0, 1].
    adjacency : np.ndarray, optional
        Binary adjacency matrix of shape ``(n_alts, n_alts)``.
        Default: circular graph where zone *i* is adjacent to
        ``(i-1) % n_alts`` and ``(i+1) % n_alts``.
    seed : int, default 1234
        Random seed for reproducibility.

    Returns
    -------
    SCLDataset
    """
    from .models.scl import _resolve_spatial_graph, _scl_log_probs_numpy

    if alt_params is None:
        alt_params = {"cost": -0.5, "time": -0.1}
    if interaction_params is None:
        interaction_params = {}

    rng = np.random.default_rng(seed)

    # --- Build adjacency matrix -----------------------------------------
    if adjacency is None:
        adjacency = np.zeros((n_alts, n_alts), dtype=np.float64)
        for i in range(n_alts):
            adjacency[i, (i - 1) % n_alts] = 1.0
            adjacency[i, (i + 1) % n_alts] = 1.0

    omega, allocation, edge_list, _ = _resolve_spatial_graph(adjacency)

    # --- Choosers and alternatives --------------------------------------
    obs_ids = pd.Index(np.arange(n_obs), name="oid")
    income = rng.standard_normal(n_obs)
    choosers = pd.DataFrame({"income": income}, index=obs_ids)

    alt_ids = pd.Index(np.arange(n_alts), name="aid")
    alternatives, alt_ids = _generate_alternatives(alt_params, n_alts, rng)

    # --- Interactions (from interaction_params keys) ------------------
    interactions = _generate_interactions(
        interaction_params, choosers, alternatives, obs_ids, alt_ids, rng
    )

    # --- Deterministic utility ------------------------------------------
    det_utility = np.zeros((n_obs, n_alts))
    for col, coef in alt_params.items():
        alt_vals = alternatives[col].to_numpy()
        det_utility += coef * np.tile(alt_vals, n_obs).reshape(n_obs, n_alts)
    for name, coef in interaction_params.items():
        det_utility += coef * interactions[name].to_numpy().reshape(n_obs, n_alts)

    # --- Compute SCL probabilities and simulate choices ------------------
    beta = np.array([alt_params[col] for col in alternatives.columns])
    design_matrix = np.tile(alternatives.to_numpy(), (n_obs, 1))
    for name in interactions:
        design_matrix = np.column_stack([design_matrix, interactions[name].to_numpy().ravel()])
    beta = np.append(beta, [interaction_params[name] for name in interactions])

    log_probs = _scl_log_probs_numpy(
        beta, rho, design_matrix, allocation, edge_list, n_obs, n_alts
    )
    probs = np.exp(log_probs)

    # Simulate choices
    choices = _simulate_choices_from_probs(probs, rng, n_obs, n_alts)
    choosers = choosers.copy()
    choosers["choice"] = choices

    # --- Build ChoiceTable -----------------------------------------------
    choice_table = _build_choice_table(
        choosers, alternatives, choosers["choice"], matrix_data=interactions
    )

    true_params = dict(alt_params)
    true_params.update(interaction_params)

    return SCLDataset(
        choosers=choosers,
        alternatives=alternatives,
        true_params=true_params,
        true_rho=rho,
        adjacency=adjacency,
        choice_table=choice_table,
        n_obs=n_obs,
        n_alts=n_alts,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Mixed logit DGP
# ---------------------------------------------------------------------------


def simulate_mixed_logit(
    n_obs: int = 10000,
    n_alts: int = 4,
    alt_params: dict[str, float] | None = None,
    random_params: dict[str, tuple[str, float, float]] | None = None,
    interaction_params: dict[str, float] | None = None,
    seed: int = 1234,
) -> MixedMNLDataset:
    """Generate synthetic mixed logit choice data with known parameters.

    The default DGP creates one fixed and one random coefficient, includes
    a chooser×alternative interaction term, and simulates choices using
    the mixed logit probability formula.

    Parameters
    ----------
    n_obs : int, default 10000
        Number of observations (decision-makers).
    n_alts : int, default 4
        Number of alternatives.
    alt_params : dict, optional
        Mapping of alternative-level column name → true fixed coefficient.
        Default: ``{"cost": -0.5}``.
    random_params : dict, optional
        Mapping of column name → ``(distribution, mean, sd)`` for random
        coefficients.  Default: ``{"time": ("normal", -0.3, 0.5)}``.
    seed : int, default 1234
        Random seed for reproducibility.

    Returns
    -------
    MixedLogitDataset
    """
    if alt_params is None:
        alt_params = {"cost": -0.5}
    if random_params is None:
        random_params = {"time": ("normal", -0.3, 0.5)}
    if interaction_params is None:
        interaction_params = {}

    rng = np.random.default_rng(seed)

    # Ensure alt_params includes all columns referenced by random_params
    alt_params = _ensure_random_param_columns(alt_params, random_params, n_alts, rng)

    # --- Choosers and alternatives --------------------------------------
    obs_ids = pd.Index(np.arange(n_obs), name="oid")
    income = rng.standard_normal(n_obs)
    choosers = pd.DataFrame({"income": income}, index=obs_ids)

    alt_ids = pd.Index(np.arange(n_alts), name="aid")
    alternatives, alt_ids = _generate_alternatives(alt_params, n_alts, rng)

    # --- Interactions (from interaction_params keys) ------------------
    interactions = _generate_interactions(
        interaction_params, choosers, alternatives, obs_ids, alt_ids, rng
    )

    # --- Deterministic utility (fixed part) ----------------------------
    det_utility = np.zeros((n_obs, n_alts))
    for col, coef in alt_params.items():
        alt_vals = alternatives[col].to_numpy()
        det_utility += coef * np.tile(alt_vals, n_obs).reshape(n_obs, n_alts)
    for name, coef in interaction_params.items():
        det_utility += coef * interactions[name].to_numpy().reshape(n_obs, n_alts)

    # --- Random utility component ----------------------------------------
    for col, (dist, mean, sd) in random_params.items():
        alt_vals = alternatives[col].to_numpy()
        # Draw per-observation random coefficients
        if dist == "normal":
            beta_random = rng.normal(mean, sd, size=n_obs)
        elif dist == "lognormal":
            beta_random = np.exp(rng.normal(mean, sd, size=n_obs))
        elif dist in ("triangular", "uniform"):
            beta_random = rng.uniform(mean - sd, mean + sd, size=n_obs)
        else:
            raise ValueError(f"Unsupported distribution: {dist}")
        det_utility += beta_random[:, None] * alt_vals[None, :]

    # --- Simulate choices (Gumbel noise) --------------------------------
    gumbel_noise = rng.gumbel(size=(n_obs, n_alts))
    sim_utility = det_utility + gumbel_noise
    choices = sim_utility.argmax(axis=1)
    choosers = choosers.copy()
    choosers["choice"] = choices

    # --- Build true_params dict ------------------------------------------
    true_params: dict[str, float] = {}
    true_params.update(alt_params)
    true_params.update(interaction_params)
    for col, (dist, mean, sd) in random_params.items():
        true_params[f"mean_{col}"] = mean
        true_params[f"sd_{col}"] = sd

    # --- Build random_params dict for MixedLogit -----------------------
    random_params_dict = {col: dist for col, (dist, _, _) in random_params.items()}

    # --- Build ChoiceTable -----------------------------------------------
    choice_table = _build_choice_table(
        choosers, alternatives, choosers["choice"], matrix_data=interactions
    )

    return MixedMNLDataset(
        choosers=choosers,
        alternatives=alternatives,
        true_params=true_params,
        random_params=random_params_dict,
        choice_table=choice_table,
        n_obs=n_obs,
        n_alts=n_alts,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# MSCL DGP
# ---------------------------------------------------------------------------


def simulate_mscl(
    n_obs: int = 10000,
    n_alts: int = 12,
    alt_params: dict[str, float] | None = None,
    rho: float = 0.7,
    random_params: dict[str, tuple[str, float, float]] | None = None,
    adjacency: np.ndarray | None = None,
    interaction_params: dict[str, float] | None = None,
    seed: int = 1234,
) -> MSCLDataset:
    """Generate synthetic MSCL choice data with known parameters.

    The default DGP creates a circular adjacency graph with 12 zones,
    one fixed and one random coefficient, includes a chooser×alternative
    interaction term, and simulates choices using the MSCL probability
    formula.

    Parameters
    ----------
    n_obs : int, default 10000
        Number of observations (decision-makers).
    n_alts : int, default 12
        Number of alternatives (zones).
    alt_params : dict, optional
        Mapping of alternative-level column name → true fixed coefficient.
        Default: ``{"cost": -0.5}``.
    rho : float, default 0.7
        True dissimilarity parameter ρ ∈ (0, 1].
    random_params : dict, optional
        Mapping of column name → ``(distribution, mean, sd)`` for random
        coefficients.  Default: ``{"time": ("normal", -0.3, 0.5)}``.
    adjacency : np.ndarray, optional
        Binary adjacency matrix of shape ``(n_alts, n_alts)``.
        Default: circular graph.
    seed : int, default 1234
        Random seed for reproducibility.

    Returns
    -------
    MSCLDataset
    """
    if alt_params is None:
        alt_params = {"cost": -0.5}
    if random_params is None:
        random_params = {"time": ("normal", -0.3, 0.5)}
    if interaction_params is None:
        interaction_params = {}

    rng = np.random.default_rng(seed)

    # Ensure alt_params includes all columns referenced by random_params
    alt_params = _ensure_random_param_columns(alt_params, random_params, n_alts, rng)

    # --- Build adjacency matrix -----------------------------------------
    if adjacency is None:
        adjacency = np.zeros((n_alts, n_alts), dtype=np.float64)
        for i in range(n_alts):
            adjacency[i, (i - 1) % n_alts] = 1.0
            adjacency[i, (i + 1) % n_alts] = 1.0

    # --- Choosers and alternatives --------------------------------------
    obs_ids = pd.Index(np.arange(n_obs), name="oid")
    income = rng.standard_normal(n_obs)
    choosers = pd.DataFrame({"income": income}, index=obs_ids)

    alt_ids = pd.Index(np.arange(n_alts), name="aid")
    alternatives, alt_ids = _generate_alternatives(alt_params, n_alts, rng)

    # --- Interactions (from interaction_params keys) ------------------
    interactions = _generate_interactions(
        interaction_params, choosers, alternatives, obs_ids, alt_ids, rng
    )

    # --- Deterministic utility (fixed part) ----------------------------
    det_utility = np.zeros((n_obs, n_alts))
    for col, coef in alt_params.items():
        alt_vals = alternatives[col].to_numpy()
        det_utility += coef * np.tile(alt_vals, n_obs).reshape(n_obs, n_alts)
    for name, coef in interaction_params.items():
        det_utility += coef * interactions[name].to_numpy().reshape(n_obs, n_alts)

    # --- Random utility component ----------------------------------------
    for col, (dist, mean, sd) in random_params.items():
        alt_vals = alternatives[col].to_numpy()
        if dist == "normal":
            beta_random = rng.normal(mean, sd, size=n_obs)
        elif dist == "lognormal":
            beta_random = np.exp(rng.normal(mean, sd, size=n_obs))
        elif dist in ("triangular", "uniform"):
            beta_random = rng.uniform(mean - sd, mean + sd, size=n_obs)
        else:
            raise ValueError(f"Unsupported distribution: {dist}")
        det_utility += beta_random[:, None] * alt_vals[None, :]

    # --- Simulate choices (Gumbel noise) --------------------------------
    gumbel_noise = rng.gumbel(size=(n_obs, n_alts))
    sim_utility = det_utility + gumbel_noise
    choices = sim_utility.argmax(axis=1)
    choosers = choosers.copy()
    choosers["choice"] = choices

    # --- Build true_params dict ------------------------------------------
    true_params: dict[str, float] = {}
    true_params.update(alt_params)
    true_params.update(interaction_params)
    for col, (dist, mean, sd) in random_params.items():
        true_params[f"mean_{col}"] = mean
        true_params[f"sd_{col}"] = sd

    # --- Build random_params dict for MSCL -----------------------------
    random_params_dict = {col: dist for col, (dist, _, _) in random_params.items()}

    # --- Build ChoiceTable -----------------------------------------------
    choice_table = _build_choice_table(
        choosers, alternatives, choosers["choice"], matrix_data=interactions
    )

    return MSCLDataset(
        choosers=choosers,
        alternatives=alternatives,
        true_params=true_params,
        true_rho=rho,
        adjacency=adjacency,
        random_params=random_params_dict,
        choice_table=choice_table,
        n_obs=n_obs,
        n_alts=n_alts,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Nested SCL DGP
# ---------------------------------------------------------------------------


def simulate_nested_scl(
    n_obs: int = 10000,
    n_alts: int = 12,
    alt_params: dict[str, float] | None = None,
    nest_rhos: dict[str, float] | None = None,
    nest_lambdas: dict[str, float] | None = None,
    adjacency: np.ndarray | None = None,
    interaction_params: dict[str, float] | None = None,
    seed: int = 1234,
) -> NestedSCLDataset:
    """Generate synthetic Nested SCL choice data with known parameters.

    The default DGP creates a circular adjacency graph with 12 zones
    partitioned into two nests (e.g., inner / outer ring), includes both
    alternative-level and chooser×alternative interaction terms, and
    simulates choices using the Nested SCL probability formula.

    Parameters
    ----------
    n_obs : int, default 10000
        Number of observations (decision-makers).
    n_alts : int, default 12
        Number of alternatives (zones).
    alt_params : dict, optional
        Mapping of alternative-level column name → true coefficient.
        Default: ``{"cost": -0.5, "time": -0.1}``.
    nest_rhos : dict, optional
        Mapping of nest name → true spatial dissimilarity ρ_m ∈ (0, 1].
        Default: ``{"inner": 0.6, "outer": 0.8}``.
    nest_lambdas : dict, optional
        Mapping of nest name → true nest dissimilarity λ_m ∈ (0, 1].
        Default: ``{"inner": 0.7, "outer": 0.9}``.
    adjacency : np.ndarray, optional
        Binary adjacency matrix of shape ``(n_alts, n_alts)``.
        Default: circular graph where zone *i* is adjacent to
        ``(i-1) % n_alts`` and ``(i+1) % n_alts``.
    seed : int, default 1234
        Random seed for reproducibility.

    Returns
    -------
    NestedSCLDataset
    """
    from .models.nested import NestingTree, NestSpec
    from .models.scl import (
        _resolve_spatial_graph,
    )

    if alt_params is None:
        alt_params = {"cost": -0.5, "time": -0.1}
    if nest_rhos is None:
        nest_rhos = {"inner": 0.6, "outer": 0.8}
    if nest_lambdas is None:
        nest_lambdas = {"inner": 0.7, "outer": 0.9}
    if interaction_params is None:
        interaction_params = {}

    rng = np.random.default_rng(seed)

    # --- Build adjacency matrix -----------------------------------------
    if adjacency is None:
        adjacency = np.zeros((n_alts, n_alts), dtype=np.float64)
        for i in range(n_alts):
            adjacency[i, (i - 1) % n_alts] = 1.0
            adjacency[i, (i + 1) % n_alts] = 1.0

    omega, allocation, edge_list, _ = _resolve_spatial_graph(adjacency)

    # --- Build nesting structure ----------------------------------------
    nest_names = list(nest_rhos.keys())
    alts_per_nest = n_alts // len(nest_names)
    if n_alts % len(nest_names) != 0:
        raise ValueError(
            f"n_alts ({n_alts}) must be evenly divisible by the number of "
            f"nests ({len(nest_names)})."
        )

    nest_specs = []
    alt_id = 0
    for name in nest_names:
        nest_specs.append(NestSpec(name, alt_ids=list(range(alt_id, alt_id + alts_per_nest))))
        alt_id += alts_per_nest
    nests = NestingTree(nest_specs)

    # --- Choosers and alternatives --------------------------------------
    obs_ids = pd.Index(np.arange(n_obs), name="oid")
    income = rng.standard_normal(n_obs)
    choosers = pd.DataFrame({"income": income}, index=obs_ids)

    alt_ids = pd.Index(np.arange(n_alts), name="aid")
    alternatives, alt_ids = _generate_alternatives(alt_params, n_alts, rng)

    # --- Interactions (from interaction_params keys) ------------------
    interactions = _generate_interactions(
        interaction_params, choosers, alternatives, obs_ids, alt_ids, rng
    )

    # --- Deterministic utility ------------------------------------------
    det_utility = np.zeros((n_obs, n_alts))
    for col, coef in alt_params.items():
        alt_vals = alternatives[col].to_numpy()
        det_utility += coef * np.tile(alt_vals, n_obs).reshape(n_obs, n_alts)
    for name, coef in interaction_params.items():
        det_utility += coef * interactions[name].to_numpy().reshape(n_obs, n_alts)

    # --- Compute Nested SCL probabilities and simulate choices --------
    nest_matrix = nests.build_nest_matrix(list(range(n_alts)))
    n_nests = len(nest_names)

    # Per-nest SCL probabilities and inclusive values
    nest_log_G = np.zeros((n_obs, n_nests), dtype=np.float64)
    log_probs_per_nest = []

    for m, name in enumerate(nest_names):
        nest_mask = nest_matrix[:, m] > 0
        nest_alts = np.where(nest_mask)[0]
        len(nest_alts)

        # Extract subgraph for this nest
        nest_adj = adjacency[np.ix_(nest_alts, nest_alts)]
        _, nest_alloc, nest_edges, _ = _resolve_spatial_graph(nest_adj)

        # _resolve_spatial_graph returns edges in local coordinates (0..n_nest_alts-1)
        # because it operates on the extracted subgraph matrix
        local_edges = nest_edges

        # Extract utilities for this nest (includes all terms: cost, time, interactions)
        V_nest = det_utility[:, nest_alts]

        log_probs_nest, log_G_nest = _scl_log_probs_from_utility_numpy(
            V_nest,
            nest_rhos[name],
            nest_alloc,
            local_edges,
        )

        # Store inclusive value
        nest_log_G[:, m] = log_G_nest

        # Store probabilities in full array positions
        full_log_probs = np.full((n_obs, n_alts), -np.inf, dtype=np.float64)
        full_log_probs[:, nest_alts] = log_probs_nest
        log_probs_per_nest.append(full_log_probs)

    # Top-level NL: compute nest probabilities
    lambda_values = np.array([nest_lambdas[name] for name in nest_names])
    nest_exponents = lambda_values[None, :] * nest_log_G  # (n_obs, n_nests)
    log_denom_top = logsumexp(nest_exponents, axis=1)  # (n_obs,)
    log_P_nest = nest_exponents - log_denom_top[:, None]  # (n_obs, n_nests)

    # Combine: P_i = P_SCL(i|m) * P_NL(m)
    log_probs_full = np.full((n_obs, n_alts), -np.inf, dtype=np.float64)
    for m, name in enumerate(nest_names):
        nest_mask = nest_matrix[:, m] > 0
        # log P_i = log P_SCL(i|m) + log P_NL(m)
        log_probs_full[:, nest_mask] = (
            log_probs_per_nest[m][:, nest_mask] + log_P_nest[:, m][:, None]
        )

    probs = np.exp(log_probs_full)
    probs = np.maximum(probs, 0.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Simulate choices
    choices = _simulate_choices_from_probs(probs, rng, n_obs, n_alts)
    choosers = choosers.copy()
    choosers["choice"] = choices

    # --- Build true_params dict ------------------------------------------
    true_params = dict(alt_params)
    true_params.update(interaction_params)

    # --- Build ChoiceTable -----------------------------------------------
    choice_table = _build_choice_table(
        choosers, alternatives, choosers["choice"], matrix_data=interactions
    )

    return NestedSCLDataset(
        choosers=choosers,
        alternatives=alternatives,
        true_params=true_params,
        true_rhos=dict(nest_rhos),
        true_lambdas=dict(nest_lambdas),
        nests=nests,
        adjacency=adjacency,
        choice_table=choice_table,
        n_obs=n_obs,
        n_alts=n_alts,
        seed=seed,
    )


def _scl_log_probs_from_utility_numpy(
    V: np.ndarray,
    rho: float,
    allocation: np.ndarray,
    edge_list: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SCL log-probabilities and inclusive value from precomputed utilities.

    Parameters
    ----------
    V : np.ndarray, shape (n_obs, n_alts)
        Precomputed utility matrix.
    rho : float
        Dissimilarity parameter in (0, 1].
    allocation : np.ndarray
        Spatial allocation matrix.
    edge_list : list[tuple[int, int]]
        List of edge tuples.

    Returns
    -------
    log_probs : np.ndarray, shape (n_obs, n_alts)
    log_inclusive_value : np.ndarray, shape (n_obs,)
    """
    n_obs, n_alts = V.shape
    inv_rho = 1.0 / rho
    exp_V = np.exp(np.clip(V, -500, 500))

    alloc_exp_V = allocation[None, :, :] * exp_V[:, :, None]
    alloc_exp_V = np.clip(alloc_exp_V, 1e-30, 1e30)
    alloc_exp_V_inv_rho = np.power(alloc_exp_V, inv_rho)

    n_edges = len(edge_list)
    if n_edges == 0:
        log_sum_exp_V = logsumexp(V, axis=1)
        log_probs = V - log_sum_exp_V[:, None]
        return log_probs, log_sum_exp_V

    connected = set()
    for i, j in edge_list:
        connected.add(i)
        connected.add(j)
    isolated = sorted(set(range(n_alts)) - connected)

    nest_vals = np.zeros((n_obs, n_edges), dtype=np.float64)
    for idx, (i, j) in enumerate(edge_list):
        term_i = alloc_exp_V_inv_rho[:, i, j]
        term_j = alloc_exp_V_inv_rho[:, j, i]
        nest_vals[:, idx] = np.power(term_i + term_j, rho)

    if isolated:
        iso_exp_V = exp_V[:, isolated]
        denom_components = np.column_stack([nest_vals, iso_exp_V])
    else:
        denom_components = nest_vals

    log_denom = np.log(np.maximum(denom_components.sum(axis=1), 1e-300))

    log_probs = np.full((n_obs, n_alts), -np.inf, dtype=np.float64)

    alt_to_edges: dict[int, list[tuple[int, bool]]] = {}
    for idx, (i, j) in enumerate(edge_list):
        alt_to_edges.setdefault(i, []).append((idx, True))
        alt_to_edges.setdefault(j, []).append((idx, False))

    for alt_i in connected:
        edge_contributions = []
        for edge_idx, is_first in alt_to_edges[alt_i]:
            i, j = edge_list[edge_idx]
            if is_first:
                my_term = alloc_exp_V_inv_rho[:, alt_i, j]
                other_term = alloc_exp_V_inv_rho[:, j, alt_i]
            else:
                my_term = alloc_exp_V_inv_rho[:, alt_i, i]
                other_term = alloc_exp_V_inv_rho[:, i, alt_i]

            log_cond = np.log(np.maximum(my_term, 1e-300)) - np.log(
                np.maximum(my_term + other_term, 1e-300)
            )
            log_nest = np.log(np.maximum(nest_vals[:, edge_idx], 1e-300)) - log_denom
            edge_contributions.append(log_cond + log_nest)

        if len(edge_contributions) == 1:
            log_probs[:, alt_i] = edge_contributions[0]
        else:
            log_probs[:, alt_i] = logsumexp(np.column_stack(edge_contributions), axis=1)

    for alt_i in isolated:
        log_probs[:, alt_i] = V[:, alt_i] - log_denom

    return log_probs, log_denom


def simulate_mnscl(
    n_obs: int = 5000,
    n_alts: int = 12,
    alt_params: dict[str, float] | None = None,
    nest_rhos: dict[str, float] | None = None,
    nest_lambdas: dict[str, float] | None = None,
    random_params: dict[str, tuple[str, float, float]] | None = None,
    adjacency: np.ndarray | None = None,
    interaction_params: dict[str, float] | None = None,
    seed: int = 1234,
) -> MNSCLDataset:
    """Generate synthetic MNSCL choice data with known parameters.

    The default DGP creates a circular adjacency graph with 12 zones
    partitioned into two nests, includes both alternative-level and
    chooser×alternative interaction terms, and simulates choices using
    the MNSCL probability formula with random coefficients.

    Parameters
    ----------
    n_obs : int, default 5000
        Number of observations (decision-makers).
    n_alts : int, default 12
        Number of alternatives (zones).
    alt_params : dict, optional
        Mapping of alternative-level column name → true coefficient.
        Default: ``{"cost": -0.5, "time": -0.1}``.
    nest_rhos : dict, optional
        Mapping of nest name → true spatial dissimilarity ρ_m ∈ (0, 1].
        Default: ``{"inner": 0.6, "outer": 0.8}``.
    nest_lambdas : dict, optional
        Mapping of nest name → true nest dissimilarity λ_m ∈ (0, 1].
        Default: ``{"inner": 0.7, "outer": 0.9}``.
    random_params : dict, optional
        Mapping of parameter name → (distribution, mean, spread).
        E.g. ``{"time": ("normal", -0.1, 0.05)}``.
        Default: ``{"time": ("normal", -0.1, 0.05)}``.
    adjacency : np.ndarray, optional
        Binary adjacency matrix of shape ``(n_alts, n_alts)``.
        Default: circular graph.
    seed : int, default 1234
        Random seed for reproducibility.

    Returns
    -------
    MNSCLDataset
    """
    from .models.nested import NestingTree, NestSpec
    from .models.scl import (
        _resolve_spatial_graph,
    )

    if alt_params is None:
        alt_params = {"cost": -0.5, "time": -0.1}
    if nest_rhos is None:
        nest_rhos = {"inner": 0.6, "outer": 0.8}
    if nest_lambdas is None:
        nest_lambdas = {"inner": 0.7, "outer": 0.9}
    if random_params is None:
        random_params = {"time": ("normal", -0.1, 0.05)}
    if interaction_params is None:
        interaction_params = {}

    rng = np.random.default_rng(seed)

    # Ensure alt_params includes all columns referenced by random_params
    alt_params = _ensure_random_param_columns(alt_params, random_params, n_alts, rng)

    # --- Build adjacency matrix -----------------------------------------
    if adjacency is None:
        adjacency = np.zeros((n_alts, n_alts), dtype=np.float64)
        for i in range(n_alts):
            adjacency[i, (i - 1) % n_alts] = 1.0
            adjacency[i, (i + 1) % n_alts] = 1.0

    omega, allocation, edge_list, _ = _resolve_spatial_graph(adjacency)

    # --- Build nesting structure ----------------------------------------
    nest_names = list(nest_rhos.keys())
    alts_per_nest = n_alts // len(nest_names)
    if n_alts % len(nest_names) != 0:
        raise ValueError(
            f"n_alts ({n_alts}) must be evenly divisible by the number of "
            f"nests ({len(nest_names)})."
        )

    nest_specs = []
    alt_id = 0
    for name in nest_names:
        nest_specs.append(NestSpec(name, alt_ids=list(range(alt_id, alt_id + alts_per_nest))))
        alt_id += alts_per_nest
    nests = NestingTree(nest_specs)

    # --- Choosers and alternatives --------------------------------------
    obs_ids = pd.Index(np.arange(n_obs), name="oid")
    income = rng.standard_normal(n_obs)
    choosers = pd.DataFrame({"income": income}, index=obs_ids)

    alt_ids = pd.Index(np.arange(n_alts), name="aid")
    alternatives, alt_ids = _generate_alternatives(alt_params, n_alts, rng)

    # --- Interactions (from interaction_params keys) ------------------
    interactions = _generate_interactions(
        interaction_params, choosers, alternatives, obs_ids, alt_ids, rng
    )

    # --- Deterministic utility -----------------------------------------
    det_utility = np.zeros((n_obs, n_alts))
    for col, coef in alt_params.items():
        alt_vals = alternatives[col].to_numpy()
        det_utility += coef * np.tile(alt_vals, n_obs).reshape(n_obs, n_alts)
    for name, coef in interaction_params.items():
        det_utility += coef * interactions[name].to_numpy().reshape(n_obs, n_alts)

    # --- Add random coefficient variation ------------------------------
    # For each random parameter, add random variation multiplied by attribute
    for param_name, (dist, mean, spread) in random_params.items():
        list(alternatives.columns).index(param_name)
        alt_vals = alternatives[param_name].to_numpy()  # shape (n_alts,)
        z = rng.standard_normal((n_obs, n_alts))
        if dist == "normal":
            random_component = (mean + spread * z) * alt_vals[None, :]
        elif dist == "lognormal":
            random_component = np.exp(mean + spread * z) * alt_vals[None, :]
        elif dist == "uniform":
            random_component = (mean + spread * (2 * rng.random((n_obs, n_alts)) - 1)) * alt_vals[
                None, :
            ]
        elif dist == "triangular":
            u = rng.random((n_obs, n_alts))
            random_component = np.where(
                u <= 0.5,
                (mean + spread * (np.sqrt(2 * u) - 1)) * alt_vals[None, :],
                (mean + spread * (1 - np.sqrt(2 * (1 - u)))) * alt_vals[None, :],
            )
        else:
            raise ValueError(f"Unknown distribution: {dist}")
        det_utility += random_component

    # --- Compute MNSCL probabilities and simulate choices --------------
    nest_matrix = nests.build_nest_matrix(list(range(n_alts)))
    n_nests = len(nest_names)

    # Per-nest SCL probabilities and inclusive values
    nest_log_G = np.zeros((n_obs, n_nests), dtype=np.float64)
    log_probs_per_nest = []

    for m, name in enumerate(nest_names):
        nest_mask = nest_matrix[:, m] > 0
        nest_alts = np.where(nest_mask)[0]
        len(nest_alts)

        # Extract subgraph for this nest
        nest_adj = adjacency[np.ix_(nest_alts, nest_alts)]
        _, nest_alloc, nest_edges, _ = _resolve_spatial_graph(nest_adj)

        # Extract utilities for this nest (with random variation)
        V_nest = det_utility[:, nest_alts]

        log_probs_nest, log_G_nest = _scl_log_probs_from_utility_numpy(
            V_nest,
            nest_rhos[name],
            nest_alloc,
            nest_edges,
        )

        # Store inclusive value
        nest_log_G[:, m] = log_G_nest

        # Store probabilities in full array positions
        full_log_probs = np.full((n_obs, n_alts), -np.inf, dtype=np.float64)
        full_log_probs[:, nest_alts] = log_probs_nest
        log_probs_per_nest.append(full_log_probs)

    # Top-level NL: compute nest probabilities
    lambda_values = np.array([nest_lambdas[name] for name in nest_names])
    nest_exponents = lambda_values[None, :] * nest_log_G  # (n_obs, n_nests)
    log_denom_top = logsumexp(nest_exponents, axis=1)  # (n_obs,)
    log_P_nest = nest_exponents - log_denom_top[:, None]  # (n_obs, n_nests)

    # Combine: P_i = P_SCL(i|m) * P_NL(m)
    log_probs_full = np.full((n_obs, n_alts), -np.inf, dtype=np.float64)
    for m, name in enumerate(nest_names):
        nest_mask = nest_matrix[:, m] > 0
        log_probs_full[:, nest_mask] = (
            log_probs_per_nest[m][:, nest_mask] + log_P_nest[:, m][:, None]
        )

    probs = np.exp(log_probs_full)
    probs = np.maximum(probs, 0.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Simulate choices
    choices = _simulate_choices_from_probs(probs, rng, n_obs, n_alts)
    choosers = choosers.copy()
    choosers["choice"] = choices

    # --- Build true_params dict -----------------------------------------
    true_params = dict(alt_params)
    true_params.update(interaction_params)

    true_random_means = {}
    true_random_spreads = {}
    random_param_dict = {}
    for param_name, (dist, mean, spread) in random_params.items():
        true_random_means[param_name] = mean
        true_random_spreads[param_name] = spread
        random_param_dict[param_name] = dist

    # --- Build ChoiceTable ----------------------------------------------
    choice_table = _build_choice_table(
        choosers, alternatives, choosers["choice"], matrix_data=interactions
    )

    return MNSCLDataset(
        choosers=choosers,
        alternatives=alternatives,
        true_params=true_params,
        true_rhos=dict(nest_rhos),
        true_lambdas=dict(nest_lambdas),
        true_random_means=true_random_means,
        true_random_spreads=true_random_spreads,
        random_params=random_param_dict,
        nests=nests,
        adjacency=adjacency,
        choice_table=choice_table,
        n_obs=n_obs,
        n_alts=n_alts,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Mixed Nested Logit DGP
# ---------------------------------------------------------------------------


def simulate_mixed_nested_logit(
    n_obs: int = 10000,
    n_alts: int = 4,
    alt_params: dict[str, float] | None = None,
    nest_lambdas: dict[str, float] | None = None,
    random_params: dict[str, tuple[str, float, float]] | None = None,
    interaction_params: dict[str, float] | None = None,
    seed: int = 1234,
) -> MixedNestedMNLDataset:
    """Generate synthetic mixed nested logit choice data with known parameters.

    The default DGP creates two nests with 2 alternatives each, includes
    both alternative-level and chooser×alternative interaction terms,
    and simulates choices using the mixed nested logit probability formula
    with random coefficients.

    Parameters
    ----------
    n_obs : int, default 10000
        Number of observations (decision-makers).
    n_alts : int, default 4
        Number of alternatives.  Must be evenly divisible by the number of
        nests in ``nest_lambdas``.
    alt_params : dict, optional
        Mapping of alternative-level column name → true fixed coefficient.
        Default: ``{"cost": -0.5}``.
    nest_lambdas : dict, optional
        Mapping of nest name → true dissimilarity parameter λ ∈ (0, 1].
        Default: ``{"transit": 0.7, "auto": 0.8}``.
    random_params : dict, optional
        Mapping of column name → ``(distribution, mean, sd)`` for random
        coefficients.  Default: ``{"time": ("normal", -0.3, 0.5)}``.
    seed : int, default 1234
        Random seed for reproducibility.

    Returns
    -------
    MixedNestedMNLDataset
    """
    from .models.nested import (
        NestingTree,
        NestSpec,
        _nested_logit_probs_numpy,
    )

    if alt_params is None:
        alt_params = {"cost": -0.5}
    if nest_lambdas is None:
        nest_lambdas = {"transit": 0.7, "auto": 0.8}
    if random_params is None:
        random_params = {"time": ("normal", -0.3, 0.5)}
    if interaction_params is None:
        interaction_params = {}

    rng = np.random.default_rng(seed)

    # Ensure alt_params includes all columns referenced by random_params
    alt_params = _ensure_random_param_columns(alt_params, random_params, n_alts, rng)

    # --- Build nesting structure ----------------------------------------
    nest_names = list(nest_lambdas.keys())
    alts_per_nest = n_alts // len(nest_names)
    if n_alts % len(nest_names) != 0:
        raise ValueError(
            f"n_alts ({n_alts}) must be evenly divisible by the number of "
            f"nests ({len(nest_names)})."
        )

    nest_specs = []
    alt_id = 0
    for name in nest_names:
        nest_specs.append(NestSpec(name, alt_ids=list(range(alt_id, alt_id + alts_per_nest))))
        alt_id += alts_per_nest
    nests = NestingTree(nest_specs)

    # --- Choosers and alternatives --------------------------------------
    obs_ids = pd.Index(np.arange(n_obs), name="oid")
    income = rng.standard_normal(n_obs)
    choosers = pd.DataFrame({"income": income}, index=obs_ids)

    alt_ids = pd.Index(np.arange(n_alts), name="aid")
    alternatives, alt_ids = _generate_alternatives(alt_params, n_alts, rng)

    # --- Interactions (from interaction_params keys) ------------------
    interactions = _generate_interactions(
        interaction_params, choosers, alternatives, obs_ids, alt_ids, rng
    )

    # --- Deterministic utility (fixed part) ----------------------------
    det_utility = np.zeros((n_obs, n_alts))
    for col, coef in alt_params.items():
        alt_vals = alternatives[col].to_numpy()
        det_utility += coef * np.tile(alt_vals, n_obs).reshape(n_obs, n_alts)
    for name, coef in interaction_params.items():
        det_utility += coef * interactions[name].to_numpy().reshape(n_obs, n_alts)

    # --- Add random coefficient variation ------------------------------
    for param_name, (dist, mean, spread) in random_params.items():
        alt_vals = alternatives[param_name].to_numpy()
        z = rng.standard_normal((n_obs, n_alts))
        if dist == "normal":
            random_component = (mean + spread * z) * alt_vals[None, :]
        elif dist == "lognormal":
            random_component = np.exp(mean + spread * z) * alt_vals[None, :]
        elif dist == "uniform":
            random_component = (mean + spread * (2 * rng.random((n_obs, n_alts)) - 1)) * alt_vals[
                None, :
            ]
        elif dist == "triangular":
            u = rng.random((n_obs, n_alts))
            random_component = np.where(
                u <= 0.5,
                (mean + spread * (np.sqrt(2 * u) - 1)) * alt_vals[None, :],
                (mean + spread * (1 - np.sqrt(2 * (1 - u)))) * alt_vals[None, :],
            )
        else:
            raise ValueError(f"Unknown distribution: {dist}")
        det_utility += random_component

    # --- Build nest matrix and simulate choices -------------------------
    alt_id_list = list(range(n_alts))
    nest_matrix = nests.build_nest_matrix(alt_id_list)

    # Convert lambdas to alpha (unconstrained) for the probability kernel
    lambda_values = np.array([nest_lambdas[name] for name in nest_names])
    alpha_values = np.log(lambda_values / (1.0 - lambda_values + 1e-30))

    # Build design matrix including interaction terms
    # Fixed coefficients: alt_params columns + interaction columns
    # Random coefficients are already included in det_utility
    fixed_cols = [col for col in alternatives.columns if col in alt_params]
    beta = np.array([alt_params[col] for col in fixed_cols])
    design_matrix = np.tile(alternatives[fixed_cols].to_numpy(), (n_obs, 1))
    for name in interactions:
        design_matrix = np.column_stack([design_matrix, interactions[name].to_numpy().ravel()])
    beta = np.append(beta, [interaction_params[name] for name in interactions])

    # Compute nested logit probabilities
    probs = _nested_logit_probs_numpy(
        beta,
        alpha_values,
        design_matrix,
        nest_matrix,
        n_obs,
        n_alts,
    )

    # Simulate choices from probabilities
    choices = _simulate_choices_from_probs(probs, rng, n_obs, n_alts)
    choosers = choosers.copy()
    choosers["choice"] = choices

    # --- Build true_params dict ------------------------------------------
    true_params = dict(alt_params)
    true_params.update(interaction_params)

    true_random_means = {}
    true_random_spreads = {}
    random_param_dict = {}
    for param_name, (dist, mean, spread) in random_params.items():
        true_random_means[param_name] = mean
        true_random_spreads[param_name] = spread
        random_param_dict[param_name] = dist

    # --- Build ChoiceTable -----------------------------------------------
    choice_table = _build_choice_table(
        choosers, alternatives, choosers["choice"], matrix_data=interactions
    )

    return MixedNestedMNLDataset(
        choosers=choosers,
        alternatives=alternatives,
        true_params=true_params,
        true_lambdas=dict(nest_lambdas),
        true_random_means=true_random_means,
        true_random_spreads=true_random_spreads,
        random_params=random_param_dict,
        nests=nests,
        choice_table=choice_table,
        n_obs=n_obs,
        n_alts=n_alts,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# SAR-MNL DGP (Smirnov 2010)
# ---------------------------------------------------------------------------


@dataclass
class SARMNLDataset:
    """Synthetic dataset drawn from a known SAR-MNL data generating process.

    Alternatives are spatial locations connected by ``W`` (alt×alt).
    Choosers select among alternatives via MNL with spatially-filtered
    and variance-normalised utilities (Smirnov 2010 PML DGP).

    Attributes
    ----------
    choosers : pd.DataFrame
        Obs-id-indexed DataFrame with chooser attributes and ``choice`` column.
    alternatives : pd.DataFrame
        Alt-id-indexed DataFrame with alternative attributes.
    interactions : dict[str, pd.Series]
        Named ``(obs_id, alt_id)``-indexed Series for chooser×alt interactions.
    true_params : dict[str, float]
        Ground-truth beta coefficients (alt-level + interaction).
    true_rho : float
        Ground-truth spatial autoregressive parameter.
    W : libpysal.graph.Graph
        Row-standardised spatial weights matrix (n_alts × n_alts) as a
        libpysal Graph.  Use ``W.sparse`` to get the scipy.sparse matrix
        for computation.
    choice_table : object
        Assembled ChoiceTable.
    n_obs : int
    n_alts : int
    seed : int
    """

    choosers: pd.DataFrame
    alternatives: pd.DataFrame
    interactions: dict[str, pd.Series]
    true_params: dict[str, float]
    true_rho: float
    W: Any  # libpysal.graph.Graph
    choice_table: Any
    n_obs: int
    n_alts: int
    seed: int


def simulate_sar_mnl(
    n_obs: int = 5000,
    n_alts: int = 50,
    alt_params: dict[str, float] | None = None,
    interaction_params: dict[str, float] | None = None,
    rho: float = 0.3,
    W=None,
    n_neighbors: int = 7,
    seed: int = 1234,
    normalize: bool = True,
) -> SARMNLDataset:
    """Generate synthetic SAR-MNL choice data with known parameters.

    The DGP follows Smirnov (2010) PML model:

    1. Build ``W`` (alt×alt, row-standardised k-nearest-neighbor Graph)
    2. Generate alternative attributes ``Z`` and chooser-alt interactions ``X``
    3. Compute base utilities: ``V_base = Zβ + Xγ``  (n_obs × n_alts)
    4. Spatial filter: ``V_filtered = (I - ρW)^{-1} V_base^T``
    5. Variance normalisation: ``D = diag((I - ρW)^{-1})``,
       ``V_star = V_filtered / D``  (divide each alt by d_jj)
    6. Add Gumbel noise: ``U = V_star + Gumbel(0, 1)``
    7. Choice = ``argmax(U)`` per chooser

    The variance normalisation (step 5) is essential — it matches the
    PML estimator's model (Smirnov 2010).  Without it, the DGP would
    not match the estimation model and parameter recovery would fail.

    Parameters
    ----------
    n_obs : int, default 5000
        Number of choosers.
    n_alts : int, default 50
        Number of alternatives (spatial locations).  Dimension of W.
        For PML dense path, keep ≤ 2000.  For CG path, can be larger.
    alt_params : dict, optional
        Mapping of alternative-level column name → true coefficient.
        Default: ``{"alt_attr": -0.5}``.
    interaction_params : dict, optional
        Mapping of interaction column name → true coefficient.
        Default: ``{"obs_x_alt": 0.8}``.
    normalize : bool, default True
        When True the DGP is the Smirnov (2010) PML form, dividing filtered
        utilities by ``diag((I - ρW)^{-1})``.  When False it is the reduced
        form ``ψ = (I - ρW)^{-1} Xβ``, for which the softmax is the exact
        choice model.  Use it to generate data matching
        ``ChoiceModel(..., estimator="reduced")``.
    rho : float, default 0.3
        True spatial autoregressive parameter.  Should be in (-1, 1).
        Smirnov 2010 MC evidence: good recovery for ρ ∈ [0, 0.5].
    W : libpysal.graph.Graph, scipy.sparse, np.ndarray, or None
        Pre-specified n_alts × n_alts spatial weights matrix.
        If None, constructed as k-nearest-neighbor Graph on random
        coordinates (matching Krisztin et al. 2022's 7-NN specification).
        A ``libpysal.graph.Graph`` is the preferred input type.
    n_neighbors : int, default 7
        Number of nearest neighbors for default W construction.
    seed : int, default 1234

    Returns
    -------
    SARMNLDataset
        Dataset with ``W`` stored as a ``libpysal.graph.Graph`` (row-standardised).
    """
    if alt_params is None:
        alt_params = {"alt_attr": -0.5}
    if interaction_params is None:
        interaction_params = {}

    rng = np.random.default_rng(seed)

    # --- Build W (alt×alt) as a libpysal Graph --------------------------
    from .models._spatial_weights import build_knn_graph, resolve_spatial_weights

    if W is None:
        coords = rng.standard_normal((n_alts, 2))
        W_graph = build_knn_graph(coords, k=n_neighbors)
        W_dense = np.asarray(W_graph.sparse.todense(), dtype=np.float64)
    else:
        W_graph, _ = resolve_spatial_weights(W, n_alts, row_standardize=True)
        W_dense = np.asarray(W_graph.sparse.todense(), dtype=np.float64)

    # --- Choosers and alternatives --------------------------------------
    obs_ids = pd.Index(np.arange(n_obs), name="oid")
    obs_feature = rng.standard_normal(n_obs)
    choosers = pd.DataFrame({"obs_feature": obs_feature}, index=obs_ids)

    alt_ids = pd.Index(np.arange(n_alts), name="aid")
    # Generate alternative columns dynamically from alt_params keys
    alt_data = {}
    for col in alt_params:
        alt_data[col] = rng.standard_normal(n_alts)
    alternatives = pd.DataFrame(alt_data, index=alt_ids)

    # --- Interactions (chooser × alternative) --------------------------
    interaction_index = pd.MultiIndex.from_product([obs_ids, alt_ids], names=["oid", "aid"])
    obs_feat_tiled = np.repeat(obs_feature, n_alts)
    interactions = {}
    for interaction_name in interaction_params:
        # interaction_name like "obs_x_cost" → interact obs_feature with "cost" column
        alt_col = interaction_name.replace("obs_x_", "", 1)
        if alt_col not in alternatives.columns:
            raise KeyError(
                f"Interaction '{interaction_name}' references alt column '{alt_col}' "
                f"not in alternatives. Available: {list(alternatives.columns)}"
            )
        alt_tiled = np.tile(alternatives[alt_col].to_numpy(), n_obs)
        interactions[interaction_name] = pd.Series(
            obs_feat_tiled * alt_tiled, index=interaction_index, name=interaction_name
        )

    # --- Base utilities: V_base = Zβ + Xγ  (n_obs × n_alts) -------------
    V_base = np.zeros((n_obs, n_alts))
    for col, coef in alt_params.items():
        V_base += coef * np.tile(alternatives[col].to_numpy(), (n_obs, 1))
    for col, coef in interaction_params.items():
        V_base += coef * interactions[col].to_numpy().reshape(n_obs, n_alts)

    # --- Spatial filter: V_filtered = (I - ρW)^{-1} V_base^T ------------
    A = np.eye(n_alts) - rho * W_dense
    V_filtered = np.linalg.solve(A, V_base.T).T  # (n_obs, n_alts)

    if normalize:
        # --- Variance normalisation: D = diag((I - ρW)^{-1}) ---------------
        Z_mat = np.linalg.inv(A)
        D = np.diag(Z_mat)  # (n_alts,)
        V_star = V_filtered / D[None, :]  # normalise each alternative by d_jj
    else:
        # Reduced form: only systematic utility is filtered, so the softmax
        # over (I - ρW)^{-1} V_base is the exact choice model.
        V_star = V_filtered

    # --- Add Gumbel noise and simulate choices -------------------------
    gumbel = rng.gumbel(size=(n_obs, n_alts))
    U = V_star + gumbel
    choices = U.argmax(axis=1)
    choosers = choosers.copy()
    choosers["choice"] = choices

    # --- Build ChoiceTable ----------------------------------------------
    true_params = dict(alt_params)
    true_params.update(interaction_params)
    choice_table = _build_choice_table(
        choosers, alternatives, choosers["choice"], matrix_data=interactions
    )

    return SARMNLDataset(
        choosers=choosers,
        alternatives=alternatives,
        interactions=interactions,
        true_params=true_params,
        true_rho=rho,
        W=W_graph,
        choice_table=choice_table,
        n_obs=n_obs,
        n_alts=n_alts,
        seed=seed,
    )
