"""Data generating process (DGP) utilities for synthetic choice data.

This module provides functions for generating synthetic choice data with
known ground-truth parameters for each model type.  Intended for parameter
recovery tests and simulation studies.

Supported DGPs
--------------
- MNL (multinomial logit)
- Nested logit
- SCL (spatially correlated logit)
- Mixed logit (random coefficients)
- MSCL (mixed spatially correlated logit)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

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
class NestedLogitDataset:
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
class MixedLogitDataset:
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


# ---------------------------------------------------------------------------
# Helper: build a ChoiceTable from choosers / alternatives / choices
# ---------------------------------------------------------------------------


def _build_choice_table(choosers, alternatives, choices, interactions=None):
    """Build a ChoiceTable from component DataFrames."""
    from locpick.data.choicetable import ChoiceTable

    return ChoiceTable.from_tables(
        choosers=choosers.drop(columns="choice", errors="ignore"),
        alternatives=alternatives,
        chosen_alternatives=choices,
        interactions=interactions,
    )


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
    * ``obs_x_alt``  — element-wise product ``obs_feature × alt_feature``
      broadcast over the full ``(n_obs × n_alts)`` grid
    * Deterministic utility::

          U_det = alt_params["alt_feature"] * alt_feature
                + interaction_params["obs_x_alt"] * obs_x_alt

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
        Mapping of interaction column name → true coefficient.
        Default: ``{"obs_x_alt": 0.95}``.
    seed : int, default 1234

    Returns
    -------
    MNLDataset
    """
    if alt_params is None:
        alt_params = {"alt_feature": -0.65}
    if interaction_params is None:
        interaction_params = {"obs_x_alt": 0.95}

    rng = np.random.default_rng(seed)

    # --- Choosers -------------------------------------------------------
    obs_ids = pd.Index(np.arange(n_obs), name="oid")
    obs_feature = rng.standard_normal(n_obs)
    choosers = pd.DataFrame({"obs_feature": obs_feature}, index=obs_ids)

    # --- Alternatives ---------------------------------------------------
    alt_ids = pd.Index(np.arange(n_alts), name="aid")
    alt_feature = np.linspace(-1.0, 1.0, n_alts)
    alternatives = pd.DataFrame({"alt_feature": alt_feature}, index=alt_ids)

    # --- Interactions (broadcast) ----------------------------------------
    interaction_index = pd.MultiIndex.from_product([obs_ids, alt_ids], names=["oid", "aid"])
    obs_feat_tiled = np.repeat(obs_feature, n_alts)  # (n_obs*n_alts,)
    alt_feat_tiled = np.tile(alt_feature, n_obs)  # (n_obs*n_alts,)
    obs_x_alt_values = obs_feat_tiled * alt_feat_tiled

    interactions: dict[str, pd.Series] = {
        "obs_x_alt": pd.Series(obs_x_alt_values, index=interaction_index, name="obs_x_alt")
    }

    # --- Deterministic utility ------------------------------------------
    det_utility = np.zeros((n_obs, n_alts))
    for col, coef in alt_params.items():
        alt_vals = alternatives[col].to_numpy()
        det_utility += coef * np.tile(alt_vals, n_obs).reshape(n_obs, n_alts)
    if "obs_x_alt" in interaction_params:
        det_utility += interaction_params["obs_x_alt"] * obs_x_alt_values.reshape(n_obs, n_alts)

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
    choice_table = _build_choice_table(choosers, alternatives, choosers["choice"], interactions)

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
    seed: int = 1234,
) -> NestedLogitDataset:
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
    from locpick.models.nested import (
        NestingTree,
        NestSpec,
        _nested_logit_probs_numpy,
    )

    if alt_params is None:
        alt_params = {"cost": -0.5, "time": -0.1}
    if nest_lambdas is None:
        nest_lambdas = {"transit": 0.7, "auto": 0.8}

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
    alternatives = pd.DataFrame(
        {
            "cost": rng.uniform(1, 10, n_alts),
            "time": rng.uniform(5, 30, n_alts),
        },
        index=alt_ids,
    )

    # --- Interactions (chooser × alternative) --------------------------
    interaction_index = pd.MultiIndex.from_product([obs_ids, alt_ids], names=["oid", "aid"])
    income_tiled = np.repeat(income, n_alts)
    cost_tiled = np.tile(alternatives["cost"].to_numpy(), n_obs)
    time_tiled = np.tile(alternatives["time"].to_numpy(), n_obs)
    interactions = {
        "income_x_cost": pd.Series(
            income_tiled * cost_tiled, index=interaction_index, name="income_x_cost"
        ),
        "income_x_time": pd.Series(
            income_tiled * time_tiled, index=interaction_index, name="income_x_time"
        ),
    }

    # --- Deterministic utility ------------------------------------------
    det_utility = np.zeros((n_obs, n_alts))
    for col, coef in alt_params.items():
        alt_vals = alternatives[col].to_numpy()
        det_utility += coef * np.tile(alt_vals, n_obs).reshape(n_obs, n_alts)
    # Add interaction terms
    det_utility += interactions["income_x_cost"].to_numpy().reshape(n_obs, n_alts) * 0.05
    det_utility += interactions["income_x_time"].to_numpy().reshape(n_obs, n_alts) * (-0.02)

    # --- Build nest matrix and simulate choices -------------------------
    alt_id_list = list(range(n_alts))
    nest_matrix = nests.build_nest_matrix(alt_id_list)

    # Convert lambdas to alpha (unconstrained) for the probability kernel
    lambda_values = np.array([nest_lambdas[name] for name in nest_names])
    # alpha = log(lambda / (1 - lambda))  (inverse logistic)
    alpha_values = np.log(lambda_values / (1.0 - lambda_values + 1e-30))

    # Compute nested logit probabilities
    beta = np.array([alt_params[col] for col in alternatives.columns])
    probs = _nested_logit_probs_numpy(
        beta,
        alpha_values,
        np.tile(alternatives.to_numpy(), (n_obs, 1)),  # design matrix
        nest_matrix,
        n_obs,
        n_alts,
    )

    # Simulate choices from probabilities
    choices = np.array([rng.choice(n_alts, p=probs[i]) for i in range(n_obs)])
    choosers = choosers.copy()
    choosers["choice"] = choices

    # --- Build ChoiceTable -----------------------------------------------
    choice_table = _build_choice_table(choosers, alternatives, choosers["choice"], interactions)

    # Include interaction params in true_params
    true_params = dict(alt_params)
    true_params["income_x_cost"] = 0.05
    true_params["income_x_time"] = -0.02

    return NestedLogitDataset(
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
    from locpick.models.scl import _resolve_spatial_graph, _scl_log_probs_numpy

    if alt_params is None:
        alt_params = {"cost": -0.5, "time": -0.1}

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
    alternatives = pd.DataFrame(
        {
            "cost": rng.uniform(1, 10, n_alts),
            "time": rng.uniform(5, 30, n_alts),
        },
        index=alt_ids,
    )

    # --- Interactions (chooser × alternative) --------------------------
    interaction_index = pd.MultiIndex.from_product([obs_ids, alt_ids], names=["oid", "aid"])
    income_tiled = np.repeat(income, n_alts)
    cost_tiled = np.tile(alternatives["cost"].to_numpy(), n_obs)
    interactions = {
        "income_x_cost": pd.Series(
            income_tiled * cost_tiled, index=interaction_index, name="income_x_cost"
        ),
    }

    # --- Deterministic utility ------------------------------------------
    det_utility = np.zeros((n_obs, n_alts))
    for col, coef in alt_params.items():
        alt_vals = alternatives[col].to_numpy()
        det_utility += coef * np.tile(alt_vals, n_obs).reshape(n_obs, n_alts)
    # Add interaction term
    det_utility += interactions["income_x_cost"].to_numpy().reshape(n_obs, n_alts) * 0.05

    # --- Compute SCL probabilities and simulate choices ------------------
    beta = np.array([alt_params[col] for col in alternatives.columns])
    design_matrix = np.tile(alternatives.to_numpy(), (n_obs, 1))

    log_probs = _scl_log_probs_numpy(
        beta, rho, design_matrix, allocation, edge_list, n_obs, n_alts
    )
    probs = np.exp(log_probs)

    # Simulate choices
    choices = np.array([rng.choice(n_alts, p=probs[i]) for i in range(n_obs)])
    choosers = choosers.copy()
    choosers["choice"] = choices

    # --- Build ChoiceTable -----------------------------------------------
    choice_table = _build_choice_table(choosers, alternatives, choosers["choice"], interactions)

    # Include interaction param in true_params
    true_params = dict(alt_params)
    true_params["income_x_cost"] = 0.05

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
    seed: int = 1234,
) -> MixedLogitDataset:
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

    rng = np.random.default_rng(seed)

    # --- Choosers and alternatives --------------------------------------
    obs_ids = pd.Index(np.arange(n_obs), name="oid")
    income = rng.standard_normal(n_obs)
    choosers = pd.DataFrame({"income": income}, index=obs_ids)

    alt_ids = pd.Index(np.arange(n_alts), name="aid")
    alternatives = pd.DataFrame(
        {
            "cost": rng.uniform(1, 10, n_alts),
            "time": rng.uniform(5, 30, n_alts),
        },
        index=alt_ids,
    )

    # --- Interactions (chooser × alternative) --------------------------
    interaction_index = pd.MultiIndex.from_product([obs_ids, alt_ids], names=["oid", "aid"])
    income_tiled = np.repeat(income, n_alts)
    cost_tiled = np.tile(alternatives["cost"].to_numpy(), n_obs)
    interactions = {
        "income_x_cost": pd.Series(
            income_tiled * cost_tiled, index=interaction_index, name="income_x_cost"
        ),
    }

    # --- Deterministic utility (fixed part) ----------------------------
    det_utility = np.zeros((n_obs, n_alts))
    for col, coef in alt_params.items():
        alt_vals = alternatives[col].to_numpy()
        det_utility += coef * np.tile(alt_vals, n_obs).reshape(n_obs, n_alts)
    # Add interaction term
    det_utility += interactions["income_x_cost"].to_numpy().reshape(n_obs, n_alts) * 0.05

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
    true_params["income_x_cost"] = 0.05
    for col, (dist, mean, sd) in random_params.items():
        true_params[f"mean_{col}"] = mean
        true_params[f"sd_{col}"] = sd

    # --- Build random_params dict for MixedLogit -----------------------
    random_params_dict = {col: dist for col, (dist, _, _) in random_params.items()}

    # --- Build ChoiceTable -----------------------------------------------
    choice_table = _build_choice_table(choosers, alternatives, choosers["choice"], interactions)

    return MixedLogitDataset(
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

    rng = np.random.default_rng(seed)

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
    alternatives = pd.DataFrame(
        {
            "cost": rng.uniform(1, 10, n_alts),
            "time": rng.uniform(5, 30, n_alts),
        },
        index=alt_ids,
    )

    # --- Interactions (chooser × alternative) --------------------------
    interaction_index = pd.MultiIndex.from_product([obs_ids, alt_ids], names=["oid", "aid"])
    income_tiled = np.repeat(income, n_alts)
    cost_tiled = np.tile(alternatives["cost"].to_numpy(), n_obs)
    interactions = {
        "income_x_cost": pd.Series(
            income_tiled * cost_tiled, index=interaction_index, name="income_x_cost"
        ),
    }

    # --- Deterministic utility (fixed part) ----------------------------
    det_utility = np.zeros((n_obs, n_alts))
    for col, coef in alt_params.items():
        alt_vals = alternatives[col].to_numpy()
        det_utility += coef * np.tile(alt_vals, n_obs).reshape(n_obs, n_alts)
    # Add interaction term
    det_utility += interactions["income_x_cost"].to_numpy().reshape(n_obs, n_alts) * 0.05

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
    true_params["income_x_cost"] = 0.05
    for col, (dist, mean, sd) in random_params.items():
        true_params[f"mean_{col}"] = mean
        true_params[f"sd_{col}"] = sd

    # --- Build random_params dict for MSCL -----------------------------
    random_params_dict = {col: dist for col, (dist, _, _) in random_params.items()}

    # --- Build ChoiceTable -----------------------------------------------
    choice_table = _build_choice_table(choosers, alternatives, choosers["choice"], interactions)

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
