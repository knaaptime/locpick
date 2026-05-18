"""Choice data assembly for location choice models.

This module provides the ``ChoiceTable`` class, which handles merging
chooser and alternative data, sampling alternatives, and producing
estimation-ready arrays for v2 location choice workflows.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from locpick._sampling.kernels import (
    HAS_NUMBA,
    _sample_unweighted_without_replacement_exclusion,
    _sample_weighted_without_replacement_1d_exclusion,
)
from locpick.data.arrays import ChoiceArrays
from locpick.data.dataset import (
    _resolve_interaction,
    build_choice_dataset,
    build_choice_dataset_from_long,
    dataset_to_long_frame,
)

# ---------------------------------------------------------------------------
# ChoiceTable class
# ---------------------------------------------------------------------------


class ChoiceTable:
    """Long-format choice data for location choice models.

    This is the primary data container for locpick v2. It handles
    merging chooser and alternative data, sampling alternatives, and
    producing estimation-ready arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with a MultiIndex of (obs_id, alt_id).
        Must contain a binary ``chosen`` column if this is estimation data.
    obs_id_col : str
        Name of the observation ID column (or first level of the index).
    alt_id_col : str
        Name of the alternative ID column (or second level of the index).
    choice_col : str or None
        Name of the binary choice column. None if this is prediction data.
    available_col : str or None
        Name of the binary availability column. None if all alternatives
        are available.
    sample_size : int or None
        Number of alternatives sampled per observation, if sampling was used.
        None if no sampling (census of all alternatives).
    """

    def __init__(
        self,
        ds: xr.Dataset,
        obs_id_col: str,
        alt_id_col: str,
        choice_col: Optional[str] = None,
        available_col: Optional[str] = None,
        sample_size: Optional[int] = None,
        matrix_data: Optional[dict[str, Union[pd.Series, np.ndarray]]] = None,
        interaction_expressions: Optional[dict[str, dict[str, str]]] = None,
    ):
        self._ds = ds.copy(deep=True)
        self._obs_id_col = obs_id_col
        self._alt_id_col = alt_id_col
        self._choice_col = choice_col
        self._available_col = available_col
        self._sample_size = sample_size
        self._matrix_data = dict(matrix_data or {})
        self._interaction_expressions = dict(interaction_expressions or {})
        self._to_arrays_cache: dict = {}
        self._frame_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Primary constructor: from separate tables with sampling
    # ------------------------------------------------------------------

    @classmethod
    def from_tables(
        cls,
        choosers: pd.DataFrame,
        alternatives: pd.DataFrame,
        chosen_alternatives: Union[str, pd.Series, None] = None,
        sample_size: Optional[int] = None,
        interactions: Optional[dict[str, pd.Series]] = None,
        matrix_data: Optional[dict[str, Union[pd.Series, np.ndarray]]] = None,
        weights: Union[str, pd.Series, None] = None,
        available: Union[str, pd.Series, None] = None,
        replace: bool = True,
        seed: Optional[int] = None,
    ) -> "ChoiceTable":
        """Build from separate chooser/alternative tables with optional sampling.

        Parameters
        ----------
        choosers : pd.DataFrame
            Table with one row per chooser, indexed by obs_id.
        alternatives : pd.DataFrame
            Table with one row per alternative, indexed by alt_id.
        chosen_alternatives : str or pd.Series or None
            Column name in ``choosers`` or a Series mapping obs_id → alt_id.
            Required for estimation data; None for prediction data.
        sample_size : int or None
            Number of alternatives to sample per chooser. None means use all
            alternatives (census). The chosen alternative is always included.
        interactions : dict[str, pd.Series] or None
            Dict mapping column names to (obs_id, alt_id)-indexed Series.
            These are chooser×alternative-specific variables like distance.
        matrix_data : dict[str, pd.Series | np.ndarray] or None
            Lazily resolved chooser×alternative matrices.
        weights : str or pd.Series or None
            Sampling weights. If str, column name in ``alternatives``.
            If Series, either alt_id-indexed (1D) or (obs_id, alt_id)-indexed (2D).
        available : str or pd.Series or None
            Binary availability mask. If str, column name in ``alternatives``.
            If Series, either alt_id-indexed or (obs_id, alt_id)-indexed.
        replace : bool
            Whether to sample with replacement. Default True.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        ChoiceTable
        """
        if seed is not None:
            np.random.seed(seed)

        # Normalize index names; if the named column exists as a data column,
        # promote it to the index.
        if choosers.index.name is None:
            choosers = choosers.copy()
            if "obs_id" in choosers.columns:
                choosers = choosers.set_index("obs_id")
            else:
                choosers.index.name = "obs_id"
        if alternatives.index.name is None:
            alternatives = alternatives.copy()
            if "alt_id" in alternatives.columns:
                alternatives = alternatives.set_index("alt_id")
            else:
                alternatives.index.name = "alt_id"

        oid_name = choosers.index.name
        aid_name = alternatives.index.name

        # Normalize chosen_alternatives to a Series (obs_id → alt_id)
        chosen_series: Optional[pd.Series] = None
        if chosen_alternatives is not None:
            if isinstance(chosen_alternatives, str):
                chosen_series = choosers[chosen_alternatives].copy()
                choosers = choosers.drop(columns=[chosen_alternatives])
            elif isinstance(chosen_alternatives, pd.DataFrame):
                # DataFrame with two columns: (obs_key, alt_id).
                # First column is the chooser key; last column is the chosen alt.
                df_c = chosen_alternatives
                obs_key_col = df_c.columns[0]
                alt_key_col = df_c.columns[-1]
                # Build a mapping from chooser key value → chooser index
                if obs_key_col == oid_name or obs_key_col in choosers.index.names:
                    chosen_series = df_c.set_index(obs_key_col)[alt_key_col]
                else:
                    # obs_key_col is a data column in choosers — look up the index
                    key_to_idx = pd.Series(choosers.index, index=choosers[obs_key_col].values)
                    chosen_series = df_c.set_index(obs_key_col)[alt_key_col]
                    chosen_series.index = chosen_series.index.map(key_to_idx)
            else:
                chosen_series = chosen_alternatives.copy()

        # Normalize weights
        weights_series: Optional[pd.Series] = None
        weights_1d = False
        weights_2d = False
        if weights is not None:
            if isinstance(weights, str):
                weights_series = alternatives[weights]
            else:
                weights_series = weights
            if len(weights_series) == len(alternatives):
                weights_1d = True
            elif len(weights_series) == len(choosers) * len(alternatives):
                weights_2d = True
            else:
                raise ValueError(
                    f"Length of weights ({len(weights_series)}) does not match "
                    f"alternatives ({len(alternatives)}) or "
                    f"choosers×alternatives ({len(choosers) * len(alternatives)})"
                )

        # Normalize available
        available_series: Optional[pd.Series] = None
        if available is not None:
            if isinstance(available, str):
                available_series = alternatives[available]
            else:
                available_series = available

        # Build the merged table
        if sample_size is None:
            df = cls._build_census(choosers, alternatives, chosen_series, oid_name, aid_name)
        else:
            df = cls._build_sampled(
                choosers,
                alternatives,
                chosen_series,
                sample_size,
                weights_series,
                weights_1d,
                weights_2d,
                replace,
                oid_name,
                aid_name,
            )

        n_obs = len(choosers)
        n_alts_eff = int(len(df) / n_obs) if n_obs > 0 else 0
        obs_ids = choosers.index.to_numpy()
        alt_ids_matrix = df[aid_name].to_numpy().reshape(n_obs, n_alts_eff)

        # Determine choice column name and values
        choice_col_name: Optional[str] = None
        chosen_arr = None
        if chosen_series is not None:
            choice_col_name = "chosen"
            chosen_arr = df["chosen"].to_numpy().reshape(n_obs, n_alts_eff)

        # Determine availability column name and values
        available_col_name: Optional[str] = None
        available_arr = None
        if available_series is not None:
            if isinstance(available, str):
                available_col_name = available
            else:
                available_col_name = available_series.name or "available"
                if (
                    isinstance(available_series.index, pd.MultiIndex)
                    and available_series.index.nlevels == 2
                ):
                    available_arr = _resolve_interaction(
                        available_series,
                        obs_ids,
                        alt_ids_matrix,
                    ).to_numpy()
                else:
                    mapped = available_series.reindex(alt_ids_matrix.reshape(-1)).to_numpy()
                    available_arr = mapped.reshape(n_obs, n_alts_eff)

        combined_matrix_data: dict[str, Union[pd.Series, np.ndarray]] = {}
        if interactions:
            combined_matrix_data.update(interactions)
        if matrix_data:
            overlap = set(combined_matrix_data).intersection(matrix_data)
            if overlap:
                overlap_list = ", ".join(sorted(overlap))
                raise ValueError(
                    f"Duplicate matrix_data names provided via interactions and matrix_data: {overlap_list}"
                )
            combined_matrix_data.update(matrix_data)

        ds = build_choice_dataset(
            obs_ids=obs_ids,
            alt_ids_matrix=alt_ids_matrix,
            chooser_df=choosers,
            alt_df=alternatives,
            chosen_arr=chosen_arr,
            interaction_data=None,
            available_arr=available_arr,
            sample_size=sample_size,
            obs_id_name=oid_name,
            alt_id_name=aid_name,
        )
        ds.attrs["alt_id_universe"] = alternatives.index.to_list()

        if available_col_name is not None and available_col_name != "available":
            if "available" in ds.data_vars:
                ds = ds.rename({"available": available_col_name})
        ds.attrs["choice_col"] = choice_col_name
        ds.attrs["available_col"] = available_col_name

        # Store sampling metadata for inclusion probability computation and
        # provenance handoff to EstimationProblem.
        if sample_size is not None:
            ds.attrs["sampling_method"] = "srswor" if not replace else "srswr"
            ds.attrs["sampling_replace"] = replace
            ds.attrs["sampling_seed"] = seed
            ds.attrs["n_alts_full"] = len(alternatives)
            ds.attrs["sampling_design"] = {
                "method": ds.attrs["sampling_method"],
                "sample_size": sample_size,
                "n_alts_full": len(alternatives),
                "replace": replace,
                "seed": seed,
            }

        return cls(
            ds=ds,
            obs_id_col=oid_name,
            alt_id_col=aid_name,
            choice_col=choice_col_name,
            available_col=available_col_name,
            sample_size=sample_size,
            matrix_data=combined_matrix_data,
            interaction_expressions=None,
        )

    # ------------------------------------------------------------------
    # Secondary constructor: from pre-built long-format data
    # ------------------------------------------------------------------

    @classmethod
    def from_long(
        cls,
        data: pd.DataFrame,
        obs_id_col: str,
        alt_id_col: str,
        choice_col: Optional[str] = None,
        available_col: Optional[str] = None,
    ) -> "ChoiceTable":
        """Build from an existing long-format DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Long-format DataFrame with obs_id and alt_id columns or index.
        obs_id_col : str
            Name of the observation ID column.
        alt_id_col : str
            Name of the alternative ID column.
        choice_col : str or None
            Name of the binary choice column. None for prediction data.
        available_col : str or None
            Name of the binary availability column.

        Returns
        -------
        ChoiceTable
        """
        df = data.copy()

        # Ensure proper index
        if obs_id_col in df.columns and alt_id_col in df.columns:
            df = df.set_index([obs_id_col, alt_id_col])
        elif not isinstance(df.index, pd.MultiIndex):
            raise ValueError(
                f"DataFrame must have columns '{obs_id_col}' and "
                f"'{alt_id_col}', or a MultiIndex with those names."
            )

        # Reset index to have obs_id and alt_id as columns
        df = df.reset_index()

        ds = build_choice_dataset_from_long(
            df=df,
            obs_id_col=obs_id_col,
            alt_id_col=alt_id_col,
            choice_col=choice_col,
            available_col=available_col,
            sample_size=None,
        )

        return cls(
            ds=ds,
            obs_id_col=obs_id_col,
            alt_id_col=alt_id_col,
            choice_col=choice_col,
            available_col=available_col,
            sample_size=None,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        """The underlying long-format DataFrame."""
        return self.to_frame()

    @property
    def obs_id_col(self) -> str:
        """Name of the observation ID column."""
        return self._obs_id_col

    @property
    def alt_id_col(self) -> str:
        """Name of the alternative ID column."""
        return self._alt_id_col

    @property
    def choice_col(self) -> Optional[str]:
        """Name of the binary choice column, or None."""
        return self._choice_col

    @property
    def available_col(self) -> Optional[str]:
        """Name of the binary availability column, or None."""
        return self._available_col

    @property
    def n_observations(self) -> int:
        """Number of observations (choosers)."""
        return int(self._ds.sizes.get("obs_id", 0))

    @property
    def n_alternatives(self) -> int:
        """Number of alternatives per observation."""
        return int(self._ds.sizes.get("alt_pos", 0))

    @property
    def sample_size(self) -> Optional[int]:
        """Number of alternatives sampled per observation, or None if census."""
        return self._sample_size

    @property
    def n_alternatives_full(self) -> int:
        """Total alternatives in the universe before sampling (if available)."""
        if self._ds.attrs.get("n_alts_full") is not None:
            return int(self._ds.attrs["n_alts_full"])
        if "alt_id_universe" in self._ds.attrs:
            return int(len(self._ds.attrs["alt_id_universe"]))
        return self.n_alternatives

    @property
    def sampling_design(self) -> Optional[dict]:
        """Sampling design provenance metadata, if sampling was used."""
        design = self._ds.attrs.get("sampling_design")
        if isinstance(design, dict):
            return dict(design)

        method = self._ds.attrs.get("sampling_method")
        if method is None:
            return None

        return {
            "method": method,
            "sample_size": self._sample_size,
            "n_alts_full": self._ds.attrs.get("n_alts_full"),
            "replace": self._ds.attrs.get("sampling_replace"),
            "seed": self._ds.attrs.get("sampling_seed"),
        }

    def to_dataset(self) -> xr.Dataset:
        """Return a defensive copy of the internal xarray Dataset."""
        return self._ds.copy(deep=True)

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def add_interaction(self, name: str, series: pd.Series) -> "ChoiceTable":
        """Merge an (obs_id, alt_id)-indexed Series as a new column.

        Parameters
        ----------
        name : str
            Column name for the interaction variable.
        series : pd.Series
            Series with a MultiIndex of (obs_id, alt_id).

        Returns
        -------
        ChoiceTable
            A new ChoiceTable with the interaction column added.
        """
        if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
            raise ValueError(
                "Interaction series must have a MultiIndex with levels (obs_id, alt_id)."
            )

        obs_ids = np.asarray(self._ds.coords["obs_id"].values)
        alt_ids_matrix = np.asarray(self._ds["alt_id_values"].values)
        obs_to_row = {obs_id: i for i, obs_id in enumerate(obs_ids)}
        obs_alt_sets = {
            obs_id: set(alt_ids_matrix[i, :].tolist()) for i, obs_id in enumerate(obs_ids)
        }

        obs_level = series.index.get_level_values(0)
        for obs_id in np.unique(obs_level):
            if obs_id not in obs_to_row:
                continue
            provided_alts = set(series.xs(obs_id, level=0).index.tolist())
            if provided_alts and provided_alts.isdisjoint(obs_alt_sets[obs_id]):
                raise KeyError(f"Interaction contains no alt_ids present for obs_id {obs_id!r}.")

        matrix_data = dict(self._matrix_data)
        matrix_data[name] = series
        interaction_expressions = dict(self._interaction_expressions)
        interaction_expressions.pop(name, None)

        ds = self._ds.copy(deep=True)
        if name in ds.data_vars:
            ds = ds.drop_vars(name)

        return ChoiceTable(
            ds=ds,
            obs_id_col=self._obs_id_col,
            alt_id_col=self._alt_id_col,
            choice_col=self._choice_col,
            available_col=self._available_col,
            sample_size=self._sample_size,
            matrix_data=matrix_data,
            interaction_expressions=interaction_expressions,
        )

    def add_interaction_expression(
        self,
        name: str,
        left: str,
        right: str,
        *,
        op: str = "product",
        missing_policy: str = "error",
    ) -> "ChoiceTable":
        """Generate a chooser-alternative interaction from existing columns.

        Parameters
        ----------
        name : str
            Name of the generated interaction column.
        left, right : str
            Source columns in the current choice table. Columns may be chooser,
            alternative, or already-aligned chooser-alternative variables.
        op : {"product"}
            Operation used to combine the sources.
        missing_policy : {"error", "allow_unavailable"}
            Missing-value policy for generated values. ``"allow_unavailable"``
            permits missing values only where the availability mask is false.

        Returns
        -------
        ChoiceTable
            A new ChoiceTable with the generated interaction column added.
        """
        if op != "product":
            raise ValueError("Only product interactions are currently supported.")
        if missing_policy not in {"error", "allow_unavailable"}:
            raise ValueError("missing_policy must be 'error' or 'allow_unavailable'.")

        available_columns = set(self._ds.data_vars)
        available_columns.update(self._matrix_data.keys())
        available_columns.update(self._interaction_expressions.keys())

        for column in (left, right):
            if column not in available_columns:
                raise ValueError(
                    f"Column '{column}' not found in choice data. "
                    f"Available columns: {sorted(available_columns)}"
                )

        # Preserve existing validation semantics at method call time while
        # keeping interaction materialization lazy until to_frame()/to_arrays().
        probe_df = self._get_frame_cached(copy=False)
        probe_values = np.asarray(probe_df[left], dtype=np.float64) * np.asarray(
            probe_df[right], dtype=np.float64
        )
        missing = np.isnan(probe_values)
        if missing.any():
            if missing_policy == "allow_unavailable" and self._available_col in probe_df.columns:
                available = np.asarray(probe_df[self._available_col], dtype=np.float64).astype(
                    bool
                )
                if np.any(missing & available):
                    raise ValueError(
                        f"Generated interaction '{name}' has missing values for available alternatives."
                    )
            else:
                raise ValueError(f"Generated interaction '{name}' contains missing values.")

        interaction_expressions = dict(self._interaction_expressions)
        interaction_expressions[name] = {
            "left": left,
            "right": right,
            "op": op,
            "missing_policy": missing_policy,
        }

        matrix_data = dict(self._matrix_data)
        matrix_data.pop(name, None)

        ds = self._ds.copy(deep=True)
        if name in ds.data_vars:
            ds = ds.drop_vars(name)

        return ChoiceTable(
            ds=ds,
            obs_id_col=self._obs_id_col,
            alt_id_col=self._alt_id_col,
            choice_col=self._choice_col,
            available_col=self._available_col,
            sample_size=self._sample_size,
            matrix_data=matrix_data,
            interaction_expressions=interaction_expressions,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return the underlying long-format DataFrame."""
        return self._get_frame_cached(copy=True)

    def _get_frame_cached(self, *, copy: bool) -> pd.DataFrame:
        """Return cached long-format data, optionally as a defensive copy."""
        if self._frame_cache is None:
            self._frame_cache = self._materialize_frame()
        if copy:
            return self._frame_cache.copy(deep=True)
        return self._frame_cache

    def _materialize_frame(self) -> pd.DataFrame:
        """Materialize long-format data from xarray plus lazy matrix columns."""
        df = dataset_to_long_frame(self._ds)
        if not self._matrix_data and not self._interaction_expressions:
            return df

        obs_ids = np.asarray(self._ds.coords["obs_id"].values)
        alt_ids_matrix = np.asarray(self._ds["alt_id_values"].values)
        n_obs, n_alts = alt_ids_matrix.shape

        for name, source in self._matrix_data.items():
            col_matrix = self._resolve_matrix_data(source, obs_ids, alt_ids_matrix)
            df[name] = np.asarray(col_matrix, dtype=np.float64).reshape(n_obs * n_alts)

        for name, expression in self._interaction_expressions.items():
            left = expression["left"]
            right = expression["right"]
            missing_policy = expression["missing_policy"]

            if left not in df.columns or right not in df.columns:
                raise ValueError(
                    f"Cannot resolve interaction '{name}': '{left}' or '{right}' is missing."
                )

            values = np.asarray(df[left], dtype=np.float64) * np.asarray(
                df[right], dtype=np.float64
            )
            missing = np.isnan(values)
            if missing.any():
                if missing_policy == "allow_unavailable" and self._available_col in df.columns:
                    available = np.asarray(df[self._available_col], dtype=np.float64).astype(bool)
                    if np.any(missing & available):
                        raise ValueError(
                            f"Generated interaction '{name}' has missing values for available alternatives."
                        )
                else:
                    raise ValueError(f"Generated interaction '{name}' contains missing values.")

            df[name] = values

        return df

    def _sampled_alt_positions(self, alt_ids_matrix: np.ndarray) -> Optional[np.ndarray]:
        """Return alternative-universe positions for sampled alt IDs if available."""
        if "alt_id_universe" not in self._ds.attrs:
            return None
        alt_universe = np.asarray(self._ds.attrs["alt_id_universe"])
        universe_index = pd.Index(alt_universe)
        flat_positions = universe_index.get_indexer(alt_ids_matrix.reshape(-1))
        return flat_positions.reshape(alt_ids_matrix.shape)

    def _resolve_matrix_data(
        self,
        source: Union[pd.Series, np.ndarray],
        obs_ids: np.ndarray,
        alt_ids_matrix: np.ndarray,
    ) -> np.ndarray:
        """Resolve matrix_data source to a dense (n_obs, n_alts) array."""
        n_obs, n_alts = alt_ids_matrix.shape

        if isinstance(source, pd.Series):
            if isinstance(source.index, pd.MultiIndex) and source.index.nlevels == 2:
                return _resolve_interaction(source, obs_ids, alt_ids_matrix).to_numpy()
            if source.index.nlevels == 1:
                mapped = source.reindex(alt_ids_matrix.reshape(-1)).to_numpy(dtype=np.float64)
                return mapped.reshape(n_obs, n_alts)
            raise ValueError("Series matrix_data must be indexed by alt_id or (obs_id, alt_id).")

        array = np.asarray(source, dtype=np.float64)
        if array.shape == (n_obs, n_alts):
            return array
        if array.ndim == 1 and array.size == n_obs * n_alts:
            return array.reshape(n_obs, n_alts)

        n_alts_full = self.n_alternatives_full
        sampled_positions = self._sampled_alt_positions(alt_ids_matrix)
        if array.shape == (n_obs, n_alts_full):
            if sampled_positions is not None:
                safe_positions = np.where(sampled_positions < 0, 0, sampled_positions)
                resolved = array[np.arange(n_obs)[:, None], safe_positions].astype(np.float64)
                resolved[sampled_positions < 0] = np.nan
                return resolved

            if np.issubdtype(alt_ids_matrix.dtype, np.integer):
                return array[np.arange(n_obs)[:, None], alt_ids_matrix.astype(int)]

        if array.ndim == 1 and array.size == n_alts_full:
            if sampled_positions is not None:
                safe_positions = np.where(sampled_positions < 0, 0, sampled_positions)
                resolved = array[safe_positions].astype(np.float64)
                resolved[sampled_positions < 0] = np.nan
                return resolved

            if np.issubdtype(alt_ids_matrix.dtype, np.integer):
                return array[alt_ids_matrix.astype(int)]

        raise ValueError(
            "Unable to align matrix_data array to sampled alternatives. "
            f"Received shape {array.shape}, expected one of "
            f"({n_obs}, {n_alts}), ({n_obs * n_alts},), ({n_obs}, {n_alts_full}), or ({n_alts_full},)."
        )

    def to_arrays(
        self,
        formula: Optional[str] = None,
        spec=None,
        weights=None,
        available=None,
        sparse: bool = False,
        sparse_threshold: float = 0.5,
    ) -> ChoiceArrays:
        """Convert to estimation-ready arrays.

        Parameters
        ----------
        formula : str or None
            Formulaic formula string. If None, all numeric columns
            (except obs_id, alt_id, chosen, available) are used.
        spec : ModelSpec or None
            ModelSpec using formula/scoped-term specification. When
            provided, ``spec.build_design_matrix(...)`` is used.
        weights : array-like or None
            Observation weights as a flat array of length n_obs * n_alts.
        available : array-like or None
            Alternative availability as a flat array of length n_obs * n_alts.
        sparse : bool
            If True, convert the design matrix to scipy.sparse.csr_matrix
            when the zero fraction exceeds ``sparse_threshold``. This is
            useful for large choice sets with many zero-valued variables
            (e.g., "has_subway_station").
        sparse_threshold : float
            Fraction of zeros required to trigger sparse conversion.
            Default 0.5 (50% zeros).

        Returns
        -------
        ChoiceArrays
        """
        # Cache key: hashable tuple of all arguments
        cache_key = (
            formula,
            id(spec) if spec is not None else None,
            tuple(weights)
            if hasattr(weights, "__iter__") and not isinstance(weights, str)
            else weights,
            tuple(available)
            if hasattr(available, "__iter__") and not isinstance(available, str)
            else available,
            sparse,
            sparse_threshold,
        )
        if cache_key in self._to_arrays_cache:
            return self._to_arrays_cache[cache_key]

        import numpy as np

        df = self._get_frame_cached(copy=False)
        n_obs = self.n_observations
        n_alts = self.n_alternatives

        # Resolve effective formula from spec if given
        if spec is not None and formula is None:
            if hasattr(spec, "prepare_data"):
                result = spec.build_design_matrix(self)
                self._to_arrays_cache[cache_key] = result
                return result
            if hasattr(spec, "formula") and spec.formula is not None:
                formula = spec.formula

        # Build design matrix from formula
        if formula is not None:
            try:
                import formulaic

                # Suppress the intercept by default: MNL utility functions
                # don't use alternative-specific constants via the formula.
                # Users can add "+ 1" explicitly to include an intercept.
                if "intercept" not in formula.lower() and formula.strip()[-2:] not in (
                    "+ 1",
                    "+1",
                ):
                    formula_str = formula + " - 1"
                else:
                    formula_str = formula
                dm = formulaic.model_matrix(formula_str, df)
            except ImportError:
                raise ImportError(
                    "formulaic is required for formula-based design matrices. "
                    "Install it with: pip install formulaic"
                )
        else:
            # Use all numeric columns except reserved ones
            reserved = {self._obs_id_col, self._alt_id_col}
            if self._choice_col:
                reserved.add(self._choice_col)
            if self._available_col:
                reserved.add(self._available_col)
            numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns if c not in reserved
            ]
            dm = df[numeric_cols].values

        # Extract chosen matrix
        chosen = None
        if self._choice_col and self._choice_col in self._ds.data_vars:
            chosen = np.asarray(self._ds[self._choice_col].values, dtype=np.float64)

        # Extract availability matrix (from kwarg or stored col)
        avail = None
        if available is not None:
            avail = np.asarray(available, dtype=np.float64).reshape(n_obs, n_alts)
        elif self._available_col and self._available_col in self._ds.data_vars:
            avail = np.asarray(self._ds[self._available_col].values, dtype=np.float64)

        # Weights
        wts = None
        if weights is not None:
            wts = np.asarray(weights, dtype=np.float64)

        # Compute inclusion probabilities for sampling correction
        inclusion_probs = None

        if self._sample_size is not None:
            from locpick._sampling.inclusion import compute_inclusion_probs

            n_alts_full = self.n_alternatives_full
            n_samples = self._sample_size
            method = self._ds.attrs.get("sampling_method", "srswor")

            # Per-alternative inclusion probabilities
            pi = compute_inclusion_probs(
                sample_size=n_samples,
                n_alts=n_alts_full,
                method=method,
            )
            # For uniform sampling all alternatives share the same probability;
            # use a scalar so we can broadcast to (n_obs, n_alts) where n_alts
            # may differ from n_alts_full after sampling.
            pi_scalar = float(pi[0])
            inclusion_probs = np.full((n_obs, n_alts), pi_scalar, dtype=np.float64)

            # The chosen alternative is always included, so its inclusion
            # probability is 1.0
            if chosen is not None:
                chosen_mask = chosen.reshape(n_obs, n_alts) > 0
                inclusion_probs[chosen_mask] = 1.0

        # Get param names
        if hasattr(dm, "columns"):
            param_names = list(dm.columns)
            design_matrix = np.asarray(dm, dtype=np.float64)
        else:
            param_names = []
            design_matrix = np.asarray(dm, dtype=np.float64)

        # Sparse design matrix (optional)
        design_matrix_sparse = None
        if sparse and design_matrix.size > 0:
            zero_fraction = 1.0 - np.count_nonzero(design_matrix) / design_matrix.size
            if zero_fraction >= sparse_threshold:
                import scipy.sparse as sp

                design_matrix_sparse = sp.csr_matrix(design_matrix)

        # Get obs_ids and alt_ids
        obs_ids = np.repeat(np.asarray(self._ds.coords["obs_id"].values), n_alts)
        alt_ids = np.asarray(self._ds["alt_id_values"].values).reshape(-1)

        result = ChoiceArrays(
            design_matrix=design_matrix,
            design_matrix_sparse=design_matrix_sparse,
            chosen=chosen,
            available=avail,
            weights=wts,
            n_obs=n_obs,
            n_alts=n_alts,
            param_names=param_names,
            inclusion_probs=inclusion_probs,
            obs_ids=obs_ids,
            alt_ids=alt_ids,
        )
        self._to_arrays_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _build_census(
        choosers: pd.DataFrame,
        alternatives: pd.DataFrame,
        chosen_series: Optional[pd.Series],
        oid_name: str,
        aid_name: str,
    ) -> pd.DataFrame:
        """Build merged table without sampling (all alternatives)."""
        obs_ids = np.repeat(choosers.index.values, len(alternatives))
        alt_ids = np.tile(alternatives.index.values, reps=len(choosers))

        df = pd.DataFrame({oid_name: obs_ids, aid_name: alt_ids})
        df = df.join(choosers, how="left", on=oid_name)
        df = df.join(alternatives, how="left", on=aid_name)

        if chosen_series is not None:
            chosen_map = chosen_series.to_dict()
            df["chosen"] = (df[aid_name] == df[oid_name].map(chosen_map)).astype(int)

        return df

    @staticmethod
    def _build_sampled(
        choosers: pd.DataFrame,
        alternatives: pd.DataFrame,
        chosen_series: Optional[pd.Series],
        sample_size: int,
        weights_series: Optional[pd.Series],
        weights_1d: bool,
        weights_2d: bool,
        replace: bool,
        oid_name: str,
        aid_name: str,
    ) -> pd.DataFrame:
        """Build merged table with alternative sampling."""
        n_obs = len(choosers)
        n_alts = len(alternatives)
        alt_ids = alternatives.index.values

        # Determine excluded alternatives (chosen ones)
        excluded_alt_ids = np.full(n_obs, -1, dtype=np.int64)
        if chosen_series is not None:
            # Map chosen alternatives to integer positions
            alt_id_to_int = {v: i for i, v in enumerate(alt_ids)}
            for i, obs_id in enumerate(choosers.index):
                chosen_alt = chosen_series.loc[obs_id]
                # .loc can return a Series when index is non-unique; take first
                if isinstance(chosen_alt, pd.Series):
                    chosen_alt = chosen_alt.iloc[0]
                if chosen_alt in alt_id_to_int:
                    excluded_alt_ids[i] = alt_id_to_int[chosen_alt]

        # Sample alternatives
        if replace:
            # With replacement — simple random sampling
            if weights_series is not None and weights_1d:
                probs = weights_series.values / weights_series.values.sum()
                sampled = np.empty((n_obs, sample_size), dtype=alt_ids.dtype)
                for i in range(n_obs):
                    available_mask = np.ones(n_alts, dtype=bool)
                    if excluded_alt_ids[i] >= 0:
                        available_mask[excluded_alt_ids[i]] = False
                    available_probs = probs.copy()
                    available_probs[~available_mask] = 0
                    available_probs /= available_probs.sum()
                    sampled[i] = np.random.choice(
                        alt_ids, size=sample_size, replace=True, p=available_probs
                    )
            else:
                sampled = np.empty((n_obs, sample_size), dtype=alt_ids.dtype)
                for i in range(n_obs):
                    available_mask = np.ones(n_alts, dtype=bool)
                    if excluded_alt_ids[i] >= 0:
                        available_mask[excluded_alt_ids[i]] = False
                    available_alts = alt_ids[available_mask]
                    sampled[i] = np.random.choice(available_alts, size=sample_size, replace=True)
        else:
            # Without replacement — use Numba kernels if available
            if weights_series is not None and weights_1d and HAS_NUMBA:
                alt_weights = weights_series.values.astype(np.float64)
                alt_log_weights = np.log(np.maximum(alt_weights, 1e-300))
                sampled_flat = _sample_weighted_without_replacement_1d_exclusion(
                    alt_ids.astype(np.int64),
                    alt_weights,
                    alt_log_weights,
                    excluded_alt_ids,
                    sample_size,
                )
                sampled = sampled_flat.reshape(n_obs, sample_size)
            elif HAS_NUMBA:
                sampled_flat = _sample_unweighted_without_replacement_exclusion(
                    alt_ids.astype(np.int64), excluded_alt_ids, sample_size
                )
                sampled = sampled_flat.reshape(n_obs, sample_size)
            else:
                # Fallback: Python loop
                sampled = np.empty((n_obs, sample_size), dtype=alt_ids.dtype)
                for i in range(n_obs):
                    available_mask = np.ones(n_alts, dtype=bool)
                    if excluded_alt_ids[i] >= 0:
                        available_mask[excluded_alt_ids[i]] = False
                    available_alts = alt_ids[available_mask]
                    sampled[i] = np.random.choice(available_alts, size=sample_size, replace=False)

        # Ensure chosen alternative is always included
        if chosen_series is not None:
            for i, obs_id in enumerate(choosers.index):
                chosen_alt = chosen_series.loc[obs_id]
                # .loc can return a Series when index has duplicates; take scalar
                if isinstance(chosen_alt, pd.Series):
                    chosen_alt = chosen_alt.iloc[0]
                if chosen_alt not in sampled[i]:
                    # Replace last position with chosen
                    sampled[i, -1] = chosen_alt

        # Build merged DataFrame
        obs_ids_expanded = np.repeat(choosers.index.values, sample_size)
        alt_ids_expanded = sampled.ravel()

        df = pd.DataFrame({oid_name: obs_ids_expanded, aid_name: alt_ids_expanded})
        df = df.join(choosers, how="left", on=oid_name)
        df = df.join(alternatives, how="left", on=aid_name)

        # Add chosen column
        if chosen_series is not None:
            chosen_map = chosen_series.to_dict()
            df["chosen"] = (df[aid_name] == df[oid_name].map(chosen_map)).astype(int)

        return df

    def __repr__(self) -> str:
        parts = [f"ChoiceTable(n_obs={self.n_observations}, n_alts={self.n_alternatives}"]
        if self._sample_size is not None:
            parts.append(f", sample_size={self._sample_size}")
        if self._choice_col:
            parts.append(f", choice_col='{self._choice_col}'")
        parts.append(")")
        return "".join(parts)

    def __len__(self) -> int:
        return self.n_observations * self.n_alternatives
