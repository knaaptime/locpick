"""Internal xarray Dataset helpers for choice data.

This module defines the canonical internal representation used by
``ChoiceTable``: an xarray Dataset with dimensions ``obs_id`` and
``alt_pos`` plus a 2D ``alt_id_values`` variable storing the actual
alternative id in each position.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


def _resolve_pairwise(
    series: pd.Series,
    obs_ids: np.ndarray,
    alt_ids_matrix: np.ndarray,
) -> xr.DataArray:
    """Map an (obs_id, alt_id)-indexed Series to (obs_id, alt_pos).

    Parameters
    ----------
    series : pd.Series
        Pairwise variable values with a two-level MultiIndex of (obs_id, alt_id).
    obs_ids : ndarray
        Observation ids in canonical order.
    alt_ids_matrix : ndarray
        Array of shape (n_obs, n_alts) with alternative ids by position.

    Returns
    -------
    xr.DataArray
        Pairwise variable values aligned to ``(obs_id, alt_pos)``.

    Raises
    ------
    ValueError
        If ``series`` is not indexed by two levels.
    KeyError
        If ``series`` contains an (obs_id, alt_id) pair whose alt_id is not
        present in that observation's alternative set.
    """
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ValueError(
            "Pairwise variable series must have a MultiIndex with levels (obs_id, alt_id)."
        )

    n_obs, n_alts = alt_ids_matrix.shape
    flat_alt = np.asarray(alt_ids_matrix).ravel()

    # Fast path: the series already covers the full (obs, alt) grid in the
    # canonical row-major order produced by the standard assembly path.  Then
    # the values map directly onto ``(n_obs, n_alts)`` with a single reshape,
    # avoiding the large intermediate lookup table and reindex below (which
    # allocate several ``n_obs * n_alts`` arrays — costly at scale).
    if len(series) == n_obs * n_alts:
        alt_level_full = series.index.get_level_values(1).to_numpy()
        if np.array_equal(alt_level_full, flat_alt) and np.array_equal(
            series.index.get_level_values(0).to_numpy(),
            np.repeat(np.asarray(obs_ids), n_alts),
        ):
            return xr.DataArray(
                series.to_numpy(dtype=np.float64).reshape(n_obs, n_alts),
                dims=("obs_id", "alt_pos"),
                coords={"obs_id": obs_ids, "alt_pos": np.arange(n_alts)},
            )

    values = np.full((n_obs, n_alts), np.nan, dtype=np.float64)

    # Vectorized (obs_id, alt_id) -> (row, alt_pos) resolution.
    #
    # Build a hash lookup from (row, alt_id) to column position once, then
    # resolve every series entry in a single vectorized reindex.  This
    # replaces a per-entry ``np.where`` scan (O(nnz * n_alts)).
    flat_rows = np.repeat(np.arange(n_obs), n_alts)
    flat_col = np.tile(np.arange(n_alts), n_obs)
    lookup = pd.Series(flat_col, index=pd.MultiIndex.from_arrays([flat_rows, flat_alt]))
    # Keep the first position for any repeated alt_id in a row (matches the
    # previous ``matches[0]`` behaviour and keeps the index unique).
    lookup = lookup[~lookup.index.duplicated(keep="first")]

    obs_level = series.index.get_level_values(0)
    alt_level = np.asarray(series.index.get_level_values(1))
    entry_rows = pd.Index(obs_ids).get_indexer(obs_level)

    # Entries whose obs_id is not in the canonical set are silently skipped.
    valid = entry_rows >= 0
    rows_v = entry_rows[valid]
    alt_v = alt_level[valid]
    vals_v = series.to_numpy()[valid]

    if rows_v.size:
        target = lookup.reindex(pd.MultiIndex.from_arrays([rows_v, alt_v])).to_numpy()
        found = ~np.isnan(target)

        # An obs present in the canonical set but with no matching alt_ids is
        # an error (all provided alts disjoint from that obs's alternatives).
        rows_with_entry = np.unique(rows_v)
        rows_with_found = np.unique(rows_v[found])
        missing = np.setdiff1d(rows_with_entry, rows_with_found)
        if missing.size:
            bad_obs = obs_ids[missing[0]]
            raise KeyError(
                f"Pairwise variable contains no alt_ids present for obs_id {bad_obs!r}."
            )

        values[rows_v[found], target[found].astype(np.int64)] = vals_v[found]

    return xr.DataArray(
        values,
        dims=("obs_id", "alt_pos"),
        coords={"obs_id": obs_ids, "alt_pos": np.arange(n_alts)},
    )


def build_choice_dataset_from_long(
    df: pd.DataFrame,
    obs_id_col: str,
    alt_id_col: str,
    choice_col: Optional[str] = None,
    available_col: Optional[str] = None,
    sample_size: Optional[int] = None,
    allow_duplicate_pairs: bool = False,
) -> xr.Dataset:
    """Build a canonical choice Dataset from long-format data."""
    if obs_id_col not in df.columns or alt_id_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{obs_id_col}' and '{alt_id_col}' columns.")

    if (not allow_duplicate_pairs) and df.duplicated([obs_id_col, alt_id_col]).any():
        raise ValueError("Duplicate (obs_id, alt_id) pairs are not allowed.")

    grouped = list(df.groupby(obs_id_col, sort=False))
    if not grouped:
        ds = xr.Dataset(coords={"obs_id": np.array([]), "alt_pos": np.array([])})
        ds["alt_id_values"] = (("obs_id", "alt_pos"), np.empty((0, 0), dtype=np.float64))
        ds.attrs.update(
            {
                "obs_id_col": obs_id_col,
                "alt_id_col": alt_id_col,
                "choice_col": choice_col,
                "available_col": available_col,
                "sample_size": sample_size,
            }
        )
        return ds

    obs_ids = np.array([obs for obs, _ in grouped])
    first_len = len(grouped[0][1])
    for obs_id, group in grouped:
        if len(group) != first_len:
            raise ValueError(
                "Non-rectangular long data: each observation must have the same "
                "number of alternatives."
            )

    len(grouped)
    n_alts = first_len

    alt_ids_matrix = np.stack([g[alt_id_col].to_numpy() for _, g in grouped], axis=0)

    ds = xr.Dataset(coords={"obs_id": obs_ids, "alt_pos": np.arange(n_alts)})
    ds["alt_id_values"] = (("obs_id", "alt_pos"), alt_ids_matrix)

    value_cols = [c for c in df.columns if c not in {obs_id_col, alt_id_col}]
    for col in value_cols:
        matrix = np.stack([g[col].to_numpy() for _, g in grouped], axis=0)
        ds[col] = (("obs_id", "alt_pos"), matrix)

    ds.attrs.update(
        {
            "obs_id_col": obs_id_col,
            "alt_id_col": alt_id_col,
            "choice_col": choice_col,
            "available_col": available_col,
            "sample_size": sample_size,
        }
    )

    validate_choice_dataset(ds)
    return ds


def build_choice_dataset(
    obs_ids: np.ndarray,
    alt_ids_matrix: np.ndarray,
    chooser_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    chosen_arr: Optional[np.ndarray] = None,
    pairwise_data: Optional[dict[str, pd.Series]] = None,
    available_arr: Optional[np.ndarray] = None,
    sample_size: Optional[int] = None,
    obs_id_name: str = "obs_id",
    alt_id_name: str = "alt_id",
) -> xr.Dataset:
    """Build a canonical choice Dataset directly from matrix inputs.

    Avoids the long-format round trip: chooser features are stored as 1-D
    arrays over ``obs_id``, and alternative features are stored as 1-D over
    ``alt_pos`` when the alt_ids matrix is constant per row (census case)
    or as 2-D ``(obs_id, alt_pos)`` matrices when alternatives were sampled.
    """
    obs_ids = np.asarray(obs_ids)
    alt_ids_matrix = np.asarray(alt_ids_matrix)
    n_obs, n_alts = alt_ids_matrix.shape

    data_vars: dict[str, tuple] = {
        "alt_id_values": (("obs_id", "alt_pos"), alt_ids_matrix),
    }

    # Chooser features → 1-D over obs_id (broadcast at materialization).
    if len(chooser_df.columns):
        chooser_aligned = (
            chooser_df
            if chooser_df.index.equals(pd.Index(obs_ids))
            else chooser_df.reindex(obs_ids)
        )
        for col in chooser_aligned.columns:
            data_vars[col] = (("obs_id",), chooser_aligned[col].to_numpy())

    # Alternative features.
    if len(alt_df.columns):
        alt_index = alt_df.index
        # Census fast path: every row of alt_ids_matrix is the canonical
        # ordering. Detect cheaply via shape + first-row equality.
        is_census = (
            sample_size is None
            and n_alts == len(alt_index)
            and np.array_equal(alt_ids_matrix[0], alt_index.to_numpy())
            and (n_obs <= 1 or np.array_equal(alt_ids_matrix[-1], alt_index.to_numpy()))
        )
        if is_census:
            for col in alt_df.columns:
                data_vars[col] = (("alt_pos",), alt_df[col].to_numpy())
        else:
            flat_ids = alt_ids_matrix.reshape(-1)
            alt_aligned = alt_df.reindex(flat_ids)
            for col in alt_df.columns:
                mat = alt_aligned[col].to_numpy().reshape(n_obs, n_alts)
                data_vars[col] = (("obs_id", "alt_pos"), mat)

    if chosen_arr is not None:
        data_vars["chosen"] = (
            ("obs_id", "alt_pos"),
            np.asarray(chosen_arr).reshape(n_obs, n_alts),
        )
    if available_arr is not None:
        data_vars["available"] = (
            ("obs_id", "alt_pos"),
            np.asarray(available_arr).reshape(n_obs, n_alts),
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"obs_id": obs_ids, "alt_pos": np.arange(n_alts)},
    )
    ds.attrs.update(
        {
            "obs_id_col": obs_id_name,
            "alt_id_col": alt_id_name,
            "choice_col": "chosen" if chosen_arr is not None else None,
            "available_col": "available" if available_arr is not None else None,
            "sample_size": sample_size,
        }
    )

    if pairwise_data:
        for name, series in pairwise_data.items():
            ds[name] = _resolve_pairwise(series, obs_ids, alt_ids_matrix)

    validate_choice_dataset(ds)
    return ds


def dataset_to_long_frame(ds: xr.Dataset) -> pd.DataFrame:
    """Convert the canonical choice Dataset to long-format DataFrame."""
    obs_id_col = ds.attrs.get("obs_id_col", "obs_id")
    alt_id_col = ds.attrs.get("alt_id_col", "alt_id")

    obs_ids = np.asarray(ds.coords["obs_id"].values)
    n_obs = int(ds.sizes.get("obs_id", 0))
    n_alts = int(ds.sizes.get("alt_pos", 0))

    alt_ids_matrix = np.asarray(ds["alt_id_values"].values)

    data: dict[str, np.ndarray] = {
        obs_id_col: np.repeat(obs_ids, n_alts),
        alt_id_col: alt_ids_matrix.reshape(-1),
    }

    for name, var in ds.data_vars.items():
        if name == "alt_id_values":
            continue
        dims = tuple(var.dims)
        values = np.asarray(var.values)

        if dims == ("obs_id", "alt_pos"):
            data[name] = values.reshape(-1)
        elif dims == ("obs_id",):
            data[name] = np.repeat(values, n_alts)
        elif dims == ("alt_pos",):
            data[name] = np.tile(values, n_obs)

    return pd.DataFrame(data)


def validate_choice_dataset(ds: xr.Dataset) -> None:
    """Validate required schema properties for the choice Dataset."""
    if "obs_id" not in ds.dims or "alt_pos" not in ds.dims:
        raise ValueError("Choice dataset must define 'obs_id' and 'alt_pos' dimensions.")

    if "alt_id_values" not in ds.data_vars:
        raise ValueError("Choice dataset must include 'alt_id_values'.")

    alt_ids = ds["alt_id_values"]
    if tuple(alt_ids.dims) != ("obs_id", "alt_pos"):
        raise ValueError("'alt_id_values' must have dims ('obs_id', 'alt_pos').")

    shape = (int(ds.sizes["obs_id"]), int(ds.sizes["alt_pos"]))
    if alt_ids.shape != shape:
        raise ValueError("'alt_id_values' shape must match (obs_id, alt_pos).")

    if "chosen" in ds.data_vars and ds["chosen"].shape != shape:
        raise ValueError("'chosen' must match shape (obs_id, alt_pos).")

    available_col = ds.attrs.get("available_col")
    if available_col and available_col in ds.data_vars and ds[available_col].shape != shape:
        raise ValueError(f"'{available_col}' must match shape (obs_id, alt_pos).")
