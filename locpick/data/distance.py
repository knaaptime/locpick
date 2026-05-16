"""Distance utilities for geo-centric location choice workflows.

This module provides distance matrix and nearest-neighbor helpers with two
modern backends:
- BallTree (haversine) for geographic coordinates (lat/lon)
- GeoPandas/Shapely distance operations for geometry-native workflows
"""

from __future__ import annotations

from itertools import tee

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

EARTH_RADIUS_METERS = 6_371_009


def great_circle_vec(lat1, lng1, lat2, lng2, earth_radius=EARTH_RADIUS_METERS):
    """Vectorized great-circle distance between points in decimal degrees."""
    phi1 = np.deg2rad(90 - lat1)
    phi2 = np.deg2rad(90 - lat2)

    theta1 = np.deg2rad(lng1)
    theta2 = np.deg2rad(lng2)

    cos = np.sin(phi1) * np.sin(phi2) * np.cos(theta1 - theta2) + np.cos(phi1) * np.cos(phi2)
    arc = np.arccos(np.clip(cos, -1.0, 1.0))
    return arc * earth_radius


def great_circle_distance_matrix(df, x, y, earth_radius=EARTH_RADIUS_METERS, return_int=True):
    """Pairwise great-circle distance matrix as a stacked Series."""
    df_dist_matrix = df.apply(
        lambda row: great_circle_vec(row[y], row[x], df[y], df[x], earth_radius=earth_radius),
        axis="columns",
    )

    if return_int:
        df_dist_matrix = df_dist_matrix.fillna(0).astype(int)

    labels = df.index.values
    df_dist_matrix.columns = labels
    df_dist_matrix.index = labels
    return df_dist_matrix.stack()


def euclidean_distance_matrix(df):
    """Pairwise Euclidean distance matrix as a stacked Series."""
    distance_vector = pdist(X=df, metric="euclidean")
    dist_matrix = squareform(distance_vector)

    labels = df.index.values
    df_dist_matrix = pd.DataFrame(data=dist_matrix, columns=labels, index=labels)
    return df_dist_matrix.stack()


def network_distance_matrix(df, x, y):
    """Placeholder for future network distance support."""
    raise NotImplementedError("network_distance_matrix is not currently implemented")


def _resolve_units_multiplier(units: str) -> float:
    units = units.lower()
    if units in {"meters", "meter", "m"}:
        return 1.0
    if units in {"kilometers", "kilometer", "km"}:
        return 1.0 / 1000.0
    raise ValueError("units must be 'meters' or 'kilometers'")


def _balltree_knn(origins: pd.DataFrame, destinations: pd.DataFrame, x: str, y: str, k: int):
    """kNN using BallTree haversine metric (expects decimal degrees input)."""
    try:
        from sklearn.neighbors import BallTree
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scikit-learn is required for BallTree haversine distance") from exc

    o_rad = np.deg2rad(origins[[y, x]].to_numpy(dtype=float))
    d_rad = np.deg2rad(destinations[[y, x]].to_numpy(dtype=float))

    tree = BallTree(d_rad, metric="haversine")
    k_eff = min(k, len(destinations))
    d_rad_out, idx = tree.query(o_rad, k=k_eff)

    return d_rad_out, idx


def _geopandas_pairwise(origins, destinations):
    """Full pairwise geometry distance using GeoPandas distance ops."""
    out = []
    for oid, geom in origins.geometry.items():
        d = destinations.geometry.distance(geom)
        s = pd.Series(d.to_numpy(), index=destinations.index)
        s.index = pd.MultiIndex.from_product([[oid], s.index])
        out.append(s)
    return pd.concat(out)


def pairwise_distance(
    origins,
    destinations,
    method: str = "auto",
    units: str = "meters",
    x: str = "lng",
    y: str = "lat",
):
    """Compute full pairwise distances from origins to destinations.

    Returns
    -------
    pandas.Series
        MultiIndex (origin_id, destination_id) with distance values.
    """
    units_mult = _resolve_units_multiplier(units)

    if method == "auto":
        method = "geopandas" if hasattr(origins, "geometry") and hasattr(destinations, "geometry") else "haversine"

    if method in {"haversine", "greatcircle"}:
        o_lat = origins[y].to_numpy(dtype=float)[:, None]
        o_lng = origins[x].to_numpy(dtype=float)[:, None]
        d_lat = destinations[y].to_numpy(dtype=float)[None, :]
        d_lng = destinations[x].to_numpy(dtype=float)[None, :]
        dist = great_circle_vec(o_lat, o_lng, d_lat, d_lng) * units_mult
        idx = pd.MultiIndex.from_product([origins.index, destinations.index], names=[origins.index.name, destinations.index.name])
        return pd.Series(dist.reshape(-1), index=idx)

    if method == "geopandas":
        try:
            import geopandas as gpd  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("geopandas is required for method='geopandas'") from exc

        if not hasattr(origins, "geometry") or not hasattr(destinations, "geometry"):
            raise TypeError("geopandas method requires GeoDataFrame inputs with a geometry column")

        if origins.crs is None or destinations.crs is None:
            raise ValueError("GeoDataFrames must have CRS set for geopandas distance")

        out = _geopandas_pairwise(origins, destinations) * units_mult
        out.index.names = [origins.index.name, destinations.index.name]
        return out

    raise ValueError("method must be one of 'auto', 'haversine', 'greatcircle', or 'geopandas'")


def nearest_neighbors(
    origins,
    destinations,
    k: int = 1,
    method: str = "auto",
    units: str = "meters",
    x: str = "lng",
    y: str = "lat",
):
    """Return k nearest neighbors per origin.

    Returns
    -------
    pandas.DataFrame
        Columns: origin_id, destination_id, rank, distance
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    units_mult = _resolve_units_multiplier(units)

    if method == "auto":
        method = "geopandas" if hasattr(origins, "geometry") and hasattr(destinations, "geometry") else "haversine"

    rows = []

    if method in {"haversine", "greatcircle"}:
        d_rad_out, idx = _balltree_knn(origins, destinations, x=x, y=y, k=k)
        d_out = d_rad_out * EARTH_RADIUS_METERS * units_mult
        for i, oid in enumerate(origins.index):
            for rank, (j, dist) in enumerate(zip(idx[i], d_out[i]), start=1):
                rows.append(
                    {
                        "origin_id": oid,
                        "destination_id": destinations.index[j],
                        "rank": rank,
                        "distance": dist,
                    }
                )
        return pd.DataFrame(rows)

    if method == "geopandas":
        if not hasattr(origins, "geometry") or not hasattr(destinations, "geometry"):
            raise TypeError("geopandas method requires GeoDataFrame inputs with a geometry column")
        for oid, geom in origins.geometry.items():
            d = destinations.geometry.distance(geom)
            topk = d.nsmallest(min(k, len(d)))
            for rank, (did, dist) in enumerate(topk.items(), start=1):
                rows.append(
                    {
                        "origin_id": oid,
                        "destination_id": did,
                        "rank": rank,
                        "distance": float(dist) * units_mult,
                    }
                )
        return pd.DataFrame(rows)

    raise ValueError("method must be one of 'auto', 'haversine', 'greatcircle', or 'geopandas'")


def distance_matrix(df, method="euclidean", x="lng", y="lat", earth_radius=EARTH_RADIUS_METERS, return_int=True):
    """Backwards-compatible distance matrix helper (v2 location: locpick.distance)."""
    if not df.index.is_unique:
        raise ValueError("The passed-in DataFrame must have a unique index")

    if method == "euclidean":
        return euclidean_distance_matrix(df=df)
    if method in {"greatcircle", "haversine"}:
        return great_circle_distance_matrix(df=df, x=x, y=y, earth_radius=earth_radius, return_int=return_int)
    if method == "network":
        return network_distance_matrix(df=df, x=x, y=y)

    raise ValueError('argument `method` must be one of "euclidean", "greatcircle", "haversine", or "network"')


def pairwise(iterable):
    """Iterate through a list pairwise."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def distance_bands(dist_vector, distances):
    """Identify all geographies located within each distance band."""
    bands = {}
    dist_matrix = dist_vector.unstack()
    for _, row in dist_matrix.iterrows():
        bands[row.name] = {}
        for band_number, (dist1, dist2) in enumerate(pairwise(distances)):
            mask = (row >= dist1) & (row < dist2)
            place_ids = row[mask].index.values
            bands[row.name][band_number] = place_ids

    return pd.DataFrame(bands).T.stack()