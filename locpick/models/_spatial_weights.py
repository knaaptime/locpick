"""Spatial weights matrix utilities for SAR-MNL models.

This module provides helpers to resolve spatial weights matrices
(``W``) connecting alternatives (spatial locations).  It follows the
convention from the sister package *bayespecon*: ``libpysal.graph.Graph``
is the canonical type, but ``scipy.sparse`` matrices and dense NumPy
arrays are also accepted for convenience.

The resolver returns both a row-standardised ``Graph`` (for storage
and return in DGP datasets) and a CSR sparse matrix (for efficient
computation inside JAX kernels).
"""

from __future__ import annotations

from typing import Union

import numpy as np
import scipy.sparse as sp


def resolve_spatial_weights(
    W,
    n_alts: int,
    row_standardize: bool = True,
):
    """Resolve alt×alt spatial weights to a libpysal Graph + CSR sparse.

    Accepts a ``libpysal.graph.Graph`` (preferred), ``scipy.sparse``
    matrix, or dense ``np.ndarray``.  Returns both a row-standardised
    ``Graph`` (for storage/return in DGP datasets) and a CSR sparse
    matrix (for efficient computation).

    Parameters
    ----------
    W : libpysal.graph.Graph, scipy.sparse, or np.ndarray
        J×J spatial weights matrix connecting alternatives (locations).
    n_alts : int
        Expected number of alternatives (for validation).
    row_standardize : bool, default True
        If True, row-standardize the weights (rows sum to 1).

    Returns
    -------
    W_graph : libpysal.graph.Graph
        Row-standardised spatial weights as a libpysal Graph.
    W_sparse : scipy.sparse.csr_array
        Row-standardised CSR sparse matrix (float64), zero diagonal.
    """
    # --- Reject legacy libpysal.weights.W -------------------------------
    if W.__class__.__module__.startswith("libpysal.weights") and not hasattr(
        W, "sparse"
    ):
        raise TypeError(
            "Legacy libpysal.weights.W is not supported. "
            "Convert via libpysal.graph.Graph.from_W(w) or pass w.sparse."
        )

    # --- Convert to CSR sparse -----------------------------------------
    if hasattr(W, "sparse"):
        # libpysal.graph.Graph
        W_sparse = sp.csr_array(W.sparse, dtype=np.float64)
    elif sp.issparse(W):
        W_sparse = sp.csr_array(W, dtype=np.float64)
    else:
        W_sparse = sp.csr_array(np.asarray(W, dtype=np.float64))

    # --- Validate shape -------------------------------------------------
    if W_sparse.shape != (n_alts, n_alts):
        raise ValueError(
            f"W shape {W_sparse.shape} does not match n_alts ({n_alts}, {n_alts})."
        )

    # --- Zero diagonal --------------------------------------------------
    W_sparse.setdiag(0.0)
    W_sparse.eliminate_zeros()

    # --- Row-standardize ------------------------------------------------
    if row_standardize:
        row_sums = np.asarray(W_sparse.sum(axis=1)).ravel()
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        W_sparse = sp.diags(1.0 / row_sums) @ W_sparse
        W_sparse = sp.csr_array(W_sparse, dtype=np.float64)

    # --- Convert back to Graph for storage/return ----------------------
    W_graph = _csr_to_graph(W_sparse)

    return W_graph, W_sparse


def _csr_to_graph(W_sparse: sp.csr_array):
    """Convert a CSR sparse matrix to a libpysal Graph."""
    from libpysal.graph import Graph

    W_coo = W_sparse.tocoo()
    return Graph.from_arrays(
        focal_ids=W_coo.row.astype(np.int32),
        neighbor_ids=W_coo.col.astype(np.int32),
        weight=W_coo.data.astype(np.float64),
    )


def build_knn_graph(
    coords: np.ndarray,
    k: int,
):
    """Build a k-nearest-neighbor libpysal Graph from coordinates.

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
        Spatial coordinates of alternatives.
    k : int
        Number of nearest neighbors.

    Returns
    -------
    libpysal.graph.Graph
        Row-standardised k-NN graph.
    """
    import geopandas as gpd
    from libpysal.graph import Graph
    from shapely.geometry import Point

    n = coords.shape[0]
    gdf = gpd.GeoDataFrame(
        {"aid": np.arange(n)},
        geometry=[Point(c) for c in coords],
        crs="EPSG:4326",
    )
    W_graph = Graph.build_knn(gdf, k=k)
    return W_graph.transform("r")