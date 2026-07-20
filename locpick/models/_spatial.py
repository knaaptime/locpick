"""Shared spatial primitives used by spatial choice-model variants.

This module exposes the graph-resolution helper, the ``rho`` link
function, and the precomputed ``EdgeStructure`` consumed by the JAX
spatial kernels.  Keeping them here lets
:class:`~locpick.models.choice_model.ChoiceModel` opt into spatial
estimation via a ``graph=`` argument in any of its configurations
(plain, nested, mixed, or mixed-nested).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Spatial graph resolution
# ---------------------------------------------------------------------------


def _resolve_spatial_graph(
    graph: Any,
    alt_ids: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], int]:
    """Resolve a spatial connectivity object into SCL-ready arrays.

    Accepts a ``libpysal.graph.Graph``, a ``scipy.sparse`` array, or a
    dense NumPy array and returns the adjacency matrix, the allocation
    matrix, and the list of paired-nest edges needed by the SCL
    probability kernel.

    Parameters
    ----------
    graph : libpysal.graph.Graph, scipy.sparse array, or np.ndarray
        Spatial connectivity structure.  ``libpysal.graph.Graph`` objects
        are converted via their ``.sparse`` property.  ``scipy.sparse``
        arrays are used directly.  Dense NumPy arrays are converted to
        CSR format.  Weights are preserved (not binarised) so that
        distance- or kernel-weighted graphs can be used directly.
    alt_ids : list, optional
        Alternative IDs that index into the graph.  If *None*, the graph
        is assumed to be ordered consistently with the design matrix
        (i.e. row/column *i* of the graph corresponds to alternative *i*).

    Returns
    -------
    omega : np.ndarray, shape (n_alts, n_alts)
        Adjacency matrix with preserved weights.  ``omega[i, j] > 0`` if
        alternatives *i* and *j* are spatially connected, 0 otherwise.
        Diagonal is zero.
    allocation : np.ndarray, shape (n_alts, n_alts)
        Row-standardised allocation parameters.  ``allocation[i, j] =
        omega[i, j] / sum_k omega[i, k]``.
    edge_list : list of (int, int)
        List of paired-nest edges ``(i, j)`` with ``i < j`` and
        ``omega[i, j] > 0``.  Each edge defines one paired nest in the
        PGNL structure.
    n_alts : int
        Number of alternatives (dimension of the graph).
    """
    import scipy.sparse as sp

    if hasattr(graph, "sparse"):
        sp_mat = sp.csr_array(graph.sparse, dtype=np.float64)
    elif sp.issparse(graph):
        sp_mat = sp.csr_array(graph, dtype=np.float64)
    else:
        sp_mat = sp.csr_array(np.asarray(graph, dtype=np.float64))

    n = sp_mat.shape[0]

    omega_sp = sp.csr_array(sp_mat, dtype=np.float64)
    omega_sp.setdiag(0.0)
    omega_sp.eliminate_zeros()

    omega = omega_sp.toarray()

    row_sums = omega.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    allocation = omega / row_sums

    coo = sp.coo_array(omega_sp)
    edge_list = sorted((int(r), int(c)) for r, c in zip(coo.row, coo.col) if r < c)

    return omega, allocation, edge_list, n


# ---------------------------------------------------------------------------
# Parameter transformation
# ---------------------------------------------------------------------------


def naturalize_rho(alpha_rho):
    r"""Transform unconstrained parameter to natural scale :math:`\rho \in (0, 1]`.

    .. math::

        \rho = \frac{1}{1 + \exp(-\alpha_\rho)}
    """
    return 1.0 / (1.0 + np.exp(-alpha_rho))


# ---------------------------------------------------------------------------
# Precomputed edge structure
# ---------------------------------------------------------------------------


class EdgeStructure:
    """Precomputed edge structure for fast SCL probability computation.

    Converts the Python-level ``edge_list`` and ``allocation`` matrix
    into flat NumPy arrays that JIT'd kernels can consume without
    Python object overhead.

    Parameters
    ----------
    edge_list : list of (int, int)
        Paired-nest edges ``(i, j)`` with ``i < j``.
    n_alts : int
        Number of alternatives.
    allocation : np.ndarray, shape (n_alts, n_alts)
        Allocation matrix.
    """

    __slots__ = (
        "n_edges",
        "n_alts",
        "edge_i",
        "edge_j",
        "alt_edge_starts",
        "alt_edge_counts",
        "alt_edge_indices",
        "alt_edge_is_first",
        "connected",
        "isolated",
        "allocation",
    )

    def __init__(
        self,
        edge_list: list[tuple[int, int]],
        n_alts: int,
        allocation: np.ndarray,
    ):
        self.n_edges = len(edge_list)
        self.n_alts = n_alts
        self.allocation = np.ascontiguousarray(allocation, dtype=np.float64)

        self.edge_i = np.zeros(self.n_edges, dtype=np.int64)
        self.edge_j = np.zeros(self.n_edges, dtype=np.int64)

        connected_set: set[int] = set()
        for idx, (i, j) in enumerate(edge_list):
            self.edge_i[idx] = i
            self.edge_j[idx] = j
            connected_set.add(i)
            connected_set.add(j)

        self.connected = np.array(sorted(connected_set), dtype=np.int64)
        self.isolated = np.array(sorted(set(range(n_alts)) - connected_set), dtype=np.int64)

        alt_edges: dict[int, list[tuple[int, bool]]] = {}
        for idx in range(self.n_edges):
            i = int(self.edge_i[idx])
            j = int(self.edge_j[idx])
            alt_edges.setdefault(i, []).append((idx, True))
            alt_edges.setdefault(j, []).append((idx, False))

        self.alt_edge_starts = np.zeros(n_alts, dtype=np.int64)
        self.alt_edge_counts = np.zeros(n_alts, dtype=np.int64)

        total_entries = sum(len(v) for v in alt_edges.values())
        self.alt_edge_indices = np.zeros(total_entries, dtype=np.int64)
        self.alt_edge_is_first = np.zeros(total_entries, dtype=np.int64)

        pos = 0
        for alt in range(n_alts):
            edges = alt_edges.get(alt, [])
            self.alt_edge_starts[alt] = pos
            self.alt_edge_counts[alt] = len(edges)
            for edge_idx, is_first in edges:
                self.alt_edge_indices[pos] = edge_idx
                self.alt_edge_is_first[pos] = 1 if is_first else 0
                pos += 1
