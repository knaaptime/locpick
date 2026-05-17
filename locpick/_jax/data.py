"""JAX-ready data containers for choice model estimation.

These containers hold pre-converted JAX arrays, built once and shared
across all solvers.  They replace the ad-hoc numpy→JAX conversion that
was duplicated inside each model's ``_build_*_jax`` closure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from locpick._compat import _JAX_AVAILABLE
from locpick._sampling.correction import get_sampling_correction

if _JAX_AVAILABLE:
    import jax.numpy as jnp


@dataclass
class EdgeDataJAX:
    """JAX-ready spatial edge structure for SCL/MSCL models.

    Converts the Python-level :class:`~locpick.models.scl.EdgeStructure`
    into flat JAX arrays suitable for vectorised computation inside
    ``jax.jit``-compiled functions.

    Parameters
    ----------
    edge_i : jnp.ndarray, shape (n_edges,)
        Source node index for each edge.
    edge_j : jnp.ndarray, shape (n_edges,)
        Target node index for each edge.
    allocation : jnp.ndarray, shape (n_alts, n_alts)
        Row-standardised allocation matrix.
    n_edges : int
        Number of edges.
    n_alts : int
        Number of alternatives.
    isolated : jnp.ndarray or None
        Indices of isolated (unconnected) alternatives.
    connected : jnp.ndarray or None
        Indices of connected alternatives.
    flat_alt_idx : jnp.ndarray or None
        Flat alt-edge table: alternative index for each (alt, edge) pair.
    flat_edge_idx : jnp.ndarray or None
        Flat alt-edge table: edge index for each (alt, edge) pair.
    flat_is_first : jnp.ndarray or None
        Flat alt-edge table: whether the alt is the "first" node in the edge.
    alloc_ij : jnp.ndarray, shape (n_edges,)
        Precomputed ``allocation[edge_i, edge_j]`` gather, cached so the
        spatial kernel does not redo the 2-D fancy index every call.
    alloc_ji : jnp.ndarray, shape (n_edges,)
        Precomputed ``allocation[edge_j, edge_i]`` gather.
    isolated_mask : jnp.ndarray, shape (n_alts,)
        Precomputed 0/1 mask marking isolated alternatives.
    """

    edge_i: "jnp.ndarray"
    edge_j: "jnp.ndarray"
    allocation: "jnp.ndarray"
    n_edges: int
    n_alts: int
    isolated: Optional["jnp.ndarray"] = None
    connected: Optional["jnp.ndarray"] = None
    flat_alt_idx: Optional["jnp.ndarray"] = None
    flat_edge_idx: Optional["jnp.ndarray"] = None
    flat_is_first: Optional["jnp.ndarray"] = None
    alloc_ij: Optional["jnp.ndarray"] = None
    alloc_ji: Optional["jnp.ndarray"] = None
    isolated_mask: Optional["jnp.ndarray"] = None

    @classmethod
    def from_edge_structure(cls, edge_struct) -> "EdgeDataJAX":
        """Build from a :class:`~locpick.models.scl.EdgeStructure`.

        Parameters
        ----------
        edge_struct : EdgeStructure
            Numba-compatible edge structure to convert.

        Returns
        -------
        EdgeDataJAX
        """
        if not _JAX_AVAILABLE:
            raise ImportError("JAX is required for EdgeDataJAX")

        n_edges = edge_struct.n_edges
        n_alts = edge_struct.n_alts

        # Build flat alt-edge table
        n_connected = edge_struct.connected.shape[0]
        n_isolated = edge_struct.isolated.shape[0]
        total_alt_edges = int(edge_struct.alt_edge_counts[edge_struct.connected].sum())

        _alt_idx_np = np.empty(total_alt_edges, dtype=np.int32)
        _edge_idx_np = np.empty(total_alt_edges, dtype=np.int32)
        _is_first_np = np.empty(total_alt_edges, dtype=np.int32)
        pos = 0
        for c_idx in range(n_connected):
            alt_i = int(edge_struct.connected[c_idx])
            start = int(edge_struct.alt_edge_starts[alt_i])
            count = int(edge_struct.alt_edge_counts[alt_i])
            for k in range(count):
                _alt_idx_np[pos] = alt_i
                _edge_idx_np[pos] = int(edge_struct.alt_edge_indices[start + k])
                _is_first_np[pos] = int(edge_struct.alt_edge_is_first[start + k])
                pos += 1

        # Cache parameter-independent spatial primitives so the SCL kernel
        # does not recompute them inside every gradient call. These are pure
        # functions of the topology (edges + allocation matrix + isolated
        # set) and stay fixed for the lifetime of an EdgeDataJAX.
        ei_np = np.asarray(edge_struct.edge_i, dtype=np.int64)
        ej_np = np.asarray(edge_struct.edge_j, dtype=np.int64)
        alloc_np = np.asarray(edge_struct.allocation, dtype=np.float64)
        alloc_ij_np = alloc_np[ei_np, ej_np] if n_edges > 0 else np.empty(0, dtype=np.float64)
        alloc_ji_np = alloc_np[ej_np, ei_np] if n_edges > 0 else np.empty(0, dtype=np.float64)

        isolated_mask_np = np.zeros(n_alts, dtype=np.float64)
        if n_isolated > 0:
            isolated_mask_np[np.asarray(edge_struct.isolated, dtype=np.int64)] = 1.0

        return cls(
            edge_i=jnp.array(edge_struct.edge_i, dtype=jnp.int32),
            edge_j=jnp.array(edge_struct.edge_j, dtype=jnp.int32),
            allocation=jnp.array(edge_struct.allocation, dtype=jnp.float64),
            n_edges=n_edges,
            n_alts=n_alts,
            isolated=jnp.array(edge_struct.isolated, dtype=jnp.int32) if n_isolated > 0 else None,
            connected=jnp.array(edge_struct.connected, dtype=jnp.int32)
            if n_connected > 0
            else None,
            flat_alt_idx=jnp.array(_alt_idx_np, dtype=jnp.int32),
            flat_edge_idx=jnp.array(_edge_idx_np, dtype=jnp.int32),
            flat_is_first=jnp.array(_is_first_np, dtype=jnp.int32),
            alloc_ij=jnp.array(alloc_ij_np, dtype=jnp.float64),
            alloc_ji=jnp.array(alloc_ji_np, dtype=jnp.float64),
            isolated_mask=jnp.array(isolated_mask_np, dtype=jnp.float64),
        )


@dataclass
class ChoiceDataJAX:
    """JAX-ready estimation data — built once, shared across solvers.

    This container holds all data needed for log-likelihood computation,
    pre-converted to JAX arrays.  It is produced by model classes and
    consumed by the :class:`Objective` builder.

    Parameters
    ----------
    design_matrix : jnp.ndarray, shape (n_obs * n_alts, k) or (n_obs, n_alts, k)
        Design matrix of explanatory variables.
    chosen : jnp.ndarray, shape (n_obs, n_alts)
        Binary indicator matrix for chosen alternatives.
    n_obs : int
        Number of observations.
    n_alts : int
        Number of alternatives per observation.
    available : jnp.ndarray, shape (n_obs, n_alts)
        Binary availability matrix.
    weights : jnp.ndarray, shape (n_obs,)
        Observation-level weights.
    inclusion_probs : jnp.ndarray or None, shape (n_obs, n_alts)
        Sampling correction rates.
    edge_data : EdgeDataJAX or None
        Spatial edge structure (for SCL/MSCL models).
    draws : jnp.ndarray or None, shape (n_obs, n_draws, k_random)
        Halton or random draws for mixed logit simulation.
    random_col_indices : list[int] or None
        Column indices of random parameters in the design matrix.
    fixed_col_indices : list[int] or None
        Column indices of fixed parameters in the design matrix.
    dist_codes : jnp.ndarray or None, shape (k_random,)
        Integer codes for random parameter distributions.
    dm_fixed : jnp.ndarray or None
        Design matrix columns for fixed parameters.
    dm_random : jnp.ndarray or None
        Design matrix columns for random parameters.
    """

    design_matrix: "jnp.ndarray"
    chosen: "jnp.ndarray"
    n_obs: int
    n_alts: int
    available: "jnp.ndarray"
    weights: "jnp.ndarray"
    inclusion_probs: Optional["jnp.ndarray"] = None
    edge_data: Optional[EdgeDataJAX] = None
    draws: Optional["jnp.ndarray"] = None
    random_col_indices: Optional[list[int]] = None
    fixed_col_indices: Optional[list[int]] = None
    dist_codes: Optional["jnp.ndarray"] = None
    dm_fixed: Optional["jnp.ndarray"] = None
    dm_random: Optional["jnp.ndarray"] = None

    @classmethod
    def from_arrays(
        cls,
        arrays,
        edge_struct=None,
        draws=None,
        random_col_indices=None,
        random_distributions=None,
    ):
        """Build from a :class:`~locpick.data.arrays.ChoiceArrays`.

        Parameters
        ----------
        arrays : ChoiceArrays
            NumPy estimation data.
        edge_struct : EdgeStructure or None
            Spatial edge structure (for SCL/MSCL).
        draws : np.ndarray or None
            Simulation draws (for MSCL).
        random_col_indices : list[int] or None
            Column indices of random parameters.
        random_distributions : list[str] or None
            Distribution names for random parameters.

        Returns
        -------
        ChoiceDataJAX
        """
        if not _JAX_AVAILABLE:
            raise ImportError("JAX is required for ChoiceDataJAX")

        import jax.numpy as jnp

        n_obs = arrays.n_obs
        n_alts = arrays.n_alts

        # Core arrays
        design_matrix = jnp.array(arrays.design_matrix, dtype=jnp.float64)
        chosen = jnp.array(arrays.chosen, dtype=jnp.float64)

        # Availability
        if arrays.available is not None:
            available = jnp.array(arrays.available, dtype=jnp.float64)
        else:
            available = jnp.ones((n_obs, n_alts), dtype=jnp.float64)

        # Weights
        if arrays.weights is not None:
            weights = jnp.array(arrays.weights, dtype=jnp.float64)
        else:
            weights = jnp.ones(n_obs, dtype=jnp.float64)

        # Sampling correction
        inclusion_probs = None
        correction = get_sampling_correction(arrays)
        if correction is not None:
            inclusion_probs = jnp.array(correction, dtype=jnp.float64)

        # Edge data
        edge_data = None
        if edge_struct is not None:
            edge_data = EdgeDataJAX.from_edge_structure(edge_struct)

        # Random parameter columns (for MSCL)
        dm_fixed = None
        dm_random = None
        dist_codes_jax = None
        if random_col_indices is not None:
            all_col_indices = list(range(arrays.design_matrix.shape[1]))
            fixed_col_indices = [i for i in all_col_indices if i not in random_col_indices]

            if fixed_col_indices:
                dm_fixed = jnp.array(arrays.design_matrix[:, fixed_col_indices], dtype=jnp.float64)
            dm_random = jnp.array(arrays.design_matrix[:, random_col_indices], dtype=jnp.float64)

            # Encode distributions as int array
            dist_map = {"normal": 0, "lognormal": 1, "triangular": 2, "uniform": 3}
            dist_codes = np.array(
                [dist_map.get(d, 0) for d in random_distributions], dtype=np.int32
            )
            dist_codes_jax = jnp.array(dist_codes, dtype=jnp.int32)
            fixed_col_indices_list = fixed_col_indices
        else:
            fixed_col_indices_list = None

        # Draws
        draws_jax = None
        if draws is not None:
            draws_jax = jnp.array(draws, dtype=jnp.float64)

        return cls(
            design_matrix=design_matrix,
            chosen=chosen,
            n_obs=n_obs,
            n_alts=n_alts,
            available=available,
            weights=weights,
            inclusion_probs=inclusion_probs,
            edge_data=edge_data,
            draws=draws_jax,
            random_col_indices=random_col_indices,
            fixed_col_indices=fixed_col_indices_list,
            dist_codes=dist_codes_jax,
            dm_fixed=dm_fixed,
            dm_random=dm_random,
        )
