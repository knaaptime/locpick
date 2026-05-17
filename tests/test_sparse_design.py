"""Tests for sparse design-matrix support in JAX data/kernel paths."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from locpick import ChoiceTable
from locpick._jax.data import ChoiceDataJAX
from locpick._jax.kernels import compute_utilities
from locpick.data.arrays import ChoiceArrays


def _make_choice_table(n_obs: int = 40, n_alts: int = 20, seed: int = 123) -> ChoiceTable:
    """Create a compact synthetic choice table with sparse-like features."""
    rng = np.random.default_rng(seed)

    choosers = pd.DataFrame(index=pd.Index(np.arange(n_obs), name="oid"))
    alternatives = pd.DataFrame(
        {
            "sparse_x": (rng.random(n_alts) < 0.15).astype(float),
            "dense_x": rng.normal(size=n_alts),
        },
        index=pd.Index(np.arange(n_alts), name="aid"),
    )

    utility = (
        -0.4 * alternatives["dense_x"].to_numpy()[None, :]
        + 0.8 * alternatives["sparse_x"].to_numpy()[None, :]
    )
    utility = utility + rng.gumbel(size=(n_obs, n_alts))
    choices = utility.argmax(axis=1)

    return ChoiceTable.from_tables(
        choosers=choosers,
        alternatives=alternatives,
        chosen_alternatives=pd.Series(choices, index=choosers.index),
    )


def test_to_arrays_force_sparse_builds_sparse_matrix():
    """force_sparse=True should populate design_matrix_sparse."""
    ct = _make_choice_table()
    arrays = ct.to_arrays(formula="sparse_x + dense_x - 1", force_sparse=True)

    assert arrays.design_matrix_sparse is not None
    assert arrays.force_sparse is True


def test_choice_data_jax_sparse_matches_dense_utilities():
    """Sparse and dense utility computation should agree numerically."""
    pytest.importorskip("jax")

    ct = _make_choice_table()
    arrays = ct.to_arrays(formula="sparse_x + dense_x - 1", force_sparse=True)
    data = ChoiceDataJAX.from_arrays(arrays)

    assert data.design_matrix_sparse is not None

    beta = np.array([0.5, -0.2], dtype=np.float64)
    v_dense = compute_utilities(
        data.design_matrix,
        beta,
        data.n_obs,
        data.n_alts,
        inclusion_probs=data.inclusion_probs,
        available=data.available,
    )
    v_sparse = compute_utilities(
        data.design_matrix,
        beta,
        data.n_obs,
        data.n_alts,
        design_matrix_sparse=data.design_matrix_sparse,
        inclusion_probs=data.inclusion_probs,
        available=data.available,
    )

    npt.assert_allclose(np.asarray(v_sparse), np.asarray(v_dense), rtol=1e-8, atol=1e-8)


def test_choice_data_jax_auto_sparse_uses_sparsity_hint():
    """ChoiceDataJAX should auto-convert highly sparse large matrices."""
    pytest.importorskip("jax")

    n_obs = 600
    n_alts = 200
    n_rows = n_obs * n_alts
    k = 2

    design = np.zeros((n_rows, k), dtype=np.float64)
    design[::80, 0] = 1.0
    design[::120, 1] = -0.5

    chosen = np.zeros((n_obs, n_alts), dtype=np.float64)
    chosen[:, 0] = 1.0

    arrays = ChoiceArrays(
        design_matrix=design,
        chosen=chosen,
        n_obs=n_obs,
        n_alts=n_alts,
        param_names=["x0", "x1"],
    )
    assert arrays._sparsity_hint is not None
    assert arrays._sparsity_hint < 0.1

    data = ChoiceDataJAX.from_arrays(arrays)
    assert data.design_matrix_sparse is not None


def test_choice_data_jax_force_dense_disables_auto_sparse():
    """force_sparse=False should suppress automatic sparse conversion."""
    pytest.importorskip("jax")

    n_obs = 600
    n_alts = 200
    n_rows = n_obs * n_alts

    design = np.zeros((n_rows, 2), dtype=np.float64)
    design[::100, 0] = 1.0

    chosen = np.zeros((n_obs, n_alts), dtype=np.float64)
    chosen[:, 0] = 1.0

    arrays = ChoiceArrays(
        design_matrix=design,
        chosen=chosen,
        n_obs=n_obs,
        n_alts=n_alts,
        param_names=["x0", "x1"],
        force_sparse=False,
    )

    data = ChoiceDataJAX.from_arrays(arrays)
    assert data.design_matrix_sparse is None
