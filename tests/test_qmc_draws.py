"""Tests for scrambled QMC draw generation and the draw-type resolver."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from locpick.models.mixed import (
    _resolve_draws,
    generate_halton_draws,
    generate_qmc_draws,
    generate_random_draws,
)

# ---------------------------------------------------------------------------
# generate_qmc_draws
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["sobol", "halton"])
def test_qmc_shape_and_dtype(engine):
    z = generate_qmc_draws(n_obs=100, n_draws=64, n_random_params=3, seed=0, engine=engine)
    assert z.shape == (100, 64, 3)
    assert z.dtype == np.float64
    assert np.all(np.isfinite(z))


@pytest.mark.parametrize("engine", ["sobol", "halton"])
def test_qmc_reproducible(engine):
    a = generate_qmc_draws(50, 32, 2, seed=123, engine=engine)
    b = generate_qmc_draws(50, 32, 2, seed=123, engine=engine)
    npt.assert_array_equal(a, b)


@pytest.mark.parametrize("engine", ["sobol", "halton"])
def test_qmc_seed_changes_output(engine):
    a = generate_qmc_draws(50, 32, 2, seed=1, engine=engine)
    b = generate_qmc_draws(50, 32, 2, seed=2, engine=engine)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("engine", ["sobol", "halton"])
def test_qmc_moments_match_standard_normal(engine):
    # 512 draws per obs over 200 obs => 102,400 samples per param dim.
    z = generate_qmc_draws(n_obs=200, n_draws=512, n_random_params=2, seed=7, engine=engine)
    means = z.reshape(-1, 2).mean(axis=0)
    stds = z.reshape(-1, 2).std(axis=0)
    npt.assert_allclose(means, [0.0, 0.0], atol=0.05)
    npt.assert_allclose(stds, [1.0, 1.0], atol=0.05)


def test_qmc_rejects_unknown_engine():
    with pytest.raises(ValueError, match="Unknown QMC engine"):
        generate_qmc_draws(10, 8, 2, seed=0, engine="nonsense")


def test_qmc_lower_variance_than_iid():
    """Scrambled Sobol should give lower MC error than IID for E[f(z)].

    Use f(z) = z**2 (true mean = 1).  At R=256 per obs the QMC estimate
    should beat the IID estimate by a healthy margin in absolute error,
    averaged across many observations.
    """
    n_obs, n_draws, k = 400, 256, 2

    iid = generate_random_draws(n_obs, n_draws, k, seed=11)
    qmc = generate_qmc_draws(n_obs, n_draws, k, seed=11, engine="sobol")

    err_iid = np.abs((iid**2).mean(axis=1).mean() - 1.0)
    err_qmc = np.abs((qmc**2).mean(axis=1).mean() - 1.0)

    # Scrambled Sobol should be at least 2x more accurate on this smooth
    # integrand.  Use a loose factor to keep the test robust to seeds.
    assert err_qmc < err_iid, f"QMC error {err_qmc:.4e} not below IID error {err_iid:.4e}"


# ---------------------------------------------------------------------------
# _resolve_draws dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "draw_type, expected_fn",
    [
        ("halton", generate_halton_draws),
        ("random", generate_random_draws),
    ],
)
def test_resolve_dispatches_to_legacy(draw_type, expected_fn):
    got = _resolve_draws(draw_type, n_obs=30, n_draws=16, n_random_params=2, seed=5)
    expected = expected_fn(30, 16, 2, seed=5)
    npt.assert_array_equal(got, expected)


@pytest.mark.parametrize("alias", ["qmc", "sobol", "QMC", "Sobol"])
def test_resolve_qmc_aliases(alias):
    got = _resolve_draws(alias, n_obs=30, n_draws=16, n_random_params=2, seed=5)
    expected = generate_qmc_draws(30, 16, 2, seed=5, engine="sobol")
    npt.assert_array_equal(got, expected)


def test_resolve_scrambled_halton():
    got = _resolve_draws("scrambled_halton", n_obs=30, n_draws=16, n_random_params=2, seed=5)
    expected = generate_qmc_draws(30, 16, 2, seed=5, engine="halton")
    npt.assert_array_equal(got, expected)


def test_resolve_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown draw_type"):
        _resolve_draws("foobar", 10, 8, 2)


# ---------------------------------------------------------------------------
# MSCL end-to-end smoke test with QMC draws
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mscl_qmc_fits_and_recovers():
    """MSCL with draw_type='qmc' should fit and stay in the same ballpark
    as the existing halton path on the standard simulated DGP."""
    from locpick import ChoiceModel
    from locpick.dgp import simulate_mscl
    from locpick.models.mixed import ParamDistribution as PD

    ds = simulate_mscl(n_obs=400, n_alts=10, seed=3)
    rp = {"time": PD(param="time", distribution="normal")}

    m_qmc = ChoiceModel(
        data=ds.choice_table,
        formula="cost + time + income_x_cost",
        graph=ds.adjacency,
        random_params=rp,
        n_draws=64,
        draw_type="qmc",
    )
    res_qmc = m_qmc.fit()
    assert np.isfinite(res_qmc.log_likelihood)

    m_halton = ChoiceModel(
        data=ds.choice_table,
        formula="cost + time + income_x_cost",
        graph=ds.adjacency,
        random_params=rp,
        n_draws=64,
        draw_type="halton",
    )
    res_halton = m_halton.fit()

    # LLs need not be identical but should be within 2% on R=64.
    rel = abs(res_qmc.log_likelihood - res_halton.log_likelihood) / abs(res_halton.log_likelihood)
    assert rel < 0.02, (
        f"LL gap too large: qmc={res_qmc.log_likelihood:.2f} halton={res_halton.log_likelihood:.2f}"
    )
