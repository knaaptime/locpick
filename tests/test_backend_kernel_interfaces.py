"""Contract tests for backend kernel interface parity."""

from __future__ import annotations

import importlib.util
import inspect

import numpy as np
import numpy.testing as npt
import pytest

from locpick._kernels.mnl_numpy import (
    mnl_log_likelihood_numpy,
    mnl_log_probs_numpy,
    mnl_probs_numpy,
)
from locpick.data import ChoiceArrays

_HAS_JAX = importlib.util.find_spec("jax") is not None
pytestmark = pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")

if _HAS_JAX:
    import jax.numpy as jnp

    from locpick._jax.builders import build_mnl_objective
    from locpick._jax.kernels import mnl_log_probs, mnl_probs


class TestBackendKernelInterfaces:
    """Tests that NumPy and JAX MNL kernels expose matching contracts."""

    def test_mnl_log_probs_signature_has_shared_contract(self):
        sig_np = inspect.signature(mnl_log_probs_numpy)
        sig_jax = inspect.signature(mnl_log_probs)

        np_names = list(sig_np.parameters.keys())
        jax_names = list(sig_jax.parameters.keys())

        assert len(np_names) == len(jax_names) == 3
        assert np_names[1:] == ["available", "inclusion_probs"]
        assert jax_names[1:] == ["available", "inclusion_probs"]

    def test_mnl_probs_signature_has_shared_contract(self):
        sig_np = inspect.signature(mnl_probs_numpy)
        sig_jax = inspect.signature(mnl_probs)

        np_names = list(sig_np.parameters.keys())
        jax_names = list(sig_jax.parameters.keys())

        assert len(np_names) == len(jax_names) == 3
        assert np_names[1:] == ["available", "inclusion_probs"]
        assert jax_names[1:] == ["available", "inclusion_probs"]

    def test_mnl_log_probs_numpy_jax_parity(self):
        rng = np.random.default_rng(123)
        n_obs = 7
        n_alts = 4

        utilities = rng.standard_normal((n_obs, n_alts))
        available = (rng.random((n_obs, n_alts)) > 0.2).astype(np.float64)
        available[np.arange(n_obs), rng.integers(0, n_alts, size=n_obs)] = 1.0
        inclusion_probs = rng.uniform(0.2, 1.0, size=(n_obs, n_alts))

        out_np = mnl_log_probs_numpy(utilities, available, inclusion_probs)
        out_jax = np.asarray(
            mnl_log_probs(
                jnp.asarray(utilities),
                jnp.asarray(available),
                jnp.asarray(inclusion_probs),
            )
        )

        npt.assert_allclose(out_jax, out_np, rtol=1e-10, atol=1e-10)

    def test_mnl_probs_numpy_jax_parity(self):
        rng = np.random.default_rng(321)
        n_obs = 6
        n_alts = 5

        utilities = rng.standard_normal((n_obs, n_alts))
        available = (rng.random((n_obs, n_alts)) > 0.3).astype(np.float64)
        available[np.arange(n_obs), rng.integers(0, n_alts, size=n_obs)] = 1.0
        inclusion_probs = rng.uniform(0.2, 1.0, size=(n_obs, n_alts))

        probs_np = mnl_probs_numpy(utilities, available, inclusion_probs)
        probs_jax = np.asarray(
            mnl_probs(
                jnp.asarray(utilities),
                jnp.asarray(available),
                jnp.asarray(inclusion_probs),
            )
        )

        npt.assert_allclose(probs_jax, probs_np, rtol=1e-10, atol=1e-10)

    def test_mnl_objective_builder_matches_numpy_ll(self):
        rng = np.random.default_rng(11)
        n_obs = 10
        n_alts = 4
        n_params = 3

        design_matrix = rng.standard_normal((n_obs * n_alts, n_params))
        chosen = np.zeros((n_obs, n_alts), dtype=np.float64)
        chosen_idx = rng.integers(0, n_alts, size=n_obs)
        chosen[np.arange(n_obs), chosen_idx] = 1.0

        available = (rng.random((n_obs, n_alts)) > 0.1).astype(np.float64)
        available[np.arange(n_obs), chosen_idx] = 1.0
        weights = rng.uniform(0.5, 2.0, size=n_obs)
        inclusion_probs = rng.uniform(0.2, 1.0, size=(n_obs, n_alts))

        arrays = ChoiceArrays(
            design_matrix=design_matrix,
            chosen=chosen,
            available=available,
            weights=weights,
            n_obs=n_obs,
            n_alts=n_alts,
            param_names=[f"x{i}" for i in range(n_params)],
            inclusion_probs=inclusion_probs,
        )
        beta = rng.standard_normal(n_params)

        objective = build_mnl_objective(arrays)
        ll_jax = objective.fn(beta)
        ll_np = mnl_log_likelihood_numpy(
            beta=beta,
            design_matrix=design_matrix,
            chosen=chosen,
            available=available,
            n_obs=n_obs,
            n_alts=n_alts,
            weights=weights,
            inclusion_probs=inclusion_probs,
        )

        npt.assert_allclose(ll_jax, ll_np, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("distribution", ["normal", "lognormal", "triangular", "uniform"])
def test_random_coefficient_realisation_matches_jax(distribution):
    """NumPy and JAX must realise the same mixing distribution.

    The NumPy reference kernel previously applied the uniform inverse-CDF to
    triangular draws, so the two backends disagreed for that distribution
    while the tests comparing them only covered the normal case.
    """
    import jax.numpy as jnp
    from jax.scipy.stats import norm

    from locpick.models.mixed import realize_random_coefficients

    rng = np.random.default_rng(0)
    draws = rng.standard_normal((6, 9, 1))
    mean, spread = np.array([0.5]), np.array([2.0])

    from_numpy = realize_random_coefficients(draws, mean, spread, [distribution])[:, :, 0]

    z = jnp.asarray(draws[:, :, 0])
    phi = norm.cdf(z)
    if distribution == "normal":
        from_jax = mean[0] + spread[0] * z
    elif distribution == "lognormal":
        from_jax = jnp.exp(jnp.clip(mean[0] + spread[0] * z, -50, 50))
    elif distribution == "uniform":
        from_jax = mean[0] + spread[0] * (2 * phi - 1)
    else:
        from_jax = jnp.where(
            phi <= 0.5,
            mean[0] + spread[0] * (jnp.sqrt(2 * phi) - 1),
            mean[0] + spread[0] * (1 - jnp.sqrt(2 * (1 - phi))),
        )

    npt.assert_allclose(from_numpy, np.asarray(from_jax), rtol=1e-5, atol=1e-5)


def test_triangular_realisation_is_not_uniform():
    """Pins the specific regression: triangular must not reduce to uniform."""
    from locpick.models.mixed import realize_random_coefficients

    rng = np.random.default_rng(1)
    draws = rng.standard_normal((8, 12, 1))
    mean, spread = np.array([0.0]), np.array([1.0])

    triangular = realize_random_coefficients(draws, mean, spread, ["triangular"])
    uniform = realize_random_coefficients(draws, mean, spread, ["uniform"])

    assert np.abs(triangular - uniform).max() > 1e-3
