import importlib.util

import numpy.testing as npt
import pytest

from locpick import MultinomialLogit, dgp

FORMULA = "alt_feature + obs_x_alt - 1"


def _fit_v2(dataset, backend, monkeypatch):
    """Fit using the v2 MultinomialLogit, controlling JAX vs NumPy via env var."""
    monkeypatch.delenv("CHOICEMODELS_MNL_BACKEND", raising=False)
    if backend != "jax":
        monkeypatch.setenv("CHOICEMODELS_MNL_BACKEND", backend)
    model = MultinomialLogit(dataset.choice_table, FORMULA)
    result = model.fit()
    return result.coefficients


def test_mnl_parameter_recovery_with_interaction_term():
    dataset = dgp.simulate_mnl(
        n_obs=4000,
        n_alts=5,
        alt_params={"alt_feature": -0.7},
        interaction_params={"obs_x_alt": 1.1},
        seed=1234,
    )
    model = MultinomialLogit(dataset.choice_table, FORMULA)
    estimated = model.fit().coefficients

    # With 4 000 observations MLE is consistent; allow 10 % relative tolerance.
    npt.assert_allclose(estimated["alt_feature"], dataset.true_params["alt_feature"], rtol=0.10)
    npt.assert_allclose(estimated["obs_x_alt"], dataset.true_params["obs_x_alt"], rtol=0.10)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_mnl_parameter_recovery_across_backends(backend, monkeypatch):
    if backend == "jax" and importlib.util.find_spec("jax") is None:
        pytest.skip("JAX backend recovery test skipped because jax is not installed")

    dataset = dgp.simulate_mnl(seed=2026)
    estimated = _fit_v2(dataset, backend, monkeypatch)

    npt.assert_allclose(estimated["alt_feature"], dataset.true_params["alt_feature"], rtol=0.10)
    npt.assert_allclose(estimated["obs_x_alt"], dataset.true_params["obs_x_alt"], rtol=0.10)


def test_mnl_backend_coefficient_consistency(monkeypatch):
    dataset = dgp.simulate_mnl(seed=909)

    est_numpy = _fit_v2(dataset, "numpy", monkeypatch)

    if importlib.util.find_spec("jax") is not None:
        est_jax = _fit_v2(dataset, "jax", monkeypatch)
        npt.assert_allclose(
            est_jax[["alt_feature", "obs_x_alt"]].to_numpy(),
            est_numpy[["alt_feature", "obs_x_alt"]].to_numpy(),
            rtol=1e-4,
            atol=1e-6,
        )
