"""Panel and correlated mixed logit."""

import warnings

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from locpick import ChoiceModel
from locpick.dgp import _build_choice_table
from locpick.models.mixed import ParamDistribution

RP1 = {"x1": ParamDistribution("normal", "x1")}
RP2 = {"x1": ParamDistribution("normal", "x1"), "x2": ParamDistribution("normal", "x2")}
SIGMA = np.array([[0.64, 0.24], [0.24, 0.25]])
MU = np.array([-1.0, 0.5])


def _panel_table(n_people, T, J, tastes, seed):
    """Choice situations nested in people; tastes are fixed within a person."""
    rng = np.random.default_rng(seed)
    n = n_people * T
    person = np.repeat(np.arange(n_people), T)
    X = rng.standard_normal((2, n, J))
    B = tastes(rng, n_people)[person]
    U = B[:, 0:1] * X[0] + B[:, 1:2] * X[1] + rng.gumbel(size=(n, J))
    oid = pd.Index(np.arange(n), name="oid")
    aid = pd.Index(np.arange(J), name="aid")
    choosers = pd.DataFrame({"person": person, "choice": U.argmax(1)}, index=oid)
    alts = pd.DataFrame({"const": np.zeros(J)}, index=aid)
    mi = pd.MultiIndex.from_product([oid, aid], names=["oid", "aid"])
    md = {f"x{k + 1}": pd.Series(X[k].ravel(), index=mi, name=f"x{k + 1}") for k in range(2)}
    return _build_choice_table(choosers, alts, choosers["choice"], matrix_data=md)


def _one_random(r, n):
    return np.column_stack([r.normal(-1.0, 0.8, n), np.full(n, 0.5)])


def _correlated(r, n):
    return MU + r.standard_normal((n, 2)) @ np.linalg.cholesky(SIGMA).T


@pytest.fixture(scope="module")
def panel_model():
    ct = _panel_table(150, 4, 4, _one_random, 0)
    model = ChoiceModel(ct, formula="x1 + x2 - 1", random_params=RP1, n_draws=50, panel="person")
    model.fit()
    return model


def test_panel_likelihood_is_product_within_person(panel_model):
    """Fitted LL = sum_n log mean_r prod_t P(y_nt | beta_nr), rebuilt in NumPy."""
    from locpick._sampling.correction import get_sampling_correction
    from locpick.models.mixed import _mixed_logit_per_draw_log_probs_numpy

    m = panel_model
    a = m._arrays
    v = m._result.coefficients.to_numpy()
    kf, kr = m._k_fixed, m._k_random
    log_p, _, _ = _mixed_logit_per_draw_log_probs_numpy(
        v[:kf],
        v[kf : kf + kr],
        m._spread_from_display(v[kf + kr :]),
        m._random_distributions,
        m._draws,
        np.asarray(a.design_matrix, dtype=np.float64),
        m._random_col_indices,
        a.n_obs,
        a.n_alts,
        available=a.available,
        inclusion_probs=get_sampling_correction(a),
    )
    chosen = np.asarray(a.chosen, dtype=np.float64).reshape(a.n_obs, a.n_alts)
    log_chosen = (log_p * chosen[:, None, :]).sum(axis=2)  # (n_obs, n_draws)
    codes, n_panels = m._panel_structure
    per_person = np.zeros((n_panels, log_chosen.shape[1]))
    np.add.at(per_person, codes, log_chosen)
    ll = np.sum(np.log(np.mean(np.exp(per_person), axis=1)))
    npt.assert_allclose(m._result.log_likelihood, ll, rtol=1e-10)


def test_panel_draws_shared_within_person(panel_model):
    codes, _ = panel_model._panel_structure
    draws = panel_model._draws
    for p in range(5):
        rows = draws[codes == p]
        assert np.all(rows == rows[0])


def test_panel_scores_and_clusters_are_per_person(panel_model):
    m = panel_model
    codes, n_panels = m._panel_structure
    assert m._observation_scores(m._arrays).shape[0] == n_panels
    # Observation-level clusters are collapsed to people ...
    groups = codes // 10
    se = m.std_errors_clustered(groups=groups)
    assert np.isfinite(se).all()
    # ... and must not split a person.
    split = codes.copy()
    split[0] = split[0] + 1000
    with pytest.raises(ValueError, match="all of a decision-maker"):
        m.std_errors_clustered(groups=split)


def test_panel_requires_mixed_logit():
    ct = _panel_table(20, 2, 3, _one_random, 1)
    with pytest.raises(ValueError, match="mixed logit"):
        ChoiceModel(ct, formula="x1 + x2 - 1", panel="person")


def test_correlated_layout_and_consistency():
    """Cholesky layout, a PD covariance, and probabilities matching the LL."""
    ct = _panel_table(800, 1, 4, _correlated, 2)
    m = ChoiceModel(ct, formula="x1 + x2 - 1", random_params=RP2, n_draws=50, correlated=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.fit()
    assert [n for n in r.coefficients.index if n.startswith("chol_")] == [
        "chol_x1_x1",
        "chol_x2_x1",
        "chol_x2_x2",
    ]
    cov = m.random_parameter_covariance()
    assert np.all(np.linalg.eigvalsh(cov.to_numpy()) > 0)
    P = m.probabilities()
    chosen = np.asarray(m._arrays.chosen, dtype=np.float64).reshape(P.shape)
    npt.assert_allclose(np.log((P * chosen).sum(axis=1)).sum(), r.log_likelihood, atol=1e-8)


def test_correlated_requires_normal_or_lognormal():
    ct = _panel_table(50, 1, 3, _correlated, 3)
    rp = {"x1": ParamDistribution("normal", "x1"), "x2": ParamDistribution("triangular", "x2")}
    m = ChoiceModel(ct, formula="x1 + x2 - 1", random_params=rp, correlated=True)
    with pytest.raises(ValueError, match="normal or lognormal"):
        m.fit()


@pytest.mark.slow
def test_panel_correlated_recovery():
    """Panel + correlated recovers the means and the Cholesky factor."""
    L = np.linalg.cholesky(SIGMA)
    truth = {
        "mean_x1": MU[0],
        "mean_x2": MU[1],
        "chol_x1_x1": L[0, 0],
        "chol_x2_x1": L[1, 0],
        "chol_x2_x2": L[1, 1],
    }
    ct = _panel_table(1000, 6, 5, _correlated, 2)
    m = ChoiceModel(
        ct,
        formula="x1 + x2 - 1",
        random_params=RP2,
        n_draws=200,
        correlated=True,
        panel="person",
    )
    r = m.fit()
    for name, value in truth.items():
        z = (r.coefficients[name] - value) / r.std_errors[name]
        assert abs(z) < 3, (name, r.coefficients[name], value, r.std_errors[name])


@pytest.mark.parametrize("family", ["mscl", "mixed_nested", "sar_mixed", "sar_mixed_nested"])
def test_panel_with_nesting_and_space(family):
    """Panels work through every mixed kernel, scoring per person."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_model_consistency import _make

    base = _make(family)  # reuse the family's data and configuration
    n_obs = base._get_arrays().n_obs
    persons = np.arange(n_obs) // 3
    model = ChoiceModel(
        base._data,
        formula=base._spec.formula,
        random_params=base._random_params,
        nests=base._nests,
        graph=base._graph_input,
        lag=base._lag,
        n_draws=10,
        panel=persons,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit()
    assert np.isfinite(result.log_likelihood)
    n_panels = model._panel_structure[1]
    assert model._observation_scores(model._arrays).shape[0] == n_panels
    # Probabilities stay per choice situation.
    assert model.probabilities().shape[0] == n_obs
