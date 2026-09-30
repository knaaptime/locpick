"""Consistency checks between estimation and post-estimation.

Each test pins down an invariant that the shape-only prediction tests cannot
see: probabilities must reproduce the maximised likelihood, reported
parameters must be the ones the kernel used, and analytic effects must match
numerical derivatives of the probabilities.
"""

import warnings

import numpy as np
import numpy.testing as npt
import pytest

from locpick import ChoiceModel, ModelSpec, dgp
from locpick.models.mixed import ParamDistribution
from locpick.models.nested import NestingTree, NestSpec, _nested_logit_probs_numpy


def _probs_ll(model):
    P = model.probabilities()
    chosen = np.asarray(model._arrays.chosen, dtype=np.float64).reshape(P.shape)
    return float(np.log((P * chosen).sum(axis=1)).sum())


RP_TIME = {"time": ParamDistribution("normal", "time")}
RP_ALT = {"alt_attr": ParamDistribution("normal", "alt_attr")}
SAR_FORMULA = "alt_attr + obs_x_alt_attr - 1"
SAR_TREE = NestingTree([NestSpec("a", list(range(10))), NestSpec("b", list(range(10, 20)))])


def _sar_data():
    return dgp.simulate_sar_mnl(
        n_obs=400, n_alts=20, seed=3, interaction_params={"obs_x_alt_attr": 0.8}
    )


def _make(family):
    if family == "mnl":
        d = dgp.simulate_nested_logit(n_obs=600, seed=1)
        return ChoiceModel(d.choice_table, formula="cost + time - 1")
    if family == "nested":
        d = dgp.simulate_nested_logit(n_obs=600, seed=1)
        return ChoiceModel(d.choice_table, formula="cost + time - 1", nests=d.nests)
    if family == "mixed":
        d = dgp.simulate_mixed_logit(n_obs=400, seed=1)
        return ChoiceModel(
            d.choice_table, formula="cost + time - 1", random_params=RP_TIME, n_draws=30
        )
    if family == "mixed_nested":
        d = dgp.simulate_mixed_nested_logit(n_obs=400, seed=1)
        return ChoiceModel(
            d.choice_table,
            formula="cost + time - 1",
            nests=d.nests,
            random_params=RP_TIME,
            n_draws=30,
        )
    if family == "scl":
        d = dgp.simulate_scl(n_obs=600, seed=1)
        return ChoiceModel(d.choice_table, formula="cost + time", graph=d.adjacency)
    if family == "mscl":
        d = dgp.simulate_mscl(n_obs=400, seed=1)
        return ChoiceModel(
            d.choice_table,
            formula="cost + time",
            graph=d.adjacency,
            random_params=RP_TIME,
            n_draws=30,
        )
    if family == "nested_scl":
        d = dgp.simulate_nested_scl(n_obs=600, seed=1)
        return ChoiceModel(d.choice_table, formula="cost + time", graph=d.adjacency, nests=d.nests)
    s = _sar_data()
    kw = {
        "sar": {},
        "sar_reduced": {"estimator": "reduced"},
        "sar_nested": {"nests": SAR_TREE},
        "sar_mixed": {"random_params": RP_ALT, "n_draws": 20},
        "sar_mixed_nested": {"nests": SAR_TREE, "random_params": RP_ALT, "n_draws": 20},
    }[family]
    return ChoiceModel(s.choice_table, formula=SAR_FORMULA, graph=s.W, lag=True, **kw)


FAMILIES = [
    "mnl",
    "nested",
    "mixed",
    "mixed_nested",
    "scl",
    "mscl",
    "nested_scl",
    "sar",
    "sar_reduced",
    "sar_nested",
    "sar_mixed",
    "sar_mixed_nested",
]


@pytest.mark.parametrize("family", FAMILIES)
def test_probabilities_reproduce_fitted_log_likelihood(family):
    """sum(log P_chosen) from probabilities() equals the maximised LL."""
    model = _make(family)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit()
    npt.assert_allclose(_probs_ll(model), result.log_likelihood, rtol=0, atol=1e-8)


def test_mixed_spread_reported_on_kernel_scale():
    """The reported SD is softplus(raw), the value the likelihood uses."""
    d = dgp.simulate_mixed_logit(n_obs=4000, seed=3)
    model = ChoiceModel(
        d.choice_table, formula="cost + time - 1", random_params=RP_TIME, n_draws=100
    )
    result = model.fit()
    raw = model._raw_params[-1]
    npt.assert_allclose(result.coefficients["sd_time"], np.logaddexp(0.0, raw), rtol=1e-12)
    # True SD is 0.5; the |raw| display this replaces reported ~0.74 here.
    assert abs(result.coefficients["sd_time"] - d.true_params["sd_time"]) < 0.2


def test_mixed_prediction_on_new_sample_size():
    """Mixed models predict on data with a different number of choosers."""
    d = dgp.simulate_mixed_logit(n_obs=300, seed=1)
    other = dgp.simulate_mixed_logit(n_obs=120, seed=2)
    model = ChoiceModel(
        d.choice_table, formula="cost + time - 1", random_params=RP_TIME, n_draws=20
    )
    model.fit()
    P = model.probabilities(data=other.choice_table)
    assert P.shape == (120, other.n_alts)
    npt.assert_allclose(P.sum(axis=1), 1.0)


def test_nested_marginal_effect_matches_numerical_derivative():
    """d log P_i / d x_i from marginal_effect() matches finite differences."""
    d = dgp.simulate_nested_logit(n_obs=300, seed=4)
    model = ChoiceModel(d.choice_table, formula="cost + time - 1", nests=d.nests)
    result = model.fit()
    me = model.marginal_effect(variable="cost").to_numpy().reshape(300, -1)

    n_obs, n_alts = me.shape
    names = [f"lambda_{n}" for n in d.nests.nest_names]
    lam = result.coefficients[names].to_numpy()
    alpha = np.log(lam / (1 - lam))
    V = model.utilities()

    def log_probs(V):
        P = _nested_logit_probs_numpy(
            np.ones(1), alpha, V.reshape(-1, 1), model._nest_matrix, n_obs, n_alts
        )
        return np.log(P)

    h = 1e-6
    numeric = np.empty_like(me)
    for j in range(n_alts):
        up, dn = V.copy(), V.copy()
        up[:, j] += h
        dn[:, j] -= h
        numeric[:, j] = (log_probs(up)[:, j] - log_probs(dn)[:, j]) / (2 * h)
    npt.assert_allclose(me, result.coefficients["cost"] * numeric, rtol=1e-5, atol=1e-8)


def test_nested_marginal_effect_reduces_to_mnl():
    """With lambda = 1 the nested formula equals the MNL formula."""
    d = dgp.simulate_nested_logit(n_obs=200, seed=5)
    model = ChoiceModel(d.choice_table, formula="cost + time - 1", nests=d.nests)
    model.fit()
    coefs = model._result.coefficients.copy()
    for n in d.nests.nest_names:
        coefs[f"lambda_{n}"] = 1.0
    object.__setattr__(model._result, "coefficients", coefs)
    probs = model.probabilities()
    me = model.marginal_effect(variable="cost").to_numpy()
    npt.assert_allclose(me, (1 - probs.ravel()) * coefs["cost"], rtol=1e-10)


def test_fixed_parameters_excluded_from_inference():
    """Held-fixed parameters get NaN SEs and do not distort free ones."""
    d = dgp.simulate_nested_logit(n_obs=800, seed=6)
    spec = ModelSpec().generic("cost").fixed("time", value=-0.1)
    model = ChoiceModel(d.choice_table, spec=spec)
    result = model.fit()
    names = list(result.coefficients.index)
    i_cost, i_time = names.index("cost"), names.index("time")
    assert np.isnan(result.std_errors.iloc[i_time])
    hess = model._objective.hessian(model._raw_params)
    npt.assert_allclose(
        result.std_errors.iloc[i_cost], 1.0 / np.sqrt(-hess[i_cost, i_cost]), rtol=1e-8
    )


def test_nonconvergence_is_reported():
    """A solver that stops early is flagged on the result and warned about."""
    d = dgp.simulate_nested_logit(n_obs=400, seed=1)
    model = ChoiceModel(
        d.choice_table,
        formula="cost + time - 1",
        nests=d.nests,
        solver_options={"maxiter": 1},
    )
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = model.fit()
    assert result.converged is False


def test_clustered_covariance_matches_reference():
    """Grouped score sums with the G/(G-1) correction match a direct loop."""
    d = dgp.simulate_nested_logit(n_obs=500, seed=7)
    model = ChoiceModel(d.choice_table, formula="cost + time - 1")
    model.fit()
    groups = np.random.default_rng(0).integers(0, 25, size=500)

    scores = model._observation_scores(model._arrays)
    B = sum(
        np.outer(scores[groups == g].sum(0), scores[groups == g].sum(0)) for g in np.unique(groups)
    )
    G = len(np.unique(groups))
    H_inv = model._get_hessian_inverse()
    expected = model._to_display_covariance(H_inv @ (B * G / (G - 1)) @ H_inv)
    npt.assert_allclose(model.covariance_clustered(groups=groups), expected, rtol=1e-10)


def test_robust_covariance_rejects_foreign_data():
    """The sandwich is defined on the estimation sample only."""
    d = dgp.simulate_nested_logit(n_obs=200, seed=1)
    other = dgp.simulate_nested_logit(n_obs=200, seed=2)
    model = ChoiceModel(d.choice_table, formula="cost + time - 1")
    model.fit()
    model.covariance_robust(data=d.choice_table)  # the estimation data is fine
    with pytest.raises(ValueError, match="estimation sample"):
        model.covariance_robust(data=other.choice_table)


class TestSARImpacts:
    @pytest.fixture(scope="class")
    def model(self):
        s = _sar_data()
        m = ChoiceModel(s.choice_table, formula=SAR_FORMULA, graph=s.W, lag=True)
        m.fit()
        return m

    def test_direct_effect_matches_numerical_derivative(self, model):
        import dataclasses

        me = model.marginal_effects(variable="alt_attr")
        base = model._arrays
        col = list(base.param_names).index("alt_attr")
        n_alts = base.n_alts
        h = 1e-5

        def probs_with(rows, delta):
            dm = np.array(base.design_matrix, dtype=np.float64, copy=True)
            dm[rows, col] += delta
            model._arrays = dataclasses.replace(base, design_matrix=dm)
            try:
                return model.probabilities()
            finally:
                model._arrays = base

        for src in (0, 7, 13):
            rows = np.arange(src, base.design_matrix.shape[0], n_alts)
            dP = (probs_with(rows, h) - probs_with(rows, -h)) / (2 * h)  # (n_obs, n_alts)
            direct = dP[:, src].mean()
            indirect = dP.mean(axis=0).sum() - direct
            npt.assert_allclose(me["direct"].iloc[src], direct, rtol=1e-5)
            npt.assert_allclose(me["indirect"].iloc[src], indirect, rtol=1e-4, atol=1e-10)
        npt.assert_allclose(me["total"].to_numpy(), 0.0)

    def test_spatial_impacts_direct_agrees(self, model):
        me = model.marginal_effects(variable="alt_attr")
        impacts = model.spatial_impacts("alt_attr")  # all sources at this J
        npt.assert_allclose(impacts["direct"], me["direct"].mean(), rtol=1e-10)


def test_sar_availability_filters_before_masking():
    """Unavailable alternatives still pass utility to neighbours when estimating."""
    s = _sar_data()
    n = s.n_obs * s.n_alts
    chosen = np.asarray(s.choice_table.to_frame()["chosen"], dtype=bool)
    avail = np.ones(n)
    rng = np.random.default_rng(1)
    drop = rng.random(n) < 0.15
    avail[drop & ~chosen] = 0.0
    model = ChoiceModel(
        s.choice_table, formula=SAR_FORMULA, graph=s.W, lag=True, availability=avail
    )
    result = model.fit()
    npt.assert_allclose(_probs_ll(model), result.log_likelihood, rtol=0, atol=1e-8)


@pytest.mark.parametrize("mixed", [False, True])
def test_sar_lowrank_filter_matches_full_filter(monkeypatch, mixed):
    """The low-rank filter gives the same likelihood and gradient."""
    import jax.numpy as jnp

    from locpick._jax import sar_kernels

    s = _sar_data()
    kw = {"random_params": RP_ALT, "n_draws": 10} if mixed else {}

    def objective():
        m = ChoiceModel(s.choice_table, formula=SAR_FORMULA, graph=s.W, lag=True, **kw)
        arrays = m._get_arrays()
        m._arrays = arrays
        m._pre_fit(arrays)
        return m._build_objective(arrays)

    # [beta..., alpha_rho] or [beta_fixed, alpha_rho, mean, raw_sd]
    x = jnp.asarray([0.7, 0.3, -0.4, -0.5] if mixed else [-0.4, 0.7, 0.3])
    arrays = ChoiceModel(s.choice_table, formula=SAR_FORMULA, graph=s.W, lag=True)._get_arrays()
    assert sar_kernels.design_lowrank(arrays) is not None
    fast = objective()
    monkeypatch.setattr(sar_kernels, "design_lowrank", lambda *a, **k: None)
    full = objective()
    npt.assert_allclose(float(fast.jax_fn(x)), float(full.jax_fn(x)), rtol=1e-12)
    npt.assert_allclose(np.asarray(fast.jax_grad(x)), np.asarray(full.jax_grad(x)), rtol=1e-10)


def test_sar_mixed_filters_random_utility():
    """SAR mixed recovers parameters when the whole utility is spatially filtered.

    The DGP filters random and fixed utility alike (reduced form, so the
    likelihood is exact).  Filtering only the fixed part, as the kernel once
    did, biases rho, the fixed coefficient and the spread upward.
    """
    import pandas as pd

    from locpick.dgp import _build_choice_table
    from locpick.models._spatial_weights import build_knn_graph

    rng = np.random.default_rng(0)
    n_obs, n_alts, rho, mu, sd, gamma = 4000, 40, 0.5, -1.0, 0.8, 0.6
    W = build_knn_graph(rng.standard_normal((n_alts, 2)), k=6)
    alts = pd.DataFrame(
        {"x": rng.standard_normal(n_alts), "z": rng.standard_normal(n_alts)},
        index=pd.Index(np.arange(n_alts), name="aid"),
    )
    b = mu + sd * rng.standard_normal(n_obs)
    V = b[:, None] * alts["x"].to_numpy()[None, :] + gamma * alts["z"].to_numpy()[None, :]
    A = np.eye(n_alts) - rho * np.asarray(W.sparse.todense())
    psi = np.linalg.solve(A, V.T).T
    choice = (psi + rng.gumbel(size=psi.shape)).argmax(axis=1)
    choosers = pd.DataFrame({"choice": choice}, index=pd.Index(np.arange(n_obs), name="oid"))
    ct = _build_choice_table(choosers, alts, choosers["choice"])

    model = ChoiceModel(
        ct,
        formula="x + z - 1",
        graph=W,
        lag=True,
        estimator="reduced",
        random_params={"x": ParamDistribution("normal", "x")},
        n_draws=200,
    )
    c = model.fit().coefficients
    assert abs(c["rho"] - rho) < 0.03
    assert abs(c["z"] - gamma) < 0.04
    assert abs(c["mean_x"] - mu) < 0.05
    assert abs(c["sd_x"] - sd) < 0.05


def test_design_lowrank_structure():
    """Rank-one and repeated-row columns factor exactly; full-rank ones do not."""
    from types import SimpleNamespace

    from locpick._jax.sar_kernels import design_lowrank

    rng = np.random.default_rng(0)
    n_obs, n_alts = 200, 15
    alt_attr = np.tile(rng.normal(size=n_alts), (n_obs, 1))
    interaction = np.outer(rng.normal(size=n_obs), rng.normal(size=n_alts))
    origins = rng.normal(size=(4, n_alts))
    distance = origins[rng.integers(0, 4, size=n_obs)]
    dm = np.column_stack([c.ravel() for c in (alt_attr, interaction, distance)])
    arrays = SimpleNamespace(n_obs=n_obs, n_alts=n_alts, design_matrix=dm, inclusion_probs=None)

    U, S_T, owner = (np.asarray(a) for a in design_lowrank(arrays))
    assert U.shape[1] == 1 + 1 + 4
    beta = np.array([0.3, -1.2, 0.8])
    npt.assert_allclose((U * beta[owner]) @ S_T, (dm @ beta).reshape(n_obs, n_alts), atol=1e-12)

    full = rng.normal(size=(n_obs * n_alts, 1))
    arrays.design_matrix = np.column_stack([dm, full])
    assert design_lowrank(arrays) is None


@pytest.mark.parametrize("family", FAMILIES)
def test_new_data_path_and_availability(family):
    """Prediction on a table reproduces the estimation probabilities, and an
    availability mask removes alternatives while keeping rows normalised."""
    model = _make(family)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit()
    P = model.probabilities()
    # Passing the table goes through fresh arrays (and, for combined models, a
    # freshly built objective with regenerated draws).
    npt.assert_allclose(model.probabilities(data=model._data), P, rtol=0, atol=1e-12)

    mask = np.random.default_rng(0).random(P.shape) > 0.3
    mask[np.arange(P.shape[0]), P.argmax(axis=1)] = True
    Pa = model.probabilities(available=mask)
    assert np.all(Pa[~mask] == 0)
    npt.assert_allclose(Pa.sum(axis=1), 1.0, atol=1e-12)
    if family == "mnl":  # IIA: restricting the set is renormalisation
        ren = P * mask
        npt.assert_allclose(Pa, ren / ren.sum(axis=1, keepdims=True), atol=1e-12)


@pytest.mark.parametrize("family", ["mscl", "nested_scl", "mixed_nested", "sar_mixed"])
def test_combined_models_predict_on_new_sample(family):
    """Combined models predict on a sample of a different size."""
    model = _make(family)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit()
    if family.startswith("sar"):
        other = dgp.simulate_sar_mnl(
            n_obs=150, n_alts=20, seed=9, interaction_params={"obs_x_alt_attr": 0.8}
        )
    else:
        maker = {
            "mscl": dgp.simulate_mscl,
            "nested_scl": dgp.simulate_nested_scl,
            "mixed_nested": dgp.simulate_mixed_nested_logit,
        }[family]
        other = maker(n_obs=150, seed=9)
    P = model.probabilities(data=other.choice_table)
    assert P.shape == (150, model._arrays.n_alts)
    npt.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
