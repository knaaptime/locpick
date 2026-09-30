"""Capacity-constrained simulation (ChoiceModel.simulate with ``capacity``)."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from locpick import ChoiceModel, dgp

sys.path.insert(0, str(Path(__file__).parent))
from test_model_consistency import _make  # noqa: E402


@pytest.fixture(scope="module")
def mnl():
    d = dgp.simulate_mnl(
        n_obs=400, n_alts=4, seed=3, interaction_params={"obs_feature_x_alt_feature": 0.8}
    )
    model = ChoiceModel(d.choice_table, formula="alt_feature + obs_feature_x_alt_feature - 1")
    model.fit()
    return model


def _alt_ids(model):
    ct = model._data
    n, J = model._arrays.n_obs, model._arrays.n_alts
    return ct.to_frame()[ct.alt_id_col].to_numpy().reshape(n, J)[0]


def _counts(sims, model, alt_ids):
    col = model._data.alt_id_col
    return (
        sims.dropna(subset=[col])
        .groupby(["draw", col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=alt_ids, fill_value=0)
    )


def test_capacities_are_respected(mnl):
    alt_ids = _alt_ids(mnl)
    cap = {alt_ids[0]: 40, alt_ids[1]: 60}
    sims = mnl.simulate(n_draws=20, seed=0, capacity=cap)
    counts = _counts(sims, mnl, alt_ids)
    assert (counts[alt_ids[0]] <= 40).all() and (counts[alt_ids[1]] <= 60).all()
    assert (sims["round"] >= 1).all()  # everyone placed: two alternatives are unconstrained
    assert sims["probability"].between(0, 1).all()


def test_mnl_lottery_matches_serial_dictatorship(mnl):
    """With IIA the lottery reproduces choosers taking their best remaining option.

    Reference: random serial dictatorship with explicit Gumbel utilities drawn
    from the model's own probabilities.  One binding capacity, where the two
    schemes coincide in distribution.
    """
    P = mnl.probabilities()
    n, J = P.shape
    alt_ids = _alt_ids(mnl)
    cap_vec = np.array([40.0, np.inf, np.inf, np.inf])
    reps = 200

    rng = np.random.default_rng(2)
    ref = np.zeros(J)
    for _ in range(reps):
        U = np.log(P) + rng.gumbel(size=P.shape)
        room = cap_vec.copy()
        for i in rng.permutation(n):
            j = int(np.argmax(np.where(room > 0, U[i], -np.inf)))
            room[j] -= 1
            ref[j] += 1
    ref /= reps

    counts = _counts(mnl.simulate(n_draws=reps, seed=1, capacity={alt_ids[0]: 40}), mnl, alt_ids)
    se = counts.std(ddof=1).to_numpy() * np.sqrt(2.0 / reps)  # SE of a difference of means
    diff = np.abs(counts.mean().to_numpy() - ref)
    assert counts[alt_ids[0]].eq(40).all()
    assert np.all(diff[1:] < 4 * se[1:] + 1e-9), (counts.mean().to_numpy(), ref)


def test_unplaced_choosers_are_reported(mnl):
    """Total capacity below demand leaves choosers unplaced, with a warning."""
    alt_ids = _alt_ids(mnl)
    cap = {a: 50 for a in alt_ids}  # 200 places for 400 choosers
    with pytest.warns(RuntimeWarning, match="could not be placed"):
        sims = mnl.simulate(seed=0, capacity=cap)
    col = mnl._data.alt_id_col
    unplaced = sims[col].isna()
    assert unplaced.sum() == 200
    assert (sims.loc[unplaced, "round"] == 0).all()
    assert _counts(sims, mnl, alt_ids).iloc[0].eq(50).all()


def test_capacity_validation(mnl):
    with pytest.raises(ValueError, match="non-negative"):
        mnl.simulate(capacity={_alt_ids(mnl)[0]: -1})


@pytest.mark.parametrize("family", ["sar", "sar_mixed", "nested_scl"])
def test_capacity_simulation_non_iia_models(family):
    """Spatial and combined models allocate within capacity, recomputing
    probabilities over each restricted choice set."""
    model = _make(family)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit()
    alt_ids = _alt_ids(model)
    n_obs = model._arrays.n_obs
    # Tight capacities on the first half of the alternatives.
    half = alt_ids[: len(alt_ids) // 2]
    cap = pd.Series(max(1, n_obs // (2 * len(alt_ids))), index=half)
    sims = model.simulate(n_draws=2, seed=0, capacity=cap)
    counts = _counts(sims, model, alt_ids)
    assert (counts[half] <= cap).all().all()
    assert sims[model._data.alt_id_col].notna().all()
