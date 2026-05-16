"""Tests for the v2 MultinomialLogit pipeline (data, spec, estimation, results).

Covers: ChoiceTable, ModelSpec, MultinomialLogit, FitResult, reporting, sampling.
"""

import numpy as np
import pandas as pd

from locpick import (
    ChoiceTable,
    FitDiagnostics,
    FitResult,
    ModelSpec,
    MultinomialLogit,
    format_coefficient_table,
    format_fit_statistics,
    sample_alternatives,
)


def make_toy_data():
    # Simple synthetic data: 3 choosers, 4 alternatives
    choosers = pd.DataFrame({"id": [1, 2, 3]})
    alternatives = pd.DataFrame(
        {
            "alt_id": [10, 11, 12, 13],
            "cost": [1.0, 2.0, 3.0, 4.0],
            "time": [5.0, 4.0, 3.0, 2.0],
        }
    )
    chosen = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "alt_id": [10, 12, 13],
        }
    )
    return choosers, alternatives, chosen


def test_choicetable_and_arrays():
    choosers, alternatives, chosen = make_toy_data()
    ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=3, seed=42)
    arrays = ct.to_arrays(formula="cost + time")
    assert arrays.design_matrix.shape[0] == arrays.n_obs * arrays.n_alts
    assert arrays.design_matrix.shape[1] == 2
    assert arrays.chosen.sum() == arrays.n_obs


def test_multinomiallogit_estimation():
    choosers, alternatives, chosen = make_toy_data()
    ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=4, seed=1)
    model = MultinomialLogit(ct, formula="cost + time")
    result = model.fit()
    assert isinstance(result, FitResult)
    assert result.coefficients.shape[0] == 2
    assert np.isfinite(result.log_likelihood)
    # Check reporting
    txt = FitDiagnostics.summary(result)
    assert "Log-likelihood" in txt
    html = FitDiagnostics.to_html(result)
    assert "<table" in html
    latex = FitDiagnostics.to_latex(result)
    assert "\\begin{tabular}" in latex


def test_formatting_and_statistics():
    choosers, alternatives, chosen = make_toy_data()
    ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=4, seed=2)
    model = MultinomialLogit(ct, formula="cost + time")
    result = model.fit()
    # Coefficient table
    table = format_coefficient_table(result)
    assert "cost" in table and "time" in table
    # Fit statistics
    stats = format_fit_statistics(result)
    assert "Log-likelihood" in stats


def test_sampling_module():
    arr = sample_alternatives(5, 10, 3, seed=123)
    assert arr.shape == (5, 3)
    arr2 = sample_alternatives(2, 5, 2, weights=np.array([1, 2, 3, 4, 5]), seed=1)
    assert arr2.shape == (2, 2)
    assert np.all(arr2 < 5)


def test_modelspec_formula_spec_estimation():
    spec = ModelSpec(formula="cost + time")
    choosers, alternatives, chosen = make_toy_data()
    ct = ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=4, seed=3)
    arrays = ct.to_arrays(spec=spec)
    assert arrays.design_matrix.shape[1] == 2
    # Estimation with formula spec
    model = MultinomialLogit(ct, spec=spec)
    result = model.fit()
    assert isinstance(result, FitResult)
    assert result.coefficients.shape[0] == 2


def test_modelspec_formula_generated_interaction():
    choosers, alternatives, chosen = make_toy_data()
    choosers = choosers.copy()
    choosers["income"] = [1.0, 2.0, 3.0]

    spec = ModelSpec(formula="income_x_cost").with_interaction(
        "income_x_cost",
        "income",
        "cost",
    )

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen,
        sample_size=4,
        seed=4,
    )
    arrays = ct.to_arrays(spec=spec)

    frame = ct.to_frame()
    expected = frame["income"].to_numpy() * frame["cost"].to_numpy()
    assert arrays.param_names == ["income_x_cost"]
    assert np.allclose(arrays.design_matrix[:, 0], expected)


def test_modelspec_formula_generated_interaction_estimation():
    choosers, alternatives, chosen = make_toy_data()
    choosers = choosers.copy()
    choosers["income"] = [1.0, 2.0, 3.0]

    spec = ModelSpec(formula="income_x_cost").with_interaction(
        "income_x_cost",
        "income",
        "cost",
    )

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen,
        sample_size=4,
        seed=5,
    )
    arrays = ct.to_arrays(spec=spec)

    frame = ct.to_frame()
    expected = frame["income"].to_numpy() * frame["cost"].to_numpy()
    assert arrays.param_names == ["income_x_cost"]
    assert np.allclose(arrays.design_matrix[:, 0], expected)


def test_modelspec_generic_scope_matches_raw_column():
    choosers, alternatives, chosen = make_toy_data()
    chosen_series = chosen["alt_id"].rename("chosen_alt")
    spec = ModelSpec().generic("cost")

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_series,
        seed=6,
    )
    arrays = ct.to_arrays(spec=spec)
    frame = ct.to_frame()

    assert arrays.param_names == ["cost"]
    assert np.allclose(arrays.design_matrix[:, 0], frame["cost"].to_numpy())


def test_modelspec_alternative_specific_scope_masks_by_alt_id():
    choosers, alternatives, chosen = make_toy_data()
    chosen_series = chosen["alt_id"].rename("chosen_alt")
    spec = ModelSpec().alternative_specific("cost", reference=10)

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_series,
        seed=7,
    )
    arrays = ct.to_arrays(spec=spec)
    frame = ct.to_frame()

    assert arrays.param_names == ["cost[11]", "cost[12]", "cost[13]"]
    for col, alt_id in enumerate([11, 12, 13]):
        expected = frame["cost"].where(frame["alt_id"] == alt_id, 0.0).to_numpy()
        assert np.allclose(arrays.design_matrix[:, col], expected)


def test_modelspec_grouped_scope_masks_by_alt_groups():
    choosers, alternatives, chosen = make_toy_data()
    chosen_series = chosen["alt_id"].rename("chosen_alt")
    spec = ModelSpec().grouped(
        "time",
        groups={
            "early": [10, 11],
            "late": [12, 13],
        },
    )

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_series,
        seed=8,
    )
    arrays = ct.to_arrays(spec=spec)
    frame = ct.to_frame()

    assert arrays.param_names == ["time[early]", "time[late]"]
    expected_early = frame["time"].where(frame["alt_id"].isin([10, 11]), 0.0).to_numpy()
    expected_late = frame["time"].where(frame["alt_id"].isin([12, 13]), 0.0).to_numpy()
    assert np.allclose(arrays.design_matrix[:, 0], expected_early)
    assert np.allclose(arrays.design_matrix[:, 1], expected_late)
