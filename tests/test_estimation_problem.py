"""Tests for the EstimationProblem layer.

Covers: EstimationProblem dataclass, from_choice_table factory, properties,
and integration with MultinomialLogit.
"""

import numpy as np
import pandas as pd
import pytest

from locpick import (
    ChoiceTable,
    EstimationProblem,
    FitResult,
    ModelSpec,
    MultinomialLogit,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def make_toy_data():
    """Simple synthetic data: 3 choosers, 4 alternatives."""
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


def make_choice_table():
    choosers, alternatives, chosen = make_toy_data()
    return ChoiceTable.from_tables(choosers, alternatives, chosen, sample_size=4, seed=1)


# ------------------------------------------------------------------
# Test: EstimationProblem dataclass basics
# ------------------------------------------------------------------


class TestEstimationProblemDataclass:
    """Test direct construction of EstimationProblem."""

    def test_basic_construction(self):
        """EstimationProblem can be constructed from ChoiceArrays."""
        ct = make_choice_table()
        arrays = ct.to_arrays(formula="cost + time")
        problem = EstimationProblem(arrays=arrays)
        assert problem.n_params == 2
        assert problem.n_obs == arrays.n_obs
        assert problem.n_alts == arrays.n_alts
        assert problem.model_type == "mnl"

    def test_default_initial_values(self):
        """Initial values default to zeros."""
        ct = make_choice_table()
        arrays = ct.to_arrays(formula="cost + time")
        problem = EstimationProblem(arrays=arrays)
        np.testing.assert_array_equal(problem.initial_values, np.zeros(2))

    def test_custom_initial_values(self):
        """Custom initial values are respected."""
        ct = make_choice_table()
        arrays = ct.to_arrays(formula="cost + time")
        problem = EstimationProblem(
            arrays=arrays,
            param_initial=[0.5, -0.3],
        )
        np.testing.assert_array_almost_equal(problem.initial_values, [0.5, -0.3])

    def test_bounds_property(self):
        """Bounds property returns param_bounds or None."""
        ct = make_choice_table()
        arrays = ct.to_arrays(formula="cost + time")
        problem_none = EstimationProblem(arrays=arrays)
        assert problem_none.bounds is None

        problem_bounds = EstimationProblem(
            arrays=arrays,
            param_bounds=[(None, None), (0.0, None)],
        )
        assert problem_bounds.bounds == [(None, None), (0.0, None)]

    def test_fixed_mask_property(self):
        """fixed_mask returns boolean array or None."""
        ct = make_choice_table()
        arrays = ct.to_arrays(formula="cost + time")
        problem_none = EstimationProblem(arrays=arrays)
        assert problem_none.fixed_mask is None

        problem_fixed = EstimationProblem(
            arrays=arrays,
            param_fixed=[True, False],
        )
        mask = problem_fixed.fixed_mask
        assert mask is not None
        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, [True, False])

    def test_repr(self):
        """repr includes key dimensions."""
        ct = make_choice_table()
        arrays = ct.to_arrays(formula="cost + time")
        problem = EstimationProblem(arrays=arrays)
        r = repr(problem)
        assert "n_obs=" in r
        assert "n_alts=" in r
        assert "n_params=" in r
        assert "model_type=" in r


# ------------------------------------------------------------------
# Test: from_choice_table factory
# ------------------------------------------------------------------


class TestFromChoiceTable:
    """Test EstimationProblem.from_choice_table factory."""

    def test_from_formula(self):
        """Create EstimationProblem from ChoiceTable + formula."""
        ct = make_choice_table()
        problem = EstimationProblem.from_choice_table(ct, formula="cost + time")
        assert problem.n_params == 2
        assert problem.design_matrix.shape[1] == 2
        assert problem.chosen.sum() == problem.n_obs

    def test_from_spec(self):
        """Create EstimationProblem from ChoiceTable + ModelSpec."""
        ct = make_choice_table()
        spec = ModelSpec(formula="cost + time")
        problem = EstimationProblem.from_choice_table(ct, spec=spec)
        assert problem.n_params == 2

    def test_formula_and_spec_mutually_exclusive(self):
        """Providing both formula and spec raises ValueError."""
        ct = make_choice_table()
        spec = ModelSpec(formula="cost + time")
        with pytest.raises(ValueError, match="both"):
            EstimationProblem.from_choice_table(ct, spec=spec, formula="cost + time")

    def test_param_names_inherited(self):
        """Param names come from ChoiceArrays."""
        ct = make_choice_table()
        problem = EstimationProblem.from_choice_table(ct, formula="cost + time")
        assert problem.param_names == ["cost", "time"]

    def test_backend_default(self):
        """Backend defaults to 'auto'."""
        ct = make_choice_table()
        problem = EstimationProblem.from_choice_table(ct, formula="cost + time")
        assert problem.backend == "auto"

    def test_solver_name_default(self):
        """Solver name defaults to 'lbfgs'."""
        ct = make_choice_table()
        problem = EstimationProblem.from_choice_table(ct, formula="cost + time")
        assert problem.solver_name == "lbfgs"


# ------------------------------------------------------------------
# Test: Integration with MultinomialLogit
# ------------------------------------------------------------------


class TestMultinomialLogitWithEstimationProblem:
    """Test that MultinomialLogit can estimate from an EstimationProblem."""

    def test_estimate_from_problem(self):
        """MultinomialLogit.fit() works with EstimationProblem."""
        ct = make_choice_table()
        problem = EstimationProblem.from_choice_table(ct, formula="cost + time")
        model = MultinomialLogit(data=ct, problem=problem)
        result = model.fit()
        assert isinstance(result, FitResult)
        assert result.coefficients.shape[0] == 2
        assert np.isfinite(result.log_likelihood)

    def test_problem_matches_legacy_result(self):
        """EstimationProblem path produces same results as legacy path."""
        ct = make_choice_table()

        # Legacy path
        model_legacy = MultinomialLogit(ct, formula="cost + time")
        result_legacy = model_legacy.fit()

        # Problem path
        problem = EstimationProblem.from_choice_table(ct, formula="cost + time")
        model_problem = MultinomialLogit(data=ct, problem=problem)
        result_problem = model_problem.fit()

        # Coefficients should be very close (same data, same solver)
        np.testing.assert_array_almost_equal(
            result_legacy.coefficients,
            result_problem.coefficients,
            decimal=4,
        )
        # Log-likelihoods should match
        assert abs(result_legacy.log_likelihood - result_problem.log_likelihood) < 1e-6

    def test_problem_with_initial_values(self):
        """EstimationProblem with custom initial values converges."""
        ct = make_choice_table()
        problem = EstimationProblem.from_choice_table(
            ct,
            formula="cost + time",
        )
        # Override initial values
        problem = EstimationProblem(
            arrays=problem.arrays,
            param_names=problem.param_names,
            param_initial=[0.1, -0.1],
        )
        model = MultinomialLogit(data=ct, problem=problem)
        result = model.fit()
        assert isinstance(result, FitResult)
        assert np.isfinite(result.log_likelihood)

    def test_problem_ignores_formula_and_spec(self):
        """When problem is provided, formula and spec are ignored."""
        ct = make_choice_table()
        problem = EstimationProblem.from_choice_table(ct, formula="cost + time")

        # Pass problem + formula — formula should be ignored
        model = MultinomialLogit(data=ct, problem=problem, formula="ignored ~ x")
        result = model.fit()
        # Should still have 2 params (from problem), not whatever "ignored ~ x" would give
        assert result.coefficients.shape[0] == 2

    def test_problem_requires_data(self):
        """MultinomialLogit with problem still needs data argument."""
        ct = make_choice_table()
        problem = EstimationProblem.from_choice_table(ct, formula="cost + time")
        # data is still required (it's positional)
        model = MultinomialLogit(data=ct, problem=problem)
        assert model._problem is problem


# ------------------------------------------------------------------
# Test: Solver bounds and fixed_mask support
# ------------------------------------------------------------------


class TestSolverBoundsAndFixed:
    """Test that the solver correctly handles bounds and fixed_mask."""

    def test_fixed_parameter_stays_at_initial(self):
        """A fixed parameter should remain at its initial value."""
        ct = make_choice_table()
        problem = EstimationProblem.from_choice_table(ct, formula="cost + time")
        # Fix the 'cost' parameter at 0.5
        problem = EstimationProblem(
            arrays=problem.arrays,
            param_names=problem.param_names,
            param_initial=[0.5, 0.0],
            param_fixed=[True, False],
        )
        model = MultinomialLogit(data=ct, problem=problem)
        result = model.fit()
        # The first parameter (cost) should be close to 0.5 (fixed)
        assert abs(result.coefficients.iloc[0] - 0.5) < 1e-6

    def test_bounds_passed_to_solver(self):
        """Bounds are passed through to the solver."""
        ct = make_choice_table()
        problem = EstimationProblem.from_choice_table(ct, formula="cost + time")
        # Set bounds: cost in [-5, 5], time in [-10, 10]
        problem = EstimationProblem(
            arrays=problem.arrays,
            param_names=problem.param_names,
            param_bounds=[(-5.0, 5.0), (-10.0, 10.0)],
        )
        model = MultinomialLogit(data=ct, problem=problem)
        result = model.fit()
        # Should converge within bounds
        assert -5.0 <= result.coefficients.iloc[0] <= 5.0
        assert -10.0 <= result.coefficients.iloc[1] <= 10.0

    def test_fixed_parameter_with_optimistix(self):
        """Optimistix should honor fixed_mask by holding fixed params at x0."""
        pytest.importorskip("optimistix")

        ct = make_choice_table()
        problem = EstimationProblem.from_choice_table(ct, formula="cost + time")
        problem = EstimationProblem(
            arrays=problem.arrays,
            param_names=problem.param_names,
            param_initial=[0.5, 0.0],
            param_fixed=[True, False],
        )

        model = MultinomialLogit(data=ct, problem=problem, solver="optimistix")
        result = model.fit()

        assert abs(result.coefficients.iloc[0] - 0.5) < 1e-6
        assert np.isfinite(result.coefficients.iloc[1])
