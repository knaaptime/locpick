"""Tests for ModelSpec v2 features: fixed, bounded, parameter metadata."""

import numpy as np
import pandas as pd
import pytest

from locpick import ChoiceTable, ModelSpec
from locpick.spec import P, X, ScopedTerm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_toy_data():
    """Create a simple choice dataset for testing."""
    choosers = pd.DataFrame({"id": [1, 2, 3]})
    alternatives = pd.DataFrame(
        {
            "alt_id": [10, 11, 12, 13],
            "cost": [1.0, 2.0, 3.0, 4.0],
            "time": [5.0, 4.0, 3.0, 2.0],
        }
    )
    chosen = pd.DataFrame({"id": [1, 2, 3], "alt_id": [10, 12, 13]})
    return choosers, alternatives, chosen


# ---------------------------------------------------------------------------
# Test: .fixed() method
# ---------------------------------------------------------------------------


class TestModelSpecFixed:
    """Tests for the .fixed() method on ModelSpec."""

    def test_fixed_creates_scoped_term(self):
        """fixed() should create a ScopedTerm with fixed=True."""
        spec = ModelSpec().fixed("distance", value=-1.0)
        assert len(spec.scoped_terms) == 1
        term = spec.scoped_terms[0]
        assert term.variable == "distance"
        assert term.fixed is True
        assert term.null_value == -1.0
        assert term.scope == "generic"

    def test_fixed_default_value(self):
        """fixed() with no value should default to 0.0."""
        spec = ModelSpec().fixed("cost")
        assert spec.scoped_terms[0].null_value == 0.0
        assert spec.scoped_terms[0].fixed is True

    def test_fixed_propagates_to_arrays(self):
        """Fixed parameters should appear in EstimationProblem metadata."""
        from locpick.data import EstimationProblem

        choosers, alternatives, chosen = _make_toy_data()
        spec = ModelSpec().fixed("cost", value=-0.5).generic("time")
        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(chosen["alt_id"].values, index=choosers.index),
            seed=42,
        )
        problem = EstimationProblem.from_choice_table(ct, spec=spec)
        assert problem.param_fixed is not None
        assert problem.param_fixed[0] is True  # cost is fixed
        assert problem.param_fixed[1] is False  # time is not fixed
        assert problem.param_initial is not None
        assert problem.param_initial[0] == -0.5  # cost initial value

    def test_fixed_with_custom_name(self):
        """fixed() should accept a custom parameter name."""
        spec = ModelSpec().fixed("cost", value=-1.0, name="beta_cost")
        assert spec.scoped_terms[0].name == "beta_cost"


# ---------------------------------------------------------------------------
# Test: .bounded() method
# ---------------------------------------------------------------------------


class TestModelSpecBounded:
    """Tests for the .bounded() method on ModelSpec."""

    def test_bounded_creates_scoped_term(self):
        """bounded() should create a ScopedTerm with bounds."""
        spec = ModelSpec().bounded("distance", upper=0.0)
        assert len(spec.scoped_terms) == 1
        term = spec.scoped_terms[0]
        assert term.variable == "distance"
        assert term.bounds == (-1e6, 0.0)

    def test_bounded_propagates_to_arrays(self):
        """Bounded parameters should appear in EstimationProblem metadata."""
        from locpick.data import EstimationProblem

        choosers, alternatives, chosen = _make_toy_data()
        spec = ModelSpec().bounded("cost", upper=0.0).generic("time")
        ct = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(chosen["alt_id"].values, index=choosers.index),
            seed=42,
        )
        problem = EstimationProblem.from_choice_table(ct, spec=spec)
        assert problem.param_bounds is not None
        assert problem.param_bounds[0] == (-1e6, 0.0)  # cost is bounded above
        # time has default bounds, so param_bounds should still be set
        # since at least one bound is non-default

    def test_bounded_with_both_bounds(self):
        """bounded() should accept both lower and upper bounds."""
        spec = ModelSpec().bounded("time", lower=-5.0, upper=5.0)
        assert spec.scoped_terms[0].bounds == (-5.0, 5.0)

    def test_bounded_with_scope(self):
        """bounded() should accept alternative_specific scope."""
        spec = ModelSpec().bounded("cost", upper=0.0, scope="alternative_specific", reference=10)
        term = spec.scoped_terms[0]
        assert term.scope == "alternative_specific"
        assert term.reference == 10
        assert term.bounds == (-1e6, 0.0)

    def test_bounded_with_groups(self):
        """bounded() should accept grouped scope."""
        spec = ModelSpec().bounded(
            "time",
            lower=-2.0,
            groups={"fast": [10, 11], "slow": [12, 13]},
            scope="grouped",
        )
        term = spec.scoped_terms[0]
        assert term.scope == "grouped"
        assert term.groups == {"fast": [10, 11], "slow": [12, 13]}


# ---------------------------------------------------------------------------
# Test: ScopedTerm metadata
# ---------------------------------------------------------------------------


class TestScopedTermMetadata:
    """Tests for ScopedTerm bounds, fixed, and null_value fields."""

    def test_default_scoped_term(self):
        """Default ScopedTerm should have no bounds, not fixed, null_value=0."""
        term = ScopedTerm(variable="cost")
        assert term.bounds is None
        assert term.fixed is False
        assert term.null_value == 0.0

    def test_bounded_scoped_term(self):
        """ScopedTerm with bounds should store them."""
        term = ScopedTerm(variable="cost", bounds=(-10, 0))
        assert term.bounds == (-10, 0)

    def test_fixed_scoped_term(self):
        """ScopedTerm with fixed=True should store it."""
        term = ScopedTerm(variable="cost", fixed=True, null_value=-0.5)
        assert term.fixed is True
        assert term.null_value == -0.5

    def test_frozen_dataclass(self):
        """ScopedTerm should be frozen (immutable)."""
        term = ScopedTerm(variable="cost")
        with pytest.raises(AttributeError):
            term.variable = "time"


# ---------------------------------------------------------------------------
# Test: ChoiceArrays parameter metadata
# ---------------------------------------------------------------------------


class TestEstimationProblemMetadata:
    """Tests for EstimationProblem param_bounds, param_fixed, param_initial."""

    def test_param_bounds_none_by_default(self):
        """EstimationProblem should have None param_bounds by default."""
        from locpick.data import ChoiceArrays, EstimationProblem

        arrays = ChoiceArrays(
            design_matrix=np.zeros((10, 2)),
            chosen=np.zeros((2, 5)),
            n_obs=2,
            n_alts=5,
            param_names=["x1", "x2"],
        )
        problem = EstimationProblem(arrays=arrays)
        assert problem.param_bounds is None
        assert problem.param_fixed is None
        assert problem.param_initial is None

    def test_param_metadata_roundtrip(self):
        """EstimationProblem should preserve param metadata."""
        from locpick.data import ChoiceArrays, EstimationProblem

        arrays = ChoiceArrays(
            design_matrix=np.zeros((10, 2)),
            chosen=np.zeros((2, 5)),
            n_obs=2,
            n_alts=5,
            param_names=["x1", "x2"],
        )
        problem = EstimationProblem(
            arrays=arrays,
            param_bounds=[(-10, 0), (-1e6, 1e6)],
            param_fixed=[True, False],
            param_initial=[-0.5, 0.0],
        )
        assert problem.param_bounds == [(-10, 0), (-1e6, 1e6)]
        assert problem.param_fixed == [True, False]
        assert problem.param_initial == [-0.5, 0.0]


def test_modelspec_utility_path_supported_for_advanced_specs():
    from locpick.data import EstimationProblem

    choosers, alternatives, chosen = _make_toy_data()
    utility = P("beta_cost", null_value=-0.5, bounds=(-5.0, 0.0), holdfast=True) * X(
        "cost"
    ) + P("beta_time") * X("time")

    spec = ModelSpec(utility=utility)
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives=pd.Series(chosen["alt_id"].values, index=choosers.index),
        seed=42,
    )

    arrays = ct.to_arrays(spec=spec)
    frame = ct.to_frame()

    assert arrays.param_names == ["beta_cost", "beta_time"]
    assert np.allclose(arrays.design_matrix[:, 0], frame["cost"].to_numpy())
    assert np.allclose(arrays.design_matrix[:, 1], frame["time"].to_numpy())

    problem = EstimationProblem.from_choice_table(ct, spec=spec)
    assert problem.param_bounds == [(-5.0, 0.0), (-1e6, 1e6)]
    assert problem.param_fixed == [True, False]
    assert problem.param_initial == [-0.5, 0.0]
