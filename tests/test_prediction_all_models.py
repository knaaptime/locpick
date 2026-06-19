"""Tests for prediction/analysis methods across all model types.

Covers: utilities(), simulate(), elasticity(), cross_elasticity(),
covariance_robust(), covariance_clustered(),
std_errors_robust(), std_errors_clustered()
for MultinomialLogit, NestedLogit, SpatiallyCorrelatedLogit,
MixedLogit, and MixedSpatiallyCorrelatedLogit.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from locpick import ChoiceModel
from locpick.dgp import (
    simulate_mixed_logit,
    simulate_mnl,
    simulate_mscl,
    simulate_nested_logit,
    simulate_scl,
)
from locpick.models.mixed import ParamDistribution

# ---------------------------------------------------------------------------
# MNL Tests
# ---------------------------------------------------------------------------


class TestMNL:
    @pytest.fixture(autouse=True)
    def setup(self):
        dataset = simulate_mnl(n_obs=500, n_alts=4, seed=42)
        self.model = ChoiceModel(
            dataset.choice_table,
            formula="alt_feature + obs_x_alt",
        )
        self.result = self.model.fit()

    def test_utilities(self):
        V = self.model.utilities()
        n_obs = self.model._arrays.n_obs
        n_alts = self.model._arrays.n_alts
        assert V.shape == (n_obs, n_alts)
        assert np.all(np.isfinite(V))

    def test_utilities_with_data(self):
        dataset = simulate_mnl(n_obs=500, n_alts=4, seed=42)
        V = self.model.utilities(data=dataset.choice_table)
        n_obs = self.model._arrays.n_obs
        n_alts = self.model._arrays.n_alts
        assert V.shape == (n_obs, n_alts)

    def test_simulate(self):
        sim = self.model.simulate(n_draws=2, seed=42)
        assert isinstance(sim, pd.DataFrame)
        assert "draw" in sim.columns
        n_obs = self.model._arrays.n_obs
        assert len(sim) == n_obs * 2

    def test_elasticity(self):
        elast = self.model.elasticity(variable="alt_feature")
        assert isinstance(elast, pd.Series)
        n_obs = self.model._arrays.n_obs
        n_alts = self.model._arrays.n_alts
        assert len(elast) == n_obs * n_alts

    def test_cross_elasticity(self):
        cross_elast = self.model.cross_elasticity(variable="alt_feature")
        assert isinstance(cross_elast, pd.Series)

    def test_covariance_robust(self):
        cov = self.model.covariance_robust()
        n_params = len(self.result.coefficients)
        assert cov.shape == (n_params, n_params)

    def test_covariance_clustered(self):
        n_obs = self.model._arrays.n_obs
        groups = np.random.randint(0, 5, size=n_obs)
        cov = self.model.covariance_clustered(groups=groups)
        n_params = len(self.result.coefficients)
        assert cov.shape == (n_params, n_params)

    def test_std_errors_robust(self):
        se = self.model.std_errors_robust()
        assert isinstance(se, pd.Series)
        assert len(se) == len(self.result.coefficients)

    def test_std_errors_clustered(self):
        n_obs = self.model._arrays.n_obs
        groups = np.random.randint(0, 5, size=n_obs)
        se = self.model.std_errors_clustered(groups=groups)
        assert isinstance(se, pd.Series)
        assert len(se) == len(self.result.coefficients)

    def test_observation_scores_cached(self):
        arrays = self.model._arrays
        scores1 = self.model._observation_scores(arrays)
        scores2 = self.model._observation_scores(arrays)
        npt.assert_array_equal(scores1, scores2)

    def test_hessian_inverse_cached(self):
        h1 = self.model._get_hessian_inverse()
        h2 = self.model._get_hessian_inverse()
        npt.assert_array_equal(h1, h2)


# ---------------------------------------------------------------------------
# Nested Logit Tests
# ---------------------------------------------------------------------------


class TestNestedLogit:
    @pytest.fixture(autouse=True)
    def setup(self):
        dataset = simulate_nested_logit(n_obs=500, n_alts=4, seed=42)
        self.model = ChoiceModel(
            dataset.choice_table,
            formula="cost + time + income_x_cost + income_x_time",
            nests=dataset.nests,
        )
        self.result = self.model.fit()

    def test_utilities(self):
        V = self.model.utilities()
        n_obs = self.model._arrays.n_obs
        n_alts = self.model._arrays.n_alts
        assert V.shape == (n_obs, n_alts)
        assert np.all(np.isfinite(V))

    def test_simulate(self):
        sim = self.model.simulate(n_draws=2, seed=42)
        assert isinstance(sim, pd.DataFrame)
        assert "draw" in sim.columns

    def test_elasticity(self):
        elast = self.model.elasticity(variable="cost")
        assert isinstance(elast, pd.Series)

    def test_cross_elasticity(self):
        cross_elast = self.model.cross_elasticity(variable="cost")
        assert isinstance(cross_elast, pd.Series)

    def test_covariance_robust(self):
        cov = self.model.covariance_robust()
        n_params = len(self.result.coefficients)
        assert cov.shape == (n_params, n_params)

    def test_covariance_clustered(self):
        n_obs = self.model._arrays.n_obs
        groups = np.random.randint(0, 5, size=n_obs)
        cov = self.model.covariance_clustered(groups=groups)
        n_params = len(self.result.coefficients)
        assert cov.shape == (n_params, n_params)

    def test_std_errors_robust(self):
        se = self.model.std_errors_robust()
        assert isinstance(se, pd.Series)

    def test_std_errors_clustered(self):
        n_obs = self.model._arrays.n_obs
        groups = np.random.randint(0, 5, size=n_obs)
        se = self.model.std_errors_clustered(groups=groups)
        assert isinstance(se, pd.Series)

    def test_observation_scores(self):
        arrays = self.model._arrays
        scores = self.model._observation_scores(arrays)
        n_params = len(self.result.coefficients)
        n_obs = self.model._arrays.n_obs
        assert scores.shape == (n_obs, n_params)

    def test_observation_scores_cached(self):
        arrays = self.model._arrays
        scores1 = self.model._observation_scores(arrays)
        scores2 = self.model._observation_scores(arrays)
        npt.assert_array_equal(scores1, scores2)


# ---------------------------------------------------------------------------
# SCL Tests
# ---------------------------------------------------------------------------


class TestSCL:
    @pytest.fixture(autouse=True)
    def setup(self):
        dataset = simulate_scl(n_obs=500, n_alts=6, seed=42)
        self.model = ChoiceModel(
            dataset.choice_table,
            formula="cost + time + income_x_cost",
            graph=dataset.adjacency,
        )
        self.result = self.model.fit()

    def test_utilities(self):
        V = self.model.utilities()
        n_obs = self.model._arrays.n_obs
        n_alts = self.model._arrays.n_alts
        assert V.shape == (n_obs, n_alts)
        assert np.all(np.isfinite(V))

    def test_simulate(self):
        sim = self.model.simulate(n_draws=2, seed=42)
        assert isinstance(sim, pd.DataFrame)
        assert "draw" in sim.columns

    def test_elasticity(self):
        with pytest.raises(NotImplementedError):
            self.model.elasticity(variable="cost")

    def test_cross_elasticity(self):
        with pytest.raises(NotImplementedError):
            self.model.cross_elasticity(variable="cost")

    def test_covariance_robust(self):
        cov = self.model.covariance_robust()
        n_params = len(self.result.coefficients)
        assert cov.shape == (n_params, n_params)

    def test_covariance_clustered(self):
        n_obs = self.model._arrays.n_obs
        groups = np.random.randint(0, 5, size=n_obs)
        cov = self.model.covariance_clustered(groups=groups)
        n_params = len(self.result.coefficients)
        assert cov.shape == (n_params, n_params)

    def test_std_errors_robust(self):
        se = self.model.std_errors_robust()
        assert isinstance(se, pd.Series)

    def test_std_errors_clustered(self):
        n_obs = self.model._arrays.n_obs
        groups = np.random.randint(0, 5, size=n_obs)
        se = self.model.std_errors_clustered(groups=groups)
        assert isinstance(se, pd.Series)

    def test_observation_scores(self):
        arrays = self.model._arrays
        scores = self.model._observation_scores(arrays)
        n_params = len(self.result.coefficients)
        n_obs = self.model._arrays.n_obs
        assert scores.shape == (n_obs, n_params)


# ---------------------------------------------------------------------------
# Mixed Logit Tests
# ---------------------------------------------------------------------------


class TestMixedLogit:
    @pytest.fixture(autouse=True)
    def setup(self):
        dataset = simulate_mixed_logit(n_obs=500, n_alts=4, seed=42)
        self.model = ChoiceModel(
            dataset.choice_table,
            formula="cost + time + income_x_cost",
            random_params={"time": ParamDistribution("normal", "time")},
            n_draws=50,
            seed=42,
        )
        self.result = self.model.fit()

    def test_utilities(self):
        V = self.model.utilities()
        n_obs = self.model._arrays.n_obs
        n_alts = self.model._arrays.n_alts
        assert V.shape == (n_obs, n_alts)
        assert np.all(np.isfinite(V))

    def test_simulate(self):
        sim = self.model.simulate(n_draws=2, seed=42)
        assert isinstance(sim, pd.DataFrame)
        assert "draw" in sim.columns

    def test_elasticity(self):
        elast = self.model.elasticity(variable="cost")
        assert isinstance(elast, pd.Series)

    def test_cross_elasticity(self):
        cross_elast = self.model.cross_elasticity(variable="cost")
        assert isinstance(cross_elast, pd.Series)

    def test_covariance_robust(self):
        cov = self.model.covariance_robust()
        n_params = len(self.result.coefficients)
        assert cov.shape == (n_params, n_params)

    def test_covariance_clustered(self):
        n_obs = self.model._arrays.n_obs
        groups = np.random.randint(0, 5, size=n_obs)
        cov = self.model.covariance_clustered(groups=groups)
        n_params = len(self.result.coefficients)
        assert cov.shape == (n_params, n_params)

    def test_std_errors_robust(self):
        se = self.model.std_errors_robust()
        assert isinstance(se, pd.Series)

    def test_std_errors_clustered(self):
        n_obs = self.model._arrays.n_obs
        groups = np.random.randint(0, 5, size=n_obs)
        se = self.model.std_errors_clustered(groups=groups)
        assert isinstance(se, pd.Series)

    def test_observation_scores(self):
        arrays = self.model._arrays
        scores = self.model._observation_scores(arrays)
        n_params = len(self.result.coefficients)
        n_obs = self.model._arrays.n_obs
        assert scores.shape == (n_obs, n_params)


# ---------------------------------------------------------------------------
# MSCL Tests
# ---------------------------------------------------------------------------


class TestMSCL:
    @pytest.fixture(autouse=True)
    def setup(self):
        dataset = simulate_mscl(n_obs=500, n_alts=6, seed=42)
        self.model = ChoiceModel(
            dataset.choice_table,
            formula="cost + time + income_x_cost",
            graph=dataset.adjacency,
            random_params={"time": ParamDistribution("normal", "time")},
            n_draws=50,
        )
        self.result = self.model.fit()

    def test_utilities(self):
        V = self.model.utilities()
        n_obs = self.model._arrays.n_obs
        n_alts = self.model._arrays.n_alts
        assert V.shape == (n_obs, n_alts)
        assert np.all(np.isfinite(V))

    def test_simulate(self):
        sim = self.model.simulate(n_draws=2, seed=42)
        assert isinstance(sim, pd.DataFrame)
        assert "draw" in sim.columns

    def test_elasticity(self):
        with pytest.raises(NotImplementedError):
            self.model.elasticity(variable="cost")

    def test_cross_elasticity(self):
        with pytest.raises(NotImplementedError):
            self.model.cross_elasticity(variable="cost")


# ---------------------------------------------------------------------------
# Cache invalidation tests
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    """Test that caches are properly cleared on re-estimation."""

    def test_mnl_cache_cleared_on_reestimate(self):
        dataset = simulate_mnl(n_obs=500, n_alts=4, seed=42)
        model = ChoiceModel(dataset.choice_table, formula="alt_feature + obs_x_alt")
        model.fit()

        # Populate caches
        V1 = model.utilities()

        # Re-estimate
        model.fit()

        # Caches should have been cleared
        assert model._utilities_cache is None

        # New values should be computed fresh
        V2 = model.utilities()
        # Values should be the same (same data, same model)
        npt.assert_array_almost_equal(V1, V2)

    def test_nested_cache_cleared_on_reestimate(self):
        dataset = simulate_nested_logit(n_obs=500, n_alts=4, seed=42)
        model = ChoiceModel(
            dataset.choice_table,
            formula="cost + time + income_x_cost + income_x_time",
            nests=dataset.nests,
        )
        model.fit()

        # Populate caches
        V1 = model.utilities()

        # Re-estimate
        model.fit()

        # Caches should have been cleared
        assert model._utilities_cache is None

        # New values should be computed fresh
        V2 = model.utilities()
        npt.assert_array_almost_equal(V1, V2)


# ---------------------------------------------------------------------------
# Protocol conformance tests
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Test that all models conform to the ChoiceModel protocol."""

    def test_mnl_is_choice_model(self):
        from locpick.models.base import ChoiceModelProtocol
        from locpick.models.choice_model import ChoiceModel

        dataset = simulate_mnl(n_obs=500, n_alts=4, seed=42)
        model = ChoiceModel(dataset.choice_table, formula="alt_feature + obs_x_alt")
        assert isinstance(model, ChoiceModelProtocol)

    def test_nested_is_choice_model(self):
        from locpick.models.base import ChoiceModelProtocol
        from locpick.models.choice_model import ChoiceModel

        dataset = simulate_nested_logit(n_obs=500, n_alts=4, seed=42)
        model = ChoiceModel(
            dataset.choice_table,
            formula="cost + time + income_x_cost + income_x_time",
            nests=dataset.nests,
        )
        assert isinstance(model, ChoiceModelProtocol)

    def test_scl_is_choice_model(self):
        from locpick.models.base import ChoiceModelProtocol
        from locpick.models.choice_model import ChoiceModel

        dataset = simulate_scl(n_obs=500, n_alts=6, seed=42)
        model = ChoiceModel(
            dataset.choice_table,
            formula="cost + time + income_x_cost",
            graph=dataset.adjacency,
        )
        assert isinstance(model, ChoiceModelProtocol)

    def test_mixed_is_choice_model(self):
        from locpick.models.base import ChoiceModelProtocol
        from locpick.models.choice_model import ChoiceModel

        dataset = simulate_mixed_logit(n_obs=500, n_alts=4, seed=42)
        model = ChoiceModel(
            dataset.choice_table,
            formula="cost + time + income_x_cost",
            random_params={"time": ParamDistribution("normal", "time")},
            n_draws=50,
            seed=42,
        )
        assert isinstance(model, ChoiceModelProtocol)

    def test_mscl_is_choice_model(self):
        from locpick.models.base import ChoiceModelProtocol
        from locpick.models.choice_model import ChoiceModel

        dataset = simulate_mscl(n_obs=500, n_alts=6, seed=42)
        model = ChoiceModel(
            dataset.choice_table,
            formula="cost + time + income_x_cost",
            graph=dataset.adjacency,
            random_params={"time": ParamDistribution("normal", "time")},
            n_draws=50,
        )
        assert isinstance(model, ChoiceModelProtocol)
