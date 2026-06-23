"""Correctness tests for the diagnostics module (LR, Wald, Hausman tests)."""

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy import stats

from locpick import ChoiceModel
from locpick.dgp import simulate_mnl, simulate_nested_logit
from locpick.results.diagnostics import (
    hausman_test,
    lr_test,
    wald_test,
)


class TestLikelihoodRatioTest:
    """Tests for the likelihood ratio test."""

    def test_lr_test_statistic_and_df(self):
        """LR statistic = 2*(LL_unrestricted - LL_restricted), df = param diff."""
        dataset = simulate_mnl(n_obs=2000, n_alts=10, seed=42)

        model_restricted = ChoiceModel(dataset.choice_table, "alt_feature - 1")
        result_restricted = model_restricted.fit()

        model_unrestricted = ChoiceModel(dataset.choice_table, "alt_feature + obs_x_alt - 1")
        result_unrestricted = model_unrestricted.fit()

        test = lr_test(result_restricted, result_unrestricted)

        expected_stat = 2 * (result_unrestricted.log_likelihood - result_restricted.log_likelihood)
        npt.assert_allclose(test.statistic, expected_stat, rtol=1e-10)
        assert test.df == result_unrestricted.n_parameters - result_restricted.n_parameters

    def test_lr_test_p_value(self):
        """P-value should come from chi2.sf(statistic, df)."""
        dataset = simulate_mnl(n_obs=2000, n_alts=10, seed=42)

        model_r = ChoiceModel(dataset.choice_table, "alt_feature - 1")
        result_r = model_r.fit()
        model_u = ChoiceModel(dataset.choice_table, "alt_feature + obs_x_alt - 1")
        result_u = model_u.fit()

        test = lr_test(result_r, result_u)
        expected_p = float(stats.chi2.sf(test.statistic, test.df))
        npt.assert_allclose(test.p_value, expected_p, rtol=1e-10)

    def test_lr_test_significant_when_restriction_false(self):
        """LR test should reject when the restricted model omits a relevant variable."""
        dataset = simulate_mnl(n_obs=5000, n_alts=10, seed=42)

        model_r = ChoiceModel(dataset.choice_table, "alt_feature - 1")
        result_r = model_r.fit()
        model_u = ChoiceModel(dataset.choice_table, "alt_feature + obs_x_alt - 1")
        result_u = model_u.fit()

        test = lr_test(result_r, result_u)
        assert test.significant_at_05, (
            f"LR test should be significant: stat={test.statistic:.4f}, p={test.p_value:.6f}"
        )

    def test_lr_test_not_significant_when_same_model(self):
        """LR test with identical models should give statistic ~0 and df=0."""
        dataset = simulate_mnl(n_obs=500, n_alts=5, seed=42)

        model1 = ChoiceModel(dataset.choice_table, "alt_feature - 1")
        result1 = model1.fit()
        model2 = ChoiceModel(dataset.choice_table, "alt_feature - 1")
        result2 = model2.fit()

        test = lr_test(result1, result2)
        npt.assert_allclose(test.statistic, 0.0, atol=1e-6)
        assert test.df == 0

    def test_lr_test_summary(self):
        """Summary should contain key information."""
        dataset = simulate_mnl(n_obs=500, n_alts=5, seed=42)
        model_r = ChoiceModel(dataset.choice_table, "alt_feature - 1")
        result_r = model_r.fit()
        model_u = ChoiceModel(dataset.choice_table, "alt_feature + obs_x_alt - 1")
        result_u = model_u.fit()

        test = lr_test(result_r, result_u)
        summary = test.summary()
        assert "LR statistic" in summary
        assert "P-value" in summary

    def test_lr_test_critical_value(self):
        """Critical value at 5% should match chi2.ppf(0.95, df)."""
        dataset = simulate_mnl(n_obs=500, n_alts=5, seed=42)
        model_r = ChoiceModel(dataset.choice_table, "alt_feature - 1")
        result_r = model_r.fit()
        model_u = ChoiceModel(dataset.choice_table, "alt_feature + obs_x_alt - 1")
        result_u = model_u.fit()

        test = lr_test(result_r, result_u)
        expected_cv = float(stats.chi2.ppf(0.95, test.df))
        npt.assert_allclose(test.critical_value_05, expected_cv, rtol=1e-10)


class TestWaldTest:
    """Tests for the Wald test."""

    def test_wald_test_single_restriction(self):
        """Wald test with a single restriction Rβ = 0."""
        beta = np.array([1.0, 2.0, 3.0])
        V = np.eye(3) * 0.1

        R = np.array([[0, 1, 0]])
        test = wald_test(beta, V, R)

        expected_stat = (2.0**2) / 0.1
        npt.assert_allclose(test.statistic, expected_stat, rtol=1e-10)
        assert test.df == 1

    def test_wald_test_multiple_restrictions(self):
        """Wald test with multiple restrictions."""
        beta = np.array([1.0, 2.0, 3.0])
        V = np.diag([0.1, 0.2, 0.3])

        R = np.array([[1, 0, 0], [0, 0, 1]])
        test = wald_test(beta, V, R)

        assert test.df == 2
        assert test.statistic > 0

    def test_wald_test_zero_restriction_passes(self):
        """Wald test should not reject when the restriction is true (β=0)."""
        beta = np.array([0.01, 0.0, -0.01])
        V = np.eye(3) * 1.0

        R = np.array([[0, 1, 0]])
        test = wald_test(beta, V, R)

        assert not test.significant_at_05, (
            f"Wald should not reject true null: stat={test.statistic:.4f}, p={test.p_value:.6f}"
        )

    def test_wald_test_nonzero_rejection(self):
        """Wald test should reject when the restriction is false (β≠0)."""
        beta = np.array([5.0, 0.0, 0.0])
        V = np.eye(3) * 0.01

        R = np.array([[1, 0, 0]])
        test = wald_test(beta, V, R)

        assert test.significant_at_05, (
            f"Wald should reject false null: stat={test.statistic:.4f}, p={test.p_value:.6f}"
        )

    def test_wald_test_p_value(self):
        """P-value should come from chi2.sf(statistic, df)."""
        beta = np.array([3.0, 1.0])
        V = np.array([[0.5, 0.1], [0.1, 0.3]])
        R = np.array([[1, 0]])

        test = wald_test(beta, V, R)
        expected_p = float(stats.chi2.sf(test.statistic, test.df))
        npt.assert_allclose(test.p_value, expected_p, rtol=1e-10)

    def test_wald_test_summary(self):
        """Summary should contain key information."""
        beta = np.array([1.0])
        V = np.array([[0.1]])
        R = np.array([[1]])
        test = wald_test(beta, V, R)
        summary = test.summary()
        assert "Statistic" in summary
        assert "P-value" in summary

    def test_wald_test_with_correlated_params(self):
        """Wald test with non-diagonal covariance matrix."""
        beta = np.array([2.0, 1.0])
        V = np.array([[0.5, 0.3], [0.3, 0.4]])
        R = np.array([[1, -1]])  # H0: beta[0] - beta[1] = 0

        test = wald_test(beta, V, R)

        r_beta = R @ beta  # 2.0 - 1.0 = 1.0
        middle = R @ V @ R.T  # scalar
        expected_stat = float(r_beta @ np.linalg.inv(middle) @ r_beta)
        npt.assert_allclose(test.statistic, expected_stat, rtol=1e-8)
        assert test.df == 1


class TestHausmanTest:
    """Tests for the Hausman specification test."""

    def test_hausman_test_identical_estimators(self):
        """Hausman test with identical estimators should give statistic ≈ 0."""
        dataset = simulate_mnl(n_obs=1000, n_alts=5, seed=42)
        model1 = ChoiceModel(dataset.choice_table, "alt_feature - 1")
        result1 = model1.fit()
        model2 = ChoiceModel(dataset.choice_table, "alt_feature - 1")
        result2 = model2.fit()

        test = hausman_test(result1, result2)

        npt.assert_allclose(test.statistic, 0.0, atol=1e-6)
        assert len(test.params) > 0

    def test_hausman_test_iia_mnl_vs_nested(self):
        """Hausman test between MNL and nested logit (classic IIA test)."""
        from locpick.models.nested import NestingTree, NestSpec

        dataset = simulate_nested_logit(n_obs=3000, n_alts=4, seed=42)
        nest_tree = NestingTree(
            nests=[
                NestSpec(name="a", alt_ids=[0, 1]),
                NestSpec(name="b", alt_ids=[2, 3]),
            ]
        )

        model_mnl = ChoiceModel(dataset.choice_table, "cost + time - 1")
        result_mnl = model_mnl.fit()

        model_nested = ChoiceModel(dataset.choice_table, "cost + time - 1", nests=nest_tree)
        result_nested = model_nested.fit()

        test = hausman_test(result_mnl, result_nested, params=["cost", "time"])

        assert test.statistic >= 0
        assert test.df > 0
        assert len(test.params) == 2

    def test_hausman_test_psd_warning(self):
        """Hausman test should flag PSD violations when V_c - V_e is not PSD."""

        class MockResult:
            def __init__(self, coefs, cov):
                self.coefficients = pd.Series(coefs, name="coefficient")
                self._cov = cov

            def covariance(self):
                return self._cov

        beta_eff = {"b0": 1.0, "b1": 2.0}
        beta_con = {"b0": 1.5, "b1": 2.5}
        V_eff = np.diag([0.1, 0.1])
        V_con = np.diag([0.05, 0.05])  # V_c - V_e is negative definite

        eff = MockResult(beta_eff, V_eff)
        con = MockResult(beta_con, V_con)

        test = hausman_test(eff, con, params=["b0", "b1"])

        assert test.psd_warning, "Should flag PSD violation"
        assert test.statistic >= 0

    def test_hausman_test_no_common_params_raises(self):
        """Hausman test should raise when no common parameters exist."""

        class MockResult:
            def __init__(self, coefs, cov):
                self.coefficients = pd.Series(coefs, name="coefficient")
                self._cov = cov

            def covariance(self):
                return self._cov

        eff = MockResult({"a": 1.0}, np.array([[0.1]]))
        con = MockResult({"b": 2.0}, np.array([[0.1]]))

        try:
            hausman_test(eff, con)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_hausman_test_summary(self):
        """Summary should contain key information."""
        dataset = simulate_mnl(n_obs=500, n_alts=5, seed=42)
        model1 = ChoiceModel(dataset.choice_table, "alt_feature - 1")
        result1 = model1.fit()
        model2 = ChoiceModel(dataset.choice_table, "alt_feature - 1")
        result2 = model2.fit()

        test = hausman_test(result1, result2)
        summary = test.summary()
        assert "Statistic" in summary
        assert "P-value" in summary

    def test_hausman_test_coef_diff(self):
        """Hausman test should return coefficient difference."""
        dataset = simulate_mnl(n_obs=500, n_alts=5, seed=42)
        model1 = ChoiceModel(dataset.choice_table, "alt_feature - 1")
        result1 = model1.fit()
        model2 = ChoiceModel(dataset.choice_table, "alt_feature - 1")
        result2 = model2.fit()

        test = hausman_test(result1, result2)
        assert test.coef_diff is not None
        assert len(test.coef_diff) == len(test.params)
