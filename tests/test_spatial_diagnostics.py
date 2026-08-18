"""Correctness, size, and power tests for the spatial and IIA diagnostics.

The tests are layered deliberately:

1. **Analytic correctness** --- the O(J) multinomial quadratic-form identity,
   the exact null moments of the residual cross-product, and the information
   block are each checked against an independent brute-force computation.
   These are deterministic and tight (1e-10 or better).
2. **Invariance** --- properties the statistics must satisfy by construction
   (score invariance to the PML normalisation, mean-zero scores).
3. **Size and power** --- Monte Carlo checks with deliberately wide bands,
   because a rejection-rate estimate from a few dozen replications is itself
   noisy (SE ~= 0.02 at 100 reps).

The stochastic tests carry generous tolerances on purpose: a test that fails
one run in twenty is worse than no test.
"""

import numpy as np
import numpy.testing as npt
import pytest

from locpick.dgp import simulate_nested_logit, simulate_sar_mnl
from locpick.models import ChoiceModel
from locpick.models._spatial_weights import resolve_spatial_weights
from locpick.results.diagnostics import (
    _mnl_diagnostic_inputs,
    _mnl_info_blocks,
    iia_test,
    lm_error_test,
    lm_lag_test,
)

FORMULA = "alt_attr + obs_x_alt_attr - 1"
INTER = {"obs_x_alt_attr": 0.8}


def _fit_mnl(n_obs=4000, n_alts=20, rho=0.0, seed=0):
    """Simulate SAR-MNL data (rho=0 gives plain MNL) and fit the null MNL."""
    ds = simulate_sar_mnl(n_obs=n_obs, n_alts=n_alts, rho=rho, seed=seed, interaction_params=INTER)
    W = resolve_spatial_weights(ds.W, n_alts, row_standardize=True)[1]
    model = ChoiceModel(ds.choice_table, formula=FORMULA)
    model.fit()
    return model, W, ds


# ---------------------------------------------------------------------------
# 1. Analytic correctness
# ---------------------------------------------------------------------------


class TestQuadraticFormIdentity:
    """The O(J) contraction must equal the explicit J x J multinomial form."""

    def test_matches_brute_force_sigma(self):
        rng = np.random.default_rng(0)
        n_obs, n_alts, k = 40, 7, 3
        P = rng.random((n_obs, n_alts))
        P /= P.sum(axis=1, keepdims=True)
        X = rng.standard_normal((n_obs, n_alts, k))
        a = rng.standard_normal((n_obs, n_alts))

        I_aa, I_ab, I_bb = _mnl_info_blocks(P, X, a)

        # Brute force: form Sigma_i = diag(P_i) - P_i P_i' explicitly.
        bf_aa, bf_ab, bf_bb = 0.0, np.zeros(k), np.zeros((k, k))
        for i in range(n_obs):
            S = np.diag(P[i]) - np.outer(P[i], P[i])
            bf_aa += a[i] @ S @ a[i]
            bf_ab += a[i] @ S @ X[i]
            bf_bb += X[i].T @ S @ X[i]

        npt.assert_allclose(I_aa, bf_aa, rtol=1e-12)
        npt.assert_allclose(I_ab, bf_ab, rtol=1e-12)
        npt.assert_allclose(I_bb, bf_bb, rtol=1e-12)

    def test_information_block_matches_fitted_model(self):
        """I_bb must reproduce the MNL information matrix from the estimator.

        This is an independent check: the diagnostic builds the information
        from probabilities and the design, while the model builds it from the
        log-likelihood Hessian.
        """
        model, W, _ = _fit_mnl(n_obs=3000, n_alts=15, seed=11)
        _, P, psi0, X, Wd = _mnl_diagnostic_inputs(model, W)
        _, _, I_bb = _mnl_info_blocks(P, X, psi0 @ Wd.T)

        I_model = np.linalg.inv(np.asarray(model._result.covariance(), dtype=float))
        npt.assert_allclose(I_bb, I_model, rtol=1e-6)


class TestResidualCrossProductMoments:
    """Exact null mean and variance of the residual cross-product."""

    @staticmethod
    def _enumerate(P_i, W):
        """Exact moments of Q = e'We for one multinomial draw, by enumeration."""
        J = len(P_i)
        Q = np.empty(J)
        for c in range(J):
            e = -P_i.copy()
            e[c] += 1.0
            Q[c] = e @ W @ e
        mean = P_i @ Q
        return mean, P_i @ Q**2 - mean**2

    def test_closed_form_moments_match_enumeration(self):
        rng = np.random.default_rng(3)
        J = 9
        W = np.zeros((J, J))
        for j in range(J):
            W[j, (j - 1) % J] = 0.5
            W[j, (j + 1) % J] = 0.5

        for _ in range(5):
            P_i = rng.random(J)
            P_i /= P_i.sum()
            mean_e, var_e = self._enumerate(P_i, W)

            # Closed forms used by lm_error_test.
            g = (W + W.T) @ P_i
            mean_c = -(P_i @ W @ P_i)
            var_c = P_i @ g**2 - (P_i @ g) ** 2

            npt.assert_allclose(mean_c, mean_e, rtol=1e-12, atol=1e-15)
            npt.assert_allclose(var_c, var_e, rtol=1e-12, atol=1e-15)

    def test_uncentred_statistic_is_biased(self):
        """The naive cross-product has a strictly negative null mean.

        Guards the centring: without it the test rejects almost surely.
        """
        rng = np.random.default_rng(5)
        J = 8
        W = np.zeros((J, J))
        for j in range(J):
            W[j, (j - 1) % J] = 0.5
            W[j, (j + 1) % J] = 0.5
        P_i = rng.random(J)
        P_i /= P_i.sum()
        mean, _ = self._enumerate(P_i, W)
        assert mean < -1e-6, "residual cross-product must be negatively biased"


# ---------------------------------------------------------------------------
# 2. Invariance properties
# ---------------------------------------------------------------------------


class TestScoreProperties:
    def test_lag_score_is_mean_zero_under_null(self):
        """E[s_rho] = 0 when there is no spatial dependence."""
        scores = []
        for seed in range(25):
            model, W, _ = _fit_mnl(n_obs=2000, n_alts=12, rho=0.0, seed=500 + seed)
            scores.append(lm_lag_test(model, W).score)
        scores = np.asarray(scores)
        t = scores.mean() / (scores.std(ddof=1) / np.sqrt(len(scores)))
        assert abs(t) < 3.5, f"lag score not mean-zero under null (t={t:.2f})"

    def test_error_score_is_mean_zero_after_centring(self):
        scores = []
        for seed in range(25):
            model, W, _ = _fit_mnl(n_obs=2000, n_alts=12, rho=0.0, seed=700 + seed)
            scores.append(lm_error_test(model, W).score)
        scores = np.asarray(scores)
        t = scores.mean() / (scores.std(ddof=1) / np.sqrt(len(scores)))
        assert abs(t) < 3.5, f"centred error score not mean-zero (t={t:.2f})"

    def test_lag_score_invariant_to_pml_normalization(self):
        """One test serves both SAR specifications.

        The PML divides filtered utilities by ``D = diag((I - rho W)^-1)``, but
        ``d/drho D`` at ``rho = 0`` is ``diag(W) = 0`` for a zero-diagonal
        ``W``.  The rho-derivative of the filtered utility is therefore
        ``W psi_0`` under both the reduced form and the PML, so the score --- and
        hence the LM statistic --- is identical.
        """
        n_alts = 15
        model, W, _ = _fit_mnl(n_obs=2000, n_alts=n_alts, rho=0.0, seed=99)
        _, _, psi0, _, Wd = _mnl_diagnostic_inputs(model, W)

        h = 1e-6

        def filtered(rho, normalize):
            S = np.linalg.inv(np.eye(n_alts) - rho * Wd)
            out = psi0 @ S.T
            return out / np.diag(S)[None, :] if normalize else out

        for normalize in (False, True):
            fd = (filtered(h, normalize) - filtered(-h, normalize)) / (2 * h)
            npt.assert_allclose(fd, psi0 @ Wd.T, rtol=1e-6, atol=1e-8)

    def test_rejects_wrong_shaped_W(self):
        model, W, _ = _fit_mnl(n_obs=500, n_alts=10, seed=1)
        with pytest.raises(ValueError, match="connecting alternatives"):
            lm_lag_test(model, np.eye(3))

    def test_error_test_requires_zero_diagonal(self):
        model, W, _ = _fit_mnl(n_obs=500, n_alts=10, seed=1)
        Wd = np.asarray(W.todense())
        np.fill_diagonal(Wd, 0.5)
        with pytest.raises(ValueError, match="zero diagonal"):
            lm_error_test(model, Wd)

    def test_requires_fitted_model(self):
        ds = simulate_sar_mnl(n_obs=200, n_alts=8, rho=0.0, seed=1, interaction_params=INTER)
        W = resolve_spatial_weights(ds.W, 8, row_standardize=True)[1]
        unfitted = ChoiceModel(ds.choice_table, formula=FORMULA)
        with pytest.raises(RuntimeError, match="estimated"):
            lm_lag_test(unfitted, W)


# ---------------------------------------------------------------------------
# 3. Size and power
# ---------------------------------------------------------------------------


class TestSizeAndPower:
    """Monte Carlo behaviour, with wide bands (rejection rates are noisy)."""

    @staticmethod
    def _rejection_rate(test_fn, rho, reps, n_obs=2500, n_alts=15, seed0=0):
        rejects = 0
        for r in range(reps):
            model, W, _ = _fit_mnl(n_obs=n_obs, n_alts=n_alts, rho=rho, seed=seed0 + r)
            rejects += test_fn(model, W).p_value < 0.05
        return rejects / reps

    def test_lm_lag_size_under_null(self):
        rate = self._rejection_rate(lm_lag_test, rho=0.0, reps=40, seed0=1000)
        assert rate <= 0.25, f"LM-lag over-rejects under the null (size={rate:.3f})"

    def test_lm_lag_power_under_spatial_dependence(self):
        rate = self._rejection_rate(lm_lag_test, rho=0.5, reps=20, seed0=2000)
        assert rate >= 0.70, f"LM-lag lacks power against rho=0.5 (power={rate:.3f})"

    def test_lm_error_size_under_null(self):
        rate = self._rejection_rate(lm_error_test, rho=0.0, reps=40, seed0=3000)
        assert rate <= 0.25, f"residual test over-rejects under the null (size={rate:.3f})"

    def test_lm_lag_more_powerful_than_error_against_lag(self):
        """The lag score targets the SAR alternative directly.

        Both statistics respond to spatial structure, but ``e'W psi_0`` is the
        score for rho while ``e'We`` is only a residual-correlation moment, so
        the former should dominate when the truth is a spatial lag.
        """
        lag_stats, err_stats = [], []
        for r in range(12):
            model, W, _ = _fit_mnl(n_obs=2500, n_alts=15, rho=0.5, seed=4000 + r)
            lag_stats.append(lm_lag_test(model, W).statistic)
            err_stats.append(lm_error_test(model, W).statistic)
        assert np.median(lag_stats) > np.median(err_stats)


# ---------------------------------------------------------------------------
# 4. IIA
# ---------------------------------------------------------------------------


class TestIIA:
    def test_no_rejection_when_iia_holds(self):
        """MNL data satisfies IIA, so the restricted fit should agree."""
        model, _, _ = _fit_mnl(n_obs=5000, n_alts=12, rho=0.0, seed=3)
        result = iia_test(model, subset=list(range(8)))
        assert result.p_value > 0.01
        assert result.df >= 1
        assert set(result.params) == {"alt_attr", "obs_x_alt_attr"}

    def test_size_under_iia(self):
        rejects = 0
        reps = 25
        for r in range(reps):
            model, _, _ = _fit_mnl(n_obs=3000, n_alts=10, rho=0.0, seed=6000 + r)
            rejects += iia_test(model, subset=list(range(7))).p_value < 0.05
        rate = rejects / reps
        assert rate <= 0.28, f"IIA test over-rejects when IIA holds (size={rate:.3f})"

    def test_power_against_nested_logit(self):
        """Nested logit violates IIA, so the test should reject.

        Restricting to a single nest is the sharpest case.  Conditional on
        choosing inside nest *m*, nested logit *is* a logit --- but over
        ``V_j / lambda_m``, so the restricted fit estimates ``beta / lambda_m``
        rather than ``beta``.  With ``lambda = 0.4`` that is a 2.5x rescaling,
        which the Hausman comparison against the full-choice-set estimate
        detects readily.
        """
        rejects, reps = 0, 10
        for r in range(reps):
            ds = simulate_nested_logit(
                n_obs=6000,
                n_alts=8,
                nest_lambdas={"nest_a": 0.4, "nest_b": 0.4},
                seed=7000 + r,
            )
            model = ChoiceModel(ds.choice_table, formula="cost + time")
            model.fit()
            try:
                rejects += iia_test(model, subset=[0, 1, 2, 3]).p_value < 0.05
            except (ValueError, np.linalg.LinAlgError):
                pass
        assert rejects >= 7, f"IIA test lacks power against nested logit ({rejects}/{reps})"

    def test_guards(self):
        model, _, _ = _fit_mnl(n_obs=1000, n_alts=10, rho=0.0, seed=2)
        with pytest.raises(ValueError, match="strict subset"):
            iia_test(model, subset=list(range(10)))
        with pytest.raises(ValueError, match="at least two"):
            iia_test(model, subset=[0])
        with pytest.raises(ValueError, match="unknown alternatives"):
            iia_test(model, subset=[0, 999])

    def test_leaves_original_model_untouched(self):
        model, _, _ = _fit_mnl(n_obs=2000, n_alts=10, rho=0.0, seed=4)
        before = model._result.coefficients.copy()
        iia_test(model, subset=list(range(6)))
        npt.assert_allclose(model._result.coefficients.values, before.values, rtol=0)
