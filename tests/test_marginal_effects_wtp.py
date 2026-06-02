"""Tests for marginal effects and willingness-to-pay (WTP) computations."""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from locpick import MNL, ChoiceTable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_dataset(n_obs=200, n_alts=4, seed=42):
    """Create a simple choice dataset for testing."""
    rng = np.random.default_rng(seed)
    choosers = pd.DataFrame(
        {
            "obsval": rng.standard_normal(n_obs),
        },
        index=pd.Index(np.arange(n_obs), name="oid"),
    )

    alternatives = pd.DataFrame(
        {
            "altval": rng.standard_normal(n_alts),
            "cost": rng.uniform(1, 10, n_alts),
            "time": rng.uniform(5, 60, n_alts),
        },
        index=pd.Index(np.arange(n_alts), name="aid"),
    )

    choices = rng.choice(np.arange(n_alts), size=n_obs)

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives=pd.Series(choices, index=choosers.index),
    )
    return ct, choosers, alternatives, choices


# ---------------------------------------------------------------------------
# Marginal Effects
# ---------------------------------------------------------------------------


class TestMarginalEffects:
    """Tests for marginal effect computation on MNL."""

    def test_marginal_effect_shape(self):
        """Marginal effects should have same length as observations * alternatives."""
        ct, _, _, _ = _make_simple_dataset(n_obs=50, n_alts=5)
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        me = model.marginal_effect(variable="cost")
        assert len(me) == ct.n_observations * ct.n_alternatives

    def test_marginal_effect_sign(self):
        """For a negative coefficient, direct ME should be negative."""
        ct, _, _, _ = _make_simple_dataset(n_obs=100, n_alts=4)
        model = MNL(ct, formula="cost + time - 1")
        result = model.fit()

        me = model.marginal_effect(variable="cost")
        # Cost coefficient is typically negative; ME should be negative
        beta_cost = result.coefficients["cost"]
        if beta_cost < 0:
            assert me.mean() < 0
        else:
            assert me.mean() > 0

    def test_cross_marginal_effect_sign(self):
        """Cross ME should have opposite sign to direct ME."""
        ct, _, _, _ = _make_simple_dataset(n_obs=100, n_alts=4)
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        me = model.marginal_effect(variable="cost")
        cross_me = model.cross_marginal_effect(variable="cost")

        # Direct and cross should have opposite signs
        assert np.sign(me.mean()) == -np.sign(cross_me.mean())

    def test_marginal_effect_vs_elasticity(self):
        """Elasticity = ME * x (for direct effects)."""
        ct, _, _, _ = _make_simple_dataset(n_obs=50, n_alts=4)
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        me = model.marginal_effect(variable="cost")
        elast = model.elasticity(variable="cost")
        df = ct.to_frame()
        x = df["cost"].values

        # Direct elasticity = ME * x
        reconstructed = me.values * x

        npt.assert_allclose(reconstructed, elast.values, rtol=1e-5)

    def test_marginal_effect_on_new_data(self):
        """ME should work on out-of-sample data."""
        ct, _, _, _ = _make_simple_dataset(n_obs=100, n_alts=4)
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        # New data
        rng = np.random.default_rng(99)
        choosers_new = pd.DataFrame(
            {"obsval": rng.standard_normal(20)},
            index=pd.Index(np.arange(20), name="oid"),
        )
        alternatives_new = pd.DataFrame(
            {
                "altval": rng.standard_normal(4),
                "cost": rng.uniform(1, 10, 4),
                "time": rng.uniform(5, 60, 4),
            },
            index=pd.Index(np.arange(4), name="aid"),
        )
        ct_new = ChoiceTable.from_tables(
            choosers_new,
            alternatives_new,
            chosen_alternatives=pd.Series(rng.choice(4, size=20), index=choosers_new.index),
        )

        me = model.marginal_effect(data=ct_new, variable="cost")
        assert len(me) == ct_new.n_observations * ct_new.n_alternatives
        assert np.all(np.isfinite(me))

    def test_average_marginal_effect_aggregations(self):
        """AME helpers should aggregate per-obs ME consistently."""
        ct, _, _, _ = _make_simple_dataset(n_obs=80, n_alts=4)
        model = MNL(ct, formula="cost + time - 1")
        model.fit()

        me = model.marginal_effect(variable="cost")
        ame_alt = model.average_marginal_effect("cost")
        ame_obs = model.average_marginal_effect("cost", by="obs")
        ame_all = model.average_marginal_effect("cost", by="overall")

        assert len(ame_alt) == ct.n_alternatives
        assert len(ame_obs) == ct.n_observations
        npt.assert_allclose(ame_all, me.mean())
        npt.assert_allclose(ame_alt.values, me.groupby(level=ame_alt.index.name).mean().values)

        # cross/elasticity counterparts
        cme = model.cross_marginal_effect(variable="cost")
        el = model.elasticity(variable="cost")
        npt.assert_allclose(
            model.average_cross_marginal_effect("cost").values,
            cme.groupby(level=ame_alt.index.name).mean().values,
        )
        npt.assert_allclose(
            model.average_elasticity("cost").values,
            el.groupby(level=ame_alt.index.name).mean().values,
        )


# ---------------------------------------------------------------------------
# WTP / VOT
# ---------------------------------------------------------------------------


class TestWTP:
    """Tests for willingness-to-pay computation."""

    def test_wtp_basic(self):
        """WTP should compute -beta_time / beta_cost."""
        ct, _, _, _ = _make_simple_dataset(n_obs=200, n_alts=4)
        model = MNL(ct, formula="cost + time - 1")
        result = model.fit()

        wtp = result.wtp(numerator="time", denominator="cost")
        beta_time = result.coefficients["time"]
        beta_cost = result.coefficients["cost"]
        expected = -beta_time / beta_cost

        npt.assert_allclose(wtp["wtp"], expected, rtol=1e-10)

    def test_wtp_has_standard_error(self):
        """WTP should include a standard error."""
        ct, _, _, _ = _make_simple_dataset(n_obs=200, n_alts=4)
        model = MNL(ct, formula="cost + time - 1")
        result = model.fit()

        wtp = result.wtp(numerator="time", denominator="cost")
        assert "se" in wtp.index
        assert np.isfinite(wtp["se"])
        assert wtp["se"] >= 0

    def test_wtp_has_t_stat_and_p_value(self):
        """WTP should include t-statistic and p-value."""
        ct, _, _, _ = _make_simple_dataset(n_obs=200, n_alts=4)
        model = MNL(ct, formula="cost + time - 1")
        result = model.fit()

        wtp = result.wtp(numerator="time", denominator="cost")
        assert "t_stat" in wtp.index
        assert "p_value" in wtp.index
        assert np.isfinite(wtp["t_stat"])
        assert 0 <= wtp["p_value"] <= 1

    def test_wtp_invalid_numerator_raises(self):
        """WTP should raise for invalid numerator."""
        ct, _, _, _ = _make_simple_dataset(n_obs=50, n_alts=4)
        model = MNL(ct, formula="cost + time - 1")
        result = model.fit()

        with pytest.raises(ValueError, match="Numerator 'income' not found"):
            result.wtp(numerator="income", denominator="cost")

    def test_wtp_invalid_denominator_raises(self):
        """WTP should raise for invalid denominator."""
        ct, _, _, _ = _make_simple_dataset(n_obs=50, n_alts=4)
        model = MNL(ct, formula="cost + time - 1")
        result = model.fit()

        with pytest.raises(ValueError, match="Denominator 'rent' not found"):
            result.wtp(numerator="time", denominator="rent")

    def test_vot_is_wtp_alias(self):
        """VOT should be equivalent to WTP(time, cost)."""
        ct, _, _, _ = _make_simple_dataset(n_obs=200, n_alts=4)
        model = MNL(ct, formula="cost + time - 1")
        result = model.fit()

        vot = result.vot(time_var="time", cost_var="cost")
        wtp = result.wtp(numerator="time", denominator="cost")

        npt.assert_allclose(vot["wtp"], wtp["wtp"], rtol=1e-10)
        npt.assert_allclose(vot["se"], wtp["se"], rtol=1e-10)

    def test_wtp_with_custom_denominator(self):
        """WTP should work with any monetary denominator."""
        ct, choosers, alternatives, choices = _make_simple_dataset(n_obs=100, n_alts=4)
        alternatives["rent"] = alternatives["cost"] * 100  # monthly rent

        ct2 = ChoiceTable.from_tables(
            choosers,
            alternatives,
            chosen_alternatives=pd.Series(choices, index=choosers.index),
        )
        model = MNL(ct2, formula="rent + time - 1")
        result = model.fit()

        wtp = result.wtp(numerator="time", denominator="rent")
        beta_time = result.coefficients["time"]
        beta_rent = result.coefficients["rent"]
        expected = -beta_time / beta_rent

        npt.assert_allclose(wtp["wtp"], expected, rtol=1e-10)
