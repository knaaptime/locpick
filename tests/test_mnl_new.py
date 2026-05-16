"""
These are tests for the refactored locpick MNL codebase.

"""

import numpy as np
import pandas as pd
import pytest

from locpick import ChoiceTable, MultinomialLogit


@pytest.fixture
def obs():
    d1 = {
        "oid": np.arange(100),
        "obsval": np.random.random(100),
        "choice": np.random.choice(np.arange(5), size=100),
    }
    return pd.DataFrame(d1).set_index("oid")


@pytest.fixture
def alts():
    d2 = {"aid": np.arange(5), "altval": np.random.random(5)}
    return pd.DataFrame(d2).set_index("aid")


def test_mnl(obs, alts):
    """
    Confirm that MNL estimation runs, using the native estimator.

    """
    formula = "obsval + altval - 1"
    ct = ChoiceTable.from_tables(obs, alts, chosen_alternatives="choice")
    m = MultinomialLogit(ct, formula=formula)
    r = m.fit()
    assert len(r.coefficients) == 2


def test_mnl_estimation(obs, alts):
    """
    Confirm that MNL returns finite coefficient estimates.

    """
    formula = "obsval + altval - 1"
    ct = ChoiceTable.from_tables(obs, alts, chosen_alternatives="choice")
    result = MultinomialLogit(ct, formula=formula).fit()
    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(result.coefficients.to_numpy()).all()


def test_mnl_prediction(obs, alts):
    """
    Confirm predicted probabilities are well-formed.

    """
    ct = ChoiceTable.from_tables(obs, alts, chosen_alternatives="choice", sample_size=5)
    m = MultinomialLogit(ct, formula="obsval + altval - 1")
    results = m.fit()

    probs = m.probabilities(ct)
    prob_sums = probs.sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-8)
