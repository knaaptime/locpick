"""Likelihood ratio test and other statistical tests for choice models.

This module provides the ``LikelihoodRatioTest`` class for comparing nested
choice models, as well as convenience functions for common hypothesis tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
from scipy.linalg import cho_factor, cho_solve

if TYPE_CHECKING:
    from .fit_result import FitResult


@dataclass
class LikelihoodRatioTest:
    """Likelihood ratio test for comparing nested models.

    Parameters
    ----------
    statistic : float
        The LR test statistic: 2 * (LL_unrestricted - LL_restricted).
    df : int
        Degrees of freedom (difference in number of parameters).
    restricted_model : FitResult, optional
        The restricted (null) model.
    unrestricted_model : FitResult, optional
        The unrestricted (alternative) model.
    """

    statistic: float
    df: int
    restricted_model: "FitResult | None" = None
    unrestricted_model: "FitResult | None" = None

    @property
    def p_value(self) -> float:
        """P-value from the chi-squared distribution."""
        return float(stats.chi2.sf(self.statistic, self.df))

    @property
    def critical_value_05(self) -> float:
        """Critical value at the 5% significance level."""
        return float(stats.chi2.ppf(0.95, self.df))

    @property
    def significant_at_05(self) -> bool:
        """Whether the test is significant at the 5% level."""
        return self.p_value < 0.05

    def summary(self) -> str:
        """Return a formatted summary of the test."""
        lines = [
            "Likelihood Ratio Test",
            "=" * 40,
            f"LR statistic:  {self.statistic:.4f}",
            f"DF:           {self.df}",
            f"P-value:       {self.p_value:.6f}",
            f"Critical (5%): {self.critical_value_05:.4f}",
            f"Significant:   {'Yes' if self.significant_at_05 else 'No'}",
        ]
        if self.restricted_model is not None:
            lines.append(f"Restricted LL:  {self.restricted_model.log_likelihood:.4f}")
        if self.unrestricted_model is not None:
            lines.append(f"Unrestricted LL: {self.unrestricted_model.log_likelihood:.4f}")
        lines.append("=" * 40)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LikelihoodRatioTest(statistic={self.statistic:.4f}, "
            f"df={self.df}, p_value={self.p_value:.6f})"
        )


def lr_test(restricted: "FitResult", unrestricted: "FitResult") -> LikelihoodRatioTest:
    """Perform a likelihood-ratio test between nested models."""
    return LikelihoodRatioTest(
        statistic=2 * (unrestricted.log_likelihood - restricted.log_likelihood),
        df=unrestricted.n_parameters - restricted.n_parameters,
        restricted_model=restricted,
        unrestricted_model=unrestricted,
    )


def wald_test(
    coefficients: np.ndarray,
    variance_covariance: np.ndarray,
    r_matrix: np.ndarray,
) -> "WaldTest":
    """Perform a Wald test for linear restrictions.

    Tests the null hypothesis Rβ = 0, where R is the restriction matrix.

    Parameters
    ----------
    coefficients : np.ndarray
        Estimated parameter vector.
    variance_covariance : np.ndarray
        Variance-covariance matrix of the estimates.
    r_matrix : np.ndarray
        Restriction matrix (q x k), where q is the number of restrictions
        and k is the number of parameters.

    Returns
    -------
    WaldTest
    """
    r_beta = r_matrix @ coefficients
    middle = r_matrix @ variance_covariance @ r_matrix.T
    # middle is PSD (sandwich of PSD covariance), so use Cholesky
    try:
        statistic = float(r_beta @ cho_solve(cho_factor(middle), r_beta))
    except np.linalg.LinAlgError:
        # Fall back to general inverse if not PD
        statistic = float(r_beta.T @ np.linalg.inv(middle) @ r_beta)
    df = r_matrix.shape[0]
    p_value = float(stats.chi2.sf(statistic, df))
    return WaldTest(statistic=statistic, df=df, p_value=p_value)


@dataclass
class WaldTest:
    """Wald test result.

    Parameters
    ----------
    statistic : float
        The Wald test statistic.
    df : int
        Degrees of freedom.
    p_value : float
        P-value from the chi-squared distribution.
    """

    statistic: float
    df: int
    p_value: float

    @property
    def significant_at_05(self) -> bool:
        """Whether the test is significant at the 5% level."""
        return self.p_value < 0.05

    def summary(self) -> str:
        """Return a formatted summary of the test."""
        lines = [
            "Wald Test",
            "=" * 40,
            f"Statistic:  {self.statistic:.4f}",
            f"DF:         {self.df}",
            f"P-value:    {self.p_value:.6f}",
            f"Significant: {'Yes' if self.significant_at_05 else 'No'}",
            "=" * 40,
        ]
        return "\n".join(lines)


@dataclass
class HausmanTest:
    """Hausman specification test result.

    Tests the null hypothesis that the efficient estimator is consistent by
    comparing it against an estimator that is consistent under both the null
    and alternative.  Under the null,

    .. math::

        H = (\\hat\\beta_c - \\hat\\beta_e)'
            (V_c - V_e)^{+}
            (\\hat\\beta_c - \\hat\\beta_e) \\;\\sim\\; \\chi^2_q

    where :math:`V_c - V_e` is positive semidefinite under the null and
    :math:`q` is its rank (equal to the number of compared parameters in the
    well-identified case).

    Parameters
    ----------
    statistic : float
        The Hausman test statistic.
    df : int
        Degrees of freedom (rank of the covariance difference).
    p_value : float
        P-value from the chi-squared distribution.
    params : list of str
        Names of the parameters compared.
    coef_diff : np.ndarray
        Difference :math:`\\hat\\beta_c - \\hat\\beta_e` for the compared parameters.
    psd_warning : bool
        ``True`` if the covariance difference had negative eigenvalues
        (suggesting violations of Hausman's regularity conditions or finite
        sample noise); the statistic is computed via the Moore-Penrose
        pseudoinverse after clipping negative eigenvalues.
    """

    statistic: float
    df: int
    p_value: float
    params: list[str]
    coef_diff: np.ndarray
    psd_warning: bool = False

    @property
    def critical_value_05(self) -> float:
        """Critical value at the 5% significance level."""
        return float(stats.chi2.ppf(0.95, self.df))

    @property
    def significant_at_05(self) -> bool:
        """Whether the test is significant at the 5% level."""
        return self.p_value < 0.05

    def summary(self) -> str:
        """Return a formatted summary of the test."""
        lines = [
            "Hausman Test",
            "=" * 40,
            f"Statistic:    {self.statistic:.4f}",
            f"DF:           {self.df}",
            f"P-value:      {self.p_value:.6f}",
            f"Critical (5%): {self.critical_value_05:.4f}",
            f"Significant:  {'Yes' if self.significant_at_05 else 'No'}",
            f"Parameters:   {', '.join(self.params)}",
        ]
        if self.psd_warning:
            lines.append(
                "Warning:      V_c - V_e was not PSD; pseudoinverse used.",
            )
        lines.append("=" * 40)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"HausmanTest(statistic={self.statistic:.4f}, "
            f"df={self.df}, p_value={self.p_value:.6f})"
        )


def hausman_test(
    efficient: "FitResult",
    consistent: "FitResult",
    params: list[str] | None = None,
    tol: float = 1e-10,
) -> HausmanTest:
    """Perform a Hausman specification test between two estimators.

    Under the null hypothesis, ``efficient`` is consistent and asymptotically
    efficient while ``consistent`` is consistent but inefficient.  Under the
    alternative, only ``consistent`` is consistent.

    Parameters
    ----------
    efficient : FitResult
        The estimator that is efficient under the null hypothesis (e.g. MNL
        when testing the IIA assumption against a nested logit).
    consistent : FitResult
        The estimator that is consistent under both the null and the
        alternative (e.g. nested logit, or MNL fit on a restricted choice set).
    params : list of str, optional
        Names of parameters to compare.  Defaults to the intersection of the
        two coefficient indices.
    tol : float, optional
        Tolerance for treating eigenvalues of :math:`V_c - V_e` as zero when
        computing the pseudoinverse rank.  Default ``1e-10``.

    Returns
    -------
    HausmanTest
    """
    eff_coefs = efficient.coefficients
    con_coefs = consistent.coefficients

    if params is None:
        params = [name for name in eff_coefs.index if name in con_coefs.index]
    if not params:
        raise ValueError(
            "No common parameters between the two FitResults to compare.",
        )

    missing_eff = [p for p in params if p not in eff_coefs.index]
    missing_con = [p for p in params if p not in con_coefs.index]
    if missing_eff or missing_con:
        raise ValueError(
            f"Parameters missing: efficient={missing_eff}, consistent={missing_con}",
        )

    idx_eff = [eff_coefs.index.get_loc(p) for p in params]
    idx_con = [con_coefs.index.get_loc(p) for p in params]

    diff = np.asarray(con_coefs.values[idx_con] - eff_coefs.values[idx_eff], dtype=np.float64)

    cov_eff_full = efficient.covariance()
    cov_con_full = consistent.covariance()
    V_e = np.asarray(cov_eff_full, dtype=np.float64)[np.ix_(idx_eff, idx_eff)]
    V_c = np.asarray(cov_con_full, dtype=np.float64)[np.ix_(idx_con, idx_con)]

    delta = V_c - V_e
    # Symmetrize to suppress numerical asymmetry
    delta = 0.5 * (delta + delta.T)

    eigvals, eigvecs = np.linalg.eigh(delta)
    max_eig = float(eigvals.max()) if eigvals.size else 0.0
    psd_warning = bool(eigvals.min() < -max(tol, tol * abs(max_eig)))

    # Moore-Penrose pseudoinverse via eigendecomposition; clip negatives to 0.
    eig_tol = max(tol, tol * abs(max_eig))
    inv_eigs = np.where(eigvals > eig_tol, 1.0 / np.where(eigvals > 0, eigvals, 1.0), 0.0)
    delta_pinv = (eigvecs * inv_eigs) @ eigvecs.T

    statistic = float(diff @ delta_pinv @ diff)
    df = int(np.sum(eigvals > eig_tol))
    if df == 0:
        df = len(params)
    p_value = float(stats.chi2.sf(statistic, df))

    return HausmanTest(
        statistic=statistic,
        df=df,
        p_value=p_value,
        params=list(params),
        coef_diff=diff,
        psd_warning=psd_warning,
    )


# ---------------------------------------------------------------------------
# Spatial score / moment diagnostics for MNL
# ---------------------------------------------------------------------------


@dataclass
class SpatialDiagnostic:
    """Result of a spatial dependence diagnostic for a fitted MNL.

    Parameters
    ----------
    test_name : str
        Human-readable test name.
    statistic : float
        Chi-squared statistic (both tests are asymptotically :math:`\\chi^2_1`).
    df : int
        Degrees of freedom.
    p_value : float
        Upper-tail p-value.
    score : float
        The centred, unstandardised score or moment.
    variance : float
        Variance used to standardise ``score``, after adjusting for the
        estimation of :math:`\\beta`.
    test_type : str
        ``"lag"`` or ``"error"``.
    details : dict
        Test-specific extras.
    """

    test_name: str
    statistic: float
    df: int
    p_value: float
    score: float
    variance: float
    test_type: str
    details: dict

    @property
    def significant_at_05(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        star = "*" if self.significant_at_05 else " "
        return (
            f"{self.test_name}: chi2({self.df}) = {self.statistic:.4f}, "
            f"p = {self.p_value:.4f}{star}"
        )

    def __repr__(self) -> str:
        return f"<{self.summary()}>"


def _mnl_diagnostic_inputs(model, W):
    """Extract ``(e, P, psi0, X, W_dense)`` from a fitted MNL and weights matrix.

    ``X`` is returned in ``(n_obs, n_alts, k)`` form and ``W`` densified, since
    the diagnostics below contract over both alternative axes.
    """
    import scipy.sparse as sp

    if getattr(model, "_result", None) is None:
        raise RuntimeError("Model must be estimated before computing diagnostics.")
    arrays = getattr(model, "_arrays", None)
    if arrays is None:
        raise RuntimeError("Fitted model does not expose estimation arrays.")

    P = np.asarray(model.probabilities(), dtype=np.float64)
    n_obs, n_alts = P.shape
    Y = np.asarray(arrays.chosen, dtype=np.float64).reshape(n_obs, n_alts)
    X = np.asarray(arrays.design_matrix, dtype=np.float64).reshape(n_obs, n_alts, -1)

    beta = np.asarray(model._result.coefficients.values, dtype=np.float64)[: X.shape[2]]
    psi0 = X @ beta

    if sp.issparse(W):
        W_dense = np.asarray(W.todense(), dtype=np.float64)
    else:
        W_dense = np.asarray(W, dtype=np.float64)
    if W_dense.shape != (n_alts, n_alts):
        raise ValueError(
            f"W must be ({n_alts}, {n_alts}) connecting alternatives; got {W_dense.shape}."
        )
    return Y - P, P, psi0, X, W_dense


def _mnl_info_blocks(P, X, a):
    """Fisher information blocks under the multinomial covariance.

    For each observation the multinomial covariance is
    ``Sigma_i = diag(P_i) - P_i P_i'``, so for any vectors ``u, v``

    .. math::
        u' \\Sigma_i v = \\sum_j P_{ij} u_j v_j
                        - \\Big(\\sum_j P_{ij}u_j\\Big)\\Big(\\sum_j P_{ij}v_j\\Big),

    which is ``O(J)`` and never forms the ``J x J`` matrix.

    Returns ``(I_aa, I_ab, I_bb)`` for the scalar direction ``a`` and the
    design ``X``.
    """
    Pa = np.einsum("nj,nj->n", P, a)
    PX = np.einsum("nj,njk->nk", P, X)

    I_aa = float(np.einsum("nj,nj,nj->", P, a, a) - np.einsum("n,n->", Pa, Pa))
    I_ab = np.einsum("nj,nj,njk->k", P, a, X) - np.einsum("n,nk->k", Pa, PX)
    I_bb = np.einsum("nj,njk,njl->kl", P, X, X) - np.einsum("nk,nl->kl", PX, PX)
    return I_aa, I_ab, I_bb


def _partial_variance(I_aa, I_ab, I_bb):
    """Bera--Yoon partial information ``I_aa - I_ab I_bb^{-1} I_ba``."""
    solved = np.linalg.solve(I_bb, I_ab)
    return float(I_aa - I_ab @ solved)


def lm_lag_test(model, W) -> SpatialDiagnostic:
    """Score (LM) test for spatial lag dependence among alternatives.

    Tests :math:`H_0: \\rho = 0` in the SAR-MNL model
    :math:`\\psi = (I - \\rho W)^{-1} X\\beta`, using only the null MNL fit ---
    no spatial model is estimated.

    Because :math:`\\partial\\psi/\\partial\\rho|_{\\rho=0} = W\\psi_0` and the
    MNL score with respect to :math:`\\psi_{ij}` is :math:`y_{ij} - P_{ij}`,

    .. math::
        s_\\rho = \\sum_i\\sum_j (y_{ij}-P_{ij})\\,(W\\hat\\psi_0)_{ij},

    the inner product of choice residuals with the spatially lagged null
    utilities.  The variance uses the Bera--Yoon partial information correcting
    for the estimation of :math:`\\beta`.

    Parameters
    ----------
    model : ChoiceModel
        A *fitted* MNL (the null model, without spatial structure).
    W : array-like or scipy.sparse
        ``(n_alts, n_alts)`` row-standardised weights connecting alternatives.

    Returns
    -------
    SpatialDiagnostic

    Notes
    -----
    The statistic is unchanged if the Smirnov variance normalisation is
    applied, because ``d/drho diag((I - rho W)^{-1})`` vanishes at ``rho = 0``
    for a zero-diagonal ``W``.  One test therefore serves both SAR
    specifications.

    Being first-order in :math:`\\rho`, it also has power against SLX-type
    *local* alternatives and cannot distinguish local from global spillover.

    The score is invariant to the MNL location normalisation only when ``W`` is
    row-standardised: residuals sum to zero across alternatives, so a constant
    shift of ``W psi_0`` cancels, and ``W1 = 1`` guarantees that a shift of
    ``psi_0`` produces exactly such a constant.
    """
    e, P, psi0, X, Wd = _mnl_diagnostic_inputs(model, W)

    a = psi0 @ Wd.T  # (W psi_0)_ij = sum_k W_jk psi_ik
    score = float(np.einsum("nj,nj->", e, a))

    I_aa, I_ab, I_bb = _mnl_info_blocks(P, X, a)
    var = _partial_variance(I_aa, I_ab, I_bb)
    if not np.isfinite(var) or var <= 0:
        raise ValueError("Non-positive partial information; rho is not identified here.")

    stat = score**2 / var
    return SpatialDiagnostic(
        test_name="LM-Lag (spatial spillover)",
        statistic=stat,
        df=1,
        p_value=float(stats.chi2.sf(stat, 1)),
        score=score,
        variance=var,
        test_type="lag",
        details={"unadjusted_information": I_aa},
    )


def lm_error_test(model, W) -> SpatialDiagnostic:
    """Test for spatial autocorrelation in MNL choice residuals.

    Uses the Moran-style cross-product :math:`T = \\tilde e' W \\tilde e` of
    choice residuals with their own spatial lag, standardised by its **exact**
    null moments.

    Parameters
    ----------
    model : ChoiceModel
        A *fitted* MNL.
    W : array-like or scipy.sparse
        ``(n_alts, n_alts)`` weights connecting alternatives, zero diagonal.

    Returns
    -------
    SpatialDiagnostic

    Notes
    -----
    **This is a moment test, not a score test against a specified spatial error
    model.**  ``locpick`` has no SEM-MNL likelihood to differentiate, and in a
    random utility framework spatially correlated errors do not yield a logit
    at all --- they yield a GEV or probit.  The statistic is the natural
    discrete-choice analogue of Moran's I on regression residuals, and it plays
    the role LM-error plays in the Gaussian case, but it is not derived from an
    alternative likelihood.  Interpret a rejection as evidence of residual
    spatial structure, not as an estimate of a spatial error parameter.

    The centring is essential and is what a naive :math:`\\tilde e' W\\tilde e`
    statistic omits.  Choice residuals are correlated *within* an observation
    (``Cov(e_ij, e_ik) = -P_ij P_ik`` for ``j != k``), so the raw cross-product
    has a strictly negative mean under the null and an uncentred test rejects
    with probability approaching one.

    Writing ``g_i = (W + W')P_i``, a single multinomial draw gives the exact
    identity ``T_i(c) = P_i'W P_i - g_{i,c}``, whence

    .. math::
        E[T_i] = -P_i'WP_i, \\qquad
        \\mathrm{Var}[T_i] = \\sum_c P_{ic}g_{ic}^2 - (P_i'g_i)^2 ,

    both exact and ``O(J)``.  The variance is then reduced by the Bera--Yoon
    term accounting for estimated :math:`\\beta`; since ``E_beta[T - E T] = 0``
    identically, the generalised information equality gives
    ``Avar = Var(h) - C I_bb^{-1} C'`` with ``C = Cov(h_i, s_i)``.
    """
    e, P, psi0, X, Wd = _mnl_diagnostic_inputs(model, W)
    if np.abs(np.diag(Wd)).max() > 0:
        raise ValueError("W must have a zero diagonal for the residual cross-product test.")

    T = float(np.einsum("nj,nj->", e @ Wd, e))

    WP = P @ Wd.T
    g = WP + P @ Wd
    gbar = np.einsum("nj,nj->n", P, g)
    expected = -float(np.einsum("nj,nj->", P, WP))

    # h_i(c) = -(g_ic - E g_i);  E[h_i] = 0 by construction.
    h = -(g - gbar[:, None])
    var_h = float(np.einsum("nj,nj,nj->", P, h, h))

    # C = Cov(h_i, s_i) with s_i(c) = X_ic - E[X_i]; then subtract C I^-1 C'.
    PX = np.einsum("nj,njk->nk", P, X)
    s = X - PX[:, None, :]
    C = np.einsum("nj,nj,njk->k", P, h, s)
    I_bb = np.einsum("nj,njk,njl->kl", P, X, X) - np.einsum("nk,nl->kl", PX, PX)
    var = var_h - float(C @ np.linalg.solve(I_bb, C))

    if not np.isfinite(var) or var <= 0:
        raise ValueError("Non-positive variance for the residual autocorrelation test.")

    stat = (T - expected) ** 2 / var
    return SpatialDiagnostic(
        test_name="Spatial autocorrelation in MNL residuals",
        statistic=stat,
        df=1,
        p_value=float(stats.chi2.sf(stat, 1)),
        score=T - expected,
        variance=var,
        test_type="error",
        details={
            "raw_cross_product": T,
            "expected_under_null": expected,
            "variance_before_beta_adjustment": var_h,
            "z_score": (T - expected) / np.sqrt(var),
        },
    )


# ---------------------------------------------------------------------------
# IIA (Hausman--McFadden)
# ---------------------------------------------------------------------------


def iia_test(
    model,
    subset: "list | np.ndarray",
    params: list[str] | None = None,
    tol: float = 1e-10,
) -> HausmanTest:
    """Hausman--McFadden test of the IIA assumption in a fitted MNL.

    Under IIA, restricting the choice set to any subset ``D`` of alternatives
    leaves the coefficients unchanged: the full-set estimator
    :math:`\\hat\\beta_f` is efficient and the restricted estimator
    :math:`\\hat\\beta_r` --- fit on ``D`` using only choosers who selected
    within ``D`` --- is consistent but inefficient.  If IIA fails, the two
    diverge.  The statistic is the Hausman form

    .. math::
        H = (\\hat\\beta_r - \\hat\\beta_f)'\\,
            (V_r - V_f)^{+}\\,
            (\\hat\\beta_r - \\hat\\beta_f) \\;\\sim\\; \\chi^2_q .

    Parameters
    ----------
    model : ChoiceModel
        A *fitted* MNL.  Refitting is performed on a restricted copy of its
        data, so the model itself is left untouched.
    subset : sequence
        Alternative ids to retain.  Must be a strict, non-empty subset of the
        alternatives, with at least two members (a one-alternative choice set
        carries no information).
    params : list of str, optional
        Coefficients to compare; defaults to those common to both fits.
    tol : float, optional
        Eigenvalue tolerance passed to :func:`hausman_test`.

    Returns
    -------
    HausmanTest

    Notes
    -----
    The test is only as good as the subset: power depends on choosing ``D`` so
    that the alternatives excluded are the ones plausibly correlated with those
    retained.  A subset chosen at random often has little power.  Note also
    that the restricted fit discards every chooser who selected outside ``D``,
    so a small ``D`` buys efficiency loss and can leave ``V_r - V_f`` poorly
    conditioned --- :class:`HausmanTest` reports ``psd_warning`` when that
    happens.

    Alternatives whose coefficients are not identified on the restricted set
    (for example an alternative-specific constant for a dropped alternative)
    should be excluded via ``params``.
    """
    from ..data.choicetable import ChoiceTable

    if getattr(model, "_result", None) is None:
        raise RuntimeError("Model must be estimated before testing IIA.")

    ct = model._data
    obs_col, alt_col, choice_col = ct.obs_id_col, ct.alt_id_col, ct.choice_col
    if choice_col is None:
        raise ValueError("IIA test requires a ChoiceTable with a choice column.")

    df = ct.to_frame()
    all_alts = np.unique(df[alt_col].to_numpy())
    keep = np.unique(np.asarray(subset))
    if keep.size < 2:
        raise ValueError("`subset` must retain at least two alternatives.")
    missing = set(keep.tolist()) - set(all_alts.tolist())
    if missing:
        raise ValueError(f"`subset` contains unknown alternatives: {sorted(missing)}")
    if keep.size >= all_alts.size:
        raise ValueError(
            "`subset` must be a strict subset; restricting to every alternative "
            "reproduces the full model and the test is degenerate."
        )

    # Keep only choosers whose selected alternative survives the restriction.
    chosen_rows = df[df[choice_col].to_numpy().astype(bool)]
    keep_set = set(keep.tolist())
    good_obs = chosen_rows[chosen_rows[alt_col].isin(keep_set)][obs_col].unique()
    if len(good_obs) == 0:
        raise ValueError("No observations chose an alternative inside `subset`.")

    sub = df[df[obs_col].isin(good_obs) & df[alt_col].isin(keep_set)].copy()

    restricted_ct = ChoiceTable.from_long(
        sub,
        obs_id_col=obs_col,
        alt_id_col=alt_col,
        choice_col=choice_col,
        available_col=ct.available_col,
    )
    restricted = type(model)(
        restricted_ct,
        formula=model._spec.formula if model._spec is not None else None,
        solver=model._solver_name if hasattr(model, "_solver_name") else "lbfgs",
    )
    restricted_result = restricted.fit()

    return hausman_test(
        efficient=model._result,
        consistent=restricted_result,
        params=params,
        tol=tol,
    )
