"""NumPy kernels for the linearized GMM SAR-MNL estimator.

Implements the two-step estimator from Carrión-Flores, Flores-Lagunes
& Guci (2018), extending Klier & McMillen (2008) to the multinomial
case.  The linearization at ρ=0 avoids matrix inversion entirely,
making it feasible for very large alternative sets where the dense
PML solve is too expensive.

Step 1: Standard MNL estimation (ignoring spatial dependence).
Step 2: Two-stage least squares (TSLS) using linearised gradients
       and instruments Z = [X, WX].
"""

from __future__ import annotations

import numpy as np


def compute_generalized_residuals(P, chosen):
    """Compute generalized residuals u_ik = d_ik - P_ik.

    Parameters
    ----------
    P : np.ndarray, shape (n_obs, n_alts)
        MNL choice probabilities from Step 1.
    chosen : np.ndarray, shape (n_obs, n_alts)
        Binary indicator matrix.

    Returns
    -------
    np.ndarray, shape (n_obs * n_alts,)
        Flattened generalized residuals.
    """
    return (chosen - P).ravel()


def compute_linearized_gradients(beta, P, X, WX, n_obs, n_alts, n_params):
    """Compute linearized gradients for the SMNL estimator at ρ=0.

    At the linearization point ρ=0: (I - ρW)^{-1} = I, so the gradients
    simplify to standard MNL gradients plus a spatial term.

    Parameters
    ----------
    beta : np.ndarray, shape (n_params,)
        Beta coefficients from Step 1 MNL.
    P : np.ndarray, shape (n_obs, n_alts)
        MNL choice probabilities from Step 1.
    X : np.ndarray, shape (n_obs, n_alts, n_params) or (n_obs * n_alts, n_params)
        Design matrix (can be in long or 3D format).
    WX : np.ndarray, shape (n_obs, n_alts, n_params)
        Spatially lagged design matrix.
    n_obs : int
    n_alts : int
    n_params : int

    Returns
    -------
    G_beta : np.ndarray, shape (n_obs * n_alts, n_params)
        Gradient matrix for beta parameters.
    G_rho : np.ndarray, shape (n_obs * n_alts,)
        Gradient vector for the spatial parameter rho.
    """
    # Reshape X to 3D if needed
    if X.ndim == 2:
        X_3d = X.reshape(n_obs, n_alts, n_params)
    else:
        X_3d = X

    # Beta gradients: G_{i,beta_k} = P_ik * (delta_{ilk} - P_il) * X_i
    # For each (obs, alt) pair and each parameter:
    # G_beta[i*J + k, p] = P[i,k] * (delta_{k,k} - P[i,k]) * X[i,k,p]
    #                     + sum_{l != k} P[i,k] * (0 - P[i,l]) * X[i,l,p]
    # = P[i,k] * (X[i,k,p] - sum_l P[i,l] * X[i,l,p])

    # Compute weighted average of X across alternatives: sum_l P_l * X_l
    # P is (n_obs, n_alts), X_3d is (n_obs, n_alts, n_params)
    # weighted_X = sum_k P[i,k] * X[i,k,:]  → (n_obs, n_params)
    weighted_X = np.einsum("ik,ikp->ip", P, X_3d)  # (n_obs, n_params)

    # G_beta[i*J + k, p] = P[i,k] * (X[i,k,p] - weighted_X[i,p])
    G_beta = np.zeros((n_obs * n_alts, n_params))
    for k in range(n_alts):
        for p in range(n_params):
            G_beta[k::n_alts, p] = P[:, k] * (X_3d[:, k, p] - weighted_X[:, p])

    # Rho gradient: G_{i,rho} = P_ik * [(WX)_i * beta - sum_l P_il * (WX)_i * beta]
    # = P_ik * (WX_i @ beta - sum_l P_il * WX_l @ beta)
    # WX is (n_obs, n_alts, n_params), beta is (n_params,)
    WX_beta = WX @ beta  # (n_obs, n_alts)
    weighted_WX_beta = np.sum(P * WX_beta, axis=1)  # (n_obs,)

    G_rho = np.zeros(n_obs * n_alts)
    for k in range(n_alts):
        G_rho[k::n_alts] = P[:, k] * (WX_beta[:, k] - weighted_WX_beta)

    return G_beta, G_rho


def _remove_dependent_columns(Z, tol=1e-10):
    """Remove linearly dependent columns via QR decomposition."""
    Q, R = np.linalg.qr(Z)
    # Check diagonal of R for near-zero values
    diag_R = np.abs(np.diag(R))
    keep = diag_R > tol * diag_R[0]
    return Z[:, keep]


def smnl_tsls(G_beta, G_rho, u, Z):
    """Two-stage least squares estimation of the linearized SMNL model.

    First stage: regress each gradient column on instruments Z.
    Second stage: regress residuals u on fitted gradients.

    Parameters
    ----------
    G_beta : np.ndarray, shape (n_obs * n_alts, n_params)
        Gradient matrix for beta parameters.
    G_rho : np.ndarray, shape (n_obs * n_alts,)
        Gradient vector for the spatial parameter rho.
    u : np.ndarray, shape (n_obs * n_alts,)
        Generalized residuals from Step 1.
    Z : np.ndarray, shape (n_obs * n_alts, n_instruments)
        Instrument matrix [X, WX] (linearly independent columns).

    Returns
    -------
    delta : np.ndarray, shape (n_params + 1,)
        Parameter updates [Delta_beta_1, ..., Delta_beta_K, Delta_rho].
    se : np.ndarray, shape (n_params + 1,)
        TSLS standard errors.
    vcov : np.ndarray, shape (n_params + 1, n_params + 1)
        TSLS covariance matrix.
    """
    # Stack gradients: G = [G_beta, G_rho]
    G = np.column_stack([G_beta, G_rho])
    n_params_total = G.shape[1]
    n = len(u)

    # First stage: project G onto instruments Z
    # G_hat = Z (Z'Z)^{-1} Z' G
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    G_hat = Z @ (ZtZ_inv @ (Z.T @ G))

    # Second stage: OLS of u on G_hat
    # delta = (G_hat' G_hat)^{-1} G_hat' u
    GtG_inv = np.linalg.inv(G_hat.T @ G_hat)
    delta = GtG_inv @ (G_hat.T @ u)

    # Residuals
    e = u - G_hat @ delta

    # Variance: sigma^2 = e'e / (n - p)
    sigma2 = e @ e / (n - n_params_total)

    # Covariance: V = sigma^2 * (G_hat' G_hat)^{-1}
    vcov = sigma2 * GtG_inv
    se = np.sqrt(np.maximum(np.diag(vcov), 0))

    return delta, se, vcov


def fit_linearized_gmm(arrays, W_sparse):
    """Two-step linearized GMM estimation (Carrión-Flores et al. 2018).

    Parameters
    ----------
    arrays : ChoiceArrays
        Estimation data arrays.
    W_sparse : scipy.sparse.csr_array
        Row-standardised alt×alt spatial weights matrix.

    Returns
    -------
    dict
        Dictionary with keys: 'beta', 'rho', 'se', 'vcov', 'log_likelihood'.
    """
    from .mnl_numpy import mnl_probs_numpy

    n_obs = arrays.n_obs
    n_alts = arrays.n_alts
    k = arrays.design_matrix.shape[1]

    dm = np.asarray(arrays.design_matrix, dtype=np.float64)
    chosen = np.asarray(arrays.chosen, dtype=np.float64).reshape(n_obs, n_alts)

    if arrays.available is not None:
        available = np.asarray(arrays.available, dtype=np.float64).reshape(n_obs, n_alts)
    else:
        available = np.ones((n_obs, n_alts), dtype=np.float64)

    # --- Step 1: Standard MNL estimation ---
    from .._solvers import get_solver

    solver = get_solver("lbfgs")

    from .._jax.builders import build_mnl_objective

    objective = build_mnl_objective(arrays)
    x0 = np.zeros(k)
    solver_result = solver.solve(
        objective=objective,
        x0=x0,
        param_names=list(arrays.param_names),
        bounds=None,
        fixed_mask=None,
    )
    beta_step1 = solver_result.coefficients

    # Compute Step 1 probabilities
    V = (dm @ beta_step1).reshape(n_obs, n_alts)
    P = mnl_probs_numpy(V, available, inclusion_probs=None)

    # --- Step 2: TSLS ---
    # Extract X from design matrix (long format → 3D)
    X_3d = dm.reshape(n_obs, n_alts, k)

    # Compute WX: spatially lagged covariates
    # W is (n_alts, n_alts), X varies across choosers
    # WX[i, j, p] = sum_k W[j, k] * X[i, k, p]
    W_dense = W_sparse.toarray()
    WX = np.einsum("jk,ikp->ijp", W_dense, X_3d)  # (n_obs, n_alts, k)

    # Generalized residuals
    u = compute_generalized_residuals(P, chosen)

    # Linearized gradients
    G_beta, G_rho = compute_linearized_gradients(beta_step1, P, X_3d, WX, n_obs, n_alts, k)

    # Instruments: [X, WX] in long format
    X_long = dm  # (n_obs * n_alts, k)
    WX_long = WX.reshape(n_obs * n_alts, k)
    Z = np.column_stack([X_long, WX_long])
    Z = _remove_dependent_columns(Z)

    # TSLS
    delta, se, vcov = smnl_tsls(G_beta, G_rho, u, Z)

    # Assemble final parameters
    beta_final = beta_step1 + delta[:k]
    rho_final = delta[k]

    # Compute log-likelihood at final params (using spatial model)
    A = np.eye(n_alts) - rho_final * W_dense
    V_base = (dm @ beta_final).reshape(n_obs, n_alts)
    V_filtered = np.linalg.solve(A, V_base.T).T
    D = np.diag(np.linalg.inv(A))
    V_star = V_filtered / D[None, :]
    log_probs = np.log(np.maximum(mnl_probs_numpy(V_star, available), 1e-30))
    ll = np.sum(chosen * log_probs)

    return {
        "beta": beta_final,
        "rho": rho_final,
        "se": se,
        "vcov": vcov,
        "log_likelihood": float(ll),
        "beta_step1": beta_step1,
    }
