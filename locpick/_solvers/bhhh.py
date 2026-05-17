"""BHHH (Berndt-Hall-Hall-Hausman) solver for maximum-likelihood estimation.

BHHH approximates the Hessian of the log-likelihood by the outer product of
per-observation scores:

.. math::

    B(\\theta) \\;=\\; \\sum_{i=1}^{N} g_i(\\theta)\\, g_i(\\theta)^\\top

where :math:`g_i = \\nabla_\\theta \\log L_i(\\theta)` is the score for
observation :math:`i`. By the information matrix equality, ``B`` is a
consistent estimator of :math:`-\\nabla^2 \\log L` near the optimum, but
requires only first derivatives — no Hessian-vector products and no full
Hessian factorisation.

The update direction is the Newton step :math:`d = B^{-1} g` (with a small
Levenberg-Marquardt ridge for numerical stability), accepted via an Armijo
backtracking line search.

Requirements
------------
The Objective must expose :meth:`~locpick._jax.objective.Objective.score_contribs`
(equivalently: ``loglike_contribs_jax`` must be set). At present this is
available for MNL and SCL.
"""

from __future__ import annotations

import numpy as np

from .protocol import SolverResult


class BHHHSolver:
    """Quasi-Newton solver using the outer-product-of-gradients Hessian.

    Parameters
    ----------
    maxiter : int, default 200
        Maximum outer iterations.
    tol : float, default 1e-6
        Convergence tolerance on the infinity norm of the gradient.
    ftol : float, default 1e-10
        Secondary convergence tolerance: relative log-likelihood change.
    ridge : float, default 1e-8
        Levenberg-Marquardt ridge added to the BHHH information matrix
        before solving for the Newton direction. Stabilises early iterates
        when :math:`G^\\top G` is ill-conditioned.
    armijo_c : float, default 1e-4
        Sufficient-increase constant for the Armijo line search.
    armijo_shrink : float, default 0.5
        Step-size shrink factor on Armijo failure.
    armijo_max : int, default 25
        Maximum backtracking steps per outer iteration.
    """

    def __init__(
        self,
        maxiter: int = 200,
        tol: float = 1e-6,
        ftol: float = 1e-10,
        ridge: float = 1e-8,
        armijo_c: float = 1e-4,
        armijo_shrink: float = 0.5,
        armijo_max: int = 25,
    ):
        self.maxiter = maxiter
        self.tol = tol
        self.ftol = ftol
        self.ridge = ridge
        self.armijo_c = armijo_c
        self.armijo_shrink = armijo_shrink
        self.armijo_max = armijo_max

    def solve(
        self,
        objective,
        x0: np.ndarray,
        param_names: list[str],
        bounds=None,
        fixed_mask=None,
        **kwargs,
    ) -> SolverResult:
        from locpick._jax.objective import Objective

        if not isinstance(objective, Objective):
            raise TypeError(f"{type(self).__name__}.solve expects an Objective instance.")
        if objective.loglike_contribs_jax is None:
            raise ValueError(
                "BHHHSolver requires Objective.loglike_contribs_jax (per-observation "
                "log-likelihoods). The active model does not expose them."
            )
        if bounds is not None and any(
            b is not None and (b[0] is not None or b[1] is not None) for b in bounds
        ):
            raise ValueError(
                "BHHHSolver does not support box constraints; use LBFGSSolver "
                "for bounded problems."
            )

        ll_fn = objective.fn
        # Per-obs scores from JAX jacrev — shape (N, K).
        score_fn = objective.score_contribs

        x_full = np.asarray(x0, dtype=np.float64).copy()
        if fixed_mask is not None and np.any(fixed_mask):
            free_mask = ~np.asarray(fixed_mask, dtype=bool)
        else:
            free_mask = np.ones_like(x_full, dtype=bool)

        n_free = int(free_mask.sum())
        ll = float(ll_fn(x_full))
        converged = False
        message = "maxiter reached"
        n_iter = 0
        last_grad_norm = np.inf

        for it in range(1, self.maxiter + 1):
            n_iter = it
            G = score_fn(x_full)  # (N, K_full)
            if G.ndim != 2:
                raise RuntimeError(f"score_contribs must return a 2-D matrix; got shape {G.shape}")
            G_free = G[:, free_mask]
            g = G_free.sum(axis=0)  # (n_free,)
            grad_norm = float(np.max(np.abs(g)))
            last_grad_norm = grad_norm
            if grad_norm < self.tol:
                converged = True
                message = f"gradient norm {grad_norm:.2e} < tol {self.tol:.0e}"
                break

            B = G_free.T @ G_free
            B_reg = B + self.ridge * np.eye(n_free)
            try:
                d = np.linalg.solve(B_reg, g)
            except np.linalg.LinAlgError:
                # Fall back to scaled steepest ascent
                d = g / max(grad_norm, 1.0)

            # Ensure ascent direction.
            slope = float(g @ d)
            if slope <= 0.0:
                d = g.copy()
                slope = float(g @ d)
                if slope <= 0.0:
                    converged = True
                    message = "zero ascent slope; stationary point"
                    break

            # Armijo backtracking on ll (maximisation: require ll_new >= ll + c*step*slope).
            step = 1.0
            accepted = False
            for _ in range(self.armijo_max):
                x_trial = x_full.copy()
                x_trial[free_mask] = x_full[free_mask] + step * d
                ll_trial = float(ll_fn(x_trial))
                if np.isfinite(ll_trial) and ll_trial >= ll + self.armijo_c * step * slope:
                    accepted = True
                    break
                step *= self.armijo_shrink

            if not accepted:
                message = "Armijo line search failed"
                break

            ll_new = ll_trial
            rel_change = abs(ll_new - ll) / max(1.0, abs(ll))
            x_full = x_trial
            ll = ll_new
            if rel_change < self.ftol:
                converged = True
                message = f"relative LL change {rel_change:.2e} < ftol {self.ftol:.0e}"
                break

        return SolverResult(
            coefficients=x_full,
            hessian=None,  # downstream code computes Hessian via Objective.hessian
            log_likelihood=ll,
            n_iterations=n_iter,
            converged=converged,
            message=message,
            solver_name="bhhh",
            raw={
                "final_grad_norm": last_grad_norm,
            },
        )
