"""L-BFGS-B solver using scipy.optimize.minimize."""

from __future__ import annotations

import numpy as np

from .protocol import SolverResult


def solve_lbfgs(
    objective,
    x0: np.ndarray,
    *,
    maxiter: int = 1000,
    tol: float = 1e-8,
    disp: bool = False,
    param_names: list[str] | None = None,
    bounds=None,
    fixed_mask=None,
    **kwargs,
) -> SolverResult:
    """Solve using L-BFGS-B.

    Parameters
    ----------
    objective : Objective
        An :class:`locpick._jax.objective.Objective` instance.
    x0 : np.ndarray
        Initial parameter values.
    maxiter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.
    disp : bool
        Ignored for SciPy L-BFGS-B (deprecated upstream).
    param_names : list[str] or None
        Names for each parameter.
    bounds : list of tuple or None
        Per-parameter bounds ``(min, max)``.
    fixed_mask : np.ndarray or None
        Boolean mask — ``True`` marks parameters held fixed.
    **kwargs
        Ignored (for API compatibility).

    Returns
    -------
    SolverResult
    """
    from scipy.optimize import minimize

    from locpick._jax.objective import Objective

    if not isinstance(objective, Objective):
        raise TypeError("solve_lbfgs expects an Objective instance.")

    obj = objective
    log_likelihood_fn = obj.fn
    gradient_fn = obj.grad
    if obj.param_names is not None and not param_names:
        param_names = obj.param_names
    if obj.bounds is not None and bounds is None:
        bounds = obj.bounds

    if fixed_mask is not None and np.any(fixed_mask):
        free_mask = ~fixed_mask
        x0_free = x0[free_mask]

        def neg_ll_free(x_free):
            x_full = x0.copy()
            x_full[free_mask] = x_free
            return -log_likelihood_fn(x_full)

        def neg_grad_free(x_free):
            x_full = x0.copy()
            x_full[free_mask] = x_free
            return -gradient_fn(x_full)[free_mask]

        bounds_free = None
        if bounds is not None:
            bounds_free = [b for b, f in zip(bounds, free_mask) if f]

        result = minimize(
            fun=neg_ll_free,
            x0=x0_free,
            jac=neg_grad_free,
            method="L-BFGS-B",
            bounds=bounds_free,
            options={
                "maxiter": maxiter,
                "ftol": tol,
            },
        )

        coefficients = x0.copy()
        coefficients[free_mask] = result.x
    else:
        result = minimize(
            fun=lambda x: -log_likelihood_fn(x),
            x0=x0,
            jac=lambda x: -gradient_fn(x),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "ftol": tol,
            },
        )
        coefficients = result.x

    hessian_inv = None
    if hasattr(result, "hess_inv"):
        try:
            hessian_inv = np.asarray(result.hess_inv.todense())
        except AttributeError:
            hessian_inv = np.asarray(result.hess_inv)

    return SolverResult(
        coefficients=coefficients,
        hessian=hessian_inv,
        log_likelihood=-result.fun,
        n_iterations=result.nit if hasattr(result, "nit") else 0,
        converged=result.success,
        message=result.message,
        solver_name="lbfgs",
        raw={"scipy_result": result},
    )


class LBFGSSolver:
    """L-BFGS-B solver (class wrapper for backward compatibility).

    .. deprecated::
        Use :func:`solve_lbfgs` directly instead.
    """

    def __init__(
        self,
        maxiter: int = 1000,
        tol: float = 1e-8,
        disp: bool = False,
    ):
        self.maxiter = maxiter
        self.tol = tol
        self.disp = disp

    def solve(
        self,
        objective,
        x0: np.ndarray,
        param_names: list[str],
        bounds=None,
        fixed_mask=None,
        **kwargs,
    ) -> SolverResult:
        return solve_lbfgs(
            objective=objective,
            x0=x0,
            maxiter=self.maxiter,
            tol=self.tol,
            disp=self.disp,
            param_names=param_names,
            bounds=bounds,
            fixed_mask=fixed_mask,
            **kwargs,
        )
