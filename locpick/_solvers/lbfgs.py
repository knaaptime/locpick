"""L-BFGS-B solver using scipy.optimize.minimize."""

from __future__ import annotations

import numpy as np

from .protocol import SolverResult


class LBFGSSolver:
    """L-BFGS-B solver using scipy.optimize.minimize.

    This is the default solver for MNL estimation. It uses the L-BFGS-B
    algorithm with JAX-computed gradients for efficient optimization.

    Parameters
    ----------
    maxiter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.
    disp : bool
        Retained for API compatibility. Ignored for SciPy L-BFGS-B because
        `disp`/`iprint` are deprecated upstream.
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
        from scipy.optimize import minimize

        # Objective-first contract: solver inputs are carried on Objective.
        from locpick._jax.objective import Objective
        if not isinstance(objective, Objective):
            raise TypeError("LBFGSSolver.solve expects an Objective instance.")

        obj = objective
        log_likelihood_fn = obj.fn
        gradient_fn = obj.grad
        if obj.param_names is not None and not param_names:
            param_names = obj.param_names
        if obj.bounds is not None and bounds is None:
            bounds = obj.bounds

        # If fixed_mask is provided, freeze those parameters by fixing
        # their values in x0 and removing them from the optimization.
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
                    "maxiter": self.maxiter,
                    "ftol": self.tol,
                },
            )

            # Reconstruct full parameter vector
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
                    "maxiter": self.maxiter,
                    "ftol": self.tol,
                },
            )
            coefficients = result.x

        # Compute inverse Hessian from the L-BFGS approximation
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
