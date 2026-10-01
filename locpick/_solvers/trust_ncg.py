"""Newton-CG trust-region solvers using SciPy + JAX Hessian-vector products.

These solvers wrap ``scipy.optimize.minimize`` with ``method="trust-ncg"``
or ``"trust-krylov"`` and supply a JAX-backed Hessian-vector product via
:meth:`locpick._jax.objective.Objective.hvp`.  Hessian-vector products
are computed in :math:`O(n)` (forward-over-reverse AD) instead of
materialising the full :math:`O(n^2)` Hessian, which makes second-order
trust-region steps competitive on small-to-medium parameter vectors.

Box constraints are not supported by ``trust-ncg`` / ``trust-krylov``;
the solver raises :class:`ValueError` if ``bounds`` are provided.  Use
:class:`locpick._solvers.lbfgs.LBFGSSolver` for bounded problems.
"""

from __future__ import annotations

import numpy as np

from .protocol import SolverResult


def _assess_convergence(result) -> tuple[bool, str]:
    """Convergence verdict for a SciPy trust-region result.

    SciPy's gradient test ``|g| < gtol`` is absolute, while a summed
    log-likelihood's gradient carries rounding noise proportional to the
    objective's magnitude.  Near the optimum of an ill-conditioned problem
    the trust region then collapses on that noise and SciPy exits with
    status 2 ("bad approximation") although the point is optimal.  That exit
    is accepted when first-order optimality holds relative to the objective's
    scale, ``|g|_inf <= sqrt(eps) * max(1, |f|)``; any other failure stands.
    """
    message = str(result.message)
    if result.success:
        return True, message
    jac = getattr(result, "jac", None)
    if getattr(result, "status", None) == 2 and jac is not None:
        g_inf = float(np.max(np.abs(jac))) if np.size(jac) else 0.0
        if g_inf <= np.sqrt(np.finfo(float).eps) * max(1.0, abs(float(result.fun))):
            return True, (
                f"{message} Accepted as converged: the gradient (|g|_inf = {g_inf:.2e}) "
                "is at the floating-point noise floor for this objective."
            )
    return False, message


class TrustNCGSolver:
    """Newton-CG trust-region solver (``scipy.optimize`` + JAX HVP).

    Parameters
    ----------
    method : {"trust-ncg", "trust-krylov"}
        Trust-region inner solver. ``trust-ncg`` is the classic
        Newton-CG; ``trust-krylov`` uses a Lanczos-Krylov subspace and
        is often more robust for indefinite Hessians.
    maxiter : int
        Maximum outer trust-region iterations.
    tol : float
        Gradient-norm convergence tolerance.
    """

    _ALLOWED_METHODS = ("trust-ncg", "trust-krylov")

    def __init__(
        self,
        method: str = "trust-ncg",
        maxiter: int = 1000,
        tol: float = 1e-8,
    ):
        if method not in self._ALLOWED_METHODS:
            raise ValueError(
                f"Unknown method {method!r}; expected one of {self._ALLOWED_METHODS}."
            )
        self.method = method
        self.maxiter = maxiter
        self.tol = tol

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

        from .._jax.objective import Objective

        if not isinstance(objective, Objective):
            raise TypeError(f"{type(self).__name__}.solve expects an Objective instance.")
        if bounds is not None and any(
            b is not None and (b[0] is not None or b[1] is not None) for b in bounds
        ):
            raise ValueError(
                f"method={self.method!r} does not support box constraints; "
                "use LBFGSSolver for bounded problems."
            )
        if objective.jax_fn is None:
            raise ValueError(
                f"{type(self).__name__} requires a JAX-backed objective "
                "(objective.jax_fn must be provided)."
            )

        obj = objective
        if obj.param_names is not None and not param_names:
            param_names = obj.param_names

        ll_fn = obj.fn
        grad_fn = obj.grad
        hvp_fn = obj.hvp

        # Handle parameter fixing by projecting to the free subspace.
        if fixed_mask is not None and np.any(fixed_mask):
            free_mask = ~fixed_mask
            x0_free = x0[free_mask]

            def neg_ll_free(x_free):
                x_full = x0.copy()
                x_full[free_mask] = x_free
                return -ll_fn(x_full)

            def neg_grad_free(x_free):
                x_full = x0.copy()
                x_full[free_mask] = x_free
                return -grad_fn(x_full)[free_mask]

            def neg_hvp_free(x_free, v_free):
                x_full = x0.copy()
                x_full[free_mask] = x_free
                v_full = np.zeros_like(x0)
                v_full[free_mask] = v_free
                return -hvp_fn(x_full, v_full)[free_mask]

            result = minimize(
                fun=neg_ll_free,
                x0=x0_free,
                jac=neg_grad_free,
                hessp=neg_hvp_free,
                method=self.method,
                options={"maxiter": self.maxiter, "gtol": self.tol},
            )
            coefficients = x0.copy()
            coefficients[free_mask] = result.x
        else:
            result = minimize(
                fun=lambda x: -ll_fn(x),
                x0=x0,
                jac=lambda x: -grad_fn(x),
                hessp=lambda x, v: -hvp_fn(x, v),
                method=self.method,
                options={"maxiter": self.maxiter, "gtol": self.tol},
            )
            coefficients = result.x

        converged, message = _assess_convergence(result)

        return SolverResult(
            coefficients=coefficients,
            hessian=None,  # SEs flow through Objective.hessian at the fit layer
            log_likelihood=-result.fun,
            n_iterations=result.nit if hasattr(result, "nit") else 0,
            converged=converged,
            message=message,
            solver_name=self.method,
            raw={"scipy_result": result},
        )


class TrustKrylovSolver(TrustNCGSolver):
    """Newton-CG trust-region with a Krylov subspace inner solver.

    Equivalent to ``TrustNCGSolver(method="trust-krylov")``.
    """

    def __init__(self, maxiter: int = 1000, tol: float = 1e-8):
        super().__init__(method="trust-krylov", maxiter=maxiter, tol=tol)
