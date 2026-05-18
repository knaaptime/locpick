"""Optimagic-backed solver for choice model estimation.

Wraps :func:`optimagic.minimize` so any algorithm available in optimagic
can be used to estimate locpick choice models.  Useful candidates for
smooth log-likelihoods with analytic / JAX gradients:

* ``"fides"`` — trust-region Newton, strong on stiff problems
  (requires ``pip install fides``).
* ``"iminuit_migrad"`` — HEP-community quasi-Newton with adaptive trust
  region (requires ``pip install iminuit``).
* ``"ipopt"`` — interior-point with exact Hessian; useful for box-
  bounded models (requires ``pip install cyipopt``).
* The ``"scipy_*"`` and ``"nlopt_*"`` families ship with the base
  ``optimagic`` install and require no extras.

See :func:`optimagic.algorithms.AVAILABLE_ALGORITHMS` for the full
algorithm catalogue available in the current environment.
"""

from __future__ import annotations

import numpy as np

from .protocol import SolverResult


class OptimagicSolver:
    """Solver wrapper around :func:`optimagic.minimize`.

    Parameters
    ----------
    algorithm : str
        Optimagic algorithm name (e.g. ``"scipy_lbfgsb"``, ``"fides"``,
        ``"iminuit_migrad"``, ``"ipopt"``, ``"nlopt_lbfgsb"``,
        ``"scipy_trust_constr"``).  Defaults to ``"scipy_lbfgsb"`` so the
        wrapper behaves like :class:`LBFGSSolver` out of the box.
    algo_options : dict or None
        Algorithm-specific options forwarded to ``optimagic.minimize`` via
        the ``algo_options`` argument.  See the optimagic documentation
        for per-algorithm option names.
    """

    def __init__(
        self,
        algorithm: str = "scipy_lbfgsb",
        algo_options: dict | None = None,
    ):
        self.algorithm = algorithm
        self.algo_options = dict(algo_options) if algo_options else {}

    def solve(
        self,
        objective,
        x0: np.ndarray,
        param_names: list[str],
        bounds=None,
        fixed_mask=None,
        **kwargs,
    ) -> SolverResult:
        import optimagic as om

        from locpick._jax.objective import Objective

        if not isinstance(objective, Objective):
            raise TypeError("OptimagicSolver.solve expects an Objective instance.")

        obj = objective
        log_likelihood_fn = obj.fn
        gradient_fn = obj.grad
        if obj.param_names is not None and not param_names:
            param_names = obj.param_names
        if obj.bounds is not None and bounds is None:
            bounds = obj.bounds

        # Resolve free / fixed parameters in the same style as LBFGSSolver.
        if fixed_mask is not None and np.any(fixed_mask):
            free_mask = ~np.asarray(fixed_mask, dtype=bool)
            x0_free = np.asarray(x0, dtype=float)[free_mask]

            def neg_ll(x_free):
                x_full = np.asarray(x0, dtype=float).copy()
                x_full[free_mask] = x_free
                return float(-log_likelihood_fn(x_full))

            def neg_grad(x_free):
                x_full = np.asarray(x0, dtype=float).copy()
                x_full[free_mask] = x_free
                return np.asarray(-gradient_fn(x_full), dtype=float)[free_mask]

            bounds_free = _build_optimagic_bounds(bounds, free_mask)
            start = x0_free
        else:
            free_mask = None

            def neg_ll(x):
                return float(-log_likelihood_fn(np.asarray(x, dtype=float)))

            def neg_grad(x):
                return np.asarray(-gradient_fn(np.asarray(x, dtype=float)), dtype=float)

            bounds_free = _build_optimagic_bounds(bounds, None)
            start = np.asarray(x0, dtype=float)

        result = om.minimize(
            fun=neg_ll,
            params=start,
            algorithm=self.algorithm,
            jac=neg_grad,
            bounds=bounds_free,
            algo_options=self.algo_options or None,
        )

        # Re-assemble full parameter vector when some entries were fixed.
        if free_mask is not None:
            coefficients = np.asarray(x0, dtype=float).copy()
            coefficients[free_mask] = np.asarray(result.params, dtype=float)
        else:
            coefficients = np.asarray(result.params, dtype=float)

        hessian_inv = None
        if getattr(result, "hess_inv", None) is not None:
            hessian_inv = np.asarray(result.hess_inv, dtype=float)

        return SolverResult(
            coefficients=coefficients,
            hessian=hessian_inv,
            log_likelihood=float(-result.fun),
            n_iterations=int(getattr(result, "n_iterations", 0) or 0),
            converged=bool(getattr(result, "success", False)),
            message=str(getattr(result, "message", "") or ""),
            solver_name=f"optimagic:{self.algorithm}",
            raw={"optimagic_result": result},
        )


def _build_optimagic_bounds(bounds, free_mask):
    """Translate per-parameter ``(lo, hi)`` bounds into ``optimagic.Bounds``.

    Returns ``None`` when no finite bound is present.
    """
    if bounds is None:
        return None

    if free_mask is not None:
        bounds = [b for b, keep in zip(bounds, free_mask) if keep]

    if not bounds:
        return None

    import optimagic as om

    lower = np.array(
        [(-np.inf if b is None or b[0] is None else b[0]) for b in bounds],
        dtype=float,
    )
    upper = np.array(
        [(np.inf if b is None or b[1] is None else b[1]) for b in bounds],
        dtype=float,
    )

    if np.all(np.isneginf(lower)) and np.all(np.isposinf(upper)):
        return None

    return om.Bounds(lower=lower, upper=upper)
