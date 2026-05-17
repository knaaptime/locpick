"""Unified optimization objective for choice model estimation.

The :class:`Objective` class encapsulates a log-likelihood function,
its gradient, and optional Hessian, along with parameter transformations
and JAX-native function references.  It replaces the ad-hoc pattern of
passing ``log_likelihood_fn`` and ``gradient_fn`` with ``.jax_fn`` attributes.

Solvers negotiate with the ``Objective`` to get the interface they need:

- :class:`LBFGSSolver` calls ``fn`` and ``grad`` (numpy ↔ python)
- JAX-native solvers (e.g. :class:`OptaxSolver`) call ``jax_fn`` and
  ``jax_grad`` directly
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from locpick._compat import _JAX_AVAILABLE
from locpick._jax.transforms import ParamTransform

if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp


@dataclass
class Objective:
    """Unified optimization objective for choice model estimation.

    Encapsulates a log-likelihood function, its gradient, and optional
    Hessian, along with parameter transformations and JAX-native function
    references.

    Parameters
    ----------
    fn : callable
        Log-likelihood function: params (numpy) → scalar (float).
        This function should *maximize* (positive LL).
    grad : callable
        Gradient function: params (numpy) → gradient (numpy array).
        Same sign convention as ``fn`` (positive gradient = uphill).
    jax_fn : callable or None
        JIT-compiled JAX log-likelihood: params (jnp) → scalar (jnp).
        When available, JAX-native solvers use this directly to avoid
        numpy↔JAX conversion overhead on every iteration.
    jax_grad : callable or None
        JIT-compiled JAX gradient: params (jnp) → gradient (jnp).
    transform : ParamTransform or None
        Parameter transformation for constrained optimization.
        When provided, solvers that don't support box constraints can
        optimize in unconstrained space and transform back.
    param_names : list[str] or None
        Names for each parameter.
    bounds : list[tuple[float, float]] or None
        Per-parameter bounds for scipy-compatible solvers.
    """

    fn: Callable
    grad: Callable
    jax_fn: Optional[Callable] = None
    jax_grad: Optional[Callable] = None
    loglike_contribs_jax: Optional[Callable] = None
    transform: Optional[ParamTransform] = None
    param_names: Optional[list[str]] = None
    bounds: Optional[list[tuple[float, float]]] = None

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def neg_fn(self, x: np.ndarray) -> float:
        """Negative log-likelihood (for minimization)."""
        return -self.fn(x)

    def neg_grad(self, x: np.ndarray) -> np.ndarray:
        """Negative gradient (for minimization)."""
        return -self.grad(x)

    def neg_jax_fn(self, params, args=None):
        """Negative log-likelihood in JAX (for minimization-style solvers)."""
        return -self.jax_fn(params)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Compute the Hessian at ``x``.

        Uses ``jax.hessian`` when JAX-native functions are available,
        otherwise falls back to finite differences.

        Parameters
        ----------
        x : np.ndarray
            Parameter vector (natural scale).

        Returns
        -------
        np.ndarray, shape (n_params, n_params)
            Hessian matrix of the log-likelihood.
        """
        if self.jax_fn is not None and _JAX_AVAILABLE:
            hess_fn = jax.hessian(self.jax_fn)
            return np.asarray(hess_fn(jnp.array(x, dtype=jnp.float64)))
        else:
            # Finite-difference Hessian
            return self._finite_diff_hessian(x)

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute the Hessian-vector product :math:`H(x)\\,v`.

        Uses forward-over-reverse autodiff (``jax.jvp`` of ``jax.grad``)
        when JAX-native functions are available, avoiding materialisation
        of the full :math:`(n\\times n)` Hessian.  Falls back to a
        finite-difference of the gradient otherwise.

        The first JAX call jit-compiles and caches the HVP closure on
        the instance so subsequent calls have no Python overhead.

        Parameters
        ----------
        x : np.ndarray
            Parameter vector (natural scale).
        v : np.ndarray
            Direction vector of the same shape as ``x``.

        Returns
        -------
        np.ndarray
            :math:`H(x)\\,v`, same shape as ``x``.
        """
        if self.jax_fn is not None and _JAX_AVAILABLE:
            cached = self.__dict__.get("_hvp_fn")
            if cached is None:
                grad_fn = jax.grad(self.jax_fn)

                @jax.jit
                def _hvp(xv, vv):
                    return jax.jvp(grad_fn, (xv,), (vv,))[1]

                cached = _hvp
                self.__dict__["_hvp_fn"] = cached
            xj = jnp.asarray(x, dtype=jnp.float64)
            vj = jnp.asarray(v, dtype=jnp.float64)
            return np.asarray(cached(xj, vj))
        # Finite-difference fallback: (g(x + h v) - g(x - h v)) / (2 h)
        h = 1e-5
        v = np.asarray(v, dtype=np.float64)
        return (self.grad(x + h * v) - self.grad(x - h * v)) / (2.0 * h)

    def score_contribs(self, x: np.ndarray) -> np.ndarray:
        """Per-observation score matrix :math:`G` of shape ``(n_obs, n_params)``.

        Each row :math:`g_i = \\nabla_\\theta \\log L_i(\\theta)` is the
        gradient of the log-likelihood contribution from observation ``i``.
        Computed via ``jax.jacrev`` of :attr:`loglike_contribs_jax`, which
        must be provided when the Objective is constructed (see
        :meth:`from_jax`).

        Used by :class:`locpick._solvers.bhhh.BHHHSolver` to assemble the
        outer-product-of-gradients Hessian approximation
        :math:`B = G^\\top G`.

        Raises
        ------
        NotImplementedError
            If ``loglike_contribs_jax`` is not available.
        """
        if self.loglike_contribs_jax is None or not _JAX_AVAILABLE:
            raise NotImplementedError(
                "score_contribs requires loglike_contribs_jax; this Objective "
                "was built without per-observation log-likelihood support."
            )
        cached = self.__dict__.get("_score_contribs_fn")
        if cached is None:
            cached = jax.jit(jax.jacrev(self.loglike_contribs_jax))
            self.__dict__["_score_contribs_fn"] = cached
        return np.asarray(cached(jnp.asarray(x, dtype=jnp.float64)))

    def _finite_diff_hessian(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian via central finite differences."""
        n = len(x)
        h = 1e-5
        hess = np.zeros((n, n))
        self.fn(x)
        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                x_pp[i] += h
                x_pp[j] += h
                x_pm[i] += h
                x_pm[j] -= h
                x_mp[i] -= h
                x_mp[j] += h
                x_mm[i] -= h
                x_mm[j] -= h
                hess[i, j] = (self.fn(x_pp) - self.fn(x_pm) - self.fn(x_mp) + self.fn(x_mm)) / (
                    4 * h * h
                )
                hess[j, i] = hess[i, j]
        return hess

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_jax(
        cls,
        ll_fn,
        grad_fn,
        param_names: Optional[list[str]] = None,
        transform: Optional[ParamTransform] = None,
        bounds: Optional[list[tuple[float, float]]] = None,
        loglike_contribs_fn: Optional[Callable] = None,
    ) -> "Objective":
        """Create an Objective from JIT-compiled JAX functions.

        This is the primary factory for JAX-accelerated models.  It wraps
        the JIT-compiled functions with numpy conversion for scipy-compatible
        solvers, while keeping the raw JAX functions available for JAX-native
        solvers.

        Parameters
        ----------
        ll_fn : callable
            JIT-compiled log-likelihood: jnp array → jnp scalar.
        grad_fn : callable
            JIT-compiled gradient: jnp array → jnp array.
        param_names : list[str] or None
            Parameter names.
        transform : ParamTransform or None
            Parameter transformation.
        bounds : list[tuple] or None
            Per-parameter bounds.

        Returns
        -------
        Objective
        """
        if not _JAX_AVAILABLE:
            raise ImportError("JAX is required for Objective.from_jax")

        # Numpy-compatible wrappers
        def log_likelihood(params: np.ndarray) -> float:
            return float(ll_fn(jnp.array(params, dtype=jnp.float64)))

        def gradient(params: np.ndarray) -> np.ndarray:
            return np.asarray(grad_fn(jnp.array(params, dtype=jnp.float64)))

        return cls(
            fn=log_likelihood,
            grad=gradient,
            jax_fn=ll_fn,
            jax_grad=grad_fn,
            loglike_contribs_jax=loglike_contribs_fn,
            transform=transform,
            param_names=param_names,
            bounds=bounds,
        )

    @classmethod
    def from_numpy(
        cls,
        ll_fn: Callable,
        grad_fn: Callable,
        param_names: Optional[list[str]] = None,
        bounds: Optional[list[tuple[float, float]]] = None,
    ) -> "Objective":
        """Create an Objective from numpy functions (no JAX).

        Parameters
        ----------
        ll_fn : callable
            Log-likelihood: numpy array → float.
        grad_fn : callable
            Gradient: numpy array → numpy array.
        param_names : list[str] or None
            Parameter names.
        bounds : list[tuple] or None
            Per-parameter bounds.

        Returns
        -------
        Objective
        """
        return cls(
            fn=ll_fn,
            grad=grad_fn,
            jax_fn=None,
            jax_grad=None,
            transform=None,
            param_names=param_names,
            bounds=bounds,
        )
