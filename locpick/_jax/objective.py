"""Unified optimization objective for choice model estimation.

The :class:`Objective` class encapsulates a log-likelihood function,
its gradient, and optional Hessian, along with parameter transformations
and JAX-native function references.  It replaces the ad-hoc pattern of
passing ``log_likelihood_fn`` and ``gradient_fn`` with ``.jax_fn`` attributes.

Solvers negotiate with the ``Objective`` to get the interface they need:

- :class:`LBFGSSolver` calls ``fn`` and ``grad`` (numpy ↔ python)
- :class:`OptimistixSolver` calls ``jax_fn`` and ``jax_grad`` directly
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .transforms import ParamTransform


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
    """Per-observation log-likelihood contributions (JAX).

    Callable: params (jnp) → jnp array of shape (n_obs,).
    Per-observation log-likelihood contributions.
    """
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
        """Negative log-likelihood in JAX (for Optimistix minimization)."""
        return -self.jax_fn(params)

    def _hvp_grad_fn(self) -> Callable:
        """Gradient function used inside HVPs, built once and reused.

        ``hvp`` / ``hessian_hvp`` are called many times per trust-region
        iteration; rebuilding ``jax.grad(self.jax_fn)`` each time re-traces
        on every call.  Prefer the pre-jitted ``jax_grad`` when present.
        """
        if not hasattr(self, "_hvp_grad_cache"):
            self._hvp_grad_cache = self.jax_grad or jax.grad(self.jax_fn)
        return self._hvp_grad_cache

    @property
    def score_contribs(self) -> Callable:
        """Per-observation score matrix function (JAX).

        Returns a callable: params (jnp) → jnp array of shape (n_obs, n_params).
        Computed via ``jax.jacrev`` of ``loglike_contribs_jax``.

        Raises
        ------
        ValueError
            If ``loglike_contribs_jax`` is not set.
        """
        if self.loglike_contribs_jax is None:
            raise ValueError("score_contribs requires loglike_contribs_jax to be set.")
        # Cache the JIT'd score function to avoid recompilation on every call
        if not hasattr(self, "_score_fn_cache"):
            self._score_fn_cache = jax.jit(jax.jacrev(self.loglike_contribs_jax))
        return self._score_fn_cache

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Hessian-vector product at ``x`` with direction ``v``.

        Uses JAX forward-over-reverse AD via ``jax.jvp`` applied to the
        gradient function.

        Parameters
        ----------
        x : np.ndarray
            Parameter vector (natural scale).
        v : np.ndarray
            Direction vector.

        Returns
        -------
        np.ndarray
            Hessian-vector product H(x) @ v.
        """
        if self.jax_fn is None:
            raise RuntimeError("hvp requires a JAX-native log-likelihood (jax_fn).")
        x_jax = jnp.array(x, dtype=jnp.float64)
        v_jax = jnp.array(v, dtype=jnp.float64)
        grad_fn = self._hvp_grad_fn()
        _, hvp = jax.jvp(grad_fn, (x_jax,), (v_jax,))
        return np.asarray(hvp)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Compute the Hessian at ``x`` via HVPs.

        Parameters
        ----------
        x : np.ndarray
            Parameter vector (natural scale).

        Returns
        -------
        np.ndarray, shape (n_params, n_params)
            Hessian matrix of the log-likelihood.
        """
        return self.hessian_hvp(x)

    def hessian_hvp(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian via Hessian-vector products (HVP).

        Uses ``jax.jvp`` on the gradient function, vectorized over all
        basis vectors via ``jax.vmap``. This is 10–100× faster than
        ``jax.hessian`` for ``n_params >= 10``.

        Parameters
        ----------
        x : np.ndarray
            Parameter vector (natural scale).

        Returns
        -------
        np.ndarray, shape (n_params, n_params)
            Hessian matrix of the log-likelihood.
        """
        if self.jax_fn is None:
            raise RuntimeError("hessian_hvp requires a JAX-native log-likelihood (jax_fn).")

        grad_fn = self._hvp_grad_fn()

        # Build (and JIT) the full-Hessian function once per objective, so
        # repeated calls reuse the compiled vmap-over-jvp rather than
        # re-tracing it each time.
        if not hasattr(self, "_hessian_fn_cache"):

            def _dense_hessian(x_jax):
                eye = jnp.eye(x_jax.shape[0], dtype=jnp.float64)

                def hvp_col(e):
                    _, hvp = jax.jvp(grad_fn, (x_jax,), (e,))
                    return hvp

                hess_cols = jax.vmap(hvp_col)(eye)
                return 0.5 * (hess_cols + hess_cols.T)  # symmetrize

            self._hessian_fn_cache = jax.jit(_dense_hessian)

        x_jax = jnp.array(x, dtype=jnp.float64)
        return np.asarray(self._hessian_fn_cache(x_jax))

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_jax(
        cls,
        ll_fn,
        grad_fn,
        loglike_contribs_jax=None,
        param_names: Optional[list[str]] = None,
        transform: Optional[ParamTransform] = None,
        bounds: Optional[list[tuple[float, float]]] = None,
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
        loglike_contribs_jax : callable or None
            JIT-compiled per-observation log-likelihood: jnp array → jnp array
            of shape (n_obs,).
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

        # Numpy-compatible wrappers — minimise JAX↔NumPy overhead
        def log_likelihood(params: np.ndarray) -> float:
            return np.asarray(ll_fn(jnp.array(params, dtype=jnp.float64))).item()

        def gradient(params: np.ndarray) -> np.ndarray:
            out = grad_fn(jnp.array(params, dtype=jnp.float64))
            # Skip copy when output is already a CPU numpy array
            return out if isinstance(out, np.ndarray) else np.asarray(out)

        return cls(
            fn=log_likelihood,
            grad=gradient,
            jax_fn=ll_fn,
            jax_grad=grad_fn,
            loglike_contribs_jax=loglike_contribs_jax,
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
