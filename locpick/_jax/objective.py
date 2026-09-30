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

#: Relative step for central differences of the gradient (eps ** (1/3)).
_FD_STEP: float = float(np.finfo(np.float64).eps ** (1.0 / 3.0))

#: Errors JAX raises when a kernel cannot be differentiated a second time
#: (e.g. a foreign-function call wrapped in a first-order custom VJP).
_AD_FAILURES = (ValueError, TypeError, NotImplementedError)


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
    log_probs_jax: Optional[Callable] = None
    """Choice log-probabilities on the estimation data (JAX).

    Callable: params (jnp) → jnp array of shape (n_obs, n_alts).  Built by
    :func:`make_log_probs_fn` from the same kernel as the likelihood, so the
    probabilities agree with the fitted model by construction.
    """

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
        # The score matrix is (n_units, n_params) with n_params small, so
        # forward mode (one pass per parameter) is the cheap direction;
        # reverse mode runs one pass per observation and its memory grows with
        # n_obs squared for simulated likelihoods.  Kernels without
        # forward-mode rules fall back to reverse mode.
        if not hasattr(self, "_score_fn_cache"):
            forward = jax.jit(jax.jacfwd(self.loglike_contribs_jax))
            reverse = jax.jit(jax.jacrev(self.loglike_contribs_jax))

            def score_fn(x):
                if not getattr(self, "_score_forward_unavailable", False):
                    try:
                        return forward(x)
                    except _AD_FAILURES:
                        self._score_forward_unavailable = True
                return reverse(x)

            self._score_fn_cache = score_fn
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
        if not getattr(self, "_second_order_ad_unavailable", False):
            x_jax = jnp.array(x, dtype=jnp.float64)
            v_jax = jnp.array(v, dtype=jnp.float64)
            try:
                _, hvp = jax.jvp(self._hvp_grad_fn(), (x_jax,), (v_jax,))
                return np.asarray(hvp)
            except _AD_FAILURES:
                self._second_order_ad_unavailable = True
        # Central difference of the exact gradient along v.
        x = np.asarray(x, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        v_norm = np.linalg.norm(v)
        if v_norm == 0.0:
            return np.zeros_like(x)
        h = _FD_STEP * max(1.0, np.linalg.norm(x)) / v_norm
        return (self.grad(x + h * v) - self.grad(x - h * v)) / (2.0 * h)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Compute the Hessian at ``x``.

        Uses exact forward-over-reverse autodiff (:meth:`hessian_hvp`) when the
        kernel supports second derivatives.  Some do not: the sparsax sparse
        solves define a first-order VJP around a foreign-function call, so no
        mode of JAX autodiff can differentiate them twice.  The Hessian is
        then taken by central differences of the *exact* gradient
        (:meth:`hessian_fd`), accurate to roughly ``1e-8`` relative, which is
        ample for standard errors.

        Parameters
        ----------
        x : np.ndarray
            Parameter vector (natural scale).

        Returns
        -------
        np.ndarray, shape (n_params, n_params)
            Hessian matrix of the log-likelihood.
        """
        if not getattr(self, "_second_order_ad_unavailable", False):
            try:
                return self.hessian_hvp(x)
            except _AD_FAILURES:
                self._second_order_ad_unavailable = True
        return self.hessian_fd(x)

    def hessian_fd(self, x: np.ndarray) -> np.ndarray:
        """Hessian by central differences of the exact gradient.

        Costs ``2 * n_params`` gradient evaluations.  The step for each
        coordinate is ``eps**(1/3) * max(1, |x_i|)``, which balances
        truncation against rounding error for a central difference.

        Parameters
        ----------
        x : np.ndarray
            Parameter vector (natural scale).

        Returns
        -------
        np.ndarray, shape (n_params, n_params)
            Symmetrised Hessian of the log-likelihood.
        """
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        hess = np.empty((n, n))
        for i in range(n):
            step = np.zeros(n)
            step[i] = _FD_STEP * max(1.0, abs(x[i]))
            hess[:, i] = (self.grad(x + step) - self.grad(x - step)) / (2.0 * step[i])
        return 0.5 * (hess + hess.T)

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
        log_probs_jax=None,
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
        log_probs_jax : callable or None
            Log-probabilities on the estimation data: jnp array → jnp array
            of shape (n_obs, n_alts).  See :func:`make_log_probs_fn`.

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
            log_probs_jax=log_probs_jax,
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


def make_log_probs_fn(contribs_fn: Callable, n_obs: int, n_alts: int, *, mixed: bool) -> Callable:
    """Build ``params -> log P`` of shape (n_obs, n_alts) from a contributions kernel.

    ``contribs_fn(params, chosen, weights, available)`` returns per-observation
    log-likelihood contributions.  Every kernel lets ``chosen`` enter only
    through the final ``(log_probs * chosen).sum(axis=1)`` reduction, which
    makes the full probability matrix recoverable without a second kernel:

    - Without mixing the contributions are linear in ``chosen``, so their
      gradient with respect to it is the log-probability matrix itself — one
      backward pass.
    - With mixing a contribution is ``log mean_r exp(sum_j chosen_j log P_rj)``,
      so setting ``chosen`` to the indicator of alternative ``j`` yields the
      simulated ``log P_j`` exactly — one evaluation per alternative.

    Parameters
    ----------
    contribs_fn : callable
        ``(params, chosen, weights, available) -> (n_obs,)`` contributions;
        ``available=None`` keeps the data's own availability.
    n_obs, n_alts : int
        Problem dimensions.
    mixed : bool
        Whether the kernel integrates over mixing draws.

    Returns
    -------
    callable
        JIT-compiled ``(params, available=None) -> (n_obs, n_alts)``
        log-probabilities; ``available`` optionally restricts the choice set.
    """
    ones = jnp.ones(n_obs, dtype=jnp.float64)

    if not mixed:

        def log_probs(params, available=None):
            zeros = jnp.zeros((n_obs, n_alts), dtype=jnp.float64)
            return jax.grad(lambda chosen: contribs_fn(params, chosen, ones, available).sum())(
                zeros
            )

    else:

        def log_probs(params, available=None):
            def one_alt(j):
                chosen = jnp.zeros((n_obs, n_alts), dtype=jnp.float64).at[:, j].set(1.0)
                return contribs_fn(params, chosen, ones, available)

            return jax.lax.map(one_alt, jnp.arange(n_alts)).T

    return jax.jit(log_probs)
