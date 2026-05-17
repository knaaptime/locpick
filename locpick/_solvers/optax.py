"""Optax-based solver for gradient-based optimization.

Uses first-order adaptive methods (Adam, etc.) from the Optax library.
These methods handle steep initial gradients well because per-parameter
adaptive learning rates naturally scale with gradient magnitude.

Requires the ``locpick[jax]`` extra and ``optax``.

.. note::

   First-order methods like Adam typically need many more iterations
   than second-order methods (L-BFGS-B) to converge.  For models
   where scipy's L-BFGS-B converges reliably, prefer :class:`LBFGSSolver`
   with the JAX backend — it benefits from fast JAX gradients while
   using scipy's robust line search.  Use :class:`OptaxSolver` when:

   * L-BFGS-B fails to converge (e.g. steep initial gradients)
   * You need GPU/TPU acceleration via JAX
   * You want to use ``jax.vmap`` for multi-start estimation
"""

from __future__ import annotations

import numpy as np

from .protocol import SolverResult


class OptaxSolver:
    """Optax-based solver using first-order adaptive methods.

    This solver uses Optax optimisers (Adam by default) for gradient-based
    maximisation of the log-likelihood.  It maximises the LL by negating
    the gradient (gradient ascent).

    When the model provides JAX-native functions (via the ``.jax_fn``
    attribute), the solver stays entirely within JAX — no numpy conversion
    on each iteration.  This enables future ``jax.vmap``-based multi-start
    estimation.

    Parameters
    ----------
    optimizer : str or optax.GradientTransformation
        Optax optimiser name or instance.  Supported names:
        ``"adam"``, ``"adamw"``, ``"rmsprop"``, ``"adabelief"``.
        Default ``"adam"``.
    learning_rate : float
        Learning rate for the optimiser.  Default ``1e-2``, which works
        well for choice models with steep initial gradients.
    maxiter : int
        Maximum number of optimisation iterations.  Default ``2000``.
    tol : float
        Convergence tolerance (relative LL change).  Default ``1e-8``.
    n_starts : int
        Number of multi-start runs.  If > 1, the solver runs
        ``n_starts`` independent optimisations from perturbed
        starting points and returns the best result.
    start_scale : float
        Standard deviation of random perturbation applied to ``x0``
        for multi-start runs (only used when ``n_starts > 1``).
    seed : int
        Random seed for multi-start perturbations.
    """

    # Map of optimizer names to optax constructors
    _OPTIMIZERS = {
        "adam": "adam",
        "adamw": "adamw",
        "rmsprop": "rmsprop",
        "adabelief": "adabelief",
    }

    def __init__(
        self,
        optimizer: str = "adam",
        learning_rate: float = 1e-2,
        maxiter: int = 2000,
        tol: float = 1e-8,
        n_starts: int = 1,
        start_scale: float = 1.0,
        seed: int = 0,
    ):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.maxiter = maxiter
        self.tol = tol
        self.n_starts = n_starts
        self.start_scale = start_scale
        self.seed = seed

    def _make_optimizer(self):
        """Create the optax optimizer."""
        import optax

        if isinstance(self.optimizer, str):
            name = self.optimizer.lower()
            if name not in self._OPTIMIZERS:
                raise ValueError(
                    f"Unknown optimizer '{self.optimizer}'. "
                    f"Choose from: {list(self._OPTIMIZERS.keys())}"
                )
            constructor = getattr(optax, self._OPTIMIZERS[name])
            return constructor(self.learning_rate)
        # If it's already an optax optimizer, use it directly
        return self.optimizer

    def solve(
        self,
        objective,
        x0: np.ndarray,
        param_names: list[str],
        bounds=None,
        fixed_mask=None,
        **kwargs,
    ) -> SolverResult:
        """Estimate parameters by maximising the log-likelihood.

        Parameters
        ----------
        objective : Objective
            Optimization objective carrying ``fn``, ``grad``, and optional
            JAX-native ``jax_fn``/``jax_grad`` references.
        x0 : np.ndarray
            Initial parameter values.
        param_names : list[str]
            Names for each parameter.
        bounds : list of tuple or None
            Per-parameter bounds ``(min, max)``.  Currently ignored —
            Optax does not support box constraints natively.
        fixed_mask : np.ndarray or None
            Boolean mask — ``True`` marks parameters held fixed.
            Currently not supported — raises ``NotImplementedError``.
        **kwargs
            Additional solver-specific options.

        Returns
        -------
        SolverResult
        """
        if fixed_mask is not None and np.any(fixed_mask):
            raise NotImplementedError("OptaxSolver does not yet support fixed parameters.")

        from locpick._jax.objective import Objective

        if not isinstance(objective, Objective):
            raise TypeError("OptaxSolver.solve expects an Objective instance.")

        obj = objective
        log_likelihood_fn = obj.fn
        gradient_fn = obj.grad
        ll_jax = obj.jax_fn
        grad_jax = obj.jax_grad
        if obj.param_names is not None and not param_names:
            param_names = obj.param_names
        if obj.bounds is not None and bounds is None:
            bounds = obj.bounds

        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:
            raise ImportError(
                "OptaxSolver requires jax and optax. "
                "Install with: pip install locpick[jax] && pip install optax"
            ) from exc

        jax.config.update("jax_enable_x64", True)

        if ll_jax is not None and grad_jax is not None:
            # Pure JAX path — no numpy conversion per iteration
            neg_grad_fn = jax.jit(lambda p: -grad_jax(p))
            ll_fn = ll_jax
        else:
            # Fallback: convert between numpy and JAX each iteration
            def neg_grad_fn(p):
                return -jnp.array(gradient_fn(np.asarray(p)), dtype=jnp.float64)

            def ll_fn(p):
                return log_likelihood_fn(np.asarray(p))

        optimizer = self._make_optimizer()

        # Single-start optimisation
        if self.n_starts == 1:
            params, n_iter, converged = self._run_single(
                neg_grad_fn, ll_fn, optimizer, jnp.array(x0, dtype=jnp.float64)
            )
        else:
            # Multi-start: run n_starts independent optimisations
            key = jax.random.PRNGKey(self.seed)
            best_ll = -jnp.inf
            best_params = jnp.array(x0, dtype=jnp.float64)
            total_iter = 0

            for i in range(self.n_starts):
                key, subkey = jax.random.split(key)
                perturbation = jax.random.normal(subkey, shape=x0.shape) * self.start_scale
                x0_i = jnp.array(x0, dtype=jnp.float64) + perturbation

                params_i, n_iter_i, _ = self._run_single(neg_grad_fn, ll_fn, optimizer, x0_i)
                total_iter += n_iter_i

                ll_i = float(ll_fn(params_i))
                if ll_i > float(best_ll):
                    best_ll = ll_i
                    best_params = params_i

            params = best_params
            n_iter = total_iter
            converged = True  # multi-start always "converges"

        # Compute Hessian at optimum for standard errors (if JAX-native)
        hessian_inv = None
        if ll_jax is not None:
            try:
                neg_ll_hess = jax.hessian(lambda p: -ll_jax(p))
                hessian_neg_ll = np.asarray(neg_ll_hess(jnp.array(params, dtype=jnp.float64)))
                hessian_inv = np.linalg.inv(hessian_neg_ll)
            except Exception:
                hessian_inv = None

        ll_val = float(ll_fn(jnp.array(params, dtype=jnp.float64)))

        return SolverResult(
            coefficients=np.asarray(params),
            hessian=hessian_inv,
            log_likelihood=ll_val,
            n_iterations=n_iter,
            converged=converged,
            message="Converged" if converged else "Max iterations reached",
            solver_name="optax",
            raw={},
        )

    def _run_single(self, neg_grad_fn, ll_fn, optimizer, x0):
        """Run a single optimisation with the given optimizer."""
        import jax.numpy as jnp
        import optax

        params = x0
        opt_state = optimizer.init(params)

        prev_ll = -jnp.inf
        converged = False
        n_iterations = 0

        for i in range(self.maxiter):
            neg_grad = neg_grad_fn(params)
            updates, opt_state = optimizer.update(neg_grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            n_iterations = i + 1

            # Check convergence every 10 iterations (for efficiency)
            if i % 10 == 0:
                ll = float(ll_fn(params))
                if abs(ll - prev_ll) < self.tol * max(1.0, abs(ll)):
                    converged = True
                    break
                prev_ll = ll

        return params, n_iterations, converged
