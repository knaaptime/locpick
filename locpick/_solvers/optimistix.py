"""Optimistix-based solver for JAX-native optimization.

Uses Optimistix solvers (BFGS, LBFGS, etc.) for fully JAX-native
optimization with JIT compilation and ``jax.vmap`` support.

Optimistix is the recommended JAX-native optimizer for choice models:

* Modular line search (BacktrackingArmijo, trust region, etc.)
* Actively maintained (by the Equinox/JAX ecosystem team)
* JIT + vmap compatible
* Interoperable with Optax first-order methods via OptaxMinimiser

.. note::

   Optimistix does not support box constraints (L-BFGS-B).  For models
   that require bounded parameters (e.g. ρ ∈ (0, 1] in SCL), the solver
   uses a sigmoid reparameterization to enforce bounds implicitly.

   For models where scipy's L-BFGS-B converges reliably, prefer
   :class:`LBFGSSolver` with the JAX backend — it uses scipy's robust
   line search while still benefiting from fast JAX gradients.

Requires the ``locpick[jax]`` extra and ``optimistix``.
"""

from __future__ import annotations

import numpy as np

from .protocol import SolverResult


class OptimistixSolver:
    """Optimistix-based solver for JAX-native optimization.

    This solver keeps all computation inside JAX, avoiding the
    Python↔JAX conversion overhead that occurs on every iteration
    when using :class:`LBFGSSolver` with JAX-computed objectives.

    When the model provides JAX-native functions (via the ``.jax_fn``
    attribute), the solver stays entirely within JAX — no numpy
    conversion on each iteration.  This enables ``jax.vmap``-based
    multi-start estimation.

    Parameters
    ----------
    method : str
        Optimistix solver name.  Supported names:
        ``"bfgs"``, ``"lbfgs"``, ``"adam"``, ``"adabelief"``.
        Default ``"lbfgs"``.
    rtol : float
        Relative tolerance for convergence.  Default ``1e-6``.
    atol : float
        Absolute tolerance for convergence.  Default ``1e-6``.
    maxiter : int
        Maximum number of optimisation steps.  Default ``500``.
    learning_rate : float
        Learning rate for Optax-based methods (``"adam"``, ``"adabelief"``).
        Ignored for BFGS/LBFGS.  Default ``1e-2``.
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

    # Map of method names to Optimistix solver constructors
    _METHODS = {
        "bfgs": "bfgs",
        "lbfgs": "lbfgs",
        "adam": "adam",
        "adabelief": "adabelief",
    }

    def __init__(
        self,
        method: str = "lbfgs",
        rtol: float = 1e-6,
        atol: float = 1e-6,
        maxiter: int = 500,
        learning_rate: float = 1e-2,
        n_starts: int = 1,
        start_scale: float = 1.0,
        seed: int = 0,
    ):
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.n_starts = n_starts
        self.start_scale = start_scale
        self.seed = seed

    def _make_solver(self):
        """Create the Optimistix solver instance."""
        import optimistix as optx

        name = self.method.lower()
        if name not in self._METHODS:
            raise ValueError(
                f"Unknown method '{self.method}'. Choose from: {list(self._METHODS.keys())}"
            )

        if name == "bfgs":
            return optx.BFGS(rtol=self.rtol, atol=self.atol)
        elif name == "lbfgs":
            return optx.LBFGS(rtol=self.rtol, atol=self.atol)
        elif name in ("adam", "adabelief"):
            import optax

            optax_ctor = optax.adam if name == "adam" else optax.adabelief
            optax_opt = optax_ctor(self.learning_rate)
            return optx.OptaxMinimiser(optax_opt, rtol=self.rtol, atol=self.atol)
        else:
            raise ValueError(f"Unhandled method: {name}")

    @staticmethod
    def _serialize_stats(stats: dict | None) -> dict:
        """Convert Optimistix/JAX stats payloads to Python scalars/lists."""
        if not isinstance(stats, dict):
            return {}

        serialized = {}
        for key, value in stats.items():
            try:
                arr = np.asarray(value)
                serialized[key] = arr.item() if arr.ndim == 0 else arr.tolist()
            except Exception:
                serialized[key] = str(value)
        return serialized

    @staticmethod
    def _status_metadata(result, optx) -> tuple[str, str, int]:
        """Return a stable (name, message, code) tuple for an Optimistix result."""
        if result == optx.RESULTS.successful:
            name = "successful"
        elif result == optx.RESULTS.nonlinear_max_steps_reached:
            name = "nonlinear_max_steps_reached"
        elif result == optx.RESULTS.nonlinear_divergence:
            name = "nonlinear_divergence"
        elif result == optx.RESULTS.nonfinite:
            name = "nonfinite"
        else:
            name = "unknown"

        try:
            message = str(optx.RESULTS[result])
        except Exception:
            message = str(result)

        code = int(getattr(result, "_value", -1))
        return name, message, code

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
            Optimization objective carrying ``fn``, optional JAX-native
            ``jax_fn``/``jax_grad``, and optional parameter metadata.
        x0 : np.ndarray
            Initial parameter values.
        param_names : list[str]
            Names for each parameter.
        bounds : list of tuple or None
            Per-parameter bounds ``(min, max)``.  When bounds are
            provided, the solver uses a sigmoid reparameterization
            to enforce them implicitly (since Optimistix does not
            support box constraints natively).
        fixed_mask : np.ndarray or None
            Boolean mask — ``True`` marks parameters held fixed.
            Supported by optimizing only free parameters and then
            reconstructing the full parameter vector.
        **kwargs
            Additional solver-specific options.

        Returns
        -------
        SolverResult
        """
        from locpick._jax.objective import Objective

        if not isinstance(objective, Objective):
            raise TypeError("OptimistixSolver.solve expects an Objective instance.")

        obj = objective
        ll_jax = obj.jax_fn
        grad_jax = obj.jax_grad
        log_likelihood_fn = obj.fn
        if obj.param_names is not None and not param_names:
            param_names = obj.param_names
        if obj.bounds is not None and bounds is None:
            bounds = obj.bounds

        try:
            import jax
            import jax.numpy as jnp
            import optimistix as optx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OptimistixSolver requires jax and optimistix. "
                "Install with: pip install locpick[jax] && pip install optimistix"
            ) from exc

        jax.config.update("jax_enable_x64", True)
        x0_full = np.asarray(x0, dtype=np.float64)

        fixed_mask_arr = None
        free_mask_arr = None
        free_indices = None
        if fixed_mask is not None:
            fixed_mask_arr = np.asarray(fixed_mask, dtype=bool)
            if fixed_mask_arr.shape != x0_full.shape:
                raise ValueError(
                    f"fixed_mask must have shape {x0_full.shape}; got {fixed_mask_arr.shape}."
                )
            free_mask_arr = ~fixed_mask_arr
            free_indices = np.where(free_mask_arr)[0]

        if ll_jax is not None and grad_jax is not None:
            # Pure JAX path — no numpy conversion per iteration
            ll_fn = ll_jax
        else:
            # Fallback: convert between numpy and JAX each iteration
            def ll_fn(p):
                return log_likelihood_fn(np.asarray(p))

        base_ll_fn = ll_fn
        if free_mask_arr is not None:
            if not np.any(free_mask_arr):
                ll_val = float(base_ll_fn(jnp.array(x0_full, dtype=jnp.float64)))
                return SolverResult(
                    coefficients=x0_full.copy(),
                    hessian=None,
                    log_likelihood=ll_val,
                    n_iterations=0,
                    converged=True,
                    message="All parameters fixed",
                    solver_name="optimistix",
                    raw={
                        "method": self.method.lower(),
                        "rtol": float(self.rtol),
                        "atol": float(self.atol),
                        "maxiter": int(self.maxiter),
                        "n_starts": int(self.n_starts),
                        "start_scale": float(self.start_scale),
                        "seed": int(self.seed),
                        "optimistix_result": "all_fixed",
                        "optimistix_message": "All parameters fixed",
                        "optimistix_code": 0,
                        "num_accepted_steps": 0,
                        "run_diagnostics": [],
                        "selected_start": 0,
                    },
                )

            free_indices_jax = jnp.array(free_indices, dtype=jnp.int32)
            x0_fixed_jax = jnp.array(x0_full, dtype=jnp.float64)

            def _expand_free(p_free):
                return x0_fixed_jax.at[free_indices_jax].set(p_free)

            def ll_fn(p_free):
                return base_ll_fn(_expand_free(p_free))

            x0 = x0_full[free_mask_arr]
            if bounds is not None:
                bounds = [b for i, b in enumerate(bounds) if free_mask_arr[i]]
            if param_names:
                param_names = [p for i, p in enumerate(param_names) if free_mask_arr[i]]
        else:
            x0 = x0_full

        # Build the negated objective (minimise negative LL = maximise LL)
        def neg_ll(params, args=None):
            return -ll_fn(params)

        # Handle bounds via sigmoid reparameterization
        if bounds is not None:
            # Sigmoid reparameterization: unconstrained α → bounded θ
            # θ = lower + (upper - lower) * sigmoid(α)
            # dθ/dα = (upper - lower) * sigmoid(α) * (1 - sigmoid(α))
            lower_bounds = jnp.array(
                [b[0] if b[0] is not None else -jnp.inf for b in bounds],
                dtype=jnp.float64,
            )
            upper_bounds = jnp.array(
                [b[1] if b[1] is not None else jnp.inf for b in bounds],
                dtype=jnp.float64,
            )
            has_lower = jnp.isfinite(lower_bounds)
            has_upper = jnp.isfinite(upper_bounds)
            has_both = has_lower & has_upper
            ~has_lower & ~has_upper

            # Convert x0 to unconstrained space (inverse sigmoid)
            x0_jax = jnp.array(x0, dtype=jnp.float64)
            # For bounded params: α = logit((θ - lower) / (upper - lower))
            # For unbounded params: α = θ
            x0_unconstrained = jnp.where(
                has_both,
                jnp.log((x0_jax - lower_bounds) / (upper_bounds - x0_jax + 1e-30) + 1e-30),
                x0_jax,
            )

            def bounded_neg_ll(alpha, args=None):
                # Transform unconstrained α → bounded θ
                theta = jnp.where(
                    has_both,
                    lower_bounds + (upper_bounds - lower_bounds) * jax.nn.sigmoid(alpha),
                    jnp.where(
                        has_lower,
                        lower_bounds + jax.nn.softplus(alpha),
                        jnp.where(
                            has_upper,
                            upper_bounds - jax.nn.softplus(-alpha),
                            alpha,  # unbounded: α = θ
                        ),
                    ),
                )
                return -ll_fn(theta)

            objective = bounded_neg_ll
            x0_opt = x0_unconstrained
        else:
            objective = neg_ll
            x0_opt = jnp.array(x0, dtype=jnp.float64)

        solver = self._make_solver()
        run_diagnostics: list[dict] = []

        # Single-start optimisation
        if self.n_starts == 1:
            params_opt, n_iter, converged, run_diag = self._run_single(objective, solver, x0_opt)
            run_diag["start_index"] = 0
            run_diag["objective_value"] = float(objective(params_opt))
            run_diagnostics.append(run_diag)
            selected_start = 0
            message = run_diag["status_message"]
        else:
            # Multi-start: run n_starts independent optimisations
            key = jax.random.PRNGKey(self.seed)
            best_val = jnp.inf
            best_params = x0_opt
            best_converged = False
            selected_start = 0
            total_iter = 0

            for i in range(self.n_starts):
                key, subkey = jax.random.split(key)
                perturbation = jax.random.normal(subkey, shape=x0.shape) * self.start_scale
                x0_i = jnp.array(x0, dtype=jnp.float64) + perturbation
                if bounds is not None:
                    # Convert perturbed start to unconstrained space
                    x0_i = jnp.where(
                        has_both,
                        jnp.log((x0_i - lower_bounds) / (upper_bounds - x0_i + 1e-30) + 1e-30),
                        x0_i,
                    )

                params_i, n_iter_i, converged_i, run_diag = self._run_single(
                    objective, solver, x0_i
                )
                total_iter += n_iter_i

                val_i = float(objective(params_i))
                run_diag["start_index"] = int(i)
                run_diag["objective_value"] = val_i
                run_diagnostics.append(run_diag)
                if val_i < float(best_val):
                    best_val = val_i
                    best_params = params_i
                    best_converged = converged_i
                    selected_start = int(i)

            params_opt = best_params
            n_iter = total_iter
            converged = bool(best_converged)
            message = run_diagnostics[selected_start]["status_message"]

        # Transform back from unconstrained space if bounded
        if bounds is not None:
            coefficients = jnp.where(
                has_both,
                lower_bounds + (upper_bounds - lower_bounds) * jax.nn.sigmoid(params_opt),
                jnp.where(
                    has_lower,
                    lower_bounds + jax.nn.softplus(params_opt),
                    jnp.where(
                        has_upper,
                        upper_bounds - jax.nn.softplus(-params_opt),
                        params_opt,  # unbounded: α = θ
                    ),
                ),
            )
        else:
            coefficients = params_opt

        coefficients = np.asarray(coefficients)
        if free_mask_arr is not None:
            full = x0_full.copy()
            full[free_mask_arr] = coefficients
            coefficients = full

        # Hessian is computed lazily by the model via Objective.hessian()
        # when standard errors are requested, rather than eagerly here.
        hessian_inv = None

        ll_val = float(base_ll_fn(jnp.array(coefficients, dtype=jnp.float64)))

        selected_diag = run_diagnostics[selected_start] if run_diagnostics else {}

        raw_payload = {
            "method": self.method.lower(),
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "maxiter": int(self.maxiter),
            "n_starts": int(self.n_starts),
            "start_scale": float(self.start_scale),
            "seed": int(self.seed),
            "optimistix_result": selected_diag.get("status_name", "unknown"),
            "optimistix_message": selected_diag.get("status_message", ""),
            "optimistix_code": selected_diag.get("status_code", -1),
            "num_accepted_steps": selected_diag.get("num_accepted_steps", n_iter),
            "run_diagnostics": run_diagnostics,
            "selected_start": int(selected_start),
        }
        if self.method.lower() in ("adam", "adabelief"):
            raw_payload["learning_rate"] = float(self.learning_rate)

        return SolverResult(
            coefficients=coefficients,
            hessian=hessian_inv,
            log_likelihood=ll_val,
            n_iterations=n_iter,
            converged=converged,
            message=message,
            solver_name="optimistix",
            raw=raw_payload,
        )

    def _run_single(self, objective, solver, x0):
        """Run a single optimisation with the given solver."""
        import optimistix as optx

        sol = optx.minimise(objective, solver, x0, max_steps=self.maxiter, throw=False)
        params = sol.value
        status_name, status_message, status_code = self._status_metadata(sol.result, optx)
        converged = status_name == "successful"

        if hasattr(sol.state, "num_accepted_steps"):
            n_iter = int(sol.state.num_accepted_steps)
        else:
            stats_dict = self._serialize_stats(sol.stats)
            n_iter = int(stats_dict.get("num_steps", self.maxiter))

        run_diag = {
            "status_name": status_name,
            "status_message": status_message,
            "status_code": status_code,
            "num_accepted_steps": int(n_iter),
            "stats": self._serialize_stats(sol.stats),
        }

        return params, n_iter, converged, run_diag
