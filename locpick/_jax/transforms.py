"""Parameter transformations for constrained optimization.

Many choice model parameters have natural constraints (e.g. ρ ∈ (0, 1],
standard deviations ≥ 0).  These transformations map unconstrained
optimization parameters to the constrained natural scale, enabling
box-constraint-free optimization.

Each transform provides:

- ``constrain(x)``: unconstrained → constrained (natural scale)
- ``log_det_jac(x)``: log |det J| of the transformation, for delta-method
  standard error computation
"""

from __future__ import annotations

import jax.numpy as jnp


class Identity:
    """Identity transformation: x → x (no constraint)."""

    def constrain(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def log_det_jac(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)


class Sigmoid:
    """Sigmoid (logistic) transformation: x → lower + (upper - lower) * σ(x).

    Maps unconstrained parameters to a bounded interval (lower, upper).
    Commonly used for the SCL dissimilarity parameter ρ ∈ (0, 1].

    Parameters
    ----------
    lower : float
        Lower bound.  Default 0.0.
    upper : float
        Upper bound.  Default 1.0.
    """

    def __init__(self, lower: float = 0.0, upper: float = 1.0):
        self.lower = lower
        self.upper = upper

    def constrain(self, x: jnp.ndarray) -> jnp.ndarray:
        """σ(x) scaled to (lower, upper)."""
        return self.lower + (self.upper - self.lower) * jax_sigmoid(x)

    def log_det_jac(self, x: jnp.ndarray) -> jnp.ndarray:
        """log |d(constrained)/d(unconstrained)| = log((upper-lower) * σ(x) * (1-σ(x)))."""
        s = jax_sigmoid(x)
        return jnp.log(jnp.maximum((self.upper - self.lower) * s * (1.0 - s), 1e-30))


class SoftPlus:
    """Softplus transformation: x → log(1 + exp(x)).

    Maps unconstrained parameters to (0, ∞).  Used for standard
    deviations and other positive parameters.
    """

    def constrain(self, x: jnp.ndarray) -> jnp.ndarray:
        """log(1 + exp(x)), numerically stable."""
        return jnp.log1p(jnp.exp(jnp.clip(x, -50, 50)))

    def log_det_jac(self, x: jnp.ndarray) -> jnp.ndarray:
        """log σ(x) where σ is the logistic sigmoid."""
        return jnp.log(jnp.maximum(jax_sigmoid(x), 1e-30))


class Tanh:
    """Tanh transformation: x → tanh(x).

    Maps unconstrained parameters to (-1, 1).  Used for the SAR
    spatial autoregressive parameter ρ ∈ (-1, 1).
    """

    def constrain(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(x)

    def log_det_jac(self, x: jnp.ndarray) -> jnp.ndarray:
        """log |d(tanh(x))/dx| = log(1 - tanh(x)^2) = log(sech^2(x))."""
        return jnp.log(jnp.maximum(1.0 - jnp.tanh(x) ** 2, 1e-30))


class Exp:
    """Exponential transformation: x → exp(x).

    Maps unconstrained parameters to (0, ∞).  Simpler than SoftPlus
    but less numerically stable near zero.
    """

    def constrain(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(jnp.clip(x, -50, 50))

    def log_det_jac(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(x, -50, 50)


class ParamTransform:
    """Per-parameter transformation list for constrained optimization.

    Holds one transform per parameter, applying them element-wise to
    the unconstrained parameter vector to produce the constrained
    (natural-scale) parameter vector.

    Parameters
    ----------
    transforms : list of transform objects
        One transform per parameter.  Each must implement
        ``constrain(x)`` and ``log_det_jac(x)``.

    Examples
    --------
    >>> from .transforms import ParamTransform, Identity, Sigmoid
    >>> # SCL model: [beta_0, beta_1, rho] → [beta_0, beta_1, sigmoid(alpha_rho)]
    >>> pt = ParamTransform([Identity(), Identity(), Sigmoid(0, 1)])
    >>> x = jnp.array([0.5, -0.1, 0.0])
    >>> pt.constrain(x)  # rho ≈ 0.5
    """

    def __init__(self, transforms: list):
        self.transforms = transforms
        self.n_params = len(transforms)

    def constrain(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply per-parameter transforms to produce constrained parameters.

        Parameters
        ----------
        x : jnp.ndarray, shape (n_params,)
            Unconstrained parameter vector.

        Returns
        -------
        jnp.ndarray, shape (n_params,)
            Constrained (natural-scale) parameter vector.
        """
        return jnp.stack([t.constrain(x[i]) for i, t in enumerate(self.transforms)])

    def log_det_jac(self, x: jnp.ndarray) -> jnp.ndarray:
        """Log |det J| of the transformation (for delta-method SEs).

        Parameters
        ----------
        x : jnp.ndarray, shape (n_params,)
            Unconstrained parameter vector.

        Returns
        -------
        jnp.ndarray, shape (n_params,)
            Per-parameter log |d(constrained)/d(unconstrained)|.
        """
        return jnp.stack([t.log_det_jac(x[i]) for i, t in enumerate(self.transforms)])

    @classmethod
    def for_scl(cls, k: int):
        """Create a transform for SCL models.

        Parameters
        ----------
        k : int
            Number of utility coefficients (beta parameters).

        Returns
        -------
        ParamTransform
            Transform with Identity for betas and Sigmoid for rho.
        """
        return cls([Identity()] * k + [Sigmoid(0, 1)])

    @classmethod
    def for_mscl(cls, k_fixed: int, k_random: int):
        """Create a transform for MSCL models.

        Parameters
        ----------
        k_fixed : int
            Number of fixed utility coefficients.
        k_random : int
            Number of random parameters.

        Returns
        -------
        ParamTransform
            Transform with Identity for fixed betas, Sigmoid for rho,
            Identity for random means, and SoftPlus for random spreads.
        """
        transforms = [Identity()] * k_fixed  # fixed betas
        transforms.append(Sigmoid(0, 1))  # rho
        transforms.extend([Identity()] * k_random)  # random means
        transforms.extend([SoftPlus()] * k_random)  # random spreads
        return cls(transforms)

    @classmethod
    def for_sar_mnl(cls, k: int):
        """Create a transform for SAR-MNL models.

        Parameters
        ----------
        k : int
            Number of utility coefficients (beta parameters).

        Returns
        -------
        ParamTransform
            Transform with Identity for betas and Tanh for rho.
        """
        return cls([Identity()] * k + [Tanh()])

    @classmethod
    def from_bounds(cls, bounds: list[tuple[float, float] | None]):
        """Create a transform from per-parameter bounds.

        Parameters
        ----------
        bounds : list of (lower, upper) tuples or None
            Per-parameter bounds.  None means unbounded (Identity).
            (lower, None) means lower-bounded (Exp).
            (None, upper) means upper-bounded.
            (lower, upper) means bounded (Sigmoid).

        Returns
        -------
        ParamTransform
        """
        transforms = []
        for b in bounds:
            if b is None:
                transforms.append(Identity())
            elif b[0] is not None and b[1] is not None:
                transforms.append(Sigmoid(b[0], b[1]))
            elif b[0] is not None:
                # Lower bound only: x → lower + softplus(x)
                transforms.append(_LowerBound(b[0]))
            elif b[1] is not None:
                # Upper bound only: x → upper - softplus(x)
                transforms.append(_UpperBound(b[1]))
            else:
                transforms.append(Identity())
        return cls(transforms)


class _LowerBound:
    """Transform: x → lower + softplus(x)."""

    def __init__(self, lower: float):
        self.lower = lower

    def constrain(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.lower + jnp.log1p(jnp.exp(jnp.clip(x, -50, 50)))

    def log_det_jac(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(jnp.maximum(jax_sigmoid(x), 1e-30))


class _UpperBound:
    """Transform: x → upper - softplus(x)."""

    def __init__(self, upper: float):
        self.upper = upper

    def constrain(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.upper - jnp.log1p(jnp.exp(jnp.clip(x, -50, 50)))

    def log_det_jac(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(jnp.maximum(jax_sigmoid(x), 1e-30))


def jax_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + jnp.exp(-x))
