"""Model specification DSL: parameter references, data references, and algebra.

This module provides the P/X algebra helpers for specifying utility functions
in location choice models:

- ``ParamRef`` — reference to a model parameter
- ``DataRef`` — reference to a data column
- ``InteractionTerm`` — chooser-alternative interaction variable
- ``ScopedTerm`` — variable with explicit coefficient scope
- ``LinearTerm`` — single term in a linear utility function
- ``LinearFunction`` — sum of LinearTerms
- ``P()`` — convenience constructor for ParamRef
- ``X()`` — convenience constructor for DataRef
- ``interaction()`` — convenience constructor for InteractionTerm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Union


@dataclass
class ParamRef:
    """Reference to a model parameter.

    Parameters
    ----------
    name : str
        Parameter name (e.g., ``"beta_distance"``).
    null_value : float
        Starting value for optimization. Default 0.0.
    bounds : tuple of float
        (lower, upper) bounds for the parameter. Default (-1e6, 1e6).
    holdfast : bool
        If True, the parameter is fixed at its null_value during estimation.
    """

    name: str
    null_value: float = 0.0
    bounds: tuple[float, float] = (-1e6, 1e6)
    holdfast: bool = False

    def __repr__(self) -> str:
        return f"P({self.name!r})"


@dataclass
class DataRef:
    """Reference to a data column.

    Parameters
    ----------
    name : str
        Column name in the choice data (e.g., ``"distance"``).
    """

    name: str

    def __repr__(self) -> str:
        return f"X({self.name!r})"


@dataclass(frozen=True)
class InteractionTerm:
    """Specification for a chooser-alternative interaction variable.

    Interaction terms describe variables whose values are built before
    estimation from existing columns in a ``ChoiceTable``. The first supported
    operation is a product, which covers common formulas such as household
    income times rent or worker sector times job density.

    Parameters
    ----------
    name : str
        Name of the generated interaction column.
    left : str
        Name of the first source column.
    right : str
        Name of the second source column.
    op : {"product"}
        Operation used to combine the two source columns.
    missing_policy : {"error", "allow_unavailable"}
        How missing generated values are handled. ``"error"`` requires all
        rows to be present. ``"allow_unavailable"`` permits missing values on
        unavailable alternatives but still requires available rows to be valid.
    """

    name: str
    left: str
    right: str
    op: Literal["product"] = "product"
    missing_policy: Literal["error", "allow_unavailable"] = "error"


@dataclass(frozen=True)
class ScopedTerm:
    """A variable with an explicit coefficient scope.

    Parameters
    ----------
    variable : str
        Column in the choice data.
    scope : {"generic", "alternative_specific", "grouped"}
        Coefficient scope used when compiling the design matrix.
    name : str or None
        Base parameter name. Defaults to ``variable``.
    groups : dict or None
        Mapping from group name to alternative ids for grouped scopes.
    reference : object or None
        Alternative id to omit from an alternative-specific scope.
    bounds : tuple of float or None
        (lower, upper) bounds for the parameter. None means unbounded.
        Only applies to generic scope; for alternative_specific and grouped,
        bounds apply to each coefficient in the group.
    fixed : bool
        If True, the parameter is held fixed at its initial value during
        estimation. The initial value is set via ``null_value``.
    null_value : float
        Starting value for optimization. Default 0.0. When ``fixed=True``,
        the parameter stays at this value.
    """

    variable: str
    scope: Literal["generic", "alternative_specific", "grouped"] = "generic"
    name: Optional[str] = None
    groups: Optional[dict[str, list]] = None
    reference: Any = None
    bounds: Optional[tuple[float, float]] = None
    fixed: bool = False
    null_value: float = 0.0


def interaction(
    name: str,
    left: str,
    right: str,
    *,
    op: Literal["product"] = "product",
    missing_policy: Literal["error", "allow_unavailable"] = "error",
) -> InteractionTerm:
    """Create an interaction term for use with ``ModelSpec``.

    Examples
    --------
    >>> interaction("income_x_rent", "income", "rent")
    InteractionTerm(name='income_x_rent', left='income', right='rent', op='product', missing_policy='error')
    """
    return InteractionTerm(
        name=name,
        left=left,
        right=right,
        op=op,
        missing_policy=missing_policy,
    )


@dataclass
class LinearTerm:
    """A single term in a linear utility function: param * data.

    Can represent:
    - ``P("beta") * X("distance")`` — a free parameter times a data column
    - ``P("beta")`` — a free parameter (constant term, data=1.0)
    - ``X("distance")`` — a fixed coefficient (param=1.0) times a data column
    - ``2.5 * X("distance")`` — a fixed coefficient times a data column

    Parameters
    ----------
    param : ParamRef or float
        The parameter reference or a fixed numeric coefficient.
    data : DataRef or float
        The data reference or a fixed numeric value (1.0 for constant).
    """

    param: Union[ParamRef, float]
    data: Union[DataRef, float]

    @property
    def is_free(self) -> bool:
        """Whether this term has a free (estimated) parameter."""
        return isinstance(self.param, ParamRef)

    @property
    def is_fixed(self) -> bool:
        """Whether this term has a fixed coefficient."""
        return isinstance(self.param, (int, float))

    def __repr__(self) -> str:
        if isinstance(self.param, ParamRef) and isinstance(self.data, DataRef):
            return f"{self.param!r} * {self.data!r}"
        elif isinstance(self.param, ParamRef):
            return f"{self.param!r}"
        elif isinstance(self.data, DataRef):
            return f"{self.param} * {self.data!r}"
        else:
            return f"{self.param} * {self.data}"


class LinearFunction:
    """Sum of LinearTerms representing a utility function.

    Constructed by adding LinearTerms together::

        utility = P("beta_dist") * X("distance") + P("beta_jobs") * X("jobs")

    Parameters
    ----------
    terms : list of LinearTerm
        The terms in the utility function.
    """

    def __init__(self, terms: Optional[list[LinearTerm]] = None):
        self.terms: list[LinearTerm] = terms or []

    def __add__(self, other: Union[LinearTerm, "LinearFunction", float]) -> "LinearFunction":
        if isinstance(other, LinearTerm):
            return LinearFunction(self.terms + [other])
        elif isinstance(other, LinearFunction):
            return LinearFunction(self.terms + other.terms)
        elif isinstance(other, (int, float)):
            return LinearFunction(self.terms + [LinearTerm(param=other, data=1.0)])
        return NotImplemented

    def __radd__(self, other: Union[LinearTerm, float]) -> "LinearFunction":
        if isinstance(other, (int, float)):
            return LinearFunction([LinearTerm(param=other, data=1.0)] + self.terms)
        return NotImplemented

    def __sub__(self, other: Union[LinearTerm, "LinearFunction", float]) -> "LinearFunction":
        if isinstance(other, LinearTerm):
            LinearTerm(param=other.param, data=other.data)
            return LinearFunction(self.terms + [other])
        elif isinstance(other, (int, float)):
            return LinearFunction(self.terms + [LinearTerm(param=-other, data=1.0)])
        return NotImplemented

    def parameters(self) -> list[ParamRef]:
        """Return all free parameters in order of first appearance."""
        seen: set[str] = set()
        params: list[ParamRef] = []
        for term in self.terms:
            if isinstance(term.param, ParamRef) and term.param.name not in seen:
                seen.add(term.param.name)
                params.append(term.param)
        return params

    def data_refs(self) -> list[DataRef]:
        """Return all data references in order of first appearance."""
        seen: set[str] = set()
        refs: list[DataRef] = []
        for term in self.terms:
            if isinstance(term.data, DataRef) and term.data.name not in seen:
                seen.add(term.data.name)
                refs.append(term.data)
        return refs

    def __repr__(self) -> str:
        if not self.terms:
            return "LinearFunction([])"
        return " + ".join(repr(t) for t in self.terms)


# ---------------------------------------------------------------------------
# Convenience functions for P/X algebra
# ---------------------------------------------------------------------------


def P(name: str, **kwargs) -> ParamRef:
    """Create a reference to a model parameter.

    Parameters
    ----------
    name : str
        Parameter name.
    **kwargs
        Additional keyword arguments passed to ``ParamRef``
        (``null_value``, ``bounds``, ``holdfast``).

    Returns
    -------
    ParamRef

    Examples
    --------
    >>> P("beta_distance")
    P('beta_distance')
    >>> P("beta_dist", bounds=(-10, 0))
    P('beta_dist')
    """
    return ParamRef(name=name, **kwargs)


def X(name: str) -> DataRef:
    """Create a reference to a data column.

    Parameters
    ----------
    name : str
        Column name in the choice data.

    Returns
    -------
    DataRef

    Examples
    --------
    >>> X("distance")
    X('distance')
    """
    return DataRef(name=name)


# ---------------------------------------------------------------------------
# Overload arithmetic operators on ParamRef and DataRef
# ---------------------------------------------------------------------------


def _mul_param_data(param: Union[ParamRef, float], data: Union[DataRef, float]) -> LinearTerm:
    """Multiply a parameter by a data reference to create a LinearTerm."""
    return LinearTerm(param=param, data=data)


# ParamRef * DataRef → LinearTerm
ParamRef.__mul__ = lambda self, other: (
    _mul_param_data(self, other) if isinstance(other, (DataRef, int, float)) else NotImplemented
)  # type: ignore[attr-defined]
ParamRef.__rmul__ = lambda self, other: (
    _mul_param_data(self, other) if isinstance(other, (DataRef, int, float)) else NotImplemented
)  # type: ignore[attr-defined]

# DataRef * ParamRef → LinearTerm
DataRef.__mul__ = lambda self, other: (
    _mul_param_data(other, self) if isinstance(other, (ParamRef, int, float)) else NotImplemented
)  # type: ignore[attr-defined]
DataRef.__rmul__ = lambda self, other: (
    _mul_param_data(other, self) if isinstance(other, (ParamRef, int, float)) else NotImplemented
)  # type: ignore[attr-defined]

# LinearTerm + LinearTerm → LinearFunction
LinearTerm.__add__ = lambda self, other: (
    LinearFunction([self, other])
    if isinstance(other, LinearTerm)
    else (LinearFunction([self]) + other if isinstance(other, LinearFunction) else NotImplemented)
)  # type: ignore[attr-defined]
LinearTerm.__radd__ = lambda self, other: (
    LinearFunction([other, self]) if isinstance(other, LinearTerm) else NotImplemented
)  # type: ignore[attr-defined]
