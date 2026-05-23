"""Model specification DSL: scoped terms and interaction variables.

This module provides the core specification helpers for location choice models:

- ``InteractionTerm`` — interaction variable (product of two columns)
- ``ScopedTerm`` — variable with explicit coefficient scope
- ``interaction()`` — convenience constructor for InteractionTerm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class InteractionTerm:
    """Specification for an interaction variable (product of two columns).

    Interaction terms describe pairwise variables whose values are computed
    before estimation from existing columns in a ``ChoiceTable``. The first
    supported operation is a product, which covers common formulas such as
    household income times rent or worker sector times job density.

    Parameters
    ----------
    name : str
        Name of the generated interaction variable column.
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
