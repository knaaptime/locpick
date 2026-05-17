"""Model specification primitives for location choice models.

This module provides ``ModelSpec`` with two complementary mechanisms:
formula/scoped-term specification and advanced P/X utility algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from .terms import (
    DataRef,
    InteractionTerm,
    LinearFunction,
    LinearTerm,
    ParamRef,
    ScopedTerm,
    interaction,
)


@dataclass
class ModelSpec:
    """Complete specification of a location choice model.

    Use formula strings for model terms, optionally augmented with
    scoped-term metadata and generated interactions. For advanced
    specifications, ``utility=...`` may be supplied using P/X algebra.

    Parameters
    ----------
    formula : str or None
        Formulaic formula string.
        E.g., ``"distance + jobs + log(accessibility)"``
    availability : str or None
        Column name for the binary availability variable.
    random_params : dict or None
        Mapping of parameter names to ParamDistribution for mixed logit.
        E.g., ``{"commute_time": ParamDistribution("normal")}``
    correlation : Any or None
        Optional model-specific correlation structure.

    Notes
    -----
    ``formula/scoped_terms`` and ``utility`` are mutually exclusive.
    """

    formula: Optional[str] = None
    utility: Optional[Any] = None
    availability: Optional[str] = None
    interactions: tuple[InteractionTerm, ...] = field(default_factory=tuple)
    scoped_terms: tuple[ScopedTerm, ...] = field(default_factory=tuple)
    random_params: Optional[dict[str, Any]] = None  # ParamDistribution (future)
    correlation: Optional[Any] = None

    # Internal (set during build)
    _design_info: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self.utility is not None and (self.formula is not None or self.scoped_terms):
            raise ValueError("ModelSpec accepts either utility or formula/scoped_terms, not both.")
        self.interactions = tuple(self.interactions or ())
        self.scoped_terms = tuple(self.scoped_terms or ())

    def _copy_with(
        self,
        *,
        interactions: Optional[tuple[InteractionTerm, ...]] = None,
        scoped_terms: Optional[tuple[ScopedTerm, ...]] = None,
    ) -> "ModelSpec":
        return ModelSpec(
            formula=self.formula,
            utility=self.utility,
            availability=self.availability,
            interactions=self.interactions if interactions is None else interactions,
            scoped_terms=self.scoped_terms if scoped_terms is None else scoped_terms,
            random_params=self.random_params,
            correlation=self.correlation,
        )

    def with_interaction(
        self,
        name: str,
        left: str,
        right: str,
        *,
        op: Literal["product"] = "product",
        missing_policy: Literal["error", "allow_unavailable"] = "error",
    ) -> "ModelSpec":
        """Return a copy of the spec with a generated interaction term."""
        terms = self.interactions + (
            interaction(
                name,
                left,
                right,
                op=op,
                missing_policy=missing_policy,
            ),
        )
        return self._copy_with(interactions=terms)

    def generic(self, variable: str, *, name: Optional[str] = None) -> "ModelSpec":
        """Return a copy with a generic coefficient for ``variable``."""
        term = ScopedTerm(variable=variable, scope="generic", name=name)
        return self._copy_with(scoped_terms=self.scoped_terms + (term,))

    def alternative_specific(
        self,
        variable: str,
        *,
        name: Optional[str] = None,
        reference: Any = None,
    ) -> "ModelSpec":
        """Return a copy with alternative-specific coefficients."""
        term = ScopedTerm(
            variable=variable,
            scope="alternative_specific",
            name=name,
            reference=reference,
        )
        return self._copy_with(scoped_terms=self.scoped_terms + (term,))

    def grouped(
        self,
        variable: str,
        groups: dict[str, list],
        *,
        name: Optional[str] = None,
    ) -> "ModelSpec":
        """Return a copy with grouped alternative coefficients."""
        term = ScopedTerm(
            variable=variable,
            scope="grouped",
            name=name,
            groups=groups,
        )
        return self._copy_with(scoped_terms=self.scoped_terms + (term,))

    def fixed(
        self,
        variable: str,
        value: float = 0.0,
        *,
        name: Optional[str] = None,
    ) -> "ModelSpec":
        """Return a copy with a fixed coefficient for ``variable``."""
        term = ScopedTerm(
            variable=variable,
            scope="generic",
            name=name,
            fixed=True,
            null_value=value,
        )
        return self._copy_with(scoped_terms=self.scoped_terms + (term,))

    def bounded(
        self,
        variable: str,
        *,
        lower: float = -1e6,
        upper: float = 1e6,
        name: Optional[str] = None,
        scope: Literal["generic", "alternative_specific", "grouped"] = "generic",
        groups: Optional[dict[str, list]] = None,
        reference: Any = None,
    ) -> "ModelSpec":
        """Return a copy with bounded coefficients for ``variable``."""
        term = ScopedTerm(
            variable=variable,
            scope=scope,
            name=name,
            groups=groups,
            reference=reference,
            bounds=(lower, upper),
        )
        return self._copy_with(scoped_terms=self.scoped_terms + (term,))

    def parameters(self) -> list[Any]:
        """Return free parameters in first-appearance order for utility specs."""
        if self.utility is None:
            return []

        seen: set[str] = set()
        params: list[ParamRef] = []
        for term in self._utility_terms():
            if isinstance(term.param, ParamRef) and term.param.name not in seen:
                seen.add(term.param.name)
                params.append(term.param)
        return params

    def data_refs(self) -> list[Any]:
        """Return utility data references in first-appearance order."""
        if self.utility is None:
            return []

        seen: set[str] = set()
        refs: list[DataRef] = []
        for term in self._utility_terms():
            if isinstance(term.data, DataRef) and term.data.name not in seen:
                seen.add(term.data.name)
                refs.append(term.data)
        return refs

    def validate(self, data) -> None:
        """Validate spec-level consistency before matrix construction."""
        if self.formula is None and not self.scoped_terms and self.utility is None:
            raise ValueError("ModelSpec requires utility, formula, or scoped_terms.")

    def build_design_matrix(self, data) -> Any:
        """Build the design matrix from this spec and the given data."""
        prepared = self.prepare_data(data)
        if self.utility is not None:
            return self._build_from_px_algebra(prepared)
        if self.scoped_terms and self.formula is None and self.utility is None:
            return self._build_from_scoped_terms(prepared)
        if self.formula is not None:
            return prepared.to_arrays(formula=self.formula)
        else:
            raise ValueError("No formula or scoped terms provided.")

    def _utility_terms(self) -> list[LinearTerm]:
        """Normalize utility expressions into a list of linear terms."""
        utility = self.utility
        if utility is None:
            return []
        if isinstance(utility, LinearFunction):
            return list(utility.terms)
        if isinstance(utility, LinearTerm):
            return [utility]
        if isinstance(utility, ParamRef):
            return [LinearTerm(param=utility, data=1.0)]
        if isinstance(utility, DataRef):
            return [LinearTerm(param=1.0, data=utility)]
        if isinstance(utility, (int, float)):
            return [LinearTerm(param=float(utility), data=1.0)]
        raise TypeError(
            "ModelSpec.utility must be a LinearFunction/LinearTerm/ParamRef/DataRef/number."
        )

    def _utility_column_specs(self) -> list[dict[str, Any]]:
        """Return ordered utility column specs with metadata and source terms."""
        specs: list[dict[str, Any]] = []
        param_to_idx: dict[str, int] = {}
        fixed_idx = 0

        for term in self._utility_terms():
            if isinstance(term.param, ParamRef):
                name = term.param.name
                existing_idx = param_to_idx.get(name)
                if existing_idx is None:
                    param_to_idx[name] = len(specs)
                    specs.append(
                        {
                            "name": name,
                            "bounds": tuple(term.param.bounds),
                            "fixed": bool(term.param.holdfast),
                            "initial": float(term.param.null_value),
                            "terms": [term],
                        }
                    )
                else:
                    existing = specs[existing_idx]
                    config = (
                        tuple(term.param.bounds),
                        bool(term.param.holdfast),
                        float(term.param.null_value),
                    )
                    existing_config = (
                        existing["bounds"],
                        existing["fixed"],
                        existing["initial"],
                    )
                    if config != existing_config:
                        raise ValueError(
                            f"Parameter '{name}' is used with inconsistent bounds/fixed/null_value."
                        )
                    existing["terms"].append(term)
            elif isinstance(term.param, (int, float)):
                fixed_idx += 1
                specs.append(
                    {
                        "name": f"__fixed_{fixed_idx}",
                        "bounds": (-1e6, 1e6),
                        "fixed": True,
                        "initial": float(term.param),
                        "terms": [term],
                    }
                )
            else:
                raise TypeError("Each utility term must use ParamRef or numeric coefficients.")

        return specs

    def prepare_data(self, data):
        """Return data with this spec's generated interactions applied."""
        prepared = data
        for term in self.interactions:
            if not hasattr(prepared, "add_interaction_expression"):
                raise TypeError(
                    "ModelSpec interactions require a ChoiceTable-like object "
                    "with add_interaction_expression()."
                )
            prepared = prepared.add_interaction_expression(
                term.name,
                term.left,
                term.right,
                op=term.op,
                missing_policy=term.missing_policy,
            )
        return prepared

    def _build_from_px_algebra(self, data) -> Any:
        """Build a design matrix from utility P/X algebra."""
        import numpy as np

        from locpick.data import ChoiceArrays

        df = data.to_frame()
        n_obs = data.n_observations
        n_alts = data.n_alternatives

        def _resolve_data_vector(source: Union[DataRef, float]) -> np.ndarray:
            if isinstance(source, DataRef):
                if source.name not in df.columns:
                    raise ValueError(
                        f"Data column '{source.name}' not found in data. "
                        f"Available columns: {list(df.columns)}"
                    )
                return df[source.name].to_numpy(dtype=np.float64)
            if isinstance(source, (int, float)):
                return np.full(len(df), float(source), dtype=np.float64)
            raise TypeError("Utility term data must be a DataRef or numeric scalar.")

        specs = self._utility_column_specs()
        if not specs:
            raise ValueError("Utility specification produced no estimable terms.")

        columns: list[np.ndarray] = []
        param_names: list[str] = []
        for spec in specs:
            vec = np.zeros(len(df), dtype=np.float64)
            for term in spec["terms"]:
                vec = vec + _resolve_data_vector(term.data)
            columns.append(vec)
            param_names.append(spec["name"])

        design_matrix = np.column_stack(columns)

        chosen = None
        if data.choice_col and data.choice_col in df.columns:
            chosen = df[data.choice_col].to_numpy(dtype=np.float64).reshape(n_obs, n_alts)

        available = None
        if data.available_col and data.available_col in df.columns:
            available = df[data.available_col].to_numpy(dtype=np.float64).reshape(n_obs, n_alts)

        return ChoiceArrays(
            design_matrix=design_matrix,
            chosen=chosen,
            available=available,
            weights=None,
            n_obs=n_obs,
            n_alts=n_alts,
            param_names=param_names,
            inclusion_probs=None,
            obs_ids=df[data.obs_id_col].values,
            alt_ids=df[data.alt_id_col].values,
        )

    def _build_from_scoped_terms(self, data) -> Any:
        """Build a design matrix from structured coefficient scopes."""
        import numpy as np
        import pandas as pd

        from locpick.data import ChoiceArrays

        df = data.to_frame()
        n_obs = data.n_observations
        n_alts = data.n_alternatives
        alt_id_col = data.alt_id_col

        columns: list[np.ndarray] = []
        param_names: list[str] = []
        param_bounds: list[tuple[float, float]] = []
        param_fixed: list[bool] = []
        param_initial: list[float] = []

        for term in self.scoped_terms:
            if term.variable not in df.columns:
                raise ValueError(
                    f"Column '{term.variable}' not found in data. "
                    f"Available columns: {list(df.columns)}"
                )

            values = df[term.variable].to_numpy(dtype=np.float64)
            base_name = term.name or term.variable

            term_bounds = term.bounds if term.bounds is not None else (-1e6, 1e6)
            term_fixed = term.fixed
            term_initial = term.null_value

            if term.scope == "generic":
                columns.append(values)
                param_names.append(base_name)
                param_bounds.append(term_bounds)
                param_fixed.append(term_fixed)
                param_initial.append(term_initial)

            elif term.scope == "alternative_specific":
                alt_ids = sorted(pd.unique(df[alt_id_col]))
                for alt_id in alt_ids:
                    if term.reference is not None and alt_id == term.reference:
                        continue
                    mask = (df[alt_id_col].to_numpy() == alt_id).astype(np.float64)
                    columns.append(values * mask)
                    param_names.append(f"{base_name}[{alt_id}]")
                    param_bounds.append(term_bounds)
                    param_fixed.append(term_fixed)
                    param_initial.append(term_initial)

            elif term.scope == "grouped":
                if not term.groups:
                    raise ValueError(f"Grouped scope for '{term.variable}' requires groups.")
                alt_values = df[alt_id_col]
                for group_name, group_alt_ids in term.groups.items():
                    mask = alt_values.isin(group_alt_ids).to_numpy(dtype=np.float64)
                    columns.append(values * mask)
                    param_names.append(f"{base_name}[{group_name}]")
                    param_bounds.append(term_bounds)
                    param_fixed.append(term_fixed)
                    param_initial.append(term_initial)
            else:
                raise ValueError(f"Unsupported coefficient scope: {term.scope}")

        design_matrix = np.column_stack(columns) if columns else np.empty((len(df), 0))

        chosen = None
        if data.choice_col and data.choice_col in df.columns:
            chosen = df[data.choice_col].to_numpy(dtype=np.float64).reshape(n_obs, n_alts)

        available = None
        if data.available_col and data.available_col in df.columns:
            available = df[data.available_col].to_numpy(dtype=np.float64).reshape(n_obs, n_alts)

        return ChoiceArrays(
            design_matrix=design_matrix,
            chosen=chosen,
            available=available,
            weights=None,
            n_obs=n_obs,
            n_alts=n_alts,
            param_names=param_names,
            inclusion_probs=None,
            obs_ids=df[data.obs_id_col].values,
            alt_ids=df[data.alt_id_col].values,
        )

    def param_metadata(self) -> dict[str, list]:
        """Return parameter metadata (bounds, fixed, initial) from scoped terms.

        Returns
        -------
        dict with keys "bounds", "fixed", "initial" — each a list aligned
        with the parameters produced by ``build_design_matrix()``.
        """
        param_bounds: list[tuple[float, float]] = []
        param_fixed: list[bool] = []
        param_initial: list[float] = []

        if self.utility is not None:
            specs = self._utility_column_specs()
            return {
                "bounds": [tuple(spec["bounds"]) for spec in specs],
                "fixed": [bool(spec["fixed"]) for spec in specs],
                "initial": [float(spec["initial"]) for spec in specs],
            }

        for term in self.scoped_terms:
            term_bounds = term.bounds if term.bounds is not None else (-1e6, 1e6)
            term_fixed = term.fixed
            term_initial = term.null_value

            if term.scope == "generic":
                param_bounds.append(term_bounds)
                param_fixed.append(term_fixed)
                param_initial.append(term_initial)
            elif term.scope == "alternative_specific":
                # We don't have data here, so we can't know the exact alt_ids.
                # Return per-term metadata; caller must expand if needed.
                param_bounds.append(term_bounds)
                param_fixed.append(term_fixed)
                param_initial.append(term_initial)
            elif term.scope == "grouped":
                if not term.groups:
                    raise ValueError(f"Grouped scope for '{term.variable}' requires groups.")
                for _ in term.groups:
                    param_bounds.append(term_bounds)
                    param_fixed.append(term_fixed)
                    param_initial.append(term_initial)

        return {
            "bounds": param_bounds,
            "fixed": param_fixed,
            "initial": param_initial,
        }

    def __repr__(self) -> str:
        if self.formula is not None:
            return f"ModelSpec(formula={self.formula!r})"
        elif self.utility is not None:
            return f"ModelSpec(utility={self.utility!r})"
        elif self.scoped_terms:
            return f"ModelSpec(scoped_terms={self.scoped_terms!r})"
        return "ModelSpec()"


# Backward-compat re-export. The real implementation is in locpick.models.mixed.
@dataclass
class ParamDistribution:
    """Distribution specification for a random parameter (mixed logit).

    .. deprecated::
        Import from ``locpick.mixed`` or ``locpick`` instead.
        This stub is kept for backward compatibility.

    Parameters
    ----------
    distribution : str
        Distribution name: ``"normal"``, ``"lognormal"``,
        ``"triangular"``, or ``"uniform"``.
    param : ParamRef or str
        The parameter to assign a random distribution to.
    """

    distribution: str
    param: Union[Any, str]

    def __post_init__(self) -> None:
        valid = {"normal", "lognormal", "triangular", "uniform"}
        if self.distribution not in valid:
            raise ValueError(
                f"Unknown distribution '{self.distribution}'. Must be one of {valid}."
            )

    @property
    def param_name(self) -> str:
        """Return the parameter name as a string."""
        if hasattr(self.param, "name"):
            return self.param.name
        return str(self.param)

    @property
    def n_params(self) -> int:
        """Number of distribution parameters (always 2: mean and spread)."""
        return 2
