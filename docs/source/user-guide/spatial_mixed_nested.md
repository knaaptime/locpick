# Mixed Nested Logit with Spatial Correlation (MixedNestedMNL + graph)

## Overview

A `MixedNestedMNL` model constructed with a spatial `graph=` argument estimates the most general model in the locpick spatial hierarchy. It combines three structures:

1. **Nested logit upper level**: alternatives are grouped into nests, each with a nest dissimilarity parameter $\lambda_m \in (0, 1]$
2. **Spatial lower levels**: within each nest, spatial correlation between contiguous alternatives is captured via a paired GNL structure with nest-specific spatial dissimilarity $\rho_m \in (0, 1]$
3. **Random coefficients**: unobserved heterogeneity is captured via mixed logit with simulated maximum likelihood

This is appropriate when alternatives are **spatially correlated**, **grouped into nests**, and decision-makers exhibit **heterogeneous preferences**.

```{warning}
The spatial mixed-nested model does **not** support alternative sampling correction. Always use the full alternative set.
```

## Mathematical Formulation

The choice probability for alternative $i$ in nest $m$ is:

$$P_i = \frac{1}{R} \sum_{r=1}^{R} P_{\text{SCL}}(i \mid m, \beta^r) \times P_{\text{NL}}(m \mid \beta^r)$$

where $\beta^r$ is the $r$-th draw of the random coefficients, approximated by simulated maximum likelihood.

## Quick Start

```python
from locpick import ChoiceTable, MixedNestedMNL
from locpick.models.nested import NestSpec, NestingTree
from locpick.models.mixed import ParamDistribution
from libpysal import graph

# Build spatial adjacency from zone geometries
g = graph.Graph.build_contiguity(tracts_gdf, rook=False)

# Define nesting structure
tree = NestingTree(
    nests=[
        NestSpec(name="urban", alt_ids=[0, 1, 2, 3]),
        NestSpec(name="suburban", alt_ids=[4, 5, 6, 7]),
    ]
)

# Define random parameters
random_params = {
    "cost": ParamDistribution(distribution="normal", param="cost"),
}

ct = ChoiceTable.from_tables(choosers, alternatives, chosen_alternatives=choices)
model = MixedNestedMNL(
    ct,
    formula="cost + time - 1",
    graph=g,
    nests=tree,
    random_params=random_params,
    n_draws=100,
)
result = model.fit()
print(result.summary())
```

## Random Parameters

Random parameters are specified as a dictionary mapping parameter names to `ParamDistribution` objects:

```python
from locpick.models.mixed import ParamDistribution

random_params = {
    "cost": ParamDistribution(distribution="normal", param="cost"),
    "time": ParamDistribution(distribution="lognormal", param="time"),
}
```

Supported distributions: `normal`, `lognormal`, `triangular`, `uniform` (see [MixedMNL with spatial correlation](spatial_mixed.md) for details).

## Draw Types

```python
# QMC draws (default, Sobol sequences — most efficient)
model = MixedNestedMNL(
    ct, formula="cost + time - 1", graph=g, nests=tree,
    random_params=random_params, n_draws=100, draw_type="qmc",
)

# Halton draws
model = MixedNestedMNL(
    ct, formula="cost + time - 1", graph=g, nests=tree,
    random_params=random_params, n_draws=250, draw_type="halton",
)

# Pseudo-random draws
model = MixedNestedMNL(
    ct, formula="cost + time - 1", graph=g, nests=tree,
    random_params=random_params, n_draws=500, draw_type="random",
)
```

## Interpreting Results

| Parameter | Interpretation |
|---|---|
| $\rho_m$ | Spatial dissimilarity within nest $m$ |
| $\lambda_m$ | Nest dissimilarity |
| $\beta$ | Fixed utility coefficients |
| $\sigma$ | Random coefficient standard deviations |

## References

- Al-Haideri et al. (2026). Cyclists' crossing behaviour at roundabouts: A Generalized Spatially Correlated Nested Logit model.
