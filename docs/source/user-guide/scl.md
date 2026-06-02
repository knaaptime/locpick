# Spatially Correlated Logit (SCL)

## Overview

`SCL` is a convenience factory that returns one of the four base estimators depending on which spatial / nesting / random-coefficient features are active:

| Construction | Returns |
|---|---|
| `SCL(..., graph=g)` | `MNL` with spatial correlation |
| `SCL(..., graph=g, nests=...)` | `NestedMNL` with spatial correlation |
| `SCL(..., graph=g, random_params=...)` | `MixedMNL` with spatial correlation |
| `SCL(..., graph=g, nests=..., random_params=...)` | `MixedNestedMNL` |

The underlying model is the Spatially Correlated Logit of Bhat & Guo (2004), which captures spatial correlation between contiguous alternatives using a paired Generalised Nested Logit (PGNL) structure with a single dissimilarity parameter $\rho$ (per nest in the nested variants).

When $\rho = 1$, the model reduces to the Multinomial Logit (MNL). Values of $\rho < 1$ indicate positive spatial correlation between adjacent alternatives — decision-makers view nearby zones as closer substitutes than distant ones.

```{note}
New code should prefer constructing `MNL`, `NestedMNL`, `MixedMNL`, or `MixedNestedMNL` directly with `graph=` for clarity. The `SCL` factory is retained for compatibility with existing scripts.
```

```{warning}
Spatial GEV models do **not** support alternative sampling correction. The MNL's uniform conditioning property does not hold for non-MNL GEV models. Always use the full alternative set (or sample without correction).
```

## Mathematical Formulation

The choice probability for alternative $i$ is:

$$P_i = \sum_{j \neq i} P_{i|ij} \times P_{ij}$$

where the conditional probability within paired nest $(i,j)$ is:

$$P_{i|ij} = \frac{(\alpha_{i,ij}\, e^{V_i})^{1/\rho}}{(\alpha_{i,ij}\, e^{V_i})^{1/\rho} + (\alpha_{j,ij}\, e^{V_j})^{1/\rho}}$$

and the nest probability is:

$$P_{ij} = \frac{\left[(\alpha_{i,ij}\, e^{V_i})^{1/\rho} + (\alpha_{j,ij}\, e^{V_j})^{1/\rho}\right]^{\rho}}{\sum_{k<l}\left[(\alpha_{k,kl}\, e^{V_k})^{1/\rho} + (\alpha_{l,kl}\, e^{V_l})^{1/\rho}\right]^{\rho}}$$

The allocation parameters are derived from the spatial adjacency structure:

$$\alpha_{i,j} = \frac{\omega_{ij}}{\sum_k \omega_{ik}}$$

where $\omega_{ij} = 1$ if zones $i$ and $j$ are spatially contiguous and $0$ otherwise.

The dissimilarity parameter is estimated via a logistic transform to enforce $\rho \in (0, 1]$:

$$\rho = \frac{1}{1 + \exp(-\alpha_\rho)}$$

## Spatial Graph Input

The `graph` parameter defines the spatial adjacency structure and accepts three formats:

1. **`libpysal.graph.Graph`** (recommended) — e.g., built from a GeoDataFrame of zones
2. **`scipy.sparse` array** — any sparse matrix format
3. **`numpy.ndarray`** — a dense adjacency matrix

All inputs are binarised (any non-zero entry becomes 1) and the diagonal is zeroed. Allocation parameters are computed automatically via row-standardisation.

## Quick Start

```python
from locpick import ChoiceTable, MNL
from libpysal import graph

# Build spatial adjacency from zone geometries
g = graph.Graph.build_contiguity(tracts_gdf, rook=False)

ct = ChoiceTable.from_tables(choosers, alternatives, chosen)
model = MNL(ct, formula="cost + time", graph=g)
result = model.fit()
print(result.summary())
```

Equivalently, using the `SCL` factory:

```python
from locpick import SCL

model = SCL(ct, formula="cost + time", graph=g)  # returns an MNL instance
```

## Using scipy.sparse Input

```python
import scipy.sparse as sp

# Build adjacency manually
adj = sp.csr_array(my_adjacency_matrix)

model = MNL(ct, formula="cost + time", graph=adj)
result = model.fit()
```

## Interpreting Results

The key parameter is **rho** ($\rho$):

| $\rho$ value | Interpretation |
|---|---|
| $\rho = 1$ | No spatial correlation (equivalent to MNL) |
| $\rho < 1$ | Positive spatial correlation between adjacent alternatives |
| $\rho \to 0$ | Very high correlation (alternatives in same nest are near-perfect substitutes) |

A likelihood-ratio test of $\rho = 1$ (MNL) vs. $\rho < 1$ (SCL) tests whether spatial correlation is statistically significant.

## Prediction

```python
# Predict on estimation data
probs = model.probabilities()

# Predict with custom parameters (last entry is alpha_rho on the unconstrained scale)
probs = model.probabilities(beta=my_full_params)
```

## References

- Bhat, C.R. and Guo, J.Y. (2004). A Mixed Spatially Correlated Logit Model: Formulation and Application to Residential Choice Modeling. *Transportation Research Part B*, 38(2), 147–168.