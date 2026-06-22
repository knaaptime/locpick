# Spatial Autoregressive Multinomial Logit (SAR-MNL)

```{note}
This user guide covers SAR-MNL estimation via ``ChoiceModel`` with
``graph=`` and ``lag=True``, which implements a spatial autoregressive
lag in the utility of alternatives (spatial locations) using the pseudo
maximum likelihood (PML) estimator from Smirnov (2010).
```

## Overview

The `ChoiceModel` class with `lag=True` models spatial spillover in the
**systematic utility** of alternatives via a spatial autoregressive (SAR)
lag:

$$V_j = \rho \sum_k w_{jk} V_k + Z_j \beta + X_{ij} \gamma$$

where $W$ is a $J \times J$ spatial weights matrix connecting alternatives
(spatial locations). The reduced-form utility is:

$$V^* = (I - \rho W)^{-1} (Z\beta + X\gamma)$$

normalised by $D = \text{diag}((I - \rho W)^{-1})$ for consistency
(Smirnov 2010). Choice probabilities follow standard MNL softmax over
the spatially-filtered, variance-normalised utilities.

### SAR vs SCL

The `graph=` parameter can be used with two spatial mechanisms:

| `lag=` | Mechanism | ρ range | Transform |
|--------|-----------|---------|-----------|
| `False` (default) | SCL (GEV paired nests) | (0, 1] | Sigmoid |
| `True` | SAR (spatial autoregressive lag) | (-1, 1) | Tanh |

SAR models spatial spillover in the **systematic utility** via the
spatial multiplier $(I - \rho W)^{-1}$. SCL models spatial correlation
via GEV paired nests between adjacent alternatives. They are different
mechanisms for the same spatial weights matrix.

### Key features

- **PML estimator** (Smirnov 2010): consistent, no log-determinant needed
- **JAX autodiff**: gradients through the spatial solve and variance normalisation
- **Dense and conjugate-gradient solve paths**: auto-selected by alternative count
- **Linearized GMM fallback** (Carrión-Flores et al. 2018): for very large J
- **libpysal Graph support**: canonical W type, matching bayespecon
- **Marginal effects**: direct, indirect, and total (LeSage & Pace 2009)
- **Composable with Mixed and Nested logit**: SAR + Mixed, SAR + Nested, SAR + Mixed + Nested

## Quick Start

```python
from locpick import ChoiceTable, ChoiceModel
from libpysal.graph import Graph

# Build spatial weights matrix connecting alternatives
W = Graph.build_knn(gdf, k=7).transform("r")

# Create choice data
ct = ChoiceTable.from_tables(choosers, alternatives, chosen_alternatives=choices)

# Estimate SAR-MNL
model = ChoiceModel(ct, formula="cost + time - 1", graph=W, lag=True)
result = model.fit()
print(result.summary())
```

## Estimation Methods

### PML (default)

The pseudo maximum likelihood estimator (Smirnov 2010) is the default.
It uses JAX autodiff through the spatial solve $(I - \rho W)^{-1}$ and
variance normalisation $\text{diag}((I - \rho W)^{-1})$.

- **Dense solve** (n_alts ≤ 2000): LU factorisation, exact gradients
- **Conjugate gradient** (n_alts > 2000): iterative solve, power-series
  diagonal approximation

```python
# Auto-select (default)
model = ChoiceModel(ct, formula="cost + time - 1", graph=W, lag=True)

# Force dense solve
model = ChoiceModel(ct, formula="cost + time - 1", graph=W, lag=True, estimator="pml")

# Force conjugate gradient
model = ChoiceModel(ct, formula="cost + time - 1", graph=W, lag=True, estimator="pml_cg")
```

### Linearized GMM

For very large alternative sets where even CG is too slow, the
linearized GMM estimator (Carrión-Flores et al. 2018) avoids matrix
inversion entirely via a two-step procedure:

1. **Step 1**: Standard MNL estimation (ignoring spatial dependence)
2. **Step 2**: Two-stage least squares (TSLS) with instruments $[X, WX]$

```python
model = ChoiceModel(ct, formula="cost + time - 1", graph=W, lag=True, estimator="linearized_gmm")
result = model.fit()
```

## Spatial Weights Matrix

`ChoiceModel` accepts `libpysal.graph.Graph` as the canonical W type
(matching the bayespecon package). `scipy.sparse` matrices and dense
NumPy arrays are also accepted for convenience.

```python
from libpysal.graph import Graph

# k-nearest-neighbor graph
W = Graph.build_knn(gdf, k=7).transform("r")

# Contiguity graph
W = Graph.build_contiguity(gdf, rook=False).transform("r")

# Distance band
W = Graph.build_distance_band(gdf, threshold=1000).transform("r")
```

## Composing with Mixed and Nested Logit

SAR composes naturally with mixed logit (random coefficients) and nested
logit (hierarchical choice). The spatial filter is applied to utilities
*before* the GEV/softmax/mixed-logit logic, so the mechanisms are
independent.

### SAR + Nested

```python
from locpick import ChoiceModel, NestingTree

nests = NestingTree(...)
model = ChoiceModel(ct, formula="cost + time - 1", graph=W, lag=True, nests=nests)
result = model.fit()
```

### SAR + Mixed

```python
from locpick import ChoiceModel, ParamDistribution

random_params = {"time": ParamDistribution("normal", "time")}
model = ChoiceModel(
    ct, formula="cost + time - 1", graph=W, lag=True,
    random_params=random_params, n_draws=200,
)
result = model.fit()
```

### SAR + Mixed + Nested

```python
model = ChoiceModel(
    ct, formula="cost + time - 1", graph=W, lag=True,
    nests=nests, random_params=random_params, n_draws=200,
)
result = model.fit()
```

## Marginal Effects

In the SAR-MNL model, a change in an attribute of alternative $j$
affects not only $j$'s utility but also neighbouring alternatives
through the spatial multiplier $(I - \rho W)^{-1}$.

```python
result = model.fit()

# Compute marginal effects for a variable
me = model.marginal_effects(variable="cost")
print(me["direct"])    # impact on own alternative
print(me["indirect"])  # spillover to neighbouring alternatives
print(me["total"])      # direct + indirect
```

## References

- Smirnov, O.A. (2010). "Modeling Spatial Discrete Choice." *Regional Science and Urban Economics*, 40, 292–298.
- Carrión-Flores, C.E., Flores-Lagunes, A., & Guci, L. (2018). "An Estimator for Discrete-Choice Models with Spatial Lag Dependence Using Large Samples." *RSUE*, 69, 77–93.
- LeSage, J.P. & Pace, R.K. (2009). *Introduction to Spatial Econometrics*.
- Krisztin, T., Piribauer, P., & Wögerer, M. (2022). "A Spatial Multinomial Logit Model for Analysing Urban Expansion." *Spatial Economic Analysis*, 17(2), 223–244.