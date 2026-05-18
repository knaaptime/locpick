# Nested Spatially Correlated Logit (NestedSCL)

## Overview

The `NestedSCL` class estimates a Nested Spatially Correlated Logit model, which combines a nested logit upper level with spatially correlated logit (SCL) lower levels. Each nest has:

- A spatial dissimilarity parameter $\rho_m \in (0, 1]$ governing correlation between spatially adjacent alternatives within the nest
- A nest dissimilarity parameter $\lambda_m \in (0, 1]$ governing correlation between alternatives in the same nest

This model is appropriate when alternatives are both **spatially correlated** (nearby zones are substitutes) and **grouped** (zones within the same district share unobserved attributes).

```{warning}
The NestedSCL model does **not** support alternative sampling correction. Always use the full alternative set.
```

## Mathematical Formulation

The choice probability for alternative $i$ in nest $m$ is:

$$P_i = P_{\text{SCL}}(i \mid m) \times P_{\text{NL}}(m)$$

where $P_{\text{SCL}}(i \mid m)$ is the SCL conditional probability within nest $m$ and $P_{\text{NL}}(m)$ is the nested logit probability of choosing nest $m$.

When $\rho_m = 1$ and $\lambda_m = 1$ for all nests, the model reduces to MNL.

## Quick Start

```python
from locpick import ChoiceTable, NestedSCL
from locpick.models.nested import NestSpec, NestingTree
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

# Estimate NestedSCL model
ct = ChoiceTable.from_tables(choosers, alternatives, chosen_alternatives=choices)
model = NestedSCL(ct, formula="cost + time - 1", graph=g, nests=tree)
result = model.fit()
print(result.summary())
```

## Interpreting Results

The key parameters are:

| Parameter | Interpretation |
|---|---|
| $\rho_m$ | Spatial dissimilarity within nest $m$. $\rho_m = 1$ means no spatial correlation within the nest |
| $\lambda_m$ | Nest dissimilarity. $\lambda_m = 1$ means no correlation within the nest (MNL-like) |
| $\beta$ | Utility coefficients |

A likelihood-ratio test comparing NestedSCL to MNL tests whether both spatial and nesting structures are jointly significant.

## Prediction

```python
# Predict on estimation data
probs = model.probabilities()

# Predict with custom parameters
probs = model.probabilities(beta=my_beta, rho=my_rho, lam=my_lam)
```

## References

- Al-Haideri et al. (2026). Cyclists' crossing behaviour at roundabouts: A Generalized Spatially Correlated Nested Logit model.
- Bhat, C.R. and Guo, J.Y. (2004). A Mixed Spatially Correlated Logit Model: Formulation and Application to Residential Choice Modeling. *Transportation Research Part B*, 38(2), 147–168.