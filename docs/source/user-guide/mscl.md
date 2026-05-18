# Mixed Spatially Correlated Logit (MSCL)

## Overview

The `MixedSCL` class estimates the MSCL model proposed by Bhat & Guo (2004), which combines a GEV-based Spatially Correlated Logit (SCL) structure with random taste variation (mixed logit). The SCL component captures spatial correlation in closed form, while the mixing distribution captures unobserved heterogeneity across decision-makers.

```{warning}
The MSCL model does **not** support alternative sampling correction. The MNL's uniform conditioning property does not hold for non-MNL GEV models. Always use the full alternative set (or sample without correction).
```

## Mathematical Formulation

The MSCL choice probability for alternative $i$ is:

$$P_i = \int (P_i \mid \beta) \, f(\beta \mid \theta) \, d\beta$$

approximated by simulated maximum likelihood:

$$\tilde{P}_i = \frac{1}{R} \sum_{r=1}^{R} P_i(\beta^r)$$

where $P_i(\beta^r)$ is the SCL choice probability conditional on the $r$-th draw of the random coefficients, and $f(\beta \mid \theta)$ is the mixing distribution.

The simulated log-likelihood is:

$$\text{SLL} = \sum_{n=1}^{N} w_n \log \left( \frac{1}{R} \sum_{r=1}^{R} L_n(\beta^r) \right)$$

### Key Computational Advantage

The SCL structure handles spatial correlation in closed form, so the simulation dimension equals the number of random parameters — **not** the number of spatial error components. In the empirical application of Bhat & Guo (2004), this reduces the integration dimension from ~500 (MMNL) to 3 (MSCL).

## Quick Start

```python
from locpick import ChoiceTable, MixedSCL
from locpick.models.mixed import ParamDistribution
from libpysal import graph

# Build spatial adjacency
g = graph.Graph.build_contiguity(tracts_gdf, rook=False)

# Define random parameters
random_params = {
    "commute_time": ParamDistribution(distribution="normal", param="commute_time"),
}

ct = ChoiceTable.from_tables(choosers, alternatives, chosen)
model = MixedSCL(
    ct,
    formula="commute_time + density + shopping_access",
    graph=g,
    random_params=random_params,
    n_draws=250,
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

Supported distributions:

| Distribution | Description | Parameters estimated |
|---|---|---|
| `normal` | $\beta \sim N(\mu, \sigma^2)$ | mean, std dev |
| `lognormal` | $\beta = \exp(\mu + \sigma \cdot z)$, $z \sim N(0,1)$ | mean, std dev |
| `triangular` | $\beta \sim \text{Triangular}(\mu-\sigma, \mu+\sigma)$ | mean, spread |
| `uniform` | $\beta \sim \text{Uniform}(\mu-\sigma, \mu+\sigma)$ | mean, spread |

## Draw Types

```python
# Halton draws (default, quasi-random — more efficient)
model = MixedSCL(
    ct, formula="cost + time", graph=g,
    random_params=random_params,
    n_draws=250,
    draw_type="halton",
)

# Pseudo-random draws
model = MixedSCL(
    ct, formula="cost + time", graph=g,
    random_params=random_params,
    n_draws=500,
    draw_type="random",
)
```

Halton draws provide better coverage of the mixing distribution with fewer draws, making them the default. Pseudo-random draws may be preferred for robustness checks.

## SCL without Random Parameters

When `random_params` is empty, the MSCL model reduces to the SCL model:

```python
model = MixedSCL(
    ct, formula="cost + time", graph=g,
    random_params={},
    n_draws=50,
)
```

For pure SCL estimation, prefer the `SCL` class directly — it avoids the overhead of the simulation loop.

## References

- Bhat, C.R. and Guo, J.Y. (2004). A Mixed Spatially Correlated Logit Model: Formulation and Application to Residential Choice Modeling. *Transportation Research Part B*, 38(2), 147–168.