# LocPick

`locpick` is a Python library for estimating **discrete choice models of location decisions** — where individuals, households, or firms choose among *spatially-defined* alternatives (neighborhoods, jobs, housing units, transit stops). These methods have broad applicability but are common in regional land-use and housing market modeling. The package is designed for:

- **Large-scale urban models**: 100K+ choosers, 1K+ alternatives
- **Sampling-based estimation**: Most alternatives are irrelevant; only a sampled subset is evaluated per chooser
- **Spatial dependence**: Nearby alternatives can affect each other or share unobserved attributes (via `graph=` on any model)
- **Heterogeneous preferences**: Mixed logit for random taste variation
- **Nested structure**: Nested logit for hierarchical choice (e.g., county → tract → block)
- **JAX-native computation**: JIT-compiled kernels, GPU acceleration, automatic differentiation

The package is **not** a general-purpose ML library. It is specifically for structural econometric models of choice where the likelihood has a closed form (or simulated approximation) and parameters have behavioral interpretations. For more transportation-oriented problems, see [larch](https://github.com/driftlesslabs/larch)

## Features

LocPick can automate the creation of choice tables for estimation or simulation, using census choice sets, uniform or weighted random sampling of alternatives, generated interaction terms, and cartesian merges. A unique feature is the implementation of *spatial* choice models, which take one of two forms.

- The [Bhat et al](https://linkinghub.elsevier.com/retrieve/pii/S0191261503000055) approach is similar to a spatial error model, assuming that nearby alternatives are more similar (closer substitutes). 
- The SAR style approach assumes that the structural utility of each alternative $V$ has a simultaneous autoregressive structure, and is estimated with either [PML](http://dx.doi.org/10.1016/j.regsciurbeco.2009.09.004) or [GMM](https://www.sciencedirect.com/science/article/pii/S0166046217300625)

It also provides tools for Monte Carlo simulation of choices given probability distributions from fitted models, with fast algorithms for independent or capacity-constrained choices.

LocPick includes estimators for classic, spatially-correlated, and simultaneous autoregressive:

- Multinomial Logit
- Nested Logit
- Mixed Logit and 
- Mixed/Nested 

models, and an internal data pipeline designed around pandas inputs, xarray-backed alignment, and NumPy/JAX-ready arrays.
