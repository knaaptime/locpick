# Sampling Correction

```{note}
This user guide is a placeholder. Full content will be added in a future release.
```

## Overview

When alternatives are sampled from a large choice set, the likelihood must include a correction term to produce consistent parameter estimates. locpick implements inclusion-probability-based correction.

## Sampling Methods

- **SRSWOR**: Simple random sampling without replacement
- **SRSWR**: Simple random sampling with replacement
- **Weighted sampling**: Probability proportional to weight

## Inclusion Probabilities

The correction term added to utilities is:

$$V_{ij} = X_{ij}\beta + \log(\pi_{ij})$$

where $\pi_{ij}$ is the inclusion probability for alternative $j$ in observation $i$.