# Multinomial Logit Estimation

```{note}
This user guide is a placeholder. Full content will be added in a future release.
```

## Overview

The `MultinomialLogit` class estimates multinomial logit (MNL) models for location choice. It supports:

- Formula/scoped-term model specification
- JAX-accelerated log-likelihood and gradient computation
- Multiple solver backends (L-BFGS-B, Optax)
- Weighted estimation and alternative availability constraints

## Quick Start

```python
from locpick import ChoiceTable, MultinomialLogit

ct = ChoiceTable.from_tables(choosers, alternatives, choice_column="choice")
model = MultinomialLogit(ct, formula="choice ~ cost + time - 1")
result = model.fit()
print(result.summary())
```

## Backend Selection

```python
# Use JAX backend (default if available)
model = MultinomialLogit(ct, formula="choice ~ cost - 1", backend="jax")

# Use NumPy backend
model = MultinomialLogit(ct, formula="choice ~ cost - 1", backend="numpy")
```