# Multinomial Logit Estimation

```{note}
This user guide is a placeholder. Full content will be added in a future release.
```

## Overview

The `ChoiceModel` class estimates multinomial logit (MNL) models for location choice. It supports:

- Formula/scoped-term model specification
- JAX-accelerated log-likelihood and gradient computation
- Multiple solver backends (L-BFGS-B, Optax)
- Weighted estimation and alternative availability constraints

## Quick Start

```python
from locpick import ChoiceTable, ChoiceModel

ct = ChoiceTable.from_tables(choosers, alternatives, chosen_alternatives=choices)
model = ChoiceModel(ct, formula="cost + time - 1")
result = model.fit()
print(result.summary())
```

## Backend Selection

```python
# JAX is the default and only production backend
# NumPy kernels exist for reference/testing only
model = ChoiceModel(ct, formula="cost + time - 1")
result = model.fit()
```