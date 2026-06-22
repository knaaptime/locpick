# Mixed Logit

```{note}
This user guide is a placeholder. Full content will be added in a future release.
```

## Overview

The `ChoiceModel` class estimates mixed logit (random coefficients) models when the `random_params` parameter is provided. Mixed logit generalizes MNL by allowing coefficients to follow random distributions.

## Quick Start

```python
from locpick import ChoiceTable, ChoiceModel
from locpick.models.mixed import ParamDistribution

ct = ChoiceTable.from_tables(choosers, alternatives, chosen_alternatives=choices)

random_params = {
    "cost": ParamDistribution(distribution="normal", param="cost"),
}

model = ChoiceModel(ct, formula="cost + time - 1", random_params=random_params, n_draws=100)
result = model.fit()
print(result.summary())
```