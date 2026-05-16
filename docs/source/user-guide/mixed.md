# Mixed Logit

```{note}
This user guide is a placeholder. Full content will be added in a future release.
```

## Overview

The `MixedLogit` class estimates mixed logit (random coefficients) models, which generalize MNL by allowing coefficients to follow random distributions.

## Quick Start

```python
from locpick import ChoiceTable, MixedLogit, ParamDistribution
from locpick.spec import P

ct = ChoiceTable.from_tables(choosers, alternatives, choice_column="choice")

distributions = [
    ParamDistribution(distribution="normal", param=P("beta_cost")),
]

model = MixedLogit(ct, formula="choice ~ cost + time - 1", distributions=distributions)
result = model.fit()
```