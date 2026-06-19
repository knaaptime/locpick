# Prediction & Simulation

```{note}
This user guide is a placeholder. Full content will be added in a future release.
```

## Overview

Prediction and simulation are **model methods**, not `FitResult` methods. After fitting, use the model object to compute probabilities, utilities, and predictions:

- `model.probabilities(data)` — Choice probabilities (returns ndarray)
- `model.utilities(data)` — Systematic utilities (returns ndarray)
- `model.predict(result, data)` — Predicted choices

```python
from locpick import ChoiceTable, ChoiceModel

ct = ChoiceTable.from_tables(choosers, alternatives, chosen_alternatives=choices)
model = ChoiceModel(ct, formula="cost + time - 1")
result = model.fit()

# Choice probabilities (n_obs × n_alts ndarray)
probs = model.probabilities(ct)

# Systematic utilities (n_obs × n_alts ndarray)
utils = model.utilities(ct)

# Predicted choices
preds = model.predict(result, ct)
```