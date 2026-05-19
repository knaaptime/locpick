# Model Specification

```{note}
This user guide is a placeholder. Full content will be added in a future release.
```

## Overview

locpick uses formula/scoped-term specification for model structure:

1. **Formula strings** — concise, familiar syntax via formulaic
2. **Pairwise variables** - unique for each chooser/alternative pair, the canonical example being distance or cost
3. **Scoped terms** — explicit coefficient scope helpers on `ModelSpec`

## Formula Strings

Typically models are specified using Wilkinson formulas

```python
from locpick import ChoiceTable, MNL

model = MNL(ct, formula="cost + time - 1")
```

The `ChoiceTable` handles data construction intelligently. In this example `cost` is fixed
for each alternative (each alternative has the same cost for all choosers); but
`distance` is unique for each chooser (each chooser lives a different distance from each
alternative)

## Pairwise Variables

Pairwise variables are chooser-alternative-specific — each value is unique per (chooser, alternative) pair, like a distance matrix.

```python
from locpick import ChoiceTable

ct = ChoiceTable.from_tables(choosers, alternatives, chosen_alternatives="choice")
ct = ct.add_pairwise_variable("time", time_series)
```


## Scoped Terms

```python
from locpick import ModelSpec, MNL

spec = ModelSpec(formula="cost + time - 1").alternative_specific("time", reference="walk")
model = MNL(ct, spec=spec)
```

## Generated Interaction Variables

Interaction variables are a special kind of pairwise variable computed as the product of two existing columns (e.g., income × rent):

```python
from locpick import ModelSpec

spec = (
    ModelSpec(formula="cost + income_x_time")
    .with_interaction("income_x_time", "income", "time")
)
```

Note you only need `with_interaction` to persist the column on the ChoiceTable,
specifically. If you only need the data in the model, you can simply specify an
interaction using formulaic notation, e.g. `income:time` and it will be computed
dynamically.