# Nested Logit

```{note}
This user guide is a placeholder. Full content will be added in a future release.
```

## Overview

The `ChoiceModel` class estimates nested logit models when the `nests` parameter is provided. Nested logit generalizes MNL by grouping alternatives into nests with correlated error terms.

## Quick Start

```python
from locpick import ChoiceTable, ChoiceModel
from locpick.models.nested import NestSpec, NestingTree

ct = ChoiceTable.from_tables(choosers, alternatives, chosen_alternatives=choices)

tree = NestingTree(
    nests=[
        NestSpec(name="auto", alt_ids=[1, 2, 3]),
        NestSpec(name="transit", alt_ids=[4, 5]),
    ]
)

model = ChoiceModel(ct, formula="cost + time - 1", nests=tree)
result = model.fit()
print(result.summary())
```