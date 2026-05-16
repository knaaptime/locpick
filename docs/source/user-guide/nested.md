# Nested Logit

```{note}
This user guide is a placeholder. Full content will be added in a future release.
```

## Overview

The `NestedLogit` class estimates nested logit models, which generalize MNL by grouping alternatives into nests with correlated error terms.

## Quick Start

```python
from locpick import ChoiceTable, NestedLogit, NestSpec

ct = ChoiceTable.from_tables(choosers, alternatives, choice_column="choice")

nests = [
    NestSpec(name="auto", alt_ids=[1, 2, 3]),
    NestSpec(name="transit", alt_ids=[4, 5]),
]

model = NestedLogit(ct, formula="choice ~ cost + time - 1", nests=nests)
result = model.fit()
```