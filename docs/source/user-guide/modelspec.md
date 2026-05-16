# Model Specification

```{note}
This user guide is a placeholder. Full content will be added in a future release.
```

## Overview

locpick uses formula/scoped-term specification for model structure:

1. **Formula strings** — concise, familiar syntax via formulaic
2. **Scoped terms** — explicit coefficient scope helpers on `ModelSpec`

## Formula Strings

```python
from locpick import ChoiceTable, MultinomialLogit

model = MultinomialLogit(ct, formula="choice ~ cost + time - 1")
```

## Scoped Terms

```python
from locpick import ModelSpec, MultinomialLogit

spec = ModelSpec().generic("cost").alternative_specific("time", reference="walk")
model = MultinomialLogit(ct, spec=spec)
```

## Generated Interactions

```python
from locpick import ModelSpec

spec = (
    ModelSpec(formula="cost + income_x_time")
    .with_interaction("income_x_time", "income", "time")
)
```