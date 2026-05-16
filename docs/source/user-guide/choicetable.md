# ChoiceTable: Data Assembly

```{note}
This user guide is a placeholder. Full content will be added in a future release.
```

## Overview

The `ChoiceTable` class is the primary data container for location choice modeling in locpick. It handles:

- Merging chooser and alternative data
- Sampling alternatives from large choice sets
- Generating chooser×alternative interaction terms
- Producing estimation-ready `ChoiceArrays`

## Quick Start

```python
from locpick import ChoiceTable, FitDiagnostics, MultinomialLogit

# Create a ChoiceTable from chooser and alternative data
ct = ChoiceTable.from_tables(
    choosers=choosers_df,
    alternatives=alternatives_df,
    choice_column="choice",
)

# Estimate an MNL model
model = MultinomialLogit(ct, formula="choice ~ cost + time - 1")
result = model.fit()
print(FitDiagnostics.summary(result))
```

## Sampling Alternatives

When the full choice set is large, you can sample a subset of alternatives:

```python
ct_sampled = ChoiceTable.from_tables(
    choosers=choosers_df,
    alternatives=alternatives_df,
    choice_column="choice",
    sample_size=25,
    seed=42,
)
```

## Interaction Terms

Chooser×alternative interactions are created with `add_interaction()`:

```python
ct_with_interactions = ct.add_interaction(
    name="dist",
    series=distance_series,
)
```