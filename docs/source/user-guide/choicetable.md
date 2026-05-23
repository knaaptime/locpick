# ChoiceTable: Data Assembly

```{note}
This user guide is a placeholder. Full content will be added in a future release.
```

## Overview

The `ChoiceTable` class is the primary data container for location choice modeling in locpick. It handles:

- Merging chooser and alternative data
- Sampling alternatives from large choice sets
- Generating chooser-alternative pairwise variables
- Producing estimation-ready `ChoiceArrays`

## Quick Start

```python
from locpick import ChoiceTable, MNL

# Create a ChoiceTable from chooser and alternative data
ct = ChoiceTable.from_tables(
    choosers=choosers_df,
    alternatives=alternatives_df,
    chosen_alternatives=choices_series,
)

# Estimate an MNL model
model = MNL(ct, formula="cost + time - 1")
result = model.fit()
print(result.summary())
```

## Sampling Alternatives

When the full choice set is large, you can sample a subset of alternatives:

```python
ct_sampled = ChoiceTable.from_tables(
    choosers=choosers_df,
    alternatives=alternatives_df,
    chosen_alternatives=choices_series,
    sample_size=25,
    seed=42,
)
```

## Pairwise Variables

Chooser-alternative pairwise variables are created with `add_pairwise_variable()`:

```python
ct_with_pairwise = ct.add_pairwise_variable(
    name="dist",
    series=distance_series,
)
```