# LocPick

LocPick is a Python library for location and destination choice modeling, with tools for assembling choice tables, sampling alternatives, fitting native choice models, prediction, simulation, and related workflow tasks.

## Features

LocPick can automate the creation of choice tables for estimation or simulation, using census choice sets, uniform or weighted random sampling of alternatives, generated interaction terms, and cartesian merges.

It also provides tools for Monte Carlo simulation of choices given probability distributions from fitted models, with fast algorithms for independent or capacity-constrained choices.

LocPick includes native Multinomial Logit, Nested Logit, and Mixed Logit estimators, and an internal data pipeline designed around pandas inputs, xarray-backed alignment, and NumPy/JAX-ready arrays.

## Installation

Install LocPick with Pip or Conda:

```bash
pip install locpick
```

```bash
conda install locpick --channel conda-forge
```

## Documentation

See the package documentation in [docs/](docs/).

More documentation lives in `CHANGELOG.md`, `CONTRIBUTING.md`, `/docs/README.md`, and `/tests/README.md`.

Current development happens in the [UDST/choicemodels](https://github.com/UDST/choicemodels) repository.

## Benchmarking

For quick local performance checks of sampled choice-table construction + MNL fit:

```bash
python benchmarks/benchmark_mnl.py --num-obs 5000 --num-alts 250 --sample-size 25
```
