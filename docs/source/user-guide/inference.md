# Inference

```{note}
This user guide is a placeholder. Full content will be added in a future release.
```

## Overview

locpick provides multiple covariance estimators for inference:

- **Hessian-based** (default): Inverse of negative Hessian
- **BHHH**: Outer product of scores
- **Robust (sandwich)**: HC0 sandwich estimator
- **Clustered**: Cluster-robust sandwich estimator

## Usage

```python
result = model.fit()

# Default standard errors
print(result.std_errors)

# BHHH standard errors
print(result.std_errors_bhhh)

# Robust standard errors
print(result.std_errors_robust)

# Clustered standard errors
print(result.std_errors_clustered(groups))
```