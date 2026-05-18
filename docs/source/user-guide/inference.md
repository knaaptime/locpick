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

# Default standard errors (from FitResult)
print(result.std_errors)

# Robust (Huber-White/sandwich) standard errors
print(model.std_errors_robust(ct))

# Clustered standard errors
print(model.std_errors_clustered(ct, groups=clusters))
```