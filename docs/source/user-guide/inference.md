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

## Standard Errors

```python
result = model.fit()

# Default standard errors (from FitResult)
print(result.std_errors)

# Robust (Huber-White/sandwich) standard errors
print(model.std_errors_robust(ct))

# Clustered standard errors
print(model.std_errors_clustered(ct, groups=clusters))
```

## Hypothesis Tests

locpick ships likelihood-ratio, Wald, and Hausman tests in
`locpick.results.diagnostics`.

### Likelihood-Ratio Test

Compare two nested models:

```python
from locpick import lr_test

restricted = MNL(ct, formula="cost + time").fit()
unrestricted = NestedMNL(ct, formula="cost + time", nests=tree).fit()

result = lr_test(restricted, unrestricted)
print(result.summary())
```

### Wald Test

Test arbitrary linear restrictions $R\beta = 0$:

```python
import numpy as np
from locpick import wald_test

# Test that the first two coefficients are jointly zero
R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
result = wald_test(
    coefficients=fit.coefficients.values,
    variance_covariance=fit.covariance(),
    r_matrix=R,
)
print(result.summary())
```

### Hausman Test

Compare an efficient estimator (consistent only under the null) against a
consistent estimator (consistent under both null and alternative). The
classic application is testing the IIA assumption of MNL against a less
restrictive nested or mixed alternative:

```python
from locpick import hausman_test

mnl_fit = MNL(ct, formula="cost + time").fit()
nested_fit = NestedMNL(ct, formula="cost + time", nests=tree).fit()

# H0: MNL is consistent (IIA holds). Compare on the common parameters.
result = hausman_test(efficient=mnl_fit, consistent=nested_fit)
print(result.summary())

# Restrict the comparison to specific parameters
result = hausman_test(mnl_fit, nested_fit, params=["cost", "time"])
```

The statistic is computed as

$$H = (\hat\beta_c - \hat\beta_e)^\top (V_c - V_e)^{+} (\hat\beta_c - \hat\beta_e) \;\sim\; \chi^2_q$$

using the Moore-Penrose pseudoinverse of $V_c - V_e$ (with `df` equal to its
numerical rank). When $V_c - V_e$ is not positive semidefinite — a common
finite-sample artefact — `HausmanTest.psd_warning` is set to `True` and the
summary flags it.
