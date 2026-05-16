# Distance Utilities

```{note}
This user guide is a placeholder. Full content will be added in a future release.
```

## Overview

locpick provides distance matrix and nearest-neighbor utilities for location choice workflows:

- `great_circle_distance_matrix()` — Haversine pairwise distances
- `euclidean_distance_matrix()` — Euclidean pairwise distances
- `distance_matrix()` — Auto-selects haversine or Euclidean
- `nearest_neighbors()` — k-nearest neighbors via BallTree
- `distance_bands()` — Distance-based spatial weights bands