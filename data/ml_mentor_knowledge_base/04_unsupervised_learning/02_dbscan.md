# DBSCAN

## Intuition
DBSCAN groups points connected through dense neighborhoods. A point with at least `min_samples` observations in its radius-`eps` neighborhood is a core point. Points reachable from core points join a cluster; isolated points may be labeled noise (`-1`). Unlike K-Means, DBSCAN does not require a cluster count and can discover non-spherical shapes.

## API
`DBSCAN(eps=0.5, min_samples=5, metric="euclidean", algorithm="auto", leaf_size=30, n_jobs=None)`. `eps` is the neighborhood radius and is the most sensitive parameter; `min_samples` controls density requirement and includes the point itself under standard interpretation. `fit_predict` returns integer labels; `-1` means noise. The algorithm may have high memory use in dense neighborhoods.

## Scaling and parameter selection
Distance meaning depends on scale and metric. Standardize or otherwise transform features when appropriate, and consider domain-specific distance. A k-distance plot can help choose an `eps` near a bend: compute distance to the `min_samples`-th neighbor and sort. This is heuristic. One global eps struggles when cluster densities differ. High dimensionality makes neighborhoods less informative.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
labels = make_pipeline(StandardScaler(), DBSCAN(eps=0.4, min_samples=6)).fit_predict(X)
```

## Evaluation and limitations
Silhouette can be computed after excluding noise only if at least two clusters remain, but ignoring noise changes the question being scored. DBSCAN has no standard predictive `predict` for assigning future unseen points; deployment may need a different method. Compare cluster count, noise fraction, stability, and domain value.

## Worked diagnostic
If every point receives `-1`, eps may be too small, min_samples too high, or scale inappropriate. If almost everything is one cluster, eps may be too large. Adjust one parameter at a time and inspect neighborhood scale.

**Q:** Does DBSCAN need K? No. **Trap:** eps is a distance, not a number of clusters. **Mistake:** compare eps values before feature scaling and interpret them as universal.
