# K-Means clustering

## What it does
K-Means partitions numeric observations into K clusters by minimizing within-cluster squared Euclidean distance (inertia). It alternates: (1) assign each point to its nearest centroid; (2) recompute each centroid as the mean of assigned points. Iteration stops when assignments/centers stabilize or a stopping criterion is reached. Because the objective is non-convex, initialization can lead to different local minima.

## Initialization and parameters
`KMeans(n_clusters=8, init="k-means++", n_init="auto", max_iter=300, tol=1e-4, random_state=None, algorithm="lloyd")` in current stable docs; defaults have changed across releases. `n_clusters` is K. `init="k-means++"` spreads initial centers to improve optimization. `n_init` controls independent restarts; select the run with lowest inertia. `max_iter` caps iterations per run. `tol` sets a relative center-shift stopping tolerance. Set `random_state` for reproducible initialization. `fit_predict(X)` fits and returns labels; `cluster_centers_`, `labels_`, `inertia_`, and `n_iter_` summarize the fit.

## Scaling is crucial
K-Means uses Euclidean distance and means. A feature with a larger unit/range can dominate assignment. Scale numeric features when units differ, fitting scaler within the analysis pipeline. Outliers can pull centroids. K-Means favors roughly compact, convex, similarly sized clusters and requires K in advance; it handles neither arbitrary curved shapes nor density variation well.

## Example
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
clusterer = make_pipeline(StandardScaler(),
                          KMeans(n_clusters=3, n_init=20, random_state=42))
labels = clusterer.fit_predict(X)
```

## Choosing K and evaluation
Inertia never increases as K increases, so the elbow is a heuristic, not proof of true cluster count. Silhouette score compares within-cluster cohesion with nearest-other-cluster separation (higher is better, range −1 to 1); it requires at least 2 clusters and at most n−1 labels and can favor convex structures. Use domain interpretability and stability too. If known labels exist, external metrics such as adjusted Rand score can compare clustering, but labels should not be smuggled into unsupervised fitting.

## Worked example
For K=2, inertia is 120; K=3 gives 75; K=4 gives 62; K=5 gives 55. The largest decrease is from 2 to 3, suggesting K=3 as a candidate. It is not a mathematical answer: inspect silhouette, stability, and whether clusters are useful.

## Q&A / pitfalls
**Q: What does `random_state` do?** Makes random initialization reproducible; it does not guarantee the globally optimal partition. **Q: Why compare multiple starts?** Reduce sensitivity to unlucky initialization. **Mistake:** choose K by minimizing inertia alone. **Exam trap:** cluster label numbers are arbitrary; label 0 has no inherent meaning. **K-Means vs DBSCAN:** K-Means requires K and favors compact groups; DBSCAN finds density-connected shapes and can mark noise.
