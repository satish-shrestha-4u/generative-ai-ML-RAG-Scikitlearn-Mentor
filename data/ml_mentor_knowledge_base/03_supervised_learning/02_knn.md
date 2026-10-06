# K-Nearest Neighbors (KNN)

## Concept
KNN is a non-parametric method that stores training observations. For a query, it finds the `k` nearest points under a distance metric. Classification votes among neighbors (optionally distance-weighted); regression averages neighbor targets. Its flexible boundary can model local structure, but prediction cost and memory grow with training-set size.

## API details
`KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30, p=2, metric="minkowski")`. `p=2` is Euclidean; `p=1` Manhattan for Minkowski distance. `weights="distance"` gives closer neighbors more influence. `algorithm` can choose brute force or tree structures; high dimensionality often undermines tree speedups. `n_neighbors` must be valid for the fitted sample count.

## Example
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
model = make_pipeline(StandardScaler(),
                      KNeighborsClassifier(n_neighbors=7, weights="distance"))
model.fit(X_train, y_train)
```

## Bias/variance intuition
Small k creates a highly local, irregular boundary and can overfit noise; larger k smooths the boundary and may underfit. Select k and distance/weight choices within CV. Odd k is sometimes used for binary voting to reduce ties, but is not a universal requirement.

## Practical issues
Scaling is usually essential because distance compares feature magnitudes. Irrelevant dimensions dilute distance (“curse of dimensionality”). Missing values need handling first. Prediction can be slow for large datasets; consider approximate neighbor systems outside basic sklearn or a different model.

## Solved example
Neighbors' labels are A, A, B, B, B. Uniform k=5 predicts B. If their distances are 1, 2, 1, 8, 9 and weights are inverse distance, the total A weight is 1+0.5=1.5 and B weight is 1+0.125+0.111≈1.236, so distance voting predicts A. This demonstrates that weighting changes the vote; exact implementation weighting follows API semantics.

**Q:** Does KNN learn coefficients? No; it retains examples. **Trap:** scale within each CV fold using a pipeline.
