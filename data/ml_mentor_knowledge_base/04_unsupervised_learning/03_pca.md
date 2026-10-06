# Principal Component Analysis (PCA)

## Concept
PCA finds orthogonal directions of maximum variance in centered numeric data. The first principal component captures the largest possible variance; each following component captures the most remaining variance subject to orthogonality. PCA is unsupervised and linear. It can compress correlated features, aid visualization, and sometimes reduce noise, but high variance is not necessarily high predictive relevance.

## API
`PCA(n_components=None, whiten=False, svd_solver="auto", tol=0.0, random_state=None)`; supported values for `n_components` include integer, fraction of variance for compatible solvers, or `"mle"` in supported cases. `fit` learns components from training data; `transform` projects; `inverse_transform` reconstructs approximately. `components_`, `explained_variance_`, and `explained_variance_ratio_` are useful. Centering occurs, but PCA does not scale each input feature to unit variance automatically.

## Scaling and leakage
If units/ranges differ, standardize before PCA; if features already share meaningful units, scaling is a modeling choice. Fit PCA only on training folds via Pipeline. PCA components may be hard to interpret because each combines original features.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
model = make_pipeline(StandardScaler(), PCA(n_components=0.95),
                      LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
```

## Choosing components and limitations
A cumulative explained-variance curve can show compression tradeoffs. Keeping 95% variance is a heuristic, not a guarantee of best supervised performance; choose component count through CV when prediction is the goal. PCA is sensitive to outliers and only models linear directions. `whiten=True` rescales component outputs to unit variance, which can help some algorithms but discards relative component variance information.

## Q&A / traps
**Q: Is PCA feature selection?** No; it creates combinations of features. **Q: Does 2D PCA preserve all structure?** No. **Trap:** fitting PCA on all data before CV leaks distributional information. **Mistake:** assume explained variance equals target relevance.
