# Support Vector Machines

## Concept
A linear SVM seeks a separating hyperplane with a large margin. The `C` parameter controls the penalty for margin violations: small C allows more violations and stronger regularization; large C fits training data more tightly. Kernels create nonlinear boundaries through similarity functions. Common choices are linear and RBF.

## API essentials
`SVC(C=1.0, kernel="rbf", degree=3, gamma="scale", probability=False, class_weight=None, ...)`. `gamma` controls the reach of RBF influence: high gamma can form narrow, complex regions; low gamma produces smoother boundaries. `degree` applies to polynomial kernels. SVC can be costly as sample size grows. `LinearSVC` is an alternative for large sparse/high-dimensional linear problems but has a different API and does not expose the same probability behavior. `probability=True` in SVC adds calibration-related fitting cost; probabilities are not necessary for ranking metrics if decision scores suffice.

## Scaling and tuning
SVMs are scale-sensitive. Put `StandardScaler` in the same pipeline. Search `C` and `gamma` on logarithmic scales, with nested parameters such as `svc__C`. Tune using a deployment-relevant metric.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
model = make_pipeline(StandardScaler(), SVC(C=2, kernel="rbf", gamma="scale"))
```

## Troubleshooting
Convergence/long runtime: reduce rows/features, use linear model, or tune computational budget. Poor CV: inspect scale, outliers, class balance, and C/gamma. Probability output: enable `probability=True` before fitting or use calibration tools when calibrated estimates matter.

**Q:** Is high C always better? No; it can overfit. **Trap:** default `gamma="scale"` is data-dependent; search it only with CV. **Mistake:** standardize all data before CV instead of embedding scaling in Pipeline.
