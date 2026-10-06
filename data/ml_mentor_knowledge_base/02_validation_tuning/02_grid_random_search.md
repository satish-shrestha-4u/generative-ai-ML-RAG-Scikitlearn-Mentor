# GridSearchCV and RandomizedSearchCV

## Hyperparameters
Learned parameters come from fitting (for example logistic coefficients). Hyperparameters are chosen before/during model selection (for example `C`, tree depth, or neighbor count). Search evaluates candidate configurations under a CV scheme and selects according to a scoring rule.

## GridSearchCV
`GridSearchCV(estimator, param_grid, scoring=None, n_jobs=None, refit=True, cv=None, ...)` exhaustively evaluates the Cartesian product of listed values. A list of dictionaries supports conditional grids, such as linear vs RBF SVM parameters. Cost is approximately candidates × folds × fit cost. `best_params_`, `best_score_`, `best_estimator_`, and `cv_results_` are common outputs. With `refit=True`, it fits the selected configuration on all data passed to search.

## RandomizedSearchCV
`RandomizedSearchCV` samples `n_iter` configurations. Lists are sampled uniformly; distributions with `.rvs` allow meaningful continuous/log-uniform sampling. It is useful for large spaces where a grid wastes evaluations. Fix `random_state` for reproducible draws. Cost is approximately `n_iter × folds × fit cost`.

## Example
```python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import loguniform
search = RandomizedSearchCV(
    pipeline,
    {"model__C": loguniform(1e-3, 1e3),
     "model__class_weight": [None, "balanced"]},
    n_iter=30, scoring="f1", cv=StratifiedKFold(5, shuffle=True, random_state=4),
    n_jobs=-1, random_state=4, refit=True)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
```

## Scoring and multi-metric selection
Choose a metric that reflects the actual objective. With multiple metrics, use `scoring={...}` and set `refit` to the metric name or callable if a best estimator is needed. Search scores are selection estimates and can be optimistic after extensive tuning; use a separate test set or nested CV for an unbiased estimate.

## Common failures
- Wrong nested parameter spelling: inspect `pipeline.get_params().keys()`; use `step__parameter`.
- Huge grid: calculate candidate count before running.
- Search across incompatible choices: use list-of-dicts.
- `best_score_` is not test performance.
- `refit=False` means no `best_estimator_` is fitted.
- Parallel fits can exhaust memory; choose `n_jobs` and `pre_dispatch` responsibly.

## Solved mini-exercise
A grid has 4 C values, 3 kernels, and 5 folds: 12 candidates and 60 CV fits (if all combinations are valid). A randomized search with `n_iter=20` and 5 folds uses 100 fits; randomized is not automatically cheaper—the budget depends on `n_iter`.
