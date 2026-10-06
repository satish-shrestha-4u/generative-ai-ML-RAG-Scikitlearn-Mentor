# Nested cross-validation

## Why nesting
If the same CV results are used both to select the best hyperparameters and to report final performance, the reported score is biased upward: among many noisy estimates, the maximum tends to be optimistic. Nested CV separates these roles.

## Structure
Outer CV estimates the whole model-selection procedure. For each outer training partition, an inner search chooses hyperparameters using only that partition. The chosen estimator is scored on the untouched outer validation fold. Aggregate outer scores to estimate generalization of “tune then fit.” Finally, for deployment, run a search on all available development data and fit the chosen model; outer-fold estimators are primarily for evaluation.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
inner = StratifiedKFold(4, shuffle=True, random_state=1)
outer = StratifiedKFold(5, shuffle=True, random_state=2)
search = GridSearchCV(pipeline, param_grid, scoring="f1", cv=inner, n_jobs=-1)
result = cross_validate(search, X, y, cv=outer, scoring="f1")
print(result["test_score"].mean(), result["test_score"].std())
```

## Interpretation and cost
With 5 outer folds, 4 inner folds, and 30 configurations, the rough number of candidate fits is 5×4×30 = 600, plus refits. Nested CV can be expensive. Use a final holdout set when data allows and the intended evaluation protocol is clear. The outer score estimates the complete search procedure, not the exact model later refit on all data.

## Traps
- Do not tune against outer scores and then quote those same outer scores as unbiased; that introduces another selection layer.
- Put preprocessing inside the searched pipeline.
- Ensure group/time split rules apply to both levels.
- For tiny datasets, nested CV estimates can be noisy; report fold scores and limitations.

## Q&A
**Q: Is nested CV always required?** No. A clean untouched test set can provide final evaluation. Nested CV is useful when data is limited and tuning bias matters. **Q: Does it make a better model?** It improves evaluation design, not necessarily predictive performance.
