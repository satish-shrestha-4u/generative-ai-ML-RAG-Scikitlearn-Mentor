# Cross-validation strategies

## K-fold logic
K-fold CV divides data into K folds. Each round trains on K−1 folds and scores on the remaining fold. The mean estimates performance across splits; fold variation communicates instability but is not a formal confidence interval in general. Larger K uses more training data per fit but costs more and does not guarantee lower uncertainty.

## Splitters and when to use them
- `KFold`: general regression/default row-wise split; `shuffle=True` plus seed for IID shuffled data.
- `StratifiedKFold`: preserves class proportions for binary/multiclass classification.
- `GroupKFold` / `StratifiedGroupKFold`: keep a group entirely in one fold when multiple rows belong to a person/site/device.
- `TimeSeriesSplit`: train on earlier observations, validate on later ones; supports expanding temporal training windows.
- `RepeatedKFold` / `RepeatedStratifiedKFold`: repeat randomized folds to examine sensitivity.
- `LeaveOneOut`: expensive and high variance for many datasets; not automatically superior.

`cross_validate` can return multiple metrics and fit/score times; `cross_val_score` returns one score per split. Most scikit-learn scorers follow “higher is better”; loss metrics are negated (for example `neg_mean_squared_error`).

## Example
```python
from sklearn.model_selection import StratifiedKFold, cross_validate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(pipeline, X, y, cv=cv,
                         scoring={"recall": "recall", "precision": "precision"},
                         return_train_score=True)
```

## API cautions
When `cv` is an integer, scikit-learn selects a default splitter based on task: typically stratified folds for binary/multiclass classifiers and KFold otherwise; default splitters do not shuffle. Specify the splitter explicitly when the split design matters. Make the pipeline the estimator passed to CV so every fold refits all learned preprocessing.

## Troubleshooting and exam traps
- A fold can fail if a rare class has fewer members than the number of folds. Reduce folds or obtain more examples; do not conceal this by reporting a misleading metric.
- Group/time dependence invalidates ordinary random CV even if code runs.
- `cross_val_score` expects higher-is-better scoring convention; negative loss scores are not errors.
- CV evaluates a procedure, not a magic universal model score. Keep the final test set untouched if available.

## Worked question
There are 100 patients with 10 measurements each. Which splitter? Use groups by patient so all measurements from one patient stay together in a fold. Random KFold can otherwise reward memorization of patient-specific signals.
