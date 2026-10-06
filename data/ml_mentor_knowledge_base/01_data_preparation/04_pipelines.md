# Pipelines

## Why pipelines
A `Pipeline` chains sequential transformers and a final estimator. Each intermediate step learns from the data passed to `.fit`; at prediction it applies the learned transformations before the final estimator. This makes train/validation behavior reproducible and allows the full workflow to be cross-validated or tuned as one object.

## API
`steps` is a sequence of `(name, estimator)` pairs; names must be unique and cannot contain `__`. Intermediate steps need `fit` and `transform`; the final step needs the relevant estimator interface. Common methods (`predict`, `predict_proba`, `score`, `transform`) are exposed when supported by the final step. `make_pipeline` creates names automatically. `memory` may cache expensive transformers. Nested parameters use `step__parameter`, e.g. `classifier__C`.

## Example: leakage-safe tuning
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

pipe = Pipeline([("scale", StandardScaler()), ("svc", SVC())])
search = GridSearchCV(pipe, {"svc__C": [0.1, 1, 10],
                             "svc__kernel": ["linear", "rbf"]},
                      cv=StratifiedKFold(5, shuffle=True, random_state=7),
                      scoring="f1", n_jobs=-1)
search.fit(X_train, y_train)
```
Each fold fits its own scaler. This prevents validation-fold means/variances from influencing the model.

## Common errors
A transformer placed after the final estimator is invalid. Names with `__` are invalid. A custom transformer should obey estimator conventions and avoid storing data-derived state in global variables. Do not separately scale the whole dataset and then place only the classifier in the pipeline. If a step can be disabled during tuning, use `'passthrough'` as a candidate.

## Q&A
**Q: Does Pipeline automatically split data?** No; it transforms the data provided to `.fit`. CV tools perform splitting. **Q: Can preprocessing parameters be tuned?** Yes, through nested parameter names. **Q: What is `fit_transform`?** It fits each transformer and returns transformed training data; normal users often call the pipeline's `.fit` instead.
