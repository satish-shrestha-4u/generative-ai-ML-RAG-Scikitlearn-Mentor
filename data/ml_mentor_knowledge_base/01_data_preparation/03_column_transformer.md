# ColumnTransformer

## Role
`ColumnTransformer` applies different transformations to selected columns and concatenates their outputs into one feature representation. It is the standard scikit-learn way to combine numeric scaling/imputation with categorical encoding while keeping preprocessing fitted within a pipeline.

## Constructor essentials
`transformers` is a list of `(name, transformer, columns)` triples. `remainder="drop"` is the default; use `"passthrough"` to retain unlisted features or supply an estimator. `sparse_threshold` controls whether combined output is sparse when some component outputs are sparse. `transformer_weights` can multiply blocks, but weighting should have a justified modeling reason. Column selectors may be names, indices, slices, masks, or callables. DataFrame column selection by names is clear and maintainable.

## Example
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
features = ColumnTransformer([
    ("numeric", numeric, numeric_columns),
    ("categorical", categorical, categorical_columns),
], remainder="drop")
model = Pipeline([("features", features),
                  ("classifier", LogisticRegression(max_iter=1000))])
model.fit(X_train, y_train)
```

## Interpretation and debugging
The output order follows transformer order and within-transformer feature order. `get_feature_names_out()` can expose names, depending on estimator support. `output_indices_` can locate transformed blocks. Inspect `get_params()` and use nested names such as `features__numeric__imputer__strategy` in a search space.

## Pitfalls
- Training and serving frames need compatible column names and dtypes.
- Remainder drop can silently omit a newly added feature; passthrough can accidentally include identifiers or leakage columns.
- Sparse one-hot output may be converted to dense if the combined density exceeds `sparse_threshold`, which can consume memory.
- Do not fit the transformer once on the complete dataset before cross-validation.

## Q&A
**Q: Does it select features?** It selects columns for transformation, not predictive feature selection. **Q: Can two transformers use the same columns?** Yes, if intentional. **Q: What happens to unlisted columns?** Determined by `remainder`.
