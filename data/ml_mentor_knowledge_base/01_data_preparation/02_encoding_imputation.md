# Categorical encoding and missing-value imputation

## Categorical variables
Most estimators operate on numeric matrices. `OneHotEncoder` creates indicator columns for categories without imposing a false numeric order. Use `handle_unknown="ignore"` when prediction data may contain unseen categories; those values produce an all-zero block for that feature. `drop="first"` can reduce exact collinearity in some unregularized linear settings, but is not universally needed and may make category interpretation asymmetric. `min_frequency`/`max_categories` can group infrequent values in recent versions. `OrdinalEncoder` maps categories to integers and is appropriate only when order is meaningful or the downstream model handles category codes appropriately; arbitrary codes mislead distance-based/linear models.

## Missingness
`SimpleImputer` learns a per-column statistic such as `mean`, `median`, `most_frequent`, or a constant. Median is often robust for skewed numeric data; most-frequent or a constant such as `"missing"` is common for categoricals. `add_indicator=True` can preserve a signal that a value was absent. `KNNImputer` uses neighboring rows and distances, so scaling and computational cost matter. `IterativeImputer` models each feature from others and is experimental in some releases; verify current version status before use. Imputation is a modeling assumption, not recovery of the unknowable true value.

## Mixed-type pipeline example
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler())])
cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore"))])
prep = ColumnTransformer([("num", num, numeric_columns),
                          ("cat", cat, categorical_columns)])
```
Fit only through the final estimator's `.fit`. The imputer and encoder learn categories/statistics from training data and reuse them at prediction.

## API details and tradeoffs
`OneHotEncoder` has `categories`, `drop`, `handle_unknown`, `sparse_output`, and frequency grouping controls; older versions used `sparse`, so check version. `SimpleImputer` has `missing_values`, `strategy`, `fill_value`, `add_indicator`, and `keep_empty_features` (availability/defaults vary by version). A column that is entirely missing in training needs deliberate handling; default behavior can vary by strategy/version. Missingness indicators only have columns for missing patterns seen during fit.

## Worked example
Training categories are `red, blue`; test contains `green`. With `handle_unknown="ignore"`, the test row receives zero in both learned color columns instead of raising an error. If the new category is important, consider grouping rare categories or updating the training data; ignoring prevents a crash but does not teach a meaning for green.

## Q&A / mistakes
**Q: Why impute inside CV?** Each fold must estimate fill values from its training fold. **Q: Is ordinal encoding the same as one-hot?** No: it introduces numeric order/distance. **Mistake:** `get_dummies` separately on train and test can create misaligned columns. Use a fitted encoder. **Trap:** an imputed value can look real; an indicator may help when the fact of missingness is predictive.
