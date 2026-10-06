# Preprocessing and feature scaling

## Why preprocessing matters
Most estimators expect a rectangular numeric feature matrix with consistent columns at fit and prediction time. Preprocessing turns raw values into an appropriate representation. It can also change geometry: scaling changes distances and regularization, encoding changes the feature space, and imputation makes missingness explicit or supplies plausible values.

## Scaling choices
`StandardScaler` centers each feature at its training mean and scales to unit variance: z=(x-mean)/std. It is often useful for linear models with regularization, KNN, SVM, PCA, and distance-based clustering. `MinMaxScaler` maps training extrema to a range (default 0–1); outliers can compress most observations. `RobustScaler` uses median and quantile range and is less influenced by extreme values. `Normalizer` rescales each row to unit norm; it is appropriate when direction matters more than magnitude, such as some text vectors, and is not a substitute for column scaling.

Tree splits generally do not require standardization because monotonic rescaling preserves candidate ordering. Yet preprocessing may still be needed for missing values or mixed types. Fit scalers on training data only; `fit_transform(X_train)` then `transform(X_valid)` prevents validation information influencing learned means/ranges.

## API notes
Typical constructor parameters include `with_mean`, `with_std` for `StandardScaler`, `feature_range` for `MinMaxScaler`, and `quantile_range` for `RobustScaler`. Sparse matrices cannot be centered with `StandardScaler(with_mean=True)` because centering would densify them; use `with_mean=False` where appropriate. Transformed data may be NumPy or sparse arrays, not a DataFrame, depending on configuration and version.

## Example
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
model.fit(X_train, y_train)
pred = model.predict(X_test)
```
The scaler is fitted separately inside each training fold if `model` is passed to cross-validation.

## Worked reasoning
Suppose income is measured in dollars (20,000–200,000) and age in years (18–90). Euclidean distance can be dominated by income's numeric range even if age is equally informative. Standardizing both features puts them on comparable variance scales. This does not prove equal predictive importance; it makes the distance/penalty operate on standardized units.

## Q&A and pitfalls
**Q: Should I scale a decision tree?** Usually not for split quality. **Q: Can I scale before train/test split?** No: estimated means and ranges then incorporate held-out observations. **Q: Does scaling remove outliers?** No; robust scaling reduces their influence on scale estimates but preserves extreme transformed values. **Mistake:** applying `fit_transform` to all data before CV. Put the transformer in a pipeline.
