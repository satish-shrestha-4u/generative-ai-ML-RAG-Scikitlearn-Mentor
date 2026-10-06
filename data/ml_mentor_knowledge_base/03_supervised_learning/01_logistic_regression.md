# Logistic Regression

## Concept
Despite its name, Logistic Regression is a classification model. It computes a linear score from features and maps it through a logistic sigmoid for binary probabilities; multiclass problems use supported strategies such as multinomial modeling or one-vs-rest depending on configuration/solver. A decision threshold turns scores into class labels. Coefficients describe changes in log-odds per one-unit feature change, conditional on other features and the chosen representation.

## Regularization and API
`LogisticRegression(C=1.0, penalty=..., solver=..., class_weight=None, max_iter=100, ...)` fits regularized coefficients. Smaller `C` means stronger regularization; larger `C` means weaker regularization. Solver/penalty compatibility is version-sensitive; consult the official API before mixing options. `class_weight="balanced"` adjusts class weights inversely to observed frequencies. `max_iter` controls optimization iterations; convergence warnings often indicate need for scaling, a different solver, or more iterations. Use `predict_proba` for probabilities and `decision_function` where supported.

## Example
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))
model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
```

## Worked interpretation
A coefficient of 0.7 means a one-unit increase in that transformed feature adds 0.7 to log-odds, holding other features fixed. Odds multiply by `exp(0.7) ≈ 2.01`. It does not mean probability increases by 70 percentage points; the probability change depends on the baseline score.

## Strengths, limits, troubleshooting
Fast, interpretable baseline; supports probability output and regularization. A linear decision boundary may underfit nonlinear patterns. Correlated features make individual coefficients unstable. One-hot categories and scaling affect interpretation. If the solver fails to converge, scale numeric features and increase `max_iter`; do not treat a warning as evidence the optimum was reached.

**Q:** Is `C` the regularization strength? It is inverse strength. **Trap:** accuracy can be misleading with imbalance; select `scoring` explicitly. **Mistake:** interpret standardized coefficients as raw-unit effects.
