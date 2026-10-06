# Regression metrics

## Core measures
MAE is average absolute error, in target units, and is less sensitive to large residuals than squared error. MSE averages squared errors; RMSE is its square root and returns to target units while emphasizing large misses. R² compares residual squared error with prediction around the target mean: 1 is perfect, 0 matches the mean baseline on that evaluated data, and it can be negative. MAPE is percentage-like but unstable/undefined near zero and can overweight small actual values.

## Residual diagnostics
A single metric hides structure. Plot residuals against predictions/features/time; look for curvature, changing variance, and systematic subgroup errors. Compare against a simple baseline. Report units and test protocol. For skewed or asymmetric costs, consider median absolute error, quantile/pinball loss, or domain-specific loss.

## API
`mean_absolute_error`, `mean_squared_error`, `root_mean_squared_error` (version-dependent availability), `r2_score`, `mean_absolute_percentage_error`, and `median_absolute_error`. Multi-output metrics may aggregate outputs; `multioutput="raw_values"` returns per-target values. Search scorers commonly use `"neg_mean_absolute_error"` or `"neg_root_mean_squared_error"` because sklearn maximizes scores.

## Solved example
Actual [2, 4], predictions [1, 6]. Absolute errors [1,2], MAE=1.5. Squared errors [1,4], MSE=2.5 and RMSE≈1.58. The second error contributes more under squared loss. R² on tiny samples is unstable and should not be interpreted without context.

## Q&A / traps
**Q: Can R² be negative?** Yes; model can be worse than predicting the evaluated-set mean. **Q: Does high R² guarantee unbiased residuals?** No. **Mistake:** report MAPE when true values include zero without understanding its behavior. **Trap:** “negative MAE” in sklearn search is a sign convention, not negative error.
