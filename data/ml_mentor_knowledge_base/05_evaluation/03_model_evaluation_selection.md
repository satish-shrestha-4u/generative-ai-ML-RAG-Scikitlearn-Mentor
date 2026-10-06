# Model evaluation and selection

## Build a fair comparison
Start with a simple baseline (e.g., `DummyClassifier`/`DummyRegressor`) and define the deployment question: prediction time, population, error costs, and operational constraints. Use the same folds and metric for candidate models. Keep preprocessing inside each candidate pipeline. Compare mean and fold variability, fit time, prediction time, memory, calibration, and interpretability as needed.

## Selection protocol
1. Reserve a final test set using random, group, or temporal separation appropriate to deployment.
2. On development data, use CV to compare families and tune parameters.
3. Select based on predeclared metric/constraints; if multiple objectives, make tradeoffs explicit.
4. Refit chosen procedure on all development data.
5. Evaluate once on test and document metric, sample size, split method, and uncertainty/limitations.

If data is scarce, nested CV estimates tuning procedure performance. After extensive experimentation against a validation set, that set also becomes a selection resource; its best score is optimistic.

## Bias, variance, and generalization
Underfitting produces poor train and validation scores; overfitting shows a strong training score and weaker held-out performance. The train-test gap is diagnostic, not a complete theory. More flexible models can lower bias but raise variance. Regularization, more representative data, simpler features, or stronger validation design may help.

## Calibration and thresholds
Ranking quality (AUC) and probability calibration answer different questions. If probabilities drive decisions, inspect calibration and choose the threshold according to cost on validation data. Class-weighting can improve class-sensitive metrics but may alter probability calibration.

## Common traps
- selecting by test score;
- comparing metrics with opposite direction/scale without understanding them;
- choosing the model with the highest noisy mean and ignoring fold spread/cost;
- assuming CV removes distribution shift;
- reporting one score without class breakdown or residual inspection.
