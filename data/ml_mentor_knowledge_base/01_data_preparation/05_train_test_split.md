# Train/test split and data discipline

## Purpose
A test set approximates future unseen observations. Training data fit the model; validation/CV data select model choices; the test set is reserved for a final, limited evaluation. Repeatedly making decisions from test results turns the test set into another validation set and biases the reported result.

## `train_test_split`
The helper returns train and test partitions. Important parameters: `test_size` or `train_size`, `random_state` for repeatability, `shuffle`, and `stratify`. For classification, `stratify=y` preserves approximate class proportions when feasible. Do not shuffle time-ordered data when future prediction is the goal. Grouped observations belonging to one person/device must not be split across partitions; use group-aware splitters.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

## Correct order
1. Define target and remove leakage/identifier fields.
2. Split raw rows according to deployment structure.
3. Fit preprocessing/model on training data, preferably as one pipeline.
4. Choose settings using training-only CV or a validation split.
5. Freeze decisions, evaluate once on test, and report metric plus uncertainty/context.

## Leakage examples
Target-derived columns, post-outcome timestamps, duplicate entities across train/test, global scaling, imputation before splitting, and selecting features based on all labels all leak information. A pipeline prevents preprocessing leakage but cannot fix a bad temporal or group split.

## Q&A / traps
**Q: Is 80/20 mandatory?** No; choose based on data size, uncertainty needs, and deployment. **Q: Does a fixed seed improve a model?** No; it makes a random split reproducible. **Trap:** a stratified random split is still wrong for forecasting if it lets future observations predict the past.
