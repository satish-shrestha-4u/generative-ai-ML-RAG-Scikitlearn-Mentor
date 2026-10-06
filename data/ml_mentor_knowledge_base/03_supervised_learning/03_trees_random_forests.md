# Decision Trees and Random Forests

## Decision Trees
A tree recursively selects feature thresholds that reduce impurity (e.g., Gini/entropy for classification, squared error for regression). Leaves predict class distributions/classes or numeric values. Trees capture nonlinear interactions, need little scaling, and are easy to visualize when small. Deep trees can memorize training data.

Key controls include `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion`, and pruning via `ccp_alpha`. A larger minimum leaf size or smaller depth regularizes. Feature importance based on impurity can favor high-cardinality variables; permutation importance on held-out data is often a useful complement.

## Random Forest
A forest fits many trees on bootstrap samples and aggregates their predictions. Random feature subsets at splits decorrelate trees; averaging reduces variance relative to a single unstable tree. `n_estimators` controls tree count, `max_features` controls split-level feature randomness, `bootstrap` controls row sampling, and tree-depth/leaf parameters control base-tree complexity. `n_jobs=-1` can parallelize fitting.

```python
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                min_samples_leaf=2, class_weight="balanced",
                                random_state=42, n_jobs=-1)
forest.fit(X_train, y_train)
```

## Evaluation and API cautions
Trees do not generally need standardized numeric features, but still require a suitable missing-data strategy and encoded categoricals according to estimator support/version. Out-of-bag estimates are available with bootstrap sampling but do not replace careful validation in every workflow. `random_state` stabilizes randomized sampling. Defaults differ by estimator/version; specify important choices deliberately.

## Worked reasoning
A training score near 1.0 with much lower CV score suggests the tree is too complex. Try shallower depth, larger `min_samples_leaf`, and compare CV, not training score alone. A forest may improve stability but can still overfit noisy leakage or provide poor extrapolation beyond observed target ranges.

## Q&A / traps
**Q: Why no scaling?** Split order is invariant to monotonic scaling. **Q: Are forests interpretable?** Less directly than one small tree; feature importance is not causal effect. **Mistake:** assume more trees solve all overfitting; more trees mainly reduce Monte Carlo variability, while tree depth controls complexity.
