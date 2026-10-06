# Integrated solved examples and practice questions

## Solved 1: leakage-safe mixed-data classification
**Task:** Predict churn from numeric tenure/spend and categorical contract/region, with missing values and imbalanced classes.

**Solution outline:** split raw rows with `stratify=y` if IID; use group/time split if customers have repeated or temporal records. Build numeric median imputation + scaling and categorical most-frequent imputation + one-hot with unknown handling inside `ColumnTransformer`; append LogisticRegression or a forest in Pipeline. Compare recall, precision, F1, and PR AUC under stratified CV. Tune threshold on validation data if costs demand it. Reserve test data until decisions are fixed.

**Why:** all learned transformations fit inside folds; class imbalance makes accuracy alone insufficient; split protocol mirrors the prediction setting.

## Solved 2: choose a clustering method
**Task:** Customer points form two crescent shapes plus scattered anomalies; K unknown.

**Answer:** Try DBSCAN after appropriate scaling/distance analysis because it can find density-connected non-convex shapes and label noise. K-Means assumes compact centroid-based groups and requires K. DBSCAN may struggle if crescent densities differ; inspect eps/min_samples sensitivity and domain usefulness. Do not interpret `-1` as a learned class.

## Solved 3: compare tuned models honestly
**Task:** Select between SVM and Random Forest after searching many parameters on 600 rows.

**Answer:** Keep a final holdout or use nested CV. Put scaling in SVM pipeline; forest can omit scaling. Use the same outer folds and metric. Search only inner folds; report outer scores and variability. Refit the chosen search on development data and use holdout once.

## Practice questions (answers below)
1. Why must `StandardScaler` be inside the CV pipeline?
2. With TP=18, FP=6, FN=12, compute precision, recall, and F1.
3. What does `C=0.1` versus `C=100` usually imply in LogisticRegression/SVC?
4. Why can K-Means inertia not select K by itself?
5. Which split strategy prevents a patient's rows appearing in both train and validation?
6. What does DBSCAN label `-1` mean?
7. How do macro and weighted F1 differ?
8. Why is a t-SNE map unsuitable as proof of global cluster separation?
9. When is nested CV useful?
10. A regression model has R²=-0.2 on test. What does that tell you?
11. Why can one-hot encoding be safer than ordinal encoding for nominal categories?
12. Why can a perfect training score and much lower CV score indicate overfit?

## Answer key
1. Each fold must estimate its own means/scales from only its training part.
2. Precision=18/24=.75; recall=18/30=.60; F1=2(.75)(.60)/1.35≈.667.
3. Smaller C means stronger regularization; larger C penalizes violations less / fits more tightly.
4. Inertia monotonically decreases as K increases; an elbow is only a heuristic.
5. GroupKFold or StratifiedGroupKFold using patient ID as group, depending on class needs.
6. Noise/unassigned point under fitted density criteria.
7. Macro averages class scores equally; weighted averages by class support.
8. t-SNE prioritizes local neighborhoods and distorts global distances/cluster sizes.
9. When tuning bias needs estimating and a separate holdout is unavailable or data is limited.
10. On that test set, it performs worse than the mean-prediction baseline under R²'s squared-error comparison.
11. Ordinal codes invent order/distance; one-hot represents categories without that assumption.
12. The model may have learned idiosyncrasies in training that do not generalize.
