# Classification metrics

## Confusion matrix
For binary classification: TP, FP, TN, FN. Precision = TP/(TP+FP), the share of predicted positives that are correct. Recall/sensitivity = TP/(TP+FN), the share of actual positives detected. Specificity = TN/(TN+FP). F1 is the harmonic mean of precision and recall. Accuracy is correct predictions / all observations.

## Choose by error cost
Accuracy is useful when classes/costs are balanced but can hide failure on a rare class. High recall matters when missed positives are costly; high precision matters when false alarms are costly. F-beta weights recall more when beta>1 and precision more when beta<1. Macro averaging weights classes equally; weighted averaging weights by support; micro aggregates counts. State the averaging mode.

## Threshold and scores
`predict` typically applies a model's default decision rule. Changing the probability threshold trades precision against recall. ROC AUC evaluates ranking across thresholds; in heavily imbalanced settings, precision-recall curves/AP can better expose positive-class performance. Log loss evaluates probability quality and penalizes confident wrong predictions. AUC does not guarantee calibrated probabilities or good performance at the deployed threshold.

## API
Use `confusion_matrix`, `classification_report`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `average_precision_score`, and `log_loss`. For ROC AUC, pass probabilities or decision scores, not hard class labels. In cross-validation use scorer names such as `"f1"`, `"recall"`, `"roc_auc"`; many loss scorers are negated by sklearn's higher-is-better convention.

## Worked example
There are 100 cases: 10 positive. Model predicts 8 positive, with 6 correct. Precision=6/8=.75; recall=6/10=.60; FN=4, FP=2. Accuracy could still be 94/100 if it correctly calls 88 negatives, but recall reveals four missed positives.

## Q&A / pitfalls
**Q: Why does classification report show macro and weighted?** They answer different questions about class balance. **Trap:** multiclass F1 requires an explicit/understood averaging convention. **Mistake:** optimize threshold on the final test set; choose it on validation data, then evaluate once.
