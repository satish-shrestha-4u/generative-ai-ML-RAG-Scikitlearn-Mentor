# Scikit-Learn Machine Learning Workflow

This document explains a complete Scikit-learn classification workflow, including:

- `train_test_split`
- `make_column_selector`
- `ColumnTransformer` / `make_column_transformer`
- Numeric and categorical preprocessing
- `Pipeline` / `make_pipeline`
- `cross_validate`
- Multiple model parameters
- `GridSearchCV`
- Final evaluation on an untouched test set
- Prevention of data leakage
- Pipeline parameter naming

The examples use a Titanic-style classification problem where `survived` is the target variable.

---

create an example of a ML model, including using train_test_splot, make_column_selector/transformer, several parameters tried, a cross valdation, a grid search cv and a testing.

Below is a complete **Scikit-learn classification example** that puts the main pieces together:

- `train_test_split` 
- `make_column_selector` 
- `ColumnTransformer` 
-  preprocessing for numeric + categorical columns 
- `Pipeline` 
-  several model parameters 
- `cross_validate` 
- `GridSearchCV` 
-  final evaluation on the untouched test set 

I'll use the Titanic dataset structure as an example.

# ---------------------------------------
# 1. IMPORTS
# ---------------------------------------

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV
)

from sklearn.compose import (
    make_column_selector,
    ColumnTransformer
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

## 1. Prepare `X` and `y`

Imagine our dataframe is called `df`, and we want to predict whether a passenger survived.

X = df.drop(columns="survived")
y = df["survived"]

Now split the data:

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

The important idea is:

FULL DATA
   │
   ├─────────────── 80% TRAIN ───────────────┐
   │                                         │
   │                              Cross-validation
   │                                         │
   │                              GridSearchCV
   │                                         │
   │                              Choose best model
   │                                         │
   └─────────────── 20% TEST ────────────────┘
                         ↓
                  Final evaluation only

You **do not use the test set during GridSearchCV**.

---

# 2. Automatically select numeric and categorical columns

numeric_selector = make_column_selector(dtype_include="number")

categorical_selector = make_column_selector(
    dtype_include=["object", "category"]
)

Instead of manually doing:

numeric_features = ["age", "fare", "siblings"]
categorical_features = ["sex", "class", "embarked"]

Scikit-learn detects them automatically.

---

# 3. Create preprocessing pipelines

For numeric columns:

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

For categorical columns:

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

So numerical data goes through:

missing values
     ↓
median imputation
     ↓
StandardScaler

Categorical data goes through:

missing values
     ↓
most frequent value
     ↓
OneHotEncoder

---

# 4. Combine them with `ColumnTransformer`

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numeric_selector
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_selector
    )
])

Conceptually:

                    X
                    │
          ┌─────────┴─────────┐
          │                   │
      Numerical          Categorical
          │                   │
      Imputer             Imputer
          │                   │
       Scaler            OneHotEncoder
          │                   │
          └─────────┬─────────┘
                    │
               Model input

---

# 5. Build the complete ML pipeline

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

Now your whole ML process is one estimator:

Raw Data
   ↓
Column selection
   ↓
Imputation
   ↓
Scaling / Encoding
   ↓
LogisticRegression
   ↓
Prediction

This is extremely useful because preprocessing will happen **inside each cross-validation fold**, which helps prevent data leakage.

---

# 6. First evaluate the basic model using cross-validation

Before tuning parameters, we can see how the default model performs.

scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring=scoring,
    return_train_score=True
)

Convert the results to a DataFrame:

cv_results = pd.DataFrame(cv_results)

print(cv_results)

For example:

   fit_time  test_accuracy  test_precision  test_recall  test_f1  test_roc_auc
0    0.03       0.81            0.79          0.73        0.76       0.85
1    0.02       0.84            0.82          0.77        0.79       0.88
2    0.02       0.80            0.76          0.75        0.75       0.84
3    0.02       0.83            0.81          0.74        0.77       0.87
4    0.03       0.82            0.78          0.78        0.78       0.86

You can calculate the averages:

print(cv_results.mean())

For example:

test_accuracy     0.820
test_precision    0.792
test_recall       0.754
test_f1           0.770
test_roc_auc      0.860

This tells you how well the **basic LogisticRegression pipeline** performs.

---

# 7. Try several parameters with GridSearchCV

Now let's tune the Logistic Regression.

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["liblinear", "lbfgs"],
    "model__class_weight": [None, "balanced"]
}

Notice the syntax:

model__C

The double underscore means:

pipeline step
     ↓
model
     ↓
parameter
     ↓
C

So:

model__C

means:

> Change the `C` parameter of the estimator inside the `"model"` pipeline step.

---

# 8. Run GridSearchCV

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    return_train_score=True
)

Now fit it:

grid_search.fit(X_train, y_train)

GridSearchCV will try every combination.

We have:

C:
5 possibilities

solver:
2 possibilities

class_weight:
2 possibilities

Therefore:

5 × 2 × 2 = 20 parameter combinations

With:

cv=5

that means approximately:

20 combinations × 5 folds
= 100 model fits

---

# 9. Look at the best parameters

print(grid_search.best_params_)

You might get:

{
    "model__C": 0.1,
    "model__class_weight": None,
    "model__solver": "liblinear"
}

And:

print(grid_search.best_score_)

For example:

0.79

That means the best parameter combination obtained an average cross-validation F1 score of approximately:

F1 = 0.79

---

# 10. Inspect all GridSearchCV results

This is very useful for certification questions.

results = pd.DataFrame(grid_search.cv_results_)

You can select the most interesting columns:

results[
    [
        "param_model__C",
        "param_model__solver",
        "param_model__class_weight",
        "mean_train_score",
        "mean_test_score",
        "rank_test_score"
    ]
].sort_values("rank_test_score")

Example:

C      solver       weight       train F1    CV F1     rank
0.1    liblinear    None          0.80       0.79       1
1      liblinear    None          0.81       0.78       2
10     lbfgs        None          0.82       0.78       3
0.1    lbfgs        balanced      0.78       0.77       4
...

This lets you compare the different configurations.

---

# 11. Get the best model

After GridSearchCV finishes:

best_model = grid_search.best_estimator_

Very importantly, `best_estimator_` contains the **whole pipeline**:

ColumnTransformer
        +
preprocessing
        +
LogisticRegression with best parameters

You do not need to rebuild it manually.

---

# 12. Finally use the test set

Only now do we use `X_test`.

y_pred = best_model.predict(X_test)

Calculate the metrics:

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

For example:

Accuracy:  0.82
Precision: 0.80
Recall:    0.76
F1:        0.78

You can also use:

print(classification_report(y_test, y_pred))

Example:

              precision    recall    f1-score

0                0.84       0.88       0.86
1                0.80       0.76       0.78

accuracy                            0.82

And the confusion matrix:

print(confusion_matrix(y_test, y_pred))

Example:

[[90 12]
 [17 54]]

---

# Complete version

Here is the whole workflow together:

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV
)

from sklearn.compose import (
    make_column_selector,
    ColumnTransformer
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# -------------------------
# X / y
# -------------------------

X = df.drop(columns="survived")
y = df["survived"]

# -------------------------
# Train/test split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------
# Column selectors
# -------------------------

numeric_selector = make_column_selector(
    dtype_include="number"
)

categorical_selector = make_column_selector(
    dtype_include=["object", "category"]
)

# -------------------------
# Numerical preprocessing
# -------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# -------------------------
# Categorical preprocessing
# -------------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# -------------------------
# ColumnTransformer
# -------------------------

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numeric_selector
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_selector
    )
])

# -------------------------
# Complete pipeline
# -------------------------

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# -------------------------
# Basic Cross-validation
# -------------------------

cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ],
    return_train_score=True
)

cv_results = pd.DataFrame(cv_results)

print("Cross-validation results:")
print(cv_results.mean())

# -------------------------
# Parameter grid
# -------------------------

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": [
        "liblinear",
        "lbfgs"
    ],
    "model__class_weight": [
        None,
        "balanced"
    ]
}

# -------------------------
# GridSearchCV
# -------------------------

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

# -------------------------
# Best model
# -------------------------

print("Best parameters:")
print(grid_search.best_params_)

print("Best CV score:")
print(grid_search.best_score_)

best_model = grid_search.best_estimator_

# -------------------------
# Final test evaluation
# -------------------------

y_pred = best_model.predict(X_test)

print("TEST RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

## The most important concept to remember

Think of the workflow as:

1. Full dataset
       ↓
2. train_test_split
       ↓
┌──────────────────────────────┐
│ TRAINING DATA                │
│                              │
│ Pipeline                     │
│   ├── ColumnTransformer      │
│   │     ├── numeric          │
│   │     └── categorical      │
│   │                          │
│   └── ML model               │
│                              │
│ Cross-validation             │
│          ↓                   │
│ GridSearchCV                 │
│          ↓                   │
│ Best hyperparameters        │
└──────────────────────────────┘
             ↓
       Best model
             ↓
┌──────────────────────────────┐
│ TEST DATA                    │
│                              │
│ predict()                    │
│      ↓                       │
│ Final metrics               │
└──────────────────────────────┘

One especially important distinction for the Scikit-learn certification is: **`cross_validate()` evaluates a configuration, whereas `GridSearchCV()` searches for a better configuration using cross-validation.** The separate test set then answers the final question: *how well does the selected model generalize to completely unseen data?*

using make_pipeline

Yes — with `make_pipeline`, the workflow is a little cleaner because Scikit-learn automatically names each step. Here’s the same kind of example using `make_pipeline` and `make_column_transformer`.

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV
)

from sklearn.compose import (
    make_column_selector,
    make_column_transformer
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

Assume we have a dataframe `df` and want to predict `survived`:

X = df.drop(columns="survived")
y = df["survived"]

Split the dataset first:

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Now create selectors for numerical and categorical columns:

numerical_selector = make_column_selector(
    dtype_include="number"
)

categorical_selector = make_column_selector(
    dtype_include=["object", "category"]
)

Create the preprocessing pipelines:

numerical_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

Then combine them:

preprocessor = make_column_transformer(
    (numerical_pipeline, numerical_selector),
    (categorical_pipeline, categorical_selector)
)

And create the final pipeline:

model = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=1000)
)

The pipeline now looks conceptually like:

X
│
├── numerical columns
│      ↓
│   SimpleImputer
│      ↓
│   StandardScaler
│
├── categorical columns
│      ↓
│   SimpleImputer
│      ↓
│   OneHotEncoder
│
└─────────────┐
              ↓
     LogisticRegression

### Cross-validation

Before tuning, you can evaluate the basic model:

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=5,
    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ],
    return_train_score=True
)

cv_results = pd.DataFrame(cv_results)

print(cv_results.mean())

This might give something like:

train_accuracy    0.84
test_accuracy     0.81
train_f1          0.80
test_f1           0.77
test_roc_auc      0.86

Now we can tune the model.

The important thing with `make_pipeline` is that **step names are generated automatically**.

You can inspect them with:

model

or:

model.named_steps

You will see something similar to:

{
    "columntransformer": ...,
    "logisticregression": LogisticRegression(...)
}

Therefore, parameters for Logistic Regression use:

logisticregression__parameter

For example:

param_grid = {
    "logisticregression__C": [0.01, 0.1, 1, 10, 100],
    "logisticregression__solver": [
        "liblinear",
        "lbfgs"
    ],
    "logisticregression__class_weight": [
        None,
        "balanced"
    ]
}

Run the grid search:

grid_search = GridSearchCV(
    model,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

Check the best combination:

print(grid_search.best_params_)

For example:

{
    "logisticregression__C": 0.1,
    "logisticregression__class_weight": None,
    "logisticregression__solver": "liblinear"
}

And the best cross-validation score:

print(grid_search.best_score_)

Then retrieve the best pipeline:

best_model = grid_search.best_estimator_

Finally, use the untouched test data:

y_pred = best_model.predict(X_test)

Evaluate it:

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

So the complete logic is:

FULL DATA
   │
   ↓
train_test_split
   │
   ├──────── TRAIN ──────────────────────────┐
   │                                         │
   │     make_pipeline                       │
   │          ↓                              │
   │     preprocessing                       │
   │          ↓                              │
   │     LogisticRegression                  │
   │          ↓                              │
   │     cross_validate                      │
   │          ↓                              │
   │     GridSearchCV                        │
   │          ↓                              │
   │     best_estimator_                     │
   │                                         │
   └──────── TEST ───────────────────────────┤
                                             ↓
                                     final prediction
                                             ↓
                                       final metrics

The key syntax difference to remember is:

# Pipeline()
Pipeline([
    ("model", LogisticRegression())
])

# parameter
"model__C"

versus:

# make_pipeline()
make_pipeline(
    LogisticRegression()
)

# automatically generated name
"logisticregression__C"

For certification questions, `make_pipeline` is especially worth remembering because **the estimator class name becomes the step name in lowercase**. For example, `StandardScaler` → `standardscaler`, and `LogisticRegression` → `logisticregression`.
