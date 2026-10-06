# ML Mentor RAG Knowledge Base

## Purpose
A structured, original teaching corpus for a retrieval-augmented assistant answering practical scikit-learn and introductory machine-learning questions. Files are intentionally divided into focused topics so a retriever can return a coherent passage rather than a giant chapter.

## Grounding and version
The corpus was checked against the official scikit-learn documentation available on 2026-10-05, using the `stable` documentation URLs listed in `06_practice/SOURCES.md`. The stable documentation changes as scikit-learn evolves. Examples use widely supported APIs, generally available in modern scikit-learn releases; estimator defaults can change. In particular, inspect the installed version with `import sklearn; print(sklearn.__version__)` and consult that version's docs before relying on a default. Examples target Python 3 with NumPy, pandas, and scikit-learn installed. They are illustrative, not a promise that every snippet is a complete runnable application.

## How to use
Put the extracted `ml_mentor_knowledge_base` directory or its contents in the project's document-ingestion location. Ingest Markdown as text; preserve headings and code fences. Chunk by heading, roughly 400–900 tokens with modest overlap, and retain the relative file path as metadata. Avoid combining unrelated topic files into one chunk. Rebuild the vector index after replacing an older corpus so stale documents do not remain retrievable.

## Grounded-answer behavior
Answer from retrieved passages first. If details are absent, state that the corpus does not specify them and ask whether the learner wants a general explanation. Distinguish documented API facts from teaching heuristics. Do not claim that the corpus is exhaustive or that an example was executed. For version-sensitive questions, state the documented version context and point to the official reference.

## Map
- `01_data_preparation/`: preprocessing, categorical encoding and imputation, column-wise transforms, pipelines, split discipline.
- `02_validation_tuning/`: cross-validation, grid/random search, nested CV.
- `03_supervised_learning/`: Logistic Regression, KNN, trees/forests, SVM.
- `04_unsupervised_learning/`: K-Means, DBSCAN, PCA, t-SNE.
- `05_evaluation/`: classification/regression metrics and model choice.
- `06_practice/`: integrated solved exercises, practice questions, official sources.

## Safety against leakage
Any operation that estimates state from data (imputation, scaling, feature selection, PCA, encoding categories, and model fitting) must learn only from the training partition or current CV training fold. Use a `Pipeline`/`ColumnTransformer` inside validation tools to enforce this.
