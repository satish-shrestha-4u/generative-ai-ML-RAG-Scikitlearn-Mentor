# t-SNE for visualization

## Appropriate use
t-SNE is a nonlinear embedding method mainly used to visualize high-dimensional data in two or three dimensions. It tries to preserve local neighborhood relationships, not global distances. Apparent cluster gaps, sizes, and relative spacing can be misleading; the map is not a reliable cluster-discovery or predictive feature-engineering answer by itself.

## API and parameters
`TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate="auto", max_iter=1000, init="pca", metric="euclidean", random_state=None)` in modern releases; parameter names/defaults have changed (older releases use `n_iter`). Check installed docs. `perplexity` loosely relates to neighborhood scale and must be less than sample count. `random_state` supports reproducibility. `fit_transform` returns embedding; standard sklearn t-SNE is primarily fit on the supplied dataset and does not provide a general out-of-sample transform in the usual workflow.

## Workflow
Scale appropriately; often reduce very high-dimensional input with PCA first, then t-SNE. Run multiple seeds and inspect whether local neighborhoods persist. Color points by known labels only as an interpretation aid, not as input to unsupervised fitting. Avoid comparing absolute coordinates across separate runs.

## Q&A
**Q: Do far-apart islands imply far-apart original groups?** Not reliably. **Q: Can I feed t-SNE coordinates to a classifier?** Usually avoid: embedding can be unstable and lacks a natural mapping for future samples. **Exam trap:** a visually separated plot is not a quantitative generalization metric. Use PCA when a stable linear projection or transform for new data is needed.
