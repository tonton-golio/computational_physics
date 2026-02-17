# Dimensionality Reduction

Dimensionality reduction helps us compress, denoise, and visualize high-dimensional data.

## PCA: linear projection
Principal component analysis (PCA) finds orthogonal directions of maximal variance.

For centered data matrix $X$:
$$
C=\frac{1}{n-1}X^\top X,\quad Z=X V_k
$$
where columns of $V_k$ are top eigenvectors of covariance matrix $C$.

Use PCA for:
- fast baseline embeddings,
- preprocessing before downstream models,
- explained-variance diagnostics.

## t-SNE: neighborhood-preserving embedding
t-SNE is a nonlinear method that preserves local neighborhoods and is useful for visual inspection of clusters. Axes in t-SNE plots are not directly interpretable coordinates.

## UMAP: scalable manifold embedding
UMAP often preserves local structure while scaling well and can show clearer global organization than t-SNE in some settings.

## Method selection guide
- **Need speed and linear interpretability:** PCA.
- **Need detailed local clusters for visualization:** t-SNE.
- **Need fast nonlinear embedding with strong practical defaults:** UMAP.

## Interactive simulations
[[simulation aml-pca-correlated-data]]

[[simulation aml-explained-variance]]

[[simulation aml-tsne-umap-comparison]]

## Practical warnings
- Standardize features before PCA in most cases.
- t-SNE/UMAP output can vary with seed and hyperparameters.
- Do not over-interpret apparent distances between far clusters in t-SNE.
