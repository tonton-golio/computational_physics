# Dimensionality Reduction

Here is a problem that comes up everywhere in science: you have data with hundreds or thousands of features, but the interesting structure lives on a much lower-dimensional surface. Dimensionality reduction finds that surface. It lets you compress data without losing the signal, strip away noise, and — perhaps most importantly — see patterns that are invisible in the raw high-dimensional space.

## PCA: finding the directions that matter

Suppose you photograph a 3D object from many angles. Each photo has millions of pixels, but the photos are not independent — they all depict the same object, just rotated. The "real" information is only three-dimensional (the rotation angles). Principal component analysis does exactly this kind of detective work: it looks at your data and asks, "Which directions carry the most variation, and which directions are mostly noise?"

Technically, PCA computes the covariance matrix of your centered data and finds its eigenvectors. The eigenvector with the largest eigenvalue points in the direction of greatest variance; the second eigenvector points in the direction of greatest remaining variance, perpendicular to the first; and so on.

For centered data matrix $X$:
$$
C=\frac{1}{n-1}X^\top X,\quad Z=X V_k
$$

Here $V_k$ contains the top $k$ eigenvectors of the covariance matrix $C$, and $Z$ is your data projected into $k$ dimensions. Each column of $V_k$ is a principal component — a direction in feature space ranked by how much variance it explains.

Use PCA when you need a fast, interpretable, linear embedding. It is the right first step for preprocessing before downstream models, for explained-variance diagnostics ("how many dimensions do we actually need?"), and as a sanity check before trying fancier methods.

## t-SNE: seeing your clusters

t-SNE is a nonlinear method designed for one thing: producing beautiful 2D visualizations where similar points cluster together. Think of it as placing your data points on a rubber sheet, then stretching and compressing the sheet so that points that were close together in high-dimensional space stay close, while points that were far apart get pushed further away.

The result is often stunning — you can *see* clusters that were hidden in 100 dimensions. But there is a loud warning you must internalize: **the distances on a t-SNE plot are not meaningful**. Two clusters that appear far apart might actually be close in the original space, and the size of clusters can be misleading. Use t-SNE for visual exploration, never for quantitative distance comparisons.

## UMAP: t-SNE's faster cousin

UMAP achieves similar goals to t-SNE — preserving local neighborhoods to reveal cluster structure — but it scales better to large datasets and often preserves more of the global organization. If t-SNE gives you beautiful clusters but you cannot tell how the clusters relate to each other, UMAP may give you a more faithful big-picture layout. That said, the same warnings about interpreting distances apply.

## Autoencoders: learned compression

Imagine a kid who has to describe a painting to a friend over the phone using only ten words. The friend then tries to redraw the painting from those ten words alone. If the redrawing is good, those ten words captured the essence of the painting. The ten-word description is the **latent representation**, and the kid-and-friend pair is an **autoencoder**.

An autoencoder is a neural network trained to reconstruct its input through a narrow bottleneck. The encoder compresses the input to a low-dimensional code; the decoder tries to reconstruct the original from that code.

$$
\mathbf{z}=f_\text{enc}(\mathbf{x}),\quad \hat{\mathbf{x}}=f_\text{dec}(\mathbf{z}),\quad \mathcal{L}=\|\mathbf{x}-\hat{\mathbf{x}}\|^2
$$

In words: we encode input $\mathbf{x}$ to a compressed code $\mathbf{z}$, decode it back to $\hat{\mathbf{x}}$, and minimize the reconstruction error. If the bottleneck has dimension $d$ and reconstruction quality is high, the data's intrinsic dimensionality is at most $d$.

Unlike PCA, autoencoders can capture curved, nonlinear manifolds. Architecture choices (depth, width, activation functions) control the kind of embedding you get. And because they are neural networks, they can be trained end-to-end with a downstream task. For the generative extension (variational autoencoders), see [Advanced Deep Learning — VAEs](/topics/advanced-deep-learning/vae).

## When to use which?

Here is a practical decision guide:

- **You need speed and linear interpretability** — use PCA. It is deterministic, fast, and gives you explained-variance scores to decide how many dimensions to keep.
- **You need detailed 2D cluster visualizations** — use t-SNE. Set perplexity carefully and run it multiple times with different seeds to check that the clusters are stable.
- **You need a fast nonlinear embedding that scales** — use UMAP. It handles large datasets well and often gives clearer global structure than t-SNE.
- **PCA underperforms and you need a learnable, nonlinear embedding** — use autoencoders. Especially valuable when you want to train the embedding end-to-end with a downstream model.

[[figure aml-dimred-comparison]]

## Interactive simulations

[[simulation aml-pca-correlated-data]]

[[simulation aml-explained-variance]]

[[simulation aml-tsne-umap-comparison]]

## Practical warnings

- Standardize features before PCA — otherwise the component with the largest scale dominates.
- t-SNE and UMAP output can vary with random seed and hyperparameters. Always run multiple times.
- Do not over-interpret apparent distances between far-apart clusters in t-SNE.
- Autoencoder embeddings depend on architecture and training. Validate that the bottleneck dimension captures meaningful structure before trusting the embedding.

## Check your understanding

- Can you explain PCA to a friend using the "photographs of a 3D object" analogy?
- Can you draw the one diagram that explains why t-SNE cluster distances are misleading?
- If you had a new dataset, how would you decide which dimensionality reduction method to try first?
