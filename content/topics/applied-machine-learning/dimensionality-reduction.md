# Dimensionality Reduction

Here is a problem that comes up everywhere in science: you have data with hundreds or thousands of features, but the interesting structure lives on a much lower-dimensional surface. Dimensionality reduction finds that surface. It lets you compress data without losing the signal, strip away noise, and -- perhaps most importantly -- *see* patterns that are invisible in the raw high-dimensional space.

## PCA: finding the directions that matter

Suppose you photograph a 3D object from many angles. Each photo has millions of pixels, but the photos are not independent -- they all depict the same object, just rotated. The "real" information is only three-dimensional (the rotation angles). Principal component analysis does exactly this kind of detective work: it looks at your data and asks, "Which directions carry the most variation, and which directions are mostly noise?"

Technically, PCA computes the covariance matrix of your centered data and finds its eigenvectors. The eigenvector with the largest eigenvalue points in the direction of greatest variance; the second eigenvector points in the direction of greatest remaining variance, perpendicular to the first; and so on.

For centered data matrix $X$:
$$
C=\frac{1}{n-1}X^\top X,\quad Z=X V_k
$$

Here $V_k$ contains the top $k$ eigenvectors of the covariance matrix $C$, and $Z$ is your data projected into $k$ dimensions. Each column of $V_k$ is a principal component -- a direction in feature space ranked by how much variance it explains.

Use PCA when you need a fast, interpretable, linear embedding. It is the right first step for preprocessing, for explained-variance diagnostics ("how many dimensions do we actually need?"), and as a sanity check before trying fancier methods.

## t-SNE: seeing your clusters

t-SNE is a nonlinear method designed for one thing: producing beautiful 2D visualizations where similar points cluster together. Think of it as placing your data points on a rubber sheet, then stretching and compressing the sheet so that nearby points in high-dimensional space stay close, while distant points get pushed further apart.

The result is often stunning -- you can *see* clusters that were hidden in 100 dimensions. But there is a loud warning you must internalize: **the distances on a t-SNE plot are not meaningful**. Two clusters that appear far apart might actually be close in the original space, and cluster sizes can be misleading. Use t-SNE for visual exploration, never for quantitative distance comparisons.

## UMAP: t-SNE's faster cousin

UMAP achieves similar goals -- preserving local neighborhoods to reveal cluster structure -- but it scales better to large datasets and often preserves more of the global organization. If t-SNE gives you beautiful clusters but you cannot tell how they relate to each other, UMAP may give you a more faithful big-picture layout. Same warnings about interpreting distances apply.

## Autoencoders: learned compression

Imagine a kid who has to describe a painting to a friend over the phone using only ten words. The friend tries to redraw the painting from those ten words alone. If the redrawing is good, those ten words captured the essence. The ten-word description is the **latent representation**, and the kid-and-friend pair is an **autoencoder**.

An autoencoder is a neural network trained to reconstruct its input through a narrow bottleneck. The encoder compresses the input to a low-dimensional code; the decoder reconstructs the original from that code.

$$
\mathbf{z}=f_\text{enc}(\mathbf{x}),\quad \hat{\mathbf{x}}=f_\text{dec}(\mathbf{z}),\quad \mathcal{L}=\|\mathbf{x}-\hat{\mathbf{x}}\|^2
$$

If the bottleneck has dimension $d$ and reconstruction quality is high, the data's intrinsic dimensionality is at most $d$. Unlike PCA, autoencoders can capture curved, nonlinear manifolds -- and because they are neural networks, they can be trained end-to-end with a downstream task.

## When to use which?

* **Speed and linear interpretability** -- PCA. Deterministic, fast, gives you explained-variance scores to decide how many dimensions to keep.
* **Detailed 2D cluster visualizations** -- t-SNE. Set perplexity carefully and run it multiple times with different seeds.
* **Fast nonlinear embedding that scales** -- UMAP. Handles large datasets well, often clearer global structure than t-SNE.
* **PCA underperforms, you need learnable nonlinear embedding** -- autoencoders. Especially valuable when you want to train the embedding end-to-end with a downstream model.

## So what did we really learn?

High-dimensional data is almost always much lower-dimensional in practice -- the interesting structure lives on a curved surface embedded in the full space. Dimensionality reduction is the act of finding that surface.

PCA is linear, deterministic, and interpretable, which makes it the right first move. When PCA's explained-variance curve has a sharp elbow, pay attention -- that is the data telling you its intrinsic dimension.

t-SNE and UMAP are visualization tools, not measurement tools. They will show you clusters, but you cannot trust the distances between them. Beautiful pictures can be deeply misleading if you forget this.

And autoencoders close the loop between dimensionality reduction and representation learning: the same compression idea, but now the embedding can be trained end-to-end with any downstream task. That latent-space idea -- compressing a complex object into a compact code -- is the conceptual bridge to everything neural networks do.

## Challenge

Take a dataset with at least 10 features. Run PCA and plot the explained-variance curve to identify the intrinsic dimensionality. Train a classifier on the full feature set and on the PCA-reduced set, and compare accuracy. At what number of retained components does performance stop improving? Now run t-SNE and UMAP on the same data and compare their layouts: do the same clusters appear? Are there apparent clusters in the 2D visualization that do not correspond to real class boundaries? Diagnose what went wrong and what went right in each method.
