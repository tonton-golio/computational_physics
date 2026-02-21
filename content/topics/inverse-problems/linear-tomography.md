# Linear Tomography

Here's a puzzle. You can't see inside the Earth. You can't drill deep enough, and even if you could, you'd only see one tiny spot. But earthquakes send seismic waves criss-crossing through the interior, and we have seismometers all over the surface recording when those waves arrive.

Each wave takes a path through the Earth. If it passes through slow material, it arrives late. If it passes through fast material, it arrives early. So the arrival time carries information about the material it crossed.

Now: given thousands of arrival times, can you reconstruct the velocity structure of the interior?

*Yes.* And the method is called **tomography** — literally, "drawing by slices." It's the same principle behind a CT scan at the hospital, except the patient is the entire planet.

---

## Forward Model: Rays Through the Earth

For a ray traveling along path $\gamma$, the travel-time anomaly (how late it is compared to a reference model) is:

$$
t_\gamma = \int_\gamma s(u)\,du,
$$

where $s(u)$ is the slowness anomaly along the path. Slow region? The integral is large. Fast region? Small.

It's like asking a tourist who walked through 50 cities which ones slowed them down. Each tourist only knows about the cities they actually visited — and each answer only lights up a few boxes on the map. But if you have *hundreds* of tourists taking different routes, you can piece together the whole map.

---

## From Rays to Matrix Form

Discretize the Earth into a grid of cells. Each cell has an unknown slowness anomaly. Each ray passes through some cells and misses others. This turns the continuous integral into a matrix equation:

$$
\mathbf{d} = \mathbf{Gm}.
$$

- **Rows** of $\mathbf{G}$: one per ray (measurement)
- **Columns** of $\mathbf{G}$: one per grid cell (unknown)
- **Entry** $G_{ij}$: the length of ray $i$ inside cell $j$ (zero if the ray misses that cell)
- $\mathbf{m}$: unknown slowness anomalies in each cell
- $\mathbf{d}$: measured travel-time anomalies

Most entries of $\mathbf{G}$ are zero — each ray only crosses a small fraction of the grid. This sparsity is what makes large-scale tomography computationally feasible.

---

## Building the Sensitivity Matrix

Here's the code that constructs $\mathbf{G}$ for a simple 2D cross-borehole geometry with diagonal rays:

```python
def make_G(N=13):
    G_right = [np.eye(N, k=1 + i).flatten() for i in range(N - 2)]
    G_left = [np.flip(np.eye(N, k=-(1 + i)), axis=0).flatten() for i in range(N - 2)]
    z = np.zeros((1, N**2))
    G = np.concatenate([z, G_left[::-1], z, z, G_right, z])
    return G * (2**0.5) * 1000
```

Each row is one ray. The non-zero entries mark which cells that ray passes through. The $\sqrt{2} \times 1000$ factor accounts for diagonal path length and unit conversion. Stare at this for a moment — each row is literally the ray's footprint on the grid.

---

## Synthetic Experiment: Test Before You Trust

A reliable inversion workflow always starts with **synthetic data**. You know the answer, so you can check whether your method recovers it.

1. Define a "true" anomaly model (something with clear features)
2. Compute noiseless data: $\mathbf{d}_{\text{true}} = \mathbf{Gm}_{\text{true}}$
3. Add controlled noise: $\mathbf{d}_{\text{obs}} = \mathbf{d}_{\text{true}} + \boldsymbol{\eta}$
4. Recover $\hat{\mathbf{m}}$ and compare with the truth

This reveals where your setup has resolving power and where it doesn't — *before* you ever touch real data. Skip this step and you're flying blind.

---

## Inversion: Regularized Reconstruction

Tomographic systems are typically underdetermined (more unknowns than data) and noisy. So we use regularized inversion:

$$
\hat{\mathbf{m}} = (\mathbf{G}^T\mathbf{G} + \epsilon^2\mathbf{I})^{-1}\mathbf{G}^T\mathbf{d}_{\text{obs}}.
$$

The target is a model that:

- fits the data within its uncertainty
- stays stable under noise
- avoids unrealistic spatial oscillations

If you've been following along, this should feel familiar — it's exactly the Tikhonov formula from the [regularization lesson](./regularization), applied to a concrete problem.

[[simulation linear-tomography]]

> **What to look for:**
> - Compare the true anomaly pattern with the recovered image — where does the reconstruction succeed? Where does it smear?
> - Change the regularization parameter and watch the image sharpen (small $\epsilon$) or blur (large $\epsilon$)
> - Notice which cells are well-resolved (crossed by many rays) and which are ghostly (sparse coverage)

---

## Resolution and Coverage

Here's something that trips up beginners: the quality of your tomographic image depends as much on your **acquisition geometry** as on your inversion algorithm.

- Dense, cross-cutting rays → high local resolution, sharp features
- Sparse, one-directional rays → elongated blobs, poor depth resolution

Two stress tests reveal this:

**Delta-function spike test.** Place a single anomaly in one cell. Invert. If the recovered image is a sharp spike, that cell is well-resolved. If it smears into a blob, the ray coverage there is poor.

**Checkerboard test.** Create an alternating pattern of positive and negative anomalies (like a checkerboard). Invert. Where the pattern recovers cleanly, you have good resolution. Where it smears or disappears, you don't. This test is standard practice in seismology — it immediately shows you the reliable and unreliable regions of your image.

The smearing pattern tells you exactly what the data can and cannot see. This is why survey design and inversion are inseparable — you can't fix bad geometry with clever algorithms.

---

## The Bridge: Linear to Nonlinear

Everything we've done so far relies on a crucial assumption: the forward model is **linear**. Travel time is a linear function of slowness. The matrix $\mathbf{G}$ doesn't depend on the model.

In the real Earth, this is almost never exactly true. Wave propagation depends on the velocity structure (rays bend). Material properties interact nonlinearly. The forward map $g(\mathbf{m})$ is a complicated function, not a simple matrix multiplication.

When the forward model is nonlinear, we can no longer find the answer with a single matrix inversion — even a regularized one. We have to stop hunting for one best model and start exploring a *family* of plausible models. That's where [Monte Carlo methods](./monte-carlo-methods) come in.

---

## Takeaway

Linear tomography turns geometry plus physics into a matrix inverse problem. The quality of reconstruction depends as much on ray coverage and regularization as on numerical solvers. And the checkerboard test is your best friend — it tells you honestly what the data can and cannot resolve.

---

## Further Reading

Nolet's *A Breviary of Seismic Tomography* is compact and insightful. Natterer's *The Mathematics of Computerized Tomography* covers the mathematical foundations.
