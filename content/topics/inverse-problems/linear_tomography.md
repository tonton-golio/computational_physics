# Linear Tomography

Linear tomography reconstructs hidden structure from line-integral measurements.
In geophysics, that means inferring subsurface slowness or density anomalies from travel-time data.

---

## Forward Model

For a ray path $\gamma$, the travel-time anomaly is

$$
t_\gamma=\int_\gamma s(u)\,du,
$$

where $s(u)$ is slowness anomaly.
After discretization on a grid:

$$
\mathbf{d}=\mathbf{Gm}.
$$

- $\mathbf{m}$: cell-wise model parameters (slowness/density anomaly)
- $\mathbf{d}$: measured travel-time anomalies
- $\mathbf{G}$: ray-path sensitivity matrix

---

## Building the Sensitivity Matrix

Each row of $\mathbf{G}$ corresponds to one ray.
Each column corresponds to one grid cell.
Entries represent ray length inside the cell (or a scaled approximation).

```python
def make_G(N=13):
    G_right = [np.eye(N, k=1 + i).flatten() for i in range(N - 2)]
    G_left = [np.flip(np.eye(N, k=-(1 + i)), axis=0).flatten() for i in range(N - 2)]
    z = np.zeros((1, N**2))
    G = np.concatenate([z, G_left[::-1], z, z, G_right, z])
    return G * (2**0.5) * 1000
```

---

## Inversion Step

Tomographic systems are typically noisy and underdetermined.
We therefore solve a regularized inverse problem:

$$
\hat{\mathbf{m}}=(\mathbf{G}^T\mathbf{G}+\epsilon^2\mathbf{I})^{-1}\mathbf{G}^T\mathbf{d}_{\text{obs}}.
$$

The target is a model that:

- fits the data within uncertainty
- remains stable under noise
- avoids unrealistic spatial oscillations

[[simulation linear-tomography]]

---

## Interpreting Resolution

If a region is crossed by many rays from multiple directions, resolution is good.
If a region is weakly sampled, uncertainty increases and artifacts can appear.

A useful stress test is a delta-like true model:

- well-covered cells reconstruct sharply
- poorly covered cells smear along acquisition geometry

This links acquisition design directly to inverse quality.

---

## Takeaway

Linear tomography turns geometry plus physics into a matrix inverse problem.
The quality of reconstruction depends as much on ray coverage and regularization as on numerical solvers.