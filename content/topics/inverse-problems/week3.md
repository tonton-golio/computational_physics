# Week 3 - Linear Tomography Workflow

This week builds a complete linear tomography pipeline:
forward modeling, synthetic data generation, regularized inversion, and resolution analysis.

---

## From Rays to Matrix Form

Travel-time anomalies along rays are line integrals of slowness.
After discretization on a grid:

$$
\mathbf{d}=\mathbf{Gm}.
$$

- rows of $\mathbf{G}$: rays/measurements
- columns of $\mathbf{G}$: model cells
- $\mathbf{m}$: unknown slowness anomalies

---

## Synthetic Experiment

A reliable workflow starts with synthetic truth:

1. define a true anomaly model
2. compute noiseless data $\mathbf{Gm}_{\text{true}}$
3. add controlled noise
4. recover $\hat{\mathbf{m}}$ and compare

This reveals where the setup is resolvable before touching real data.

[[simulation linear-tomography]]

---

## Resolution and Coverage

Tomographic quality is driven by acquisition geometry:

- dense, cross-cutting rays -> higher local resolution
- sparse one-directional rays -> elongated uncertainty

This is why survey design and inversion are inseparable.

---

## Link to Deep Dive

For derivations and implementation details, see:

- [Linear Tomography Notes](./linear_tomography)
- [Least Squares and Tikhonov](./Tikonov)

---

## Week 3 Takeaway

Linear tomography is a matrix inverse problem constrained by physics and geometry.
Good reconstructions come from the combination of coverage, noise modeling, and regularization.
