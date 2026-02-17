# Inverse Problems

Inverse problems ask a simple question: **what hidden system produced the data we measured?**
You observe effects, then infer causes.

Typical examples include medical imaging, seismic exploration, and climate model calibration.
The common challenge is that inverse problems are often **ill-posed**: solutions can be non-unique, unstable, or sensitive to noise.

[[figure mri-scan]]

[[figure spect-scan]]

[[figure seismic-tomography]]

---

## Learning Path

Use this path if you are new to the topic:

1. **Foundations**: [Introduction](./intro)
2. **Information + uncertainty**: [Week 1](./week1)
3. **Regularization and optimization**: [Week 2](./week2)
4. **Linear tomography workflow**: [Week 3](./week3)
5. **Nonlinear inversion and Monte Carlo**: [Week 4](./week4)

Deep dives:

- [Least Squares and Tikhonov](./Tikonov)
- [Linear Tomography Notes](./linear_tomography)

---

## Interactive Simulations in This Module

- Information theory: `entropy-demo`, `kl-divergence`
- Optimization: `steepest-descent`, `tikhonov-regularization`
- Linear inverse problems: `linear-tomography`
- Monte Carlo inversion: `monte-carlo-integration`, `vertical-fault-mcmc`, `glacier-thickness-mcmc`, `sphere-in-cube-mc`

This module is designed for short theory bursts followed by interactive exploration.
