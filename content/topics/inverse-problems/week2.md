# Week 2 - Regularization and Stable Inference

In week 1, we asked whether a model matches data.
Now we ask a harder question: **is the inferred model stable and physically plausible?**

---

## Linear Inverse Setup

We start from

$$
\mathbf{d}=\mathbf{Gm}+\boldsymbol{\eta},
$$

where $\boldsymbol{\eta}$ is measurement noise.
When $\mathbf{G}$ is ill-conditioned or rank-deficient, direct inversion is unstable.

---

## Tikhonov Objective

Use a regularized objective:

$$
J(\mathbf{m})=\|\mathbf{d}-\mathbf{Gm}\|^2+\epsilon^2\|\mathbf{m}\|^2.
$$

This balances:

- **data fit** (first term)
- **model complexity control** (second term)

Closed-form minimizer:

$$
\hat{\mathbf{m}}=(\mathbf{G}^T\mathbf{G}+\epsilon^2\mathbf{I})^{-1}\mathbf{G}^T\mathbf{d}.
$$

[[simulation tikhonov-regularization]]

---

## Optimization View

The same objective can be solved iteratively, which is useful for larger or nonlinear systems.

[[simulation steepest-descent]]

Interpretation:

- if the learning rate is too large, iterates oscillate or diverge
- if too small, convergence is slow
- regularization smooths the landscape and improves robustness

---

## Practical Guidance

When selecting $\epsilon$:

1. sweep a log-scale range
2. inspect residual vs model norm
3. prefer the simplest model that still explains data uncertainty

This is often more important than chasing the minimum residual.

---

## Week 2 Takeaway

Regularization is not an optional tweak.
It is the core mechanism that turns ill-posed inversion into a usable scientific workflow.
