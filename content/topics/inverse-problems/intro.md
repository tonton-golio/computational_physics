# Introduction to Inverse Problems

An **inverse problem** starts from measurements and asks for the hidden parameters that produced them.

$$
\mathbf{d} = g(\mathbf{m})
$$

- $\mathbf{d}$: observed data
- $\mathbf{m}$: unknown model parameters
- $g$: forward model (physics + measurement process)

In practice, we know $\mathbf{d}$ and we can evaluate $g$, but computing $g^{-1}$ is usually impossible or unstable.

---

## Why Inverse Problems Are Hard

Following Hadamard, an inverse problem can fail to be well-posed in three ways:

1. **No exact solution** exists
2. **Multiple solutions** fit the data
3. **Instability**: small data noise causes large model changes

That is why regularization, priors, and uncertainty quantification are central to this field.

---

## Canonical Examples

### Medical and geophysical imaging

We infer hidden structure from indirect signals:

- MRI and SPECT infer tissue properties from measured responses
- Seismic tomography infers subsurface velocity/slowness from travel times

[[figure mri-scan]]

[[figure spect-scan]]

[[figure seismic-tomography]]

### Waveform inversion

A simplified acoustic model is:

$$
\frac{1}{\kappa(x)}\frac{\partial^2 p}{\partial t^2}(x,t) - \nabla\cdot\left(\frac{1}{\rho(x)}\nabla p(x,t)\right)=S(x,t)
$$

Given source and boundary conditions, we measure $p(x_n,t)$ at sensors and infer acceptable $\kappa(x)$ and $\rho(x)$.

---

## A Quick Instability Example (Hadamard)

For a 2D heat-flow setup, one obtains

$$
T(x,y)=\frac{1}{n^2}\sin(nx)\sinh(nx).
$$

- At $y=0$, the boundary data can remain bounded
- At $y>0$, $\sup |T(x,y)|$ can blow up as $n\to\infty$

So tiny perturbations at the boundary can produce huge changes in the interior solution: a textbook unstable inverse setting.

---

## Bayesian Viewpoint

Regularization can be interpreted probabilistically.
A common form is:

$$
\sigma(\mathbf{m}) = \rho_m(\mathbf{m})\,L(\mathbf{m}), \qquad L(\mathbf{m}) = \rho_d(g(\mathbf{m}))
$$

Here, prior knowledge $\rho_m$ and data likelihood $L$ combine into a posterior over models.
In high dimensions, we often sample this posterior with Monte Carlo methods.

---

## What Comes Next

- [Week 1](./week1): information, entropy, and uncertainty
- [Week 2](./week2): least squares and Tikhonov regularization
- [Week 3](./week3): linear tomography pipeline
- [Week 4](./week4): nonlinear inversion and Monte Carlo
