# Inverse Problems

You measure something. You want to know what caused it. That's an inverse problem — and it's one of the hardest, most beautiful challenges in computational science.

The catch? The answer is almost never unique, and the obvious approach (just invert the equations) almost always blows up. This module gives you the tools to handle that — to stabilize inversion, quantify uncertainty, and extract honest answers from noisy, incomplete data.

[[figure mri-scan]]

[[figure spect-scan]]

[[figure seismic-tomography]]

---

## The Journey

Here's where we're going. Think of it as a subway map — each stop builds on the last:

```
 ╔══════════════╗     ╔═══════════════════╗     ╔════════════════════╗
 ║  1. WHY IT'S ║────▶║  2. REGULARIZATION ║────▶║  3. ITERATIVE       ║
 ║    HARD      ║     ║  The First Rescue  ║     ║  METHODS            ║
 ╚══════════════╝     ╚═══════════════════╝     ╚════════════════════╝
                                                          │
       ╔════════════════════╗     ╔══════════════════╗    │
       ║  5. LINEAR         ║◀────║  4. BAYESIAN     ║◀───╯
       ║  TOMOGRAPHY        ║     ║  INVERSION       ║
       ╚════════════════════╝     ╚══════════════════╝
                │
                ▼
 ╔══════════════════╗     ╔══════════════════╗     ╔═══════════════════╗
 ║  6. MONTE CARLO  ║────▶║  7. GEOPHYSICAL  ║────▶║  8. INFORMATION   ║
 ║  METHODS         ║     ║  CASE STUDIES    ║     ║  & ENTROPY        ║
 ╚══════════════════╝     ╚══════════════════╝     ╚═══════════════════╝
```

**Ill-posed → Stabilize → Iterate → Probabilistic view → Full workflow → Explore the posterior → Real applications → How much did we really learn?**

---

## Learning Path

1. **Why it's hard**: [Introduction to Inverse Problems](./foundations) — Hadamard, instability, and why naive inversion fails
2. **The first rescue**: [Regularization — The First Rescue](./regularization) — penalizing wildness, the L-curve, finding the sweet spot
3. **Scaling up**: [Iterative Methods and Large-Scale Tricks](./tikhonov) — when the formula is too expensive and you need to walk toward the answer
4. **Thinking probabilistically**: [Bayesian Inversion](./bayesian-inversion) — regularization as prior belief, from point estimates to posteriors
5. **A complete workflow**: [Linear Tomography](./linear-tomography) — from seismic rays to subsurface images
6. **Exploring the posterior**: [Monte Carlo Methods](./monte-carlo-methods) — sampling when analysis fails
7. **Real applications**: [Geophysical Inversion Examples](./geophysical-inversion) — faults, glaciers, and why the answer is always a distribution
8. **The deep question**: [Information, Entropy, and Uncertainty](./information-entropy) — measuring how much the data actually taught us

---

## What You'll Be Able to Do

By the end of this module, you will be able to:

- Formulate any inverse problem as parameter estimation from indirect measurements
- Recognize ill-posedness and choose appropriate regularization
- Set up and solve linear tomographic inversions
- Run MCMC to explore nonlinear posteriors
- Quantify what the data can and cannot resolve
- Look at any inverse problem and say, calmly, "I know how to tame you"

---

## Interactive Simulations

Each lesson includes hands-on simulations. Drag sliders, watch posteriors breathe, break things on purpose:

- Optimization: `steepest-descent`, `tikhonov-regularization`
- Linear inverse problems: `linear-tomography`
- Monte Carlo inversion: `monte-carlo-integration`, `sphere-in-cube-mc`, `vertical-fault-mcmc`, `glacier-thickness-mcmc`
- Information theory: `entropy-demo`, `kl-divergence`

Short theory bursts, then interactive exploration. That's the rhythm.

---

## Prerequisites

- Linear algebra: matrix operations, eigenvalues, least squares
- Calculus: derivatives, integrals, Taylor series
- Probability and statistics: distributions, Bayes' theorem, expectation
- Programming in Python with NumPy
