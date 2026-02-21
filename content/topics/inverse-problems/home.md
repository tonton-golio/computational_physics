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
 ║  1. WHY IT'S ║────▶║  2. REGULARIZATION ║────▶║  3. BAYESIAN       ║
 ║    HARD      ║     ║  The First Rescue  ║     ║  INVERSION         ║
 ╚══════════════╝     ╚═══════════════════╝     ╚════════════════════╝
                                                          │
       ╔════════════════════╗     ╔══════════════════╗    │
       ║  5. LINEAR         ║◀────║  4. ITERATIVE    ║◀───╯
       ║  TOMOGRAPHY        ║     ║  METHODS         ║
       ╚════════════════════╝     ╚══════════════════╝
                │
                ▼
 ╔══════════════════╗     ╔══════════════════╗     ╔═══════════════════╗
 ║  6. MONTE CARLO  ║────▶║  7. GEOPHYSICAL  ║────▶║  8. INFORMATION   ║
 ║  METHODS         ║     ║  CASE STUDIES    ║     ║  & ENTROPY        ║
 ╚══════════════════╝     ╚══════════════════╝     ╚═══════════════════╝
```

**Ill-posed → Stabilize → Probabilistic meaning → Scale up → Full workflow → Explore the posterior → Real applications → How much did we really learn?**

The key move: Bayesian inversion comes right after regularization. Once you see that the Tikhonov penalty *is* a Gaussian prior, everything else clicks — iterative methods become "how to find the MAP when the matrix is huge," and Monte Carlo becomes "how to explore beyond the MAP."

---

## What You'll Be Able to Do

By the end of this module, you will be able to:

- Formulate any inverse problem as parameter estimation from indirect measurements
- Recognize ill-posedness and choose appropriate regularization
- Explain *why* regularization works (it's a prior, not a trick)
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
