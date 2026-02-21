# Introduction to Inverse Problems

**Question:** You record surface displacements after an earthquake. Can you uniquely determine how deep the fault is and how much it slipped?

**Answer in advance:** No. Many different fault geometries produce almost identical surface movements.

This is the hallmark of an *inverse problem*. The forward problem (given the fault, predict the surface) is easy and unique. The inverse problem is hard, non-unique, and unstable. Welcome to the club.

You drop a stone into a pond, and ripples spread outward. That's a **forward problem** — given the cause, predict the effect. Easy enough. Now imagine you're standing at the edge of the pond and all you see are ripples. Can you figure out where the stone landed? *That's* an inverse problem. You observe effects and try to work backwards to the cause.

$$
\mathbf{d} = g(\mathbf{m})
$$

- $\mathbf{d}$: the data you actually measured (the ripples)
- $\mathbf{m}$: the hidden model parameters you want (where the stone fell, how big it was)
- $g$: the forward model — the physics that connects cause to effect

In practice, we know $\mathbf{d}$ and we can evaluate $g$, but computing $g^{-1}$? That's where the trouble begins. It's usually impossible, or worse — it's possible but wildly unstable.

---

## Why Inverse Problems Are Hard

Jacques Hadamard figured this out over a century ago. He said a problem is **well-posed** if it satisfies three conditions:

1. A solution **exists**
2. The solution is **unique**
3. The solution **depends continuously on the data** (small changes in data produce small changes in the answer)

Inverse problems gleefully violate all three. Here's what goes wrong:

**No exact solution.** Your measurements have noise. The data you recorded doesn't perfectly match any physical model. So there's no model that exactly reproduces what you measured.

**Multiple solutions.** Several different models can produce nearly identical data. The Earth doesn't care which of ten different fault geometries you pick — they all wiggle the surface the same way.

**Instability.** This is the killer. Tiny perturbations in data can send your inferred model flying off to absurd values. Let's see this happen right now.

---

## Let's Break Something on Purpose

Here's the Hadamard instability example, and we're going to *feel* it. Consider a 2D heat-flow problem where the temperature satisfies Laplace's equation. One family of solutions is:

$$
T(x,y) = \frac{1}{n^2}\sin(nx)\sinh(ny).
$$

At the boundary $y = 0$, the temperature is $T(x,0) = 0$ — perfectly calm. The derivative $\partial T / \partial y$ at the boundary is $\frac{1}{n}\sin(nx)$, which gets *smaller* as $n$ increases. So the boundary data looks more and more innocent.

But look what happens in the interior. Pick $y = 1$ and watch:

| Frequency $n$ | Boundary signal $\sim 1/n$ | Interior $T(x,1) \sim \sinh(n)/n^2$ |
|:-:|:-:|:-:|
| $n = 1$ | 1.0 | 1.18 |
| $n = 10$ | 0.1 | $1.1 \times 10^{2}$ |
| $n = 100$ | 0.01 | $\sim 10^{41}$ |

Read that last row again. The boundary signal is *one hundredth* of the original, barely a whisper. But the interior solution has exploded to $10^{41}$. That's not a number — that's a catastrophe.

This is what instability looks like. Your data says "everything is fine." Your reconstruction says "the interior is on fire." A tiny high-frequency perturbation in the measurements — something you'd dismiss as noise — gets amplified into a nonsensical answer.

That queasy feeling in your stomach? Good. Hold onto it. It's the reason this entire course exists.

---

## Canonical Examples

These aren't abstract curiosities. Inverse problems show up everywhere that matters.

### Medical and geophysical imaging

In MRI and SPECT imaging, you measure electromagnetic or nuclear responses at the surface of the body and try to reconstruct what's happening inside. In seismic tomography, earthquakes send waves through the Earth's interior and we record the wiggles at the surface — then try to figure out what the interior looks like.

[[figure mri-scan]]

[[figure spect-scan]]

[[figure seismic-tomography]]

Every one of these is an inverse problem. Every one of them is ill-posed. And every one of them requires the tools we'll build in this course.

### Waveform inversion

A simplified acoustic wave equation:

$$
\frac{1}{\kappa(x)}\frac{\partial^2 p}{\partial t^2}(x,t) - \nabla\cdot\left(\frac{1}{\rho(x)}\nabla p(x,t)\right) = S(x,t)
$$

We record the pressure $p(x_n, t)$ at sensors and try to infer the material properties $\kappa(x)$ and $\rho(x)$. The forward problem is well-posed. The inverse? Spectacularly ill-posed.

---

## The Road Ahead

The central challenge is clear: inverse problems are ill-posed, and naive inversion will betray you. But there's good news. Over the next seven lessons, we'll build a toolkit that tames these problems:

- [Regularization — The First Rescue](./regularization): how to penalize wildness and stabilize inversion
- [Bayesian Inversion](./bayesian-inversion): the probabilistic viewpoint — regularization is a prior in disguise
- [Iterative Methods and Large-Scale Tricks](./tikhonov): when the problem is too big for a formula
- [Linear Tomography](./linear-tomography): a complete inversion workflow from rays to images
- [Monte Carlo Methods](./monte-carlo-methods): exploring the full space of plausible models
- [Geophysical Inversion Examples](./geophysical-inversion): fault and glacier case studies with real uncertainty
- [Information, Entropy, and Uncertainty](./information-entropy): measuring how much the data actually taught us

By the time we finish, you will be able to look at any inverse problem and say, calmly, "I know how to tame you."

---

## Further Reading

Tarantola's *Inverse Problem Theory* is the gold standard. Aster, Borchers & Thurber is excellent for the linear algebra foundations. But work through the demos first — the intuition matters more than the theorems at this stage.
