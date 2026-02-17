# Week 4 - Nonlinear Inversion and Monte Carlo

Nonlinear inverse problems rarely admit closed-form solutions.
Instead, we estimate parameters with stochastic sampling and posterior exploration.

---

## Why Monte Carlo

Suppose your posterior over parameters is $p(\mathbf{x}\mid \mathbf{d})$.
Direct integration is often intractable in high dimensions, so we sample.

For basic rejection sampling:

$$
p_{\text{accept}}=\frac{p(\mathbf{x}_{\text{cand}})}{M},
\qquad M\ge \max_{\mathbf{x}} p(\mathbf{x}).
$$

With proposal density $q$:

$$
p_{\text{accept}}=\frac{p(\mathbf{x}_{\text{cand}})}{M q(\mathbf{x}_{\text{cand}})}.
$$

[[simulation monte-carlo-integration]]

---

## Markov Chain Monte Carlo (MCMC)

MCMC avoids independent sampling from a hard posterior.
It builds a chain that spends more time in high-probability regions.

A common Metropolis-style acceptance rule is:

$$
p_{\text{accept}}=\min\left(1,\frac{p(\mathbf{x}_{\text{new}})}{p(\mathbf{x}_{\text{old}})}\right).
$$

Practical diagnostics to monitor:

- burn-in length
- acceptance rate
- chain mixing
- autocorrelation

---

## Example: Vertical Fault Inversion

Geometry and synthetic setup:

[[figure vertical-fault-diagram]]

Posterior exploration:

[[simulation vertical-fault-mcmc]]

Key insight: multiple parameter combinations can produce similar data, so posterior shape matters more than a single point estimate.

---

## Example: Glacier Thickness Inversion

Forward geometry:

[[figure glacier-valley-diagram]]

Sampling-based inversion:

[[simulation glacier-thickness-mcmc]]

This case highlights the trade-off between fit quality and physically plausible smooth thickness profiles.

---

## Monte Carlo Geometry Intuition

Before full inversion, it helps to build geometric intuition with volume estimation:

[[simulation sphere-in-cube-mc]]

As sample count grows, Monte Carlo error decays roughly as $O(N^{-1/2})$.

---

## Week 4 Takeaway

For nonlinear inverse problems, uncertainty is part of the answer.
Monte Carlo methods turn inversion from "find one best model" into "characterize a credible family of models."

