# Geophysical Inversion Examples

This is where everything we've built comes together. We've developed regularization, the Bayesian framework, and Monte Carlo sampling. Now let's point these tools at real geophysical problems and see what happens.

Two case studies: a fault beneath the surface, and a glacier hiding its bed. In both cases, the answer isn't a number — it's a distribution.

---

## Example 1: Vertical Fault Inversion

### The Problem

An earthquake ruptures along a fault buried beneath the surface. The rupture displaces the ground — GPS stations and satellite radar measure how much the surface moved. The question: what is the geometry of the fault? How deep does it go? How much did it slip?

[[figure vertical-fault-diagram]]

### The Forward Model

For a vertical strike-slip fault in an elastic half-space, the surface displacement at a point $x$ depends on:

* **Fault depth** $D$: how deep the top of the fault is
* **Fault slip** $s$: how much the two sides moved past each other
* **Fault dip** $\delta$: the angle of the fault plane

The displacement is a nonlinear function of these parameters. Given the geometry, we can compute what the surface *should* look like — that's the forward model. But given the surface measurements, multiple fault configurations can produce nearly identical displacements.

### The Data

Suppose we measured surface displacements at 20 stations along a profile perpendicular to the fault:

* Station 1 (0.5 km from fault): 3.1 cm displacement
* Station 5 (2.5 km): 2.4 cm
* Station 10 (5.0 km): 1.2 cm
* Station 15 (7.5 km): 0.4 cm
* Station 20 (10.0 km): 0.1 cm

The displacements decay with distance — but *how* they decay encodes the fault geometry. A shallow fault produces a sharp near-field signal. A deep fault produces a broader, gentler pattern. A steeply dipping fault looks different from a gently dipping one.

### Running the MCMC

We set up a Metropolis sampler (see [Monte Carlo Methods](./monte-carlo-methods) for the algorithm) to explore the posterior over fault parameters — starting from an initial guess, proposing random perturbations, and accepting or rejecting via the likelihood ratio. After 50,000 iterations, the chain has mapped out the posterior.

[[simulation vertical-fault-mcmc]]

[[simulation tradeoff-cloud]]

### What the Cloud of Answers Actually Says

Watch the simulation carefully. The chain doesn't settle on a single answer — it wanders through a cloud of plausible fault configurations. Some things are well-determined: the total seismic moment (roughly, depth times slip) is tightly constrained because it controls the total amount of surface displacement. But there's a **trade-off**: a shallow fault with large slip can produce similar surface displacements as a deeper fault with smaller slip.

This shows up as an elongated, tilted cloud in the depth-vs-slip scatter plot. The data alone cannot break this trade-off. You'd need additional information — maybe InSAR data from a different viewing angle, or teleseismic waveforms — to shrink the posterior further.

The Earth doesn't care which of these fault models you pick — they all explain the surface measurements equally well. The whole cloud is the honest answer. A single "best" model would hide this fundamental ambiguity.

---

## Example 2: Glacier Bed Topography

### The Problem

A glacier flows down a valley, and we want to know the shape of the bedrock underneath. Why? Because bed topography controls ice flow, determines how the glacier will respond to climate warming, and governs whether meltwater can drain or gets trapped.

But the bed is buried under hundreds of meters of ice. We can't see it directly.

[[figure glacier-valley-diagram]]

### What We Can Measure

We have two types of data:

* **Surface velocity**: ice flows faster where it's thicker — roughly proportional to the fourth power of thickness (for $n = 3$ in Glen's flow law, the shallow-ice approximation gives velocity $\sim h^{n+1} = h^4$). Measured by tracking features in satellite images.
* **Surface elevation**: precisely measured by GPS or lidar. The bed elevation equals surface elevation minus ice thickness.

The forward model connects bed topography to surface observables through the equations of ice flow. It's nonlinear — the relationship between bed shape and surface velocity involves a power law and depends on the local slope.

### Running the MCMC

We parameterize the bed as a smooth curve (control points with interpolation) and explore the posterior with the same Metropolis algorithm from [Monte Carlo methods](./monte-carlo-methods): propose perturbations to each control point, run the forward model, accept or reject. After 30,000 iterations, we have a family of plausible bed profiles.

[[simulation glacier-thickness-mcmc]]

The result isn't one bed profile — it's a family of plausible profiles. Where the glacier is thin and surface data is dense, the posterior is narrow — we know the bed pretty well. Near the glacier center, where the ice is thickest, many different bed shapes produce similar surface patterns and the posterior is wide. Some sampled beds show an overdeepening; others don't — the data simply cannot resolve whether this feature exists. That matters hugely for predictions of future glacier retreat, because warm ocean water can intrude into an overdeepening and accelerate melting.

The smoothness prior keeps the bed shapes geologically realistic while the likelihood pulls them toward the observations. You could create a bumpy bed that fits the data marginally better, but it would be geologically absurd.

---

## The Lesson: The Answer Is Always a Distribution

Look at both examples one more time. In neither case did we produce "the answer." We produced a collection of answers — a posterior distribution — and that distribution tells us:

* **What's well-determined:** features that appear in every sample
* **What's uncertain:** features that vary wildly between samples
* **What trade-offs exist:** which parameters can compensate for each other

This is the payoff of the entire course. [Regularization](./regularization) stabilized the inversion. The [Bayesian framework](./bayesian-inversion) gave it probabilistic meaning. [Iterative methods](./tikhonov) scaled it up. [Tomography](./linear-tomography) showed the complete linear workflow. And [Monte Carlo](./monte-carlo-methods) gave us the computational machinery to explore nonlinear posteriors.

The answer to an inverse problem is never a single number. The answer is always a distribution. That distribution is where the science lives.

---

## Resolution and Coverage

Both examples raise a practical question: where in the model can the data actually resolve structure, and where is it effectively blind?

Two standard diagnostic tests answer this:

**Spike test.** Place a single compact anomaly at one location in the model. Generate synthetic data and invert. If the recovered anomaly is sharp and localized, that region is well-resolved. If it smears into a broad blob, the data geometry there is poor.

**Checkerboard test.** Create an alternating pattern of positive and negative anomalies across the model grid. Generate synthetic data and invert. Where the checkerboard recovers cleanly, you have good resolution. Where it degrades into gray mush, you don't.

The smearing pattern tells you exactly what the data can and cannot see — and this depends on the acquisition geometry, not on the inversion algorithm. You can't fix bad survey design with clever math.

---

## Connection to Regularization

Recall from [Bayesian Inversion](./bayesian-inversion) that Tikhonov regularization corresponds to a Gaussian prior (under Gaussian noise and a quadratic penalty). MCMC generalizes this — instead of finding a single regularized optimum, we explore the full posterior distribution. These examples show why that matters:

* The posterior is **multimodal**: the fault example can have distinct solution families
* Parameter **trade-offs** create elongated or curved posterior shapes that a point estimate would miss
* **Uncertainty estimates** are as scientifically important as the best-fit model itself

---

So here's what stays with you from these examples: a trade-off between depth and slip is not a failure of the inversion — it's a fact about the data. Well-constrained and poorly-constrained regions coexist in every real inversion, and identifying which is which matters as much as the best-fit model itself. The smoothness prior isn't arbitrary aesthetics — it encodes the physical fact that geological structures don't change abruptly at the meter scale. And when two different models explain your data equally well, the right response is to report both and quantify what additional data would break the ambiguity.

## What Comes Next

These examples show the posterior distribution as a collection of samples — histograms, scatter plots, families of curves. But how much did the data actually teach us? We can see that the fault depth is uncertain, but is that uncertainty 20% of the prior range or 80%? Is the data worth collecting at all? To answer these questions precisely requires a way to measure information — to quantify the reduction in uncertainty from prior to posterior in consistent units.

That measurement is Shannon entropy and KL divergence, and they provide the deepest unifying framework for everything in this topic.

## Let's Make Sure You Really Got It
1. In the fault inversion example, the total seismic moment (proportional to depth times slip) is well-constrained while depth and slip individually are not. Explain geometrically why this happens using the shape of the posterior in the depth-slip plane.
2. In the glacier example, uncertainty in the reconstructed bed is largest where the ice is thickest. What is the physical reason for this — what property of the forward model causes thick ice to be harder to constrain?
3. The fault inversion uses a nonlinear forward model, which means the posterior cannot be computed analytically. Explain in your own words why nonlinearity of the forward model destroys the possibility of an analytical closed-form posterior, even if the prior and noise model are both Gaussian.

## Challenge

Design a synthetic "survey design" experiment for the glacier problem. Given a fixed budget of $N = 10$ velocity measurement points along the glacier surface, determine the optimal placement of those points to minimize the posterior uncertainty in the bed topography. Define a scalar measure of total uncertainty (for example, the average posterior variance over bed control points), implement the MCMC workflow for each proposed survey design, and compare several designs — uniform spacing, clustering near the glacier center, clustering near the edges. Can you find a placement that substantially outperforms uniform spacing? What does the optimal design tell you about which part of the forward model carries the most information about the bed?
