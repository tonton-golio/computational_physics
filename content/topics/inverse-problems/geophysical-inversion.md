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

- **Fault depth** $D$: how deep the top of the fault is
- **Fault slip** $s$: how much the two sides moved past each other
- **Fault dip** $\delta$: the angle of the fault plane

The displacement is a nonlinear function of these parameters. Given the geometry, we can compute what the surface *should* look like — that's the forward model. But given the surface measurements, multiple fault configurations can produce nearly identical displacements.

### The Data

Suppose we measured surface displacements at 20 stations along a profile perpendicular to the fault:

- Station 1 (0.5 km from fault): 3.1 cm displacement
- Station 5 (2.5 km): 2.4 cm
- Station 10 (5.0 km): 1.2 cm
- Station 15 (7.5 km): 0.4 cm
- Station 20 (10.0 km): 0.1 cm

The displacements decay with distance — but *how* they decay encodes the fault geometry. A shallow fault produces a sharp near-field signal. A deep fault produces a broader, gentler pattern. A steeply dipping fault looks different from a gently dipping one.

### Running the MCMC

We set up a Metropolis sampler to explore the posterior over fault parameters. The algorithm:

1. Start from an initial guess (say, $D = 5$ km, $s = 1$ m, $\delta = 90°$)
2. Propose a small random perturbation to each parameter
3. Compute the forward model for the proposed fault
4. Compare with the data — if the fit improves, accept. If it worsens, accept with probability proportional to the likelihood ratio.
5. Repeat 50,000 times.

[[simulation vertical-fault-mcmc]]

### What the Posterior Tells Us

Watch the simulation carefully. The chain doesn't settle on a single answer — it wanders through a cloud of plausible fault configurations. Some things are well-determined: the total seismic moment (roughly, depth × slip) is tightly constrained because it controls the total amount of surface displacement. But there's a **trade-off**: a shallow fault with large slip can produce similar surface displacements as a deeper fault with smaller slip.

This shows up as an elongated, tilted cloud in the depth-vs-slip scatter plot. The data alone cannot break this trade-off. You'd need additional information — maybe InSAR data from a different viewing angle, or teleseismic waveforms — to shrink the posterior further.

The Earth doesn't care which of these fault models you pick — they all explain the surface measurements equally well. The whole cloud of models is the honest answer. A single "best" model would hide this fundamental ambiguity.

---

## Example 2: Glacier Bed Topography

### The Problem

A glacier flows down a valley, and we want to know the shape of the bedrock underneath. Why? Because bed topography controls ice flow, determines how the glacier will respond to climate warming, and governs whether meltwater can drain or gets trapped.

But the bed is buried under hundreds of meters of ice. We can't see it directly.

[[figure glacier-valley-diagram]]

### What We Can Measure

We have two types of data:

- **Surface velocity**: ice flows faster where it's thicker (roughly proportional to the fourth power of thickness — the Glen flow law). Measured by tracking features in satellite images.
- **Surface elevation**: precisely measured by GPS or lidar. The bed elevation equals surface elevation minus ice thickness.

The forward model connects bed topography to surface observables through the equations of ice flow. It's nonlinear — the relationship between bed shape and surface velocity involves a power law and depends on the local slope.

### Running the MCMC

We parameterize the bed as a smooth curve (a set of control points with interpolation) and explore the posterior with MCMC:

1. Start from a flat-bed initial guess
2. Propose random perturbations to each control point
3. Run the forward model (ice flow simulation) for the proposed bed
4. Compare predicted surface velocity and elevation with observations
5. Accept or reject using the Metropolis criterion
6. After 30,000 iterations, examine the posterior distribution over bed shapes

[[simulation glacier-thickness-mcmc]]

### A Family of Possible Beds

The result isn't one bed profile — it's a family of plausible profiles. Some key features emerge:

**Well-constrained regions:** Where the glacier is thin and surface data is dense, the posterior is narrow. We know the bed pretty well there.

**Poorly-constrained regions:** Near the glacier center, where the ice is thickest, many different bed shapes produce similar surface patterns. The posterior is wide. We're honestly uncertain.

**The overdeepening question:** Some sampled bed profiles show a deep trough near the glacier center (an "overdeepening"). Others don't. The data alone cannot resolve whether this feature exists. This matters hugely for predictions of future glacier retreat — if there's an overdeepening, warm ocean water can intrude and accelerate melting.

Notice the trade-off between fit quality and physical plausibility: you could create a bed profile with lots of bumps and ridges that fits the data marginally better, but it would be geologically absurd. The prior (smoothness constraint) keeps the bed shapes realistic, while the likelihood (data fit) pulls them toward the observations.

---

## The Lesson: The Answer Is Always a Distribution

Look at both examples one more time. In neither case did we produce "the answer." We produced a collection of answers — a posterior distribution — and that distribution tells us:

- **What's well-determined:** features that appear in every sample
- **What's uncertain:** features that vary wildly between samples
- **What trade-offs exist:** which parameters can compensate for each other

This is the payoff of the entire course. Regularization (Lessons 2–3) stabilized the inversion. The Bayesian framework (Lesson 4) gave it probabilistic meaning. Tomography (Lesson 5) showed the complete linear workflow. And Monte Carlo (Lesson 6) gave us the computational machinery to explore nonlinear posteriors.

The answer to an inverse problem is never a single number. The answer is always a distribution. That distribution is where the science lives.

---

## Connection to Regularization

Recall from [Bayesian Inversion](./bayesian-inversion) that Tikhonov regularization corresponds to a Gaussian prior. MCMC generalizes this — instead of finding a single regularized optimum, we explore the full posterior distribution. These examples show why that matters:

- The posterior is **multimodal**: the fault example can have distinct solution families
- Parameter **trade-offs** create elongated or curved posterior shapes that a point estimate would miss
- **Uncertainty estimates** are as scientifically important as the best-fit model itself

---

## Takeaway

For nonlinear inverse problems, uncertainty is not a nuisance to be minimized — it is part of the answer. Monte Carlo methods turn inversion from "find one best model" into "characterize a credible family of models." The examples here — a fault and a glacier — are simple enough to visualize but rich enough to show the essential features: trade-offs, ambiguity, and the irreducible honesty of a posterior distribution.

---

## Further Reading

Mosegaard & Tarantola's *Monte Carlo sampling of solutions to inverse problems* (JGR, 1995) is the classic paper on MCMC in geophysics. Sambridge's *Geophysical Inversion with a Neighbourhood Algorithm* offers an alternative sampling strategy. For the fault mechanics, Okada's (1985) dislocation model is the standard forward model. For glaciers, Gudmundsson's work on inverse methods for ice flow is excellent. But the real learning happens when you run the simulations and watch the posterior breathe.
