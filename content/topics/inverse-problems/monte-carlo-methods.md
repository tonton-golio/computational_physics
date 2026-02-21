# Monte Carlo Methods

Imagine you're a hiker trying to find the deepest point in a dark valley. You can't see the terrain — you can only feel the ground under your feet. You take a step. If it goes downhill, great, keep going. If it goes uphill, you *usually* step back — but sometimes, with a small probability, you go uphill anyway. Why? Because the downhill direction might lead to a shallow puddle, while the uphill step might take you over a ridge to a much deeper valley beyond.

That's the essential idea behind Markov Chain Monte Carlo. But let's build up to it.

---

## Why Sampling?

In the [Bayesian lesson](./bayesian-inversion), we wrote down the posterior distribution $p(\mathbf{m} \mid \mathbf{d})$. That's the full answer to the inverse problem — it tells you which models are plausible and which aren't.

But here's the problem: for anything beyond simple linear-Gaussian cases, you can't compute the posterior analytically. It lives in a high-dimensional space, it might have multiple peaks, curved ridges, and heavy tails. You can't integrate it, you can't visualize it directly, and you certainly can't summarize it with a single number.

What you *can* do is **sample** from it. Generate thousands of models, each drawn from the posterior distribution. The collection of samples *is* the answer — means, variances, correlations, confidence intervals all come from the samples.

---

## Monte Carlo Integration: The Basic Idea

The simplest form of Monte Carlo is remarkably straightforward. Want to compute the integral $\int f(\mathbf{x})\,p(\mathbf{x})\,d\mathbf{x}$? Draw $N$ samples from $p$, evaluate $f$ at each one, and average:

$$
\int f(\mathbf{x})\,p(\mathbf{x})\,d\mathbf{x} \approx \frac{1}{N}\sum_{i=1}^N f(\mathbf{x}_i).
$$

The error shrinks as $O(N^{-1/2})$ — regardless of dimension. That's the magic. In one dimension, quadrature rules are faster. In ten dimensions, they choke. In a hundred dimensions, Monte Carlo is the only game in town.

---

## Geometric Intuition: Throwing Darts

Before full inversion, let's build intuition. Suppose you want to estimate the volume of a sphere inscribed in a cube. Throw random darts at the cube. Count how many land inside the sphere. The ratio estimates the volume fraction.

[[simulation sphere-in-cube-mc]]

Things to look for in the simulation:

* Watch the estimate converge — noisy at first, then settling down
* Increase the dimension and see how the sphere's volume fraction collapses (the curse of dimensionality in action)
* Notice the $1/\sqrt{N}$ convergence: 100x more samples for 10x more accuracy

[[simulation monte-carlo-integration]]

---

## The Problem with Rejection Sampling

For simple distributions, you can use **rejection sampling**: propose a candidate from some easy distribution, accept it with probability proportional to the target density:

$$
p_{\text{accept}} = \frac{p(\mathbf{x}_{\text{cand}})}{M}, \qquad M \ge \max_{\mathbf{x}} p(\mathbf{x}).
$$

With a proposal density $q$:

$$
p_{\text{accept}} = \frac{p(\mathbf{x}_{\text{cand}})}{M\,q(\mathbf{x}_{\text{cand}})}.
$$

This works fine in low dimensions. But in high dimensions, the acceptance rate plummets. Think about it: in 100 dimensions, almost all the volume of the cube is in the corners, far from the target distribution. You'd throw billions of darts and accept almost none.

We need a smarter strategy.

---

## Markov Chain Monte Carlo (MCMC)

Here's the key insight: instead of generating independent samples from a hard distribution, build a **chain** of samples where each step depends on the previous one. Design the chain so that it spends more time in high-probability regions — exactly where the posterior puts its weight.

The simplest version is the **Metropolis algorithm**:

1. Start at some model $\mathbf{m}_{\text{current}}$
2. Propose a new model: $\mathbf{m}_{\text{new}} = \mathbf{m}_{\text{current}} + \text{random perturbation}$
3. Accept the new model with probability:

$$
p_{\text{accept}} = \min\left(1, \frac{p(\mathbf{m}_{\text{new}})}{p(\mathbf{m}_{\text{current}})}\right)
$$

4. If accepted, move to $\mathbf{m}_{\text{new}}$. If rejected, stay put. Either way, record the current position.
5. Repeat. Many, many times.

Imagine the posterior probability landscape as a hilly terrain. The Metropolis walker always prefers to walk uphill (better fit). But occasionally it takes a small downhill step — with probability exactly equal to the ratio of probabilities. That tiny chance of going downhill is what lets the walker escape local valleys and explore the entire mountain range.

The beautiful thing: if the new model is *more* probable, you always accept. If it's *less* probable, you sometimes accept anyway — and the probability of accepting is exactly the ratio of the densities. This is what keeps you from getting trapped in local optima.

After enough steps, the chain "forgets" where it started and its samples are drawn from the true posterior. The histogram of visited models *is* the posterior distribution.

---

## Practical Diagnostics: Is the Chain Working?

Running MCMC is easy. Knowing whether it *worked* is the hard part. Here's what to watch:

**Burn-in.** The chain starts wherever you initialized it, which might be far from the posterior. The initial samples are garbage — discard them. How many? Plot the chain's trajectory and look for where it "settles down."

**Acceptance rate.** If you accept everything (rate ~100%), your steps are too tiny and you're barely moving. If you accept almost nothing (rate ~1%), your steps are too big and you keep proposing implausible models. For Gaussian targets, optimal acceptance is around 23% in high dimensions, 44% in one dimension.

**Mixing.** Does the chain explore the full posterior, or does it get stuck in one region? Multiple chains started from different points should converge to the same distribution.

**Autocorrelation.** Consecutive samples are correlated (each step is a small perturbation of the last). Effective sample size is the total number of samples divided by the autocorrelation time. A chain of 100,000 samples with autocorrelation time 50 gives you only ~2,000 independent samples.

---

## Big Ideas
* The $O(N^{-1/2})$ convergence of Monte Carlo integration is dimension-independent — the same rate in one dimension as in a thousand. That is the entire reason MCMC exists as a discipline.
* The Metropolis acceptance rule is not an approximation. When the chain has mixed, the histogram of visited states is the exact posterior. You are not fitting — you are sampling.
* Acceptance rate, burn-in, and autocorrelation time are not optional diagnostics. They are the only evidence you have that the chain actually worked.
* A stuck chain with low acceptance rate can look exactly like a well-mixed chain in the wrong metric. Multiple chains started from different points are your insurance policy.

## What Comes Next

Monte Carlo methods give you samples from the posterior. But knowing what to do with those samples — and what they mean scientifically — requires applying them to real problems. The theoretical machinery becomes concrete when you run MCMC on an earthquake fault geometry and watch the sampler reveal a trade-off between fault depth and slip that no point estimate could show you. It becomes real when the glacier bed example demonstrates that some features are sharply constrained by the data while others remain genuinely unknown, no matter how long the chain runs.

The geophysical examples also show the full inversion workflow in context: how the forward model feeds the likelihood, how a smoothness prior keeps the solutions physically plausible, and how the resulting posterior distribution — not any single model — is the honest scientific answer to the original question.

## Check Your Understanding
1. Why does rejection sampling fail in high dimensions, even when the proposal distribution is reasonable? Describe what happens geometrically as dimension increases.
2. You run 100,000 Metropolis steps and compute the acceptance rate as 78%. Should you be worried, and if so, what should you change? What if the rate is 2%?
3. Two independent Markov chains are started from very different initial points. After 10,000 steps they have converged to the same stationary distribution. What does this tell you — and what does it still not guarantee?

## Challenge

Implement a Metropolis sampler for a two-dimensional posterior that has two separated modes — for example, a mixture of two Gaussians placed far apart. Run a single chain and record how often it successfully crosses between modes as a function of the proposal step size. Then implement a simple parallel tempering scheme (run chains at different "temperatures" and occasionally swap states between adjacent chains) and measure the mode-crossing rate again. How much does tempering improve exploration, and how does it scale with the separation between modes?

