# Bayesian Statistics

## A Different Way of Thinking

You've been using frequentist methods: p-values, confidence intervals, maximum likelihood. Probability as long-run frequency.

**Bayesian statistics** flips the question. Probability represents *belief*. You start with a prior belief, observe data, and update. The result is a **posterior distribution** -- everything you know about the parameter after seeing the data.

Neither framework is "correct." Frequentist: "how surprising is this data if the hypothesis is true?" Bayesian: "given the data, what should I believe?" Both are useful. The best practitioners speak both languages.

And here's the thing -- you've been doing Bayesian reasoning in disguise all along. The likelihood function from [PDFs](./probability-density-functions)? Same engine. The nuisance parameter constraints from [advanced fitting](./advanced-fitting-calibration)? Gaussian priors. Regularization in [ML](./machine-learning-data-analysis)? Bayesian priors again.

## Bayes' Theorem

$$
P(H|D) = \frac{P(D|H) \cdot P(H)}{P(D)}
$$

* $P(H|D)$: the **posterior** -- your updated belief. What you want.
* $P(D|H)$: the **likelihood** -- probability of data given hypothesis. Same function you've been using since PDFs.
* $P(H)$: the **prior** -- what you believed before seeing data.
* $P(D)$: the **evidence** -- a normalization constant. Often the hardest to compute, but you can frequently sidestep it.

The recipe: **posterior $\propto$ likelihood $\times$ prior**. Data update your belief.

## Example: Is This Coin Fair?

Let $\theta$ be the probability of heads. Watch how your belief evolves with each flip.

**Prior**: no reason to think the coin is biased. Uniform: $P(\theta) = 1$ for $\theta \in [0, 1]$.

**Flip 1**: heads. Likelihood is $P(D|\theta) = \theta$. Posterior: $P(\theta|D) \propto \theta$. Values near 1 are slightly more plausible. But the distribution is broad -- one flip tells you almost nothing.

**Flips 1-10**: 7 heads, 3 tails. The likelihood from the binomial:

$$
P(D|\theta) = \binom{10}{7} \theta^7 (1-\theta)^3
$$

Posterior: $P(\theta|D) \propto \theta^7 (1-\theta)^3$. That's a Beta(8, 4) distribution. Its peak is at $\theta = 0.7$ -- same as the MLE. But you get *more*: the full shape tells you how uncertain you are. The 95% credible interval runs roughly from 0.40 to 0.93. Still wide -- 10 flips haven't pinned it down.

**Flips 1-110**: 70 heads out of 110. The posterior tightens dramatically. Now the 95% credible interval is roughly 0.55 to 0.73. More data means more certainty.

**Flips 1-1000**: say 680 heads. The posterior is now razor-sharp, clustered tightly around 0.68. Whatever prior you started with barely matters anymore. This is the beautiful part: with enough data, reasonable people with different priors converge to the same conclusion. The data overwhelm the prior.

> **Challenge.** Grab a coin. Write your prior for $P(\text{heads})$ before flipping. Flip 10 times. After each flip, mentally update. Notice your confidence growing or shifting. That's Bayesian updating -- your brain does this naturally.

[[simulation prior-influence]]

## Choosing Priors

The prior is the controversial bit. Where does it come from?

* **Informative priors**: genuine prior knowledge. Previous experiments measured the quantity? Use that result. This is a great strength -- a principled way to combine old and new evidence.
* **Weakly informative priors**: gently constrain to physically reasonable ranges without committing to specific values.
* **Non-informative (flat) priors**: "let the data speak." But a flat prior on $\theta$ isn't flat on $\theta^2$ or $\ln\theta$. The parameterization matters.

In practice, the prior matters most when data are scarce. With abundant data, the likelihood dominates. If conclusions depend sensitively on the prior, that's telling you the data are insufficient -- which is itself useful to know.

[[simulation shrinkage-plot]]

## The Disease Test: Bayes in Action

A disease affects 1 in 1000 people. A test has 99% sensitivity (catches 99% of true cases) and 95% specificity (5% false positive rate). You test positive. How worried should you be?

Your gut says "very worried." Bayes says: not so fast. With 1000 people, about 1 has the disease (and tests positive) while 50 of the healthy 999 also test positive (the 5% false positive rate). So roughly 1 out of 51 positive tests is a true case -- about 2%. The prior (1/1000) matters enormously. Rare diseases stay unlikely even after a positive test.

## Bayesian vs Frequentist Confidence

A 95% **confidence interval** (frequentist): if you repeated the experiment many times, 95% of the intervals would contain the true value. Says nothing about *this* interval.

A 95% **credible interval** (Bayesian): given data and prior, 95% probability the parameter is in here. This is often what people *think* a confidence interval means.

For large samples with weak priors, the two are numerically similar. The distinction matters with small samples or strong prior information.

> **Challenge.** Explain Bayes' theorem using the rare disease example. A positive test doesn't mean you have it -- the prior (how rare the disease is) matters enormously. One minute.

## Big Ideas

* The likelihood you've been using since the beginning is Bayesian inference's engine too -- prior and posterior are new, but the data's voice is the same.
* Credible intervals say the parameter is in this range with 95% probability; confidence intervals say 95% of such intervals contain the truth. People want the Bayesian answer but usually get the frequentist one.
* If conclusions depend on the prior, that's a signal the data are insufficient -- useful information, not a flaw.
* With enough data, reasonable priors wash out. The philosophical difference matters mainly when data are scarce.

## What Comes Next

Bayesian thinking reframes inference as belief updating. The next section shows how Bayesian priors appear in disguise inside frequentist fitting -- constraint terms on nuisance parameters are Gaussian priors, and the two philosophies converge in practice.

## Check Your Understanding

1. Disease: 1 in 1000. Test: 99% sensitivity, 95% specificity. You test positive. Walk through Bayes step by step. Why is the answer far less than 99%?
2. Strong informative prior, posterior barely moved. "The prior was too strong." What's the equally valid alternative interpretation?
3. Frequentist CI and Bayesian credible interval are numerically identical. Do they mean the same thing?

## Challenge

Coin bias $\theta$, uniform prior. After 10 flips with 3 heads, compute the posterior (a Beta distribution). A colleague starts with Beta(10, 10) centered at 0.5. Compare the two posteriors: peaks, widths, and how much the prior matters. How many flips until the two posteriors become nearly indistinguishable?
