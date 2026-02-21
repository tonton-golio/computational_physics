# Bayesian Statistics

## A Different Way of Thinking

Throughout this course, you've used **frequentist** methods: p-values, confidence intervals, maximum likelihood. These treat probability as a long-run frequency — if you repeat the experiment many times, how often does this outcome occur?

**Bayesian statistics** takes a fundamentally different view. Probability represents a *degree of belief*. You start with some prior belief about a parameter, observe data, and update that belief. The result is a **posterior distribution** — a complete description of what you know about the parameter after seeing the data.

Neither approach is "correct" — they answer different questions. Frequentist methods ask "how surprising is this data if the hypothesis is true?" Bayesian methods ask "given the data, what should I believe about the parameter?" Both are useful, and the best practitioners are fluent in both.

You might think this is a completely new idea. But actually, you've been doing something very close to Bayesian reasoning all along. The likelihood function from [probability density functions](./probability-density-functions)? That's the engine of Bayesian inference too. The nuisance parameter constraints from [advanced fitting](./advanced-fitting-calibration)? Those are priors in disguise. The regularization penalties in [machine learning](./machine-learning-data-analysis)? Bayesian priors again. This section makes the connection explicit.

## Bayes' Theorem

The mathematical engine of Bayesian statistics is **Bayes' theorem**. For a hypothesis $H$ and observed data $D$:

$$
P(H|D) = \frac{P(D|H) \cdot P(H)}{P(D)}
$$

Each term plays a specific role:

* $P(H|D)$ is the **posterior**: your updated belief about $H$ after seeing the data. This is what you want.
* $P(D|H)$ is the **likelihood**: the probability of the data given the hypothesis. This is the same likelihood function you've been using since [probability density functions](./probability-density-functions) — the connection between Bayesian and frequentist methods runs deep.
* $P(H)$ is the **prior**: your belief about $H$ before seeing the data. This is where existing knowledge (or honest ignorance) enters.
* $P(D)$ is the **evidence** (or marginal likelihood): a normalization constant ensuring the posterior integrates to 1. Often the hardest part to compute, but you can frequently sidestep it.

The formula reads as a recipe: **posterior $\propto$ likelihood $\times$ prior**. The data update your belief, with the likelihood acting as the bridge.

## Example: Is This Coin Fair?

Suppose someone hands you a coin. You want to know whether it's fair. Let $\theta$ be the probability of heads. Let's walk through this step by step — the way you'd actually think about it.

**Prior**: You have no strong reason to think the coin is biased. So you start with a uniform prior: $P(\theta) = 1$ for $\theta \in [0, 1]$. Every value is equally plausible before you see any data.

**Flip 1**: You flip the coin once and get heads. Your likelihood is $P(D|\theta) = \theta$. The posterior is $P(\theta|D) \propto \theta \times 1 = \theta$. After one head, values of $\theta$ near 1 look slightly more plausible than values near 0. But the distribution is broad — one flip doesn't tell you much.

**Flips 1-10**: You flip 9 more times and get a total of 7 heads and 3 tails. The likelihood from the binomial distribution is:

$$
P(D|\theta) = \binom{10}{7} \theta^7 (1-\theta)^3
$$

The posterior becomes:

$$
P(\theta|D) \propto \theta^7 (1-\theta)^3
$$

This is a Beta(8, 4) distribution. Its peak is at $\theta = 7/10 = 0.7$ — the same as the maximum likelihood estimate. But the Bayesian answer gives you *more*: the full shape of the distribution tells you how uncertain you are. The 95% credible interval runs roughly from 0.40 to 0.93. That's a wide range — 10 flips haven't pinned down $\theta$ very tightly.

**Flips 1-110**: You flip the coin 100 more times and get 70 heads total (out of 110). The posterior tightens dramatically — more data means more certainty. Now the 95% credible interval might run from 0.55 to 0.73. The prior matters less and less as data accumulate.

This is a reassuring property: with enough data, reasonable people with different priors converge to the same conclusion. The data overwhelm the prior.

> **Challenge.** Grab a coin. Before you flip it, write down your prior belief about $P(\text{heads})$. Now flip it 10 times. After each flip, mentally update your belief: is the coin fair, or is it biased? Notice how your confidence grows (or shifts) with each flip. That's Bayesian updating in real time — your brain does this naturally.

## Choosing Priors

The prior is the most controversial part of Bayesian statistics. Where does it come from? Here's the menu:

* **Informative priors** encode genuine prior knowledge. If previous experiments have measured a quantity, use that result as your prior. This is one of the great strengths of Bayesian methods — they provide a principled way to combine old and new evidence.
* **Weakly informative priors** gently constrain parameters to physically reasonable ranges without strongly favoring specific values. For example, a half-normal prior on a variance parameter ensures it stays positive without committing to a particular magnitude.
* **Non-informative (flat) priors** attempt to "let the data speak." A uniform prior on $\theta$ says every value is equally plausible. This sounds objective, but it's not quite: a flat prior on $\theta$ is *not* flat on $\theta^2$ or $\ln\theta$. The choice of parameterization matters.

In practice, the prior is most important when data are scarce. With abundant data, the likelihood dominates and the prior washes out. If your conclusions depend sensitively on the prior, that's a signal that the data are insufficient to answer the question — which is itself a useful thing to know.

## Bayesian vs Frequentist Confidence

A 95% **confidence interval** (frequentist) means: if you repeated the experiment many times, 95% of the intervals would contain the true value. It says nothing about *this particular* interval.

A 95% **credible interval** (Bayesian) means: given the data and the prior, there is a 95% probability that the parameter lies in this interval. This is often what people *think* a confidence interval means.

For large samples with weak priors, the two intervals are numerically similar. The philosophical difference becomes practically important for small samples or when strong prior information is available.

> **Challenge.** Explain Bayes' theorem to a friend using only the example of diagnosing a rare disease. A positive test result doesn't mean you have the disease — it depends on how rare the disease is (the prior). One minute.

## Big Ideas

* The likelihood function you have been using since the beginning is the engine of Bayesian inference too — the prior and posterior are new, but the data's voice is the same function in both frameworks.
* A 95% credible interval says the parameter is in this range with 95% probability (given the data and prior); a 95% confidence interval says 95% of intervals constructed this way will contain the true value. People want the Bayesian answer but are usually given the frequentist one.
* If your conclusions depend strongly on the prior, that is not a flaw of Bayesian statistics — it is a signal that the data are insufficient to answer the question, which is itself important to know.
* With enough data, reasonable priors wash out and Bayesian and frequentist answers converge. The philosophical difference becomes practically important only in the regime where data are scarce.

## What Comes Next

Bayesian thinking reframes inference as belief updating: prior knowledge plus data equals posterior belief. This framework becomes especially powerful in complex models where you have many parameters, some of interest and some nuisance.

The next section on advanced fitting shows how Bayesian priors appear in disguise inside frequentist fitting — constraint terms on nuisance parameters are Gaussian priors. The two philosophies converge in practice, and understanding both gives you tools that neither alone provides.

## Check Your Understanding

1. A disease affects 1 in 1000 people. A test for it has 99% sensitivity and 95% specificity. If you test positive, what is the probability you actually have the disease? Walk through Bayes' theorem step by step and explain why the answer is far less than 99%.
2. You fit a Bayesian model with a strong informative prior and get a posterior that barely moved from the prior. A colleague concludes "the prior was too strong." What is the equally valid alternative interpretation?
3. A frequentist confidence interval and a Bayesian credible interval for the same parameter are numerically identical in a particular analysis. Does this mean they mean the same thing? Explain the distinction carefully.

## Challenge

You are estimating the bias $\theta$ of a coin. You start with a uniform prior. After 10 flips yielding 3 heads, compute the posterior distribution analytically (it will be a Beta distribution). Now suppose a colleague had started with a strongly informative prior centered at $\theta = 0.5$ (a Beta(10, 10) prior). Compute or sketch their posterior after seeing the same data. Compare the two posteriors: where do they peak, how wide are they, and how much does the prior choice matter? Now consider: how many flips would it take for the two posteriors to be nearly indistinguishable, regardless of the starting prior?
