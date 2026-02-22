# Probability Density Functions

You've summarized data you already have. Now the deeper question: what *process* generated it? Describe the mechanism with a mathematical function, and you unlock prediction, extrapolation, and honest uncertainty. That function is the **probability density function**.

## What is a PDF?

A PDF is a density, not a probability -- the same way mass density isn't mass. You need to integrate over a region to get an actual probability:

$$
P(a \leq X \leq b) = \int_{a}^{b} f(x) \, dx
$$

The **cumulative distribution function** (CDF) accumulates probability from $-\infty$:

$$
F_X(x) = \int_{-\infty}^{x} f(x') \, dx'.
$$

It answers a simple question: what fraction of outcomes fall below $x$?

## Common Distributions

Different physical processes give rise to different distributions. Learn the situation, and the distribution follows.

### Binomial

Flip a coin $N$ times, each with probability $p$ of heads. How many heads? Imagine you're actually doing this -- you flip ten times and get seven heads. Is $p = 0.5$? Probably not. Is $p = 0.7$? More like it. We'll formalize that intuition soon.

The binomial distribution counts the outcomes:

$$
\begin{aligned}
f(n;N,p) &= \frac{N!}{n!(N-n)!}p^n(1-p)^{N-n}\\
\langle f(n;N,p)\rangle &= Np \\
\sigma^2 &= Np(1-p)
\end{aligned}
$$

Any time you have a fixed number of independent yes/no trials with the same probability, the binomial is your distribution.

### Poisson

Now imagine a huge number of trials ($N\rightarrow \infty$), each with a tiny success probability ($p\rightarrow 0$), but the expected count stays finite ($Np\rightarrow\lambda$). The binomial simplifies to:

$$
f(n, \lambda) = \frac{\lambda^n}{n!}e^{-\lambda}
$$

**Example**: Millions of people drive every day ($N\rightarrow\infty$), each with a tiny crash probability ($p\rightarrow 0$), but some number of crashes happen every year ($\lambda\neq 0$).

[[simulation poisson-to-gaussian]]

And here's the gorgeous part: the Poisson's mean and variance are both $\lambda$. So the error on a count is just $\sqrt{N}$. A histogram bin with 100 events has uncertainty $\pm 10$. You'll use this constantly.

The sum of independent Poissons is itself Poisson: $\lambda = \lambda_a + \lambda_b$. And when $\lambda$ gets large ($\gtrsim 20$), it approaches a Gaussian. Nature keeps converging on that bell curve.

### Gaussian

The star of the show:

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left[-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right]
$$

Why does it appear everywhere? The central limit theorem (next section) gives the answer: average many independent contributions, and the result tends toward a Gaussian no matter what the individual contributions look like. Measurement errors -- sums of many small disturbances -- are almost always Gaussian. Nature's favorite trick.

[[simulation applied-stats-sim-2]]

### Student's t-Distribution

The Gaussian works beautifully with plenty of data. But with small samples, the estimated standard deviation is itself uncertain, and the Gaussian underestimates the tails. The **Student's t-distribution** has heavier tails to account for this. As sample size grows, it converges to the Gaussian. For small-sample work, always prefer the $t$-distribution.

## Maximum Likelihood Estimation

You've met the distributions. Now: given data, how do you figure out *which* distribution and parameters generated it?

Think detective story. The data are clues at the scene. The parameters are the culprit. Your job: which culprit makes these clues least surprising?

### The Detective's Question

You flip a coin 10 times and get 7 heads. If $p = 0.5$, seven heads is possible but not the most likely result. If $p = 0.7$, it is. If $p = 0.99$, getting *only* seven is surprising -- you'd expect more. The answer is around 0.7. That's what the detective does: find the $p$ that makes the evidence least surprising.

Now here's where it gets fun. Flip 10 more times. Get 6 heads this time. Now your total is 13 heads out of 20, so $\hat{p} = 0.65$. Each batch of data nudges your estimate. With 100 flips, the estimate tightens. With 1000, it's razor-sharp. The more evidence the detective gathers, the more confident the verdict.

> **Challenge.** Grab a coin and flip it 10 times. How many heads? What $p$ makes your result least surprising? If you got 6, the MLE is $\hat{p} = 0.6$. Simple -- but that simplicity hides a powerful principle.

[[simulation likelihood-surface]]

Mathematically, the **likelihood** is the probability of the data as a function of the parameters:

$$
\mathcal{L}(\theta) = \prod_i f(x_i; \theta)
$$

In practice you maximize the **log-likelihood** (products of small numbers cause numerical grief):

$$
\ln\mathcal{L}(\theta) = \sum_i \ln f(x_i; \theta)
$$

For Gaussian data, minimizing $-2\ln\mathcal{L}$ is equivalent to minimizing $\chi^2$ -- a connection we'll exploit heavily in [chi-square fitting](./chi-square-method).

The likelihood is the bridge between frequentist and Bayesian statistics. In the [Bayesian framework](./bayesian-statistics), you multiply it by a *prior* to get a *posterior*. Same engine, different questions.

### Why MLE Works Well

Three desirable properties. **Consistent**: more data means the estimate homes in on the truth, like an archer improving with practice. **Asymptotically normal**: for large samples, the estimator becomes Gaussian, making confidence intervals straightforward. **Efficient**: among all consistent estimators, MLE achieves the smallest possible variance -- the Cramer-Rao bound. No method can do better in the large-sample limit.

### The Fitting Procedure

1. **Choose a model**: which PDF describes the data?
2. **Write the likelihood** for your observations.
3. **Optimize**: find the $\theta$ that maximizes $\mathcal{L}$ (typically numerically).
4. **Assess**: does the model actually describe what you see?

Now you have distributions and a method for fitting them. But why do Gaussian errors keep appearing everywhere? That's the mystery the central limit theorem solves next.

> **Challenge.** Explain MLE to a friend using only the coin example. No equations -- just "which $p$ makes my data least surprising?" One minute.

## Big Ideas

* A PDF is a density, not a probability -- integrate over a region to get a probability, just like integrating mass density to get mass.
* The Poisson's mean and variance are the same number, so uncertainty on any count is just its square root.
* MLE asks "which parameters make my data least surprising?" -- the same question a good detective asks about a suspect.
* No consistent estimator beats MLE in the large-sample limit (Cramer-Rao bound).

## What Comes Next

You have distributions and a fitting method. But why does the Gaussian show up *everywhere*, even when the underlying process has nothing to do with a bell curve? The central limit theorem answers this -- and once you have it, you can track how uncertainties in raw measurements flow through to derived quantities. That's error propagation, and together with the CLT it's the foundation for almost everything ahead.

## Check Your Understanding

1. You count 25 photons in one second. What's your best estimate of the true rate, and what's the uncertainty? (Hint: Poisson.)
2. A friend fits a Gaussian to 8 data points and calls it definitive. What distribution should they use instead, and why do the tails matter?
3. You flip a coin 20 times, get 14 heads. Roughly what $p$ maximizes the likelihood? How would 14 out of 200 change things?

## Challenge

You collect 50 measurements and see two peaks in the histogram -- neither Gaussian nor Poisson. Describe a strategy for fitting with MLE. What model would you try first? How would you choose between a single component and a two-component mixture? What would the residuals tell you?
