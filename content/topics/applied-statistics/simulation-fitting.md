# Simulation Methods

What if you could run your experiment a million times? Not in the real world — that would take forever and cost a fortune — but inside a computer, where electrons are cheap and time runs fast.

The central limit theorem told you that averages converge to a Gaussian. Error propagation showed you how uncertainties flow through calculations. But both rely on analytical formulas that assume smoothness, linearity, or known distributions. When those assumptions break down — and in real life, they often do — you let the computer do the experiment for you.

Instead of deriving a formula, you generate millions of random samples, push them through your calculation, and look at the result. This is **Monte Carlo simulation** — named after the famous casino, because it runs on random numbers.

## Producing Random Numbers

Every Monte Carlo method starts with random numbers drawn from a specific distribution. Computers generate **uniform** random numbers natively, so the challenge is converting those into samples from whatever distribution you need. Two methods cover almost every case.

## Transformation Method

The transformation method is elegant and efficient when it works. The idea: if you can invert the CDF (the cumulative distribution function from lesson 2), you can transform uniform random numbers into any distribution you want.

The recipe is short:

1. Verify the PDF is normalized.
2. Compute the CDF.
3. Invert it.

$$
F(x) = \int_{-\infty}^x f(x') \, dx'
$$

Draw a uniform random number $p \in [0,1]$ and compute $x = F^{-1}(p)$. The resulting $x$ follows the target distribution. That's it. The CDF maps probabilities to values, and its inverse maps values back.

### Example: Exponential Distribution

Consider the exponential distribution, which models waiting times between Poisson events:

$$
f(x) = \lambda \exp(-\lambda x), \quad x \in [0, \infty)
$$

This is normalized. The CDF is:

$$
F(x) = 1 - \exp(-\lambda x)
$$

Inverting gives:

$$
F^{-1}(p) = -\frac{\ln(1-p)}{\lambda}
$$

To sample: draw $p$ uniformly from $[0,1]$ and compute $x = F^{-1}(p)$. Each $x$ is a random draw from the exponential distribution. You've just turned uniform noise into structured randomness.

## Accept-Reject Method

The transformation method requires an invertible CDF, which is not always available. What if your distribution is some complicated function you can evaluate but can't integrate analytically?

The **accept-reject method** (also called the von Neumann method) is more general: it works for any distribution you can evaluate, even if you can't integrate or invert it.

The idea: draw samples from a simple proposal distribution and keep only those that "pass" a random acceptance test. Given a target PDF $f(x)$ and a proposal distribution $g(x)$ with $f(x) \leq M \cdot g(x)$ for some constant $M$:

1. Sample $x$ from $g(x)$.
2. Sample $u$ uniformly from $[0, 1]$.
3. Accept $x$ if $u \leq f(x) / (M \cdot g(x))$; otherwise reject and repeat.

The accepted samples follow the target distribution exactly. The efficiency depends on how tightly the proposal $g$ envelops the target $f$ — a loose envelope wastes many samples, like casting a wide net when you only want one kind of fish.

```python
def accept_reject_sample(target_pdf, proposal_sample, proposal_pdf, M, num_samples):
    samples = []
    while len(samples) < num_samples:
        x = proposal_sample()
        u = random.uniform(0, 1)
        if u <= target_pdf(x) / (M * proposal_pdf(x)):
            samples.append(x)
    return samples
```

## Why Monte Carlo Scales Well

Here is the key advantage of Monte Carlo, and it's genuinely surprising. The uncertainty on a Monte Carlo estimate decreases as $1/\sqrt{N}$, where $N$ is the number of samples. This convergence rate holds *regardless of the dimensionality* of the problem.

By contrast, deterministic numerical integration (like the trapezoidal rule) converges as $1/N^{2/d}$, where $d$ is the number of dimensions. In one dimension, deterministic methods win easily. But as $d$ grows, their convergence collapses while Monte Carlo stays at $1/\sqrt{N}$.

This is why Monte Carlo is the tool of choice for high-dimensional problems — integrating over many parameters, propagating errors through complex models, or simulating physical processes with many degrees of freedom. It connects directly to the CLT: every Monte Carlo estimate is an average, and the CLT guarantees that average will be approximately Gaussian with a calculable uncertainty.

Now that you can fake the universe a million times in a computer, watch what happens when you ask: "I have a model and some data — how well does the model fit?" That's chi-square fitting, and it's where we go next.

> **Challenge.** Explain the accept-reject method to a friend using only the analogy of throwing darts at a dartboard. One minute. (Hint: the dartboard is the proposal distribution, and you only keep darts that land under the target curve.)
