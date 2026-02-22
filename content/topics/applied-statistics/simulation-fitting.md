# Simulation Methods

What if you could run your experiment a million times? Not in the real world -- that'd take forever and cost a fortune -- but inside a computer, where electrons are cheap and time runs fast.

The CLT and error propagation rely on analytical formulas. When those assumptions break down, you let the computer do the experiment for you. Generate millions of random samples, push them through your calculation, and look at what happens. That's **Monte Carlo simulation** -- named after the casino, because it runs on random numbers.

## Producing Random Numbers

Every Monte Carlo method starts with random numbers from a specific distribution. Computers generate **uniform** random numbers natively, so the challenge is converting those into whatever distribution you need.

## Transformation Method

If you can invert the CDF, you can transform uniform random numbers into any distribution:

1. Verify the PDF is normalized.
2. Compute the CDF.
3. Invert it.

$$
F(x) = \int_{-\infty}^x f(x') \, dx'
$$

Draw a uniform $p \in [0,1]$, compute $x = F^{-1}(p)$. Done. The CDF maps probabilities to values, its inverse maps them back.

### Example: Exponential Distribution

The exponential models waiting times between Poisson events:

$$
f(x) = \lambda \exp(-\lambda x), \quad x \in [0, \infty)
$$

CDF: $F(x) = 1 - \exp(-\lambda x)$. Invert: $F^{-1}(p) = -\ln(1-p)/\lambda$. Draw $p$ uniformly, compute $x$. You've just turned uniform noise into structured randomness.

## Accept-Reject Method

What if you can't invert the CDF? The **accept-reject method** works for *any* distribution you can evaluate, even if you can't integrate it.

Here's the dartboard picture. You want to sample from some complicated target curve. Draw a big rectangle (your "dartboard") that completely covers the curve. Throw darts at the dartboard uniformly at random. Keep only the darts that land *under* the curve. The kept darts follow the target distribution exactly. Darts above the curve? Toss them.

More formally: given target $f(x)$ and proposal $g(x)$ with $f(x) \leq M \cdot g(x)$:

1. Sample $x$ from $g(x)$.
2. Sample $u$ uniformly from $[0, 1]$.
3. Accept $x$ if $u \leq f(x) / (M \cdot g(x))$; otherwise reject.

The tighter your dartboard wraps the target curve, the fewer darts you waste. A loose envelope means throwing a lot of darts to keep only a few -- like casting a wide net for one kind of fish. The art is finding a proposal $g$ that hugs $f$ closely.

[[simulation accept-reject]]

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

[[simulation monte-carlo-integration-stats]]

## Why Monte Carlo Scales Well

Here's the genuinely surprising part. Monte Carlo uncertainty decreases as $1/\sqrt{N}$, regardless of how many dimensions your problem has. Deterministic integration (like the trapezoidal rule) converges as $1/N^{2/d}$ -- in one dimension it wins easily, but as $d$ grows, it collapses. Monte Carlo stays at $1/\sqrt{N}$ no matter what.

This is why Monte Carlo dominates in high-dimensional problems -- many-parameter fits, complex error propagation, physical simulations with many degrees of freedom.

Now that you can fake the universe a million times, watch what happens when you ask: "how well does my model fit the data?" That's chi-square fitting, next.

> **Challenge.** Explain accept-reject to a friend using only the dartboard analogy. The dartboard is the proposal, you only keep darts under the target curve. One minute.

## Big Ideas

* Monte Carlo is just running a virtual experiment millions of times -- turning "what's the uncertainty?" into "how much do the outputs spread when I wiggle the inputs?"
* The transformation method is elegant: invert the CDF, and uniform noise becomes any distribution you want.
* Monte Carlo's $1/\sqrt{N}$ convergence is dimension-independent -- this is why it wins in high-dimensional problems.
* Accept-reject works for *any* evaluable distribution. Tighter proposals waste fewer samples.

## What Comes Next

You can now generate random samples from arbitrary distributions and run virtual experiments. The next step: given real data and a model, find the parameters that best describe what you see. That's chi-square fitting -- the workhorse of model fitting in the physical sciences, and it connects directly to the MLE principle from the PDFs section.

## Check Your Understanding

1. You want samples from $f(x) = 3x^2$ on $[0, 1]$. Walk through the transformation method: CDF, inverse, and how a uniform number produces a sample.
2. You use a uniform proposal for a highly peaked distribution. What fraction of samples gets accepted, and why is this a problem?
3. "Monte Carlo is always slower than analytical error propagation." When is this false?

## Challenge

You have $z = a^2 \ln(b/c)$ with Gaussian uncertainties on $a$, $b$, $c$. Describe the Monte Carlo algorithm in pseudocode to estimate the distribution of $z$. Compare the mean and standard deviation to analytical error propagation. When do you expect the two to disagree significantly?
