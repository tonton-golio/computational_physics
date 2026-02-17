# Simulation and More Fitting

## Producing Random Numbers

For producing random numbers from arbitrary distributions, there are two main approaches: the **transformation method** and the **accept-reject method**.

## Transformation Method

The transformation method proceeds in three steps:

1. Verify the PDF is normalized.
2. Compute the cumulative distribution function (CDF).
3. Invert the CDF.

$$
F(x) = \int_{-\infty}^x f(x') \, dx'
$$

### Example: Exponential Distribution

Consider the exponential distribution:

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

To sample, draw $p$ uniformly from $[0,1]$ and compute $x = F^{-1}(p)$.

## Accept-Reject Method

The **accept-reject method** (also called the von Neumann method) generates random samples from a proposal distribution and accepts or rejects them according to acceptance criteria.

The idea: given a target PDF $f(x)$ and a proposal distribution $g(x)$ with $f(x) \leq M \cdot g(x)$ for some constant $M$:

1. Sample $x$ from $g(x)$.
2. Sample $u$ uniformly from $[0, 1]$.
3. Accept $x$ if $u \leq f(x) / (M \cdot g(x))$; otherwise reject and repeat.

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

## Comparison of Methods

**Monte Carlo integration** converges as $1/\sqrt{N}$ regardless of dimensionality, while deterministic numerical methods (e.g., trapezoidal rule) converge as $1/N^{2/d}$, where $d$ is the dimensionality. This means Monte Carlo becomes increasingly competitive for high-dimensional problems.

[[simulation applied-stats-sim-3]]
