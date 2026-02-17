# Probability Density Functions

## Probability Density Functions (PDFs)

A **probability density function** is a function of a continuous random variable whose integral across an interval gives the probability that the value of the variable lies within that interval:

$$
f_X(x) = \frac{1}{\Delta x}\int_{x_0}^{x_0+\Delta x} f(x) \, dx
$$

When fitting with PDFs, we should consider the error stemming from the bin-widths.

We may also consider the **cumulative distribution function**: the integral from $-\infty$,

$$
F_X(x) = \int_{-\infty}^{x} f(x') \, dx'.
$$

## Distributions

### Binomial

$N$ trials, $p$ chance of success, how many successes should you expect?

$$
\begin{aligned}
f(n;N,p) &= \frac{N!}{n!(N-n)!}p^n(1-p)^{N-n}\\
\langle f(n;N,p)\rangle &= Np \\
\sigma^2 &= Np(1-p)
\end{aligned}
$$

Note that this is a **discrete** distribution.

[[simulation applied-stats-sim-1]]

### Poisson

If $N\rightarrow \infty$ and $p\rightarrow 0$, but $Np\rightarrow\lambda$ (some finite number), then the binomial approaches a **Poisson** distribution:

$$
f(n, \lambda) = \frac{\lambda^n}{n!}e^{-\lambda}
$$

**Example**: A large number of people go into traffic every day ($N\rightarrow\infty$), the probability of being killed in traffic is tiny ($p\rightarrow 0$), but some people do get killed in traffic every year ($\lambda\neq 0$).

The Poisson distribution has mean and variance both equal to $\lambda$. The error on a Poisson count is the square root of that count.

**A useful case**: The error to assign a bin in a histogram if there are reasonable statistics ($N \approx 5{-}20$) in each bin. If there are low statistics in a bin, we cannot make the Gaussian approximation.

The sum of independent Poissons is a Poisson: $\lambda = \lambda_a + \lambda_b$.

If $\lambda \rightarrow \infty$, the Poisson approaches a Gaussian (practically, $\lambda \gtrsim 20$).

### Gaussian

The **normal distribution**:

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left[-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right]
$$

Given a large number of samples, we usually observe this distribution (by the central limit theorem), and it is convenient to work with analytically.

### Student's t-Distribution

Useful for **low statistics**, because it accounts for the uncertainty in the estimated $\mu$ and $\sigma$. When the number of samples $\rightarrow\infty$, the $t$-distribution converges to the Gaussian. Its heavier tails make it more robust for small sample sizes.

## Maximum Likelihood Estimation

We seek the parameters $\theta$ that maximize the **likelihood** $\mathcal{L}$ given the observed data:

$$
\mathcal{L}(\theta) = \prod_i f(x_i; \theta)
$$

We are prone to rounding errors when multiplying many small numbers together, so we instead maximize the **log-likelihood**. Assuming Gaussian errors:

$$
-2\ln(\mathcal{L}) = \chi^2 + \text{const.}
$$

Properties of maximum likelihood estimators:
- **Consistent**: Converges to the true value as $N \to \infty$.
- **Asymptotically normal**: Errors become Gaussian for large $N$.
- **Efficient**: Reaches the minimum variance bound (Cramer-Rao bound) for large $N$.

The fitting procedure:
1. Choose a model and compute the likelihood for the data.
2. Optimize parameters to maximize $\mathcal{L}$ (or equivalently, minimize $-\ln\mathcal{L}$).
3. Test the fit by comparing with the data distribution.

Note: There are different methods depending on whether we use individual data points (unbinned) or histogram bins and counts (binned). Consider the binned approach for very large samples since the unbinned likelihood calculation can be expensive.

## The Likelihood Ratio Test

If we have two different hypotheses, the test statistic is

$$
d = -2\ln\frac{\mathcal{L}_\text{null}}{\mathcal{L}_\text{alt}} = -2\ln(\mathcal{L}_\text{null}) + 2\ln(\mathcal{L}_\text{alt})
$$

Under the null hypothesis, $d$ follows a $\chi^2$ distribution with degrees of freedom equal to the difference in the number of parameters between the two models. Greater degrees of freedom will necessarily yield a higher likelihood, so the test accounts for model complexity.

[[simulation applied-stats-sim-2]]
