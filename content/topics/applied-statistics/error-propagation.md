# Central Limit Theorem and Error Propagation

## Central Limit Theorem

The central limit theorem answers the question: *why do statistics in the limit of large $N$ tend toward a Gaussian?*

The distribution of the mean of $N$ independent samples from any distribution (with finite variance) approaches a Gaussian as $N$ increases. This holds regardless of the original distribution's shape.

[[simulation applied-stats-sim-3]]

For the CLT to work well, the standard deviations of the contributing distributions must be similar. Distributions with heavy tails (like the Cauchy distribution) may require truncation since they lack finite variance.

**The central limit theorem ensures that uncertainties are Gaussian distributed** (under suitable conditions), which is why Gaussian error analysis is so widely applicable.

## Error Propagation

If we have a function $y(x_i)$ and we know the uncertainty $\sigma(x_i)$, how do we find $\sigma(y)$? It depends on the gradient of $y$ with respect to $x$:

$$
\sigma(y) = \frac{\partial y}{\partial x}\sigma(x_i)
$$

If $y$ is smooth around $x_i$ this works well. The slope should be relatively constant over the uncertainty range in $x_i$.

For multiple variables with correlations, the general formula is:

$$
\sigma_y^2 = \sum_{i,j}^n \frac{\partial y}{\partial x_i} \frac{\partial y}{\partial x_j} V_{ij}
$$

If there are no correlations, only the diagonal terms (individual errors) contribute. This lets us identify which parameters dominate the total uncertainty.

### Addition

$$
y = x_1 + x_2 \implies \sigma_y^2 = \sigma_{x_1}^2 + \sigma_{x_2}^2 + 2V_{x_1, x_2}
$$

### Multiplication

$$
y = x_1 x_2 \implies \sigma_y^2 = (x_2\sigma_{x_1})^2 + (x_1\sigma_{x_2})^2 + 2x_1 x_2 V_{x_1, x_2}
$$

Dividing by $y^2$ gives the relative uncertainties. By using negative error correlations, we can cancel out errors (e.g., Harrison's gridiron pendulum).

### Simulating Error Propagation

Choose random inputs $x_i$ and record the output $y$. If $y(x)$ is not smooth, this will not yield Gaussian-distributed $y$, even with Gaussian errors in $x_i$.
