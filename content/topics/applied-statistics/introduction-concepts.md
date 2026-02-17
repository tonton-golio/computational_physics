# Introduction and General Concepts

## Measures of Central Tendency

### Mean

The mean tells us about the bulk magnitude-tendency of data.

### Geometric Mean

The $n$-th root of the product:

$$
\bar{x}_\text{geo} = \left( \prod_i^n x_i\right)^{1/n} = \exp\left(\frac{1}{n}\sum_{i}^n\ln x_i \right)
$$

This is equivalent to the arithmetic mean in log-scale.

```python
def geometric(arr):
    return np.prod(arr)**(1/len(arr))

def geometric(arr):
    # A sum of logs is less prone to
    # under-/over-flow than a product.
    return np.exp(np.mean(np.log(arr)))
```

### Arithmetic Mean

The equal-weight center:

$$
\hat{\mu} = \bar{x} = \langle x \rangle = \frac{1}{N}\sum_i^N x_i
$$

```python
def arithmetic(arr):
    return np.sum(arr) / len(arr)
    # or equivalently: np.mean(arr)
```

### Median

The counting center of the data (the middle value when sorted).

```python
def median(arr):
    n = len(arr)
    return arr[n//2] if n % 2 == 0 else arr[n//2]
```

### Mode

The most typical data point (the value that occurs most frequently).

### Harmonic Mean

$$
\bar{x}_\text{harm} = \left[\frac{1}{n}\sum_i^n \frac{1}{x_i}\right]^{-1}
$$

```python
def harmonic(x):
    return (np.sum(x**(-1)) / len(x))**(-1)
```

### Truncated Mean

The arithmetic mean of the data with its tails cut off:

```python
def truncated(arr, k):
    arr_sorted = np.sort(arr)
    return np.mean(arr_sorted[k:-k])
```

## Standard Deviation

Standard deviation measures how much data points deviate from the dataset mean:

$$
\hat{\sigma} = \sqrt{\frac{1}{N}\sum_i^N (x_i - \mu)^2}
$$

In estimating this, we assume we know the true mean. In practice we use the sample mean $\bar{x}$, which requires **Bessel's correction** (dividing by $N-1$ instead of $N$ to account for the lost degree of freedom):

$$
\tilde{\sigma} = \sqrt{\frac{1}{N-1}\sum_i(x_i-\bar{x})^2}
$$

## Normal Distribution

The normal (Gaussian) distribution is fundamental in statistics.

[[simulation applied-stats-sim-2]]

## Weighted Mean

How to average data with different uncertainties and what is the uncertainty on the average?

$$
\hat{\mu} = \frac{\sum x_i / \sigma_i^2}{\sum 1 / \sigma_i^2}, \qquad \hat{\sigma}_\mu = \sqrt{\frac{1}{\sum 1/\sigma_i^2}}
$$

The uncertainty decreases with the square root of the number of samples:

$$
\hat{\sigma}_\mu = \hat{\sigma}/\sqrt{N}.
$$

## Correlation

Correlation measures whether a feature varies in concordance with another. From the variance we obtain the **covariance**:

$$
\begin{aligned}
V = \sigma^2 &= \frac{1}{N}\sum_i^N(x_i-\mu)^2 = E[(x-\mu)^2] = E[x^2] - \mu^2 \\
V_{xy} &= E[(x_i-\mu_x)(y_i-\mu_y)] =
\begin{bmatrix}
\sigma_{11}^2 & \sigma_{12}^2 & \ldots\\
\sigma_{21}^2 & \ldots & \ldots
\end{bmatrix}
\end{aligned}
$$

Normalizing by the widths gives **Pearson's (linear) correlation coefficient**:

$$
\rho_{xy} = \frac{V_{xy}}{\sigma_x\sigma_y}, \qquad -1 \leq \rho_{xy} \leq 1
$$

Note that $\rho_{xy} = 0$ only indicates the absence of *linear* correlation. Always plot the data to check for non-linear relationships.

### Rank Correlation

**Rank correlation** tests for non-linear monotonic relationships. It compares the ranking between two sets. **Spearman's $\rho$** and **Kendall's $\tau$** are the most common:

$$
\rho_S = 1 - \frac{6\sum_i (r_i - s_i)^2}{n^3-n}
$$

Kendall's $\tau$ compares the number of concordant pairs to discordant pairs.

### Other Non-Linear Correlation Measures

- Maximal Information Coefficient (MIC)
- Mutual Information (MI)
- Distance Correlation (DC)

## Linear Regression

Linear regression models the relationship between a dependent variable and one or more independent variables. The simple linear regression model is:

$$
y = \beta_0 + \beta_1 x + \epsilon
$$

where $\beta_0$ is the intercept, $\beta_1$ is the slope, and $\epsilon$ is the error term.

[[simulation applied-stats-sim-1]]

