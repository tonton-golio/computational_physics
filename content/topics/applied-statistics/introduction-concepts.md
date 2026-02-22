# Introduction and General Concepts

You have a pile of numbers -- pendulum timings, blood pressures, photon counts. Before you build any model, you need two things: *where* does the data cluster, and *how much does it scatter?*

## Measures of Central Tendency

The "center" of a dataset depends on the question you're asking.

### Arithmetic Mean

[[simulation which-average]]

Add everything up, divide by how many you have:

$$
\hat{\mu} = \bar{x} = \frac{1}{N}\sum_i^N x_i
$$

```python
def arithmetic_mean(arr):
    return np.sum(arr) / len(arr)
    # or equivalently: np.mean(arr)
```

The arithmetic mean is sensitive to extremes. One wild reading of 10.5 (a bumped table, maybe) drags the mean away from where most of the data sits.

### Geometric Mean

When your data are multiplicative -- growth rates, ratios, concentrations spanning orders of magnitude -- take the $n$-th root of the product:

$$
\bar{x}_\text{geo} = \left( \prod_i^n x_i\right)^{1/n} = \exp\left(\frac{1}{n}\sum_{i}^n\ln x_i \right)
$$

This is the arithmetic mean in log-space. If your data span several orders of magnitude, the geometric mean lives where the data actually cluster.

```python
def geometric_mean(arr):
    # A sum of logs is less prone to
    # under-/over-flow than a product.
    return np.exp(np.mean(np.log(arr)))
```

### Median

The middle value when sorted. Even wild outliers don't shift it. If you had that bumped-table reading, the median wouldn't flinch.

```python
def median(arr):
    s = np.sort(arr)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2
```

### Mode

The most frequently occurring value. For continuous data, estimate it from a histogram or kernel density estimate.

### Harmonic Mean

$$
\bar{x}_\text{harm} = \left[\frac{1}{n}\sum_i^n \frac{1}{x_i}\right]^{-1}
$$

The harmonic mean is right when you're averaging *rates*. Drive 60 km/h for the first half and 40 km/h for the second half. The harmonic mean (48 km/h) gives the correct average speed, not the arithmetic mean (50 km/h).

```python
def harmonic_mean(x):
    return (np.sum(x**(-1)) / len(x))**(-1)
```

### Truncated Mean

A compromise: compute the mean after tossing the most extreme values on both ends.

```python
def truncated_mean(arr, k):
    arr_sorted = np.sort(arr)
    return np.mean(arr_sorted[k:-k])
```

> **Challenge.** Measure ten of anything -- walking time, temperature, words per sentence. Compute the mean and the median. Are they close? If not, you probably have outliers or skew. That's already telling you something.

## Measures of Spread

Two datasets can have the same mean but look completely different. You need a number that says *how much the data wanders*.

### Standard Deviation

Think of it as the "average distance" from the center:

$$
\hat{\sigma} = \sqrt{\frac{1}{N}\sum_i^N (x_i - \mu)^2}
$$

Here's the catch: if you use the sample mean $\bar{x}$ instead of the true mean $\mu$, the formula systematically underestimates the spread. Why? Because $\bar{x}$ is always a little closer to the data than $\mu$ is -- it was *computed* from the data, so it cheats slightly. **Bessel's correction** fixes this -- think of it as paying for the drinks you actually drank, not the drinks someone else ordered:

$$
\tilde{\sigma} = \sqrt{\frac{1}{N-1}\sum_i(x_i-\bar{x})^2}
$$

### Weighted Mean

When measurements come with different uncertainties, you should trust precise ones more. Imagine a seesaw: a precise measurement (small $\sigma$) is a heavy person near the fulcrum -- it has more say in where the balance point lands.

$$
\hat{\mu} = \frac{\sum x_i / \sigma_i^2}{\sum 1 / \sigma_i^2}, \qquad \hat{\sigma}_\mu = \sqrt{\frac{1}{\sum 1/\sigma_i^2}}
$$

[[simulation weighted-mean]]

Three measurements of a length: $10.2 \pm 0.1$, $10.5 \pm 0.5$, $10.1 \pm 0.2$ cm. The weighted mean pulls toward 10.2 because that's the most precise one. This is your first taste of a powerful idea: *let the data tell you how much to trust each piece*. You'll see it again in [likelihood](./probability-density-functions) and [chi-square fitting](./chi-square-method).

## From Individual Variables to Relationships

So far you've described single variables. But the juicy questions are about *relationships*. Does temperature drive reaction rate? That's where **correlation** enters.

### Covariance and Pearson's Correlation

Covariance measures whether $x$ and $y$ move together:

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

Normalizing by the widths gives **Pearson's correlation**, which lives between $-1$ and $+1$:

$$
\rho_{xy} = \frac{V_{xy}}{\sigma_x\sigma_y}, \qquad -1 \leq \rho_{xy} \leq 1
$$

Here's the catch: $\rho_{xy} = 0$ does **not** mean the variables are unrelated. Picture a perfect parabola $y = x^2$. Complete dependence -- but Pearson is zero, because the relationship isn't linear. A perfect circle? Same thing. Always plot the data.

### Beyond Pearson

Pearson only sees linear relationships. **Spearman's $\rho$** replaces values with ranks and catches any monotonic trend -- like study hours vs. exam scores, where the first few hours help enormously but later ones add less. For truly complex, non-monotonic dependencies, information-theoretic measures (Mutual Information, Distance Correlation) detect relationships of any shape, at the cost of harder interpretation.

> **Challenge.** Explain to a 12-year-old why the average can be misleading when there's an outlier. One minute.

## Big Ideas

* The "right" average depends on structure: arithmetic for additive quantities, geometric for multiplicative, harmonic for rates.
* Bessel's correction ($N-1$) exists because the sample mean cheats -- it's always a bit closer to the data than the true mean.
* Correlation zero does not mean independence. A perfect circle has $\rho = 0$. Always plot first.
* The weighted mean lets uncertainty tell you how much to trust each reading.

## What Comes Next

Everything here describes data you already have. But what *process* generated it? If you can describe the mechanism with a mathematical function, you unlock prediction, extrapolation, and principled uncertainty. That function is the probability density function -- and the weighted mean you just computed will reappear as a special case of maximum likelihood estimation.

## Check Your Understanding

1. Salary data where the CEO earns 200x the median worker. Which central tendency gives a more honest picture of a typical employee, and why?
2. Two datasets have identical means and standard deviations but very different shapes. What does this tell you about what those summaries capture and miss?
3. Two variables have $\rho = 0$. A classmate says they're independent. What example kills this claim?

## Challenge

Design a scheme for estimating average commute time in a city. Some people commute three hours each way. Write down: which central tendency you'd report and why, how you'd quantify spread, and what a "weighted" version looks like if some respondents kept more careful logs. Then consider: if a journalist reports the arithmetic mean and a union reports the median, and the numbers differ wildly, who is lying?
