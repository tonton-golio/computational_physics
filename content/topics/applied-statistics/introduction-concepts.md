# Introduction and General Concepts

You have a pile of numbers. Maybe they're pendulum timings, patient blood pressures, or photon counts from a distant star. Before you build any model or run any test, you need to answer two questions: *where* does the data cluster, and *how much does it scatter?*

These sound simple. They aren't. The "center" of a dataset is not a single concept, and neither is "spread." Different situations demand different summaries. Picking the wrong one can lead you astray before you've even started your analysis.

Alex, our experimentalist, has just measured the gravitational acceleration ten times and gotten values between 9.78 and 9.84 m/s$^2$. What should Alex report as "the" value? And how uncertain should Alex be? That's what this section is about.

## Measures of Central Tendency

The "center" of a dataset depends on what question you're asking. Here is the menu, ordered from most familiar to most specialized.

### Arithmetic Mean

The most familiar average. You add up all the values and divide by how many there are:

$$
\hat{\mu} = \bar{x} = \langle x \rangle = \frac{1}{N}\sum_i^N x_i
$$

```python
def arithmetic_mean(arr):
    return np.sum(arr) / len(arr)
    # or equivalently: np.mean(arr)
```

The arithmetic mean is sensitive to extreme values — a single outlier can drag it far from where most of the data sits. If Alex's ten measurements include one wild reading of 10.5 (a bumped table, perhaps), the mean shifts noticeably. That's not always what you want.

### Geometric Mean

When your data are multiplicative in nature (growth rates, ratios, concentrations spanning orders of magnitude), the geometric mean is more appropriate. It is the $n$-th root of the product:

$$
\bar{x}_\text{geo} = \left( \prod_i^n x_i\right)^{1/n} = \exp\left(\frac{1}{n}\sum_{i}^n\ln x_i \right)
$$

This is equivalent to taking the arithmetic mean in log-space, which is why it works well for data that are log-normally distributed. If your data span several orders of magnitude, the geometric mean lives where the data actually cluster — not where the arithmetic mean gets dragged by the largest values.

```python
def geometric_mean(arr):
    # A sum of logs is less prone to
    # under-/over-flow than a product.
    return np.exp(np.mean(np.log(arr)))
```

### Median

The middle value when the data are sorted. Unlike the mean, the median is robust to outliers — even wild values at the extremes do not shift it much. If Alex had that one bumped-table reading of 10.5, the median wouldn't flinch.

```python
def median(arr):
    s = np.sort(arr)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2
```

### Mode

The most frequently occurring value. For continuous data, the mode is estimated from a histogram or kernel density estimate. It tells you where the data piles up the most — useful when your distribution has a clear peak.

### Harmonic Mean

$$
\bar{x}_\text{harm} = \left[\frac{1}{n}\sum_i^n \frac{1}{x_i}\right]^{-1}
$$

The harmonic mean is the right choice when you're averaging *rates*. Suppose you drive 60 km/h for the first half of a trip and 40 km/h for the second half. The harmonic mean (48 km/h) gives the correct average speed, not the arithmetic mean (50 km/h). The difference matters.

```python
def harmonic_mean(x):
    return (np.sum(x**(-1)) / len(x))**(-1)
```

### Truncated Mean

A compromise between the sensitivity of the mean and the robustness of the median: compute the arithmetic mean after throwing away the most extreme values on both ends.

```python
def truncated_mean(arr, k):
    arr_sorted = np.sort(arr)
    return np.mean(arr_sorted[k:-k])
```

> **Try this at home.** Take ten measurements of anything — your walking time to class, the temperature outside, the number of words per sentence in a book. Compute the mean and the median. Are they close? If not, you probably have outliers or a skewed distribution. That's already telling you something about the data.

## Measures of Spread

Knowing the center is only half the story. Two datasets can have the same mean but look completely different — one tightly clustered, the other scattered across a huge range. You need a number that captures *how much the data wanders*.

### Standard Deviation

Standard deviation measures how much data points typically deviate from the mean. Think of it as the "average distance" from the center:

$$
\hat{\sigma} = \sqrt{\frac{1}{N}\sum_i^N (x_i - \mu)^2}
$$

This formula assumes you know the true mean $\mu$. In practice, you use the sample mean $\bar{x}$, which introduces a subtle bias. Here is the surprising thing: the sample mean is always a little closer to the data than the true mean is, so the raw formula systematically underestimates the spread. **Bessel's correction** compensates by dividing by $N-1$ instead of $N$, accounting for the lost degree of freedom:

$$
\tilde{\sigma} = \sqrt{\frac{1}{N-1}\sum_i(x_i-\bar{x})^2}
$$

### Weighted Mean

When measurements come with different uncertainties, you should trust the precise ones more. The weighted mean does exactly this — measurements with smaller error bars get larger weights:

$$
\hat{\mu} = \frac{\sum x_i / \sigma_i^2}{\sum 1 / \sigma_i^2}, \qquad \hat{\sigma}_\mu = \sqrt{\frac{1}{\sum 1/\sigma_i^2}}
$$

The uncertainty on the weighted mean decreases with the number of samples:

$$
\hat{\sigma}_\mu = \hat{\sigma}/\sqrt{N}.
$$

Alex has three measurements of the same length: $10.2 \pm 0.1$, $10.5 \pm 0.5$, and $10.1 \pm 0.2$ cm. The weighted mean pulls toward 10.2, because that measurement is the most precise. This is your first taste of a powerful idea: *let the data tell you how much to trust each piece of information*. You'll see this idea again when we meet likelihood in lesson 2 and chi-square fitting in lesson 5.

## From Individual Variables to Relationships

So far you've described single variables in isolation: their centers and their spreads. But the most interesting questions in science are about *relationships*. Does one quantity vary in step with another? If temperature goes up, does reaction rate go up too? This is where **correlation** enters.

### Covariance and Pearson's Correlation

The natural extension of variance to two variables is the **covariance** — it measures whether $x$ and $y$ tend to move together:

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

Normalizing by the widths of each distribution gives **Pearson's (linear) correlation coefficient**, which lives between $-1$ and $+1$:

$$
\rho_{xy} = \frac{V_{xy}}{\sigma_x\sigma_y}, \qquad -1 \leq \rho_{xy} \leq 1
$$

A value of $+1$ means perfect positive linear relationship; $-1$ means perfect negative; $0$ means no *linear* relationship. But here is the catch — and this is important: $\rho_{xy} = 0$ does **not** mean the variables are unrelated.

Picture four scatter plots. In the first, points fall neatly along a rising line — Pearson catches this perfectly. In the second, the same rising trend with more scatter — Pearson is smaller but still positive. In the third, the points form a perfect parabola — $y = x^2$ — but Pearson is zero, because the relationship isn't linear. In the fourth, the points form a perfect circle — complete dependence, but Pearson sees nothing at all. Always plot the data.

### Rank Correlation: Beyond Linearity

What if the relationship between two variables is monotonic but not linear? Study hours and exam scores might be positively related, but not in a straight line — the first few hours of study help enormously, while the last few add less. **Rank correlation** handles this by comparing the *rankings* rather than the raw values.

**Spearman's $\rho$** replaces each value with its rank and computes the Pearson correlation on the ranks:

$$
\rho_S = 1 - \frac{6\sum_i (r_i - s_i)^2}{n^3-n}
$$

**Kendall's $\tau$** takes a different approach: it counts concordant pairs (both variables increase together) versus discordant pairs (one goes up while the other goes down).

### Beyond Monotonic Relationships

Pearson catches linear relationships. Spearman and Kendall catch monotonic ones. But what about truly complex dependencies — where $y$ might go up, then down, then up again as $x$ increases? For these, you need more powerful tools:

- **Maximal Information Coefficient (MIC)**: Scans across many possible grids to find the strongest relationship of any shape. High MIC means the variables are related somehow, even if the pattern is complex.
- **Mutual Information (MI)**: Borrowed from information theory, MI measures how much knowing $x$ reduces your uncertainty about $y$. It captures any kind of dependency, linear or not.
- **Distance Correlation (DC)**: Unlike Pearson, distance correlation equals zero *only* when the variables are truly independent. It provides a rigorous test for any form of association.

These measures are more computationally expensive, so they are typically reserved for exploratory analysis when you suspect non-trivial structure. The progression from Pearson to Spearman to MIC/MI/DC mirrors a trade-off: each step captures more general relationships, at the cost of simplicity and interpretability.

> **A note on what's coming.** Everything in this section describes data you already have. But the deeper question is: what *process* generated that data? If you can describe the underlying mechanism with a mathematical function, you unlock the ability to predict, extrapolate, and quantify uncertainty. That's the idea of a probability density function — and it's where we go next.

> **Challenge.** Explain to a friend, using only words a 12-year-old would understand, why the average can be misleading when there's an outlier. You have one minute.

---

**What we just learned.** You now have a toolkit for describing data: means (arithmetic, geometric, harmonic, truncated) for the center; standard deviation and weighted mean for spread; and correlation coefficients (Pearson, Spearman, Kendall, MIC) for relationships between variables. The key insight is that no single summary captures everything — always plot the data, always ask what kind of structure you're looking for, and always consider whether the "obvious" summary is actually the right one.
