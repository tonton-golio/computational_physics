# Introduction and General Concepts

You have a pile of numbers. Maybe they're pendulum timings, patient blood pressures, or photon counts from a distant star. Before you build any model or run any test, you need to answer two questions: *where* does the data cluster, and *how much does it scatter?*

These sound simple. They aren't. The "center" of a dataset is not a single concept, and neither is "spread." Different situations demand different summaries. Picking the wrong one can lead you astray before you've even started your analysis.

Alex, our experimentalist, has just measured the gravitational acceleration ten times and gotten values between 9.78 and 9.84 m/s$^2$. What should Alex report as "the" value? And how uncertain should Alex be? That's what this section is about.

## Measures of Central Tendency

The "center" of a dataset depends on what question you're asking. Here is the menu, ordered from most familiar to most specialized.

### Arithmetic Mean

[[simulation which-average]]

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

> **Challenge.** Take ten measurements of anything — your walking time to class, the temperature outside, the number of words per sentence in a book. Compute the mean and the median. Are they close? If not, you probably have outliers or a skewed distribution. That's already telling you something about the data.

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

When measurements come with different uncertainties, you should trust the precise ones more. The weighted mean does exactly this — measurements with smaller error bars get larger weights.

Imagine each measurement standing on a seesaw. A precise measurement (small $\sigma$) is a heavy person near the fulcrum — it has more say in where the balance point lands. The weighted mean is simply where the seesaw balances.

$$
\hat{\mu} = \frac{\sum x_i / \sigma_i^2}{\sum 1 / \sigma_i^2}, \qquad \hat{\sigma}_\mu = \sqrt{\frac{1}{\sum 1/\sigma_i^2}}
$$

The uncertainty on the weighted mean decreases with the number of samples:

$$
\hat{\sigma}_\mu = \hat{\sigma}/\sqrt{N}.
$$

[[simulation weighted-mean]]

Alex has three measurements of the same length: $10.2 \pm 0.1$, $10.5 \pm 0.5$, and $10.1 \pm 0.2$ cm. The weighted mean pulls toward 10.2, because that measurement is the most precise. This is your first taste of a powerful idea: *let the data tell you how much to trust each piece of information*. You'll see this idea again when we meet likelihood in [probability density functions](./probability-density-functions) and [chi-square fitting](./chi-square-method).

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

### Beyond Pearson: Other Correlation Measures

Pearson's $\rho$ only detects *linear* relationships — a perfect parabola or circle scores zero. A family of alternative measures captures progressively more general patterns. **Spearman's $\rho$** replaces values with ranks and then computes Pearson on those ranks, so it catches any monotonic trend (e.g., study hours vs. exam scores, where the first few hours help enormously but later ones add less). **Kendall's $\tau$** counts concordant vs. discordant pairs and is more robust when the sample is small or contains many ties. For truly complex, non-monotonic dependencies, information-theoretic measures such as the **Maximal Information Coefficient (MIC)**, **Mutual Information (MI)**, and **Distance Correlation (DC)** can detect relationships of any shape, at the cost of higher computation and harder interpretation. The progression from Pearson to Spearman to MIC/MI/DC mirrors a trade-off: each step captures more general relationships, at the cost of simplicity.

| Measure | What it captures | When to use |
|---|---|---|
| Pearson $\rho$ | Linear association | Default first check; data roughly bivariate-normal |
| Spearman $\rho$ | Any monotonic trend | Ordinal data, outliers, or curved-but-monotonic relationships |
| Kendall $\tau$ | Monotonic trend (rank-based) | Small samples or many tied values |
| MIC | Any functional relationship | Exploratory screening for complex patterns |
| Mutual Information | Any statistical dependency | High-dimensional or strongly non-linear data |
| Distance Correlation | Any dependency (zero iff independent) | Rigorous independence testing |

Everything in this section describes data you already have. But the deeper question is: what *process* generated that data? If you can describe the underlying mechanism with a mathematical function, you unlock the ability to predict, extrapolate, and quantify uncertainty. That's the idea of a probability density function — and it's where we go next.

> **Challenge.** Explain to a friend, using only words a 12-year-old would understand, why the average can be misleading when there's an outlier. You have one minute.

## Big Ideas

* The "right" average depends on the structure of your data: arithmetic means for additive quantities, geometric means for multiplicative ones, harmonic means for rates — there is no universal champion.
* Bessel's correction ($N-1$ instead of $N$) exists because the sample mean is always closer to the data than the true mean is, so the naive formula systematically underestimates spread.
* A correlation of zero does not mean independence — a perfect circle has Pearson's $\rho = 0$. Always plot before you summarize.
* Giving imprecise measurements equal weight with precise ones throws away information; the weighted mean is how you let uncertainty tell you how much to trust each reading.

## What Comes Next

Everything in this section describes data you already have — summaries, centers, spreads, relationships. But the deeper question is: what *process* generated that data? If you can describe the underlying mechanism with a mathematical function, you unlock the ability to predict, extrapolate, and quantify uncertainty in a principled way.

That function is the probability density function, and it is where the course goes next. The weighted mean you computed here will reappear there as a special case of maximum likelihood estimation — the same idea, made general.

## Check Your Understanding

1. You have salary data for a company where the CEO earns 200 times the median worker's salary. Which measure of central tendency gives a more honest picture of what a typical employee earns, and why?
2. Two datasets have identical means and standard deviations but very different shapes. What does this tell you about what the mean and standard deviation do and do not capture?
3. Two variables have a Pearson correlation of $\rho = 0$. A classmate concludes they are independent. What example would you use to show this conclusion is wrong?

## Challenge

Design a measurement scheme for estimating the average commute time in a city. Your dataset will inevitably include some extreme outliers (people who commute three hours each way). Write down: which measure of central tendency you would report and why, how you would quantify the spread, and what a "weighted" version of your estimate might look like if some respondents kept more careful time logs than others. Then consider: if a journalist reports the arithmetic mean and a union reports the median, and the numbers are very different, who is lying?
