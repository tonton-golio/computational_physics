# Chi-Square Method

You have a model and some data. The model says straight line; the data scatter around it. The question is sharp: *how well does this model actually describe the data?*

## Linear Regression

The simplest model -- a straight line through data:

$$
y = \beta_0 + \beta_1 x + \epsilon
$$

But this treats all points equally. What if some measurements are much more precise? You need to weight the fit -- listen more carefully to reliable measurements, less to noisy ones.

[[simulation applied-stats-sim-1]]

## The Chi-Square Statistic

Remember the likelihood function from [probability density functions](./probability-density-functions)? For Gaussian errors, maximizing the likelihood is the same as minimizing this:

$$
\chi^2(\theta) = \sum_i^N \frac{(y_i - f(x_i,\theta))^2}{\sigma_i^2}
$$

Each residual is weighted by the measurement's precision. Precise measurements (small $\sigma_i$) pull hard; noisy ones whisper. This isn't a coincidence -- if each $y_i$ is Gaussian with mean $f(x_i, \theta)$, then $-2\ln\mathcal{L} = \chi^2 + \text{const}$. Chi-square and MLE are two faces of one coin.

Degrees of freedom:

$$
N_\text{dof} = N_\text{samples} - N_\text{fit parameters}
$$

### Interpreting the Fit

A good fit has $\chi^2$ roughly equal to $N_\text{dof}$. That's it -- the reduced chi-squared $\chi^2_\nu = \chi^2/N_\text{dof}$ should be around 1.

Way above 1? Either the model is wrong or the error bars are too small. Way below 1? The error bars are probably too big -- the fit is suspiciously good. Figure out which.

If errors are large, different models all fit similarly -- the data can't distinguish them. With small, precise errors, even slight model defects produce large $\chi^2$. That's a feature: better data demands better models.

Plot the **residuals** $(y_i - f(x_i, \theta))/\sigma_i$. They should scatter like a standard normal (mean 0, SD 1) with no visible pattern. A trend in the residuals is your early warning system -- the model is missing something.

```python
chi2_prob = stats.chi2.sf(chi2_value, N_dof)
```

Note: the weighted mean from [introduction](./introduction-concepts) is just fitting a constant to data -- a special case of chi-square fitting.

### Chi-Square for Binned Data

For large datasets, bin into a histogram:

$$
\chi^2 = \sum_{i\in \text{bins}} \frac{(O_i-E_i)^2}{E_i}
$$

where $O_i$ is observed and $E_i$ expected. Exclude empty bins.

### Why Chi-Square Is Powerful

[[simulation applied-stats-sim-5]]

Near the minimum, the $\chi^2$ surface is a bowl. The curvature of that bowl directly gives you parameter uncertainties. Steep, narrow bowl = tight constraint. Shallow, wide bowl = large uncertainty.

### Uncertainties in $x$

If both $x$ and $y$ have errors, iterate: fit without $x$ errors first, then fold them in:

$$
\sigma_{y_i}^{\text{new}} = \sqrt{\sigma_{y_i}^2 + \left( \frac{\partial y}{\partial x}\bigg|_{x_i} \sigma_{x_i} \right)^2}
$$

Repeat. Converges in one or two rounds.

### Reporting Errors

Distinguish **statistical** (shrinks with more data) from **systematic** (doesn't):

$$
a = (0.24 \pm 0.05_\text{stat} \pm 0.07_\text{syst}) \times 10^4 \; \text{kg}
$$

> **Challenge.** Explain chi-square fitting as a target-and-arrows analogy. Each arrow lands near the bullseye; closer arrows get more credit. One minute.

## Big Ideas

* Minimizing $\chi^2$ and maximizing likelihood are the same operation when errors are Gaussian.
* $\chi^2 \approx N_\text{dof}$ means good fit. Much larger means wrong model or underestimated errors.
* Precise measurements don't just give better answers -- they demand better models.
* Residuals are not a formality: patterns tell you the model is missing something, even when overall $\chi^2$ looks fine.

## What Comes Next

Chi-square tells you how well a model fits and what the best parameters are. But it doesn't directly answer: *is there a real effect at all, or is what you see just noise?* That's hypothesis testing -- the formal framework for deciding between competing explanations.

## Check Your Understanding

1. You fit a line to 20 points (2 parameters) and get $\chi^2 = 36$. Good fit? Compute the reduced chi-squared and explain.
2. A fit gives $\chi^2_\nu = 0.3$. "Great fit -- even better than expected!" What does this actually suggest?
3. Residuals show a clear sinusoidal pattern, but $\chi^2/N_\text{dof} \approx 1$. What's happening?

## Challenge

Two datasets measure the same quantity. Dataset A: 50 precise measurements ($\sigma_i \approx 0.1$). Dataset B: 200 less precise ($\sigma_i \approx 1.0$). Same linear model for both. Which fit has more degrees of freedom? Which is more likely to reveal model defects? If both give $\chi^2_\nu \approx 1$, does that mean the model is equally good for both?
