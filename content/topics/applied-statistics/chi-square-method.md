# Chi-Square Method

Alex has a model and some data. The model predicts a straight line; the data scatter around it. The question is sharp: *how well does this model actually describe the data?* And what are the best-fit parameters?

You now have the tools to describe data (lesson 1), model it with distributions (lesson 2), understand why errors are Gaussian (lesson 3), and simulate complex scenarios (lesson 4). The next step is to connect models to data quantitatively. This is the domain of the **chi-square method** — the workhorse of model fitting in the physical sciences.

## Linear Regression

The simplest model is a straight line through data:

$$
y = \beta_0 + \beta_1 x + \epsilon
$$

where $\beta_0$ is the intercept, $\beta_1$ is the slope, and $\epsilon$ is the error term. This is a starting point, but it treats all data points equally. What if some measurements are much more precise than others? You need a way to weight the fit — to listen more carefully to the reliable measurements and less to the noisy ones.

[[simulation applied-stats-sim-1]]

## The Chi-Square Statistic

Remember the likelihood function from lesson 2? You learned to find the parameters that make the data most probable. For Gaussian errors, that's equivalent to minimizing this:

$$
\chi^2(\theta) = \sum_i^N \frac{(y_i - f(x_i,\theta))^2}{\sigma_i^2}
$$

This is the sum of squared residuals, each divided by the variance of that measurement. Precise measurements (small $\sigma_i$) pull the fit strongly; noisy ones (large $\sigma_i$) have less influence. The connection to MLE is exact: minimizing $\chi^2$ *is* maximizing the likelihood when errors are Gaussian. That's not a coincidence — it has to be this way. Let's see why.

If each measurement $y_i$ is Gaussian with mean $f(x_i, \theta)$ and variance $\sigma_i^2$, then $-2\ln\mathcal{L} = \chi^2 + \text{const}$. Minimizing one is the same as minimizing the other. The chi-square method and maximum likelihood are two faces of the same coin.

The number of **degrees of freedom** is:

$$
N_\text{dof} = N_\text{samples} - N_\text{fit parameters}
$$

### Interpreting the Fit

How do you know if a fit is good? The $\chi^2$ value should be roughly equal to the number of degrees of freedom. More precisely, you compute the **p-value** — the probability of getting a $\chi^2$ at least this large if the model is correct:

$$
\text{Prob}(\chi^2 = 65.7, N_\text{dof}=42) = 0.011
$$

A small p-value (say, below 0.05) suggests the model does not describe the data well. But be careful: it could also mean the error bars are underestimated. Alex learned this the hard way — a "bad" fit that turned out to be perfectly fine once the systematic errors were properly accounted for.

If errors are large, different models will all fit similarly well — the data cannot distinguish between them. With small, precise errors, even slight model deficiencies produce large $\chi^2$ values. This is a feature, not a bug: better data demands better models.

Plot the **residuals** $\frac{y_i-f(x_i, \theta)}{\sigma_i}$ — these should scatter like a standard normal distribution (mean 0, standard deviation 1) with no visible pattern. Trends in the residuals are your early warning system: the model is missing something.

```python
chi2_prob = stats.chi2.sf(chi2_value, N_dof)
```

Note: the weighted mean from lesson 1 is actually a special case of chi-squared fitting — it is just fitting a constant to the data.

### Chi-Square for Binned Data

For large datasets, it is often practical to bin the data into a histogram first:

$$
\chi^2 = \sum_{i\in \text{bins}} \frac{(O_i-E_i)^2}{E_i}
$$

where $O_i$ is the observed count and $E_i$ is the expected count in each bin. Empty bins should be excluded (they cause division by zero), at the cost of some loss of resolution.

### Why Chi-Square Is Powerful

[[simulation applied-stats-sim-5]]

The power of $\chi^2$ comes from its geometry. Near the minimum, the $\chi^2$ surface is approximately parabolic — like a bowl. The curvature of that bowl directly gives you the uncertainties on the fitted parameters. A steep, narrow bowl means the parameter is tightly constrained; a shallow, wide bowl means large uncertainty. This parabolic structure is what makes $\chi^2$-based confidence intervals so straightforward.

### Uncertainties in $x$

So far you've assumed errors only in $y$. If both $x$ and $y$ have uncertainties, the procedure is iterative: fit without $x$ errors first, then fold them in using error propagation (from lesson 3):

$$
\sigma_{y_i}^{\text{new}} = \sqrt{\sigma_{y_i}^2 + \left( \frac{\partial y}{\partial x}\bigg|_{x_i} \sigma_{x_i} \right)^2}
$$

Repeat the fit with the updated errors. This converges quickly — usually one or two iterations suffice.

### Reporting Errors

When you report a result, distinguish between **statistical** uncertainty (from the finite size of your dataset) and **systematic** uncertainty (from imperfect knowledge of the experimental setup). The standard format is:

$$
a = (0.24 \pm 0.05_\text{stat} \pm 0.07_\text{syst}) \times 10^4 \; \text{kg}
$$

## Looking Ahead

The chi-square method tells you how well a model fits the data and gives you parameter uncertainties. But it doesn't directly answer the question: *is there a real effect, or is what I see just noise?* That's the territory of **hypothesis testing**, which we take up next. There, you'll formalize the idea of comparing competing explanations — building directly on the likelihood and $\chi^2$ framework developed here.

> **Challenge.** Explain chi-square fitting to a friend using only the analogy of a target and arrows. Each arrow lands somewhere near the bullseye; closer arrows get more "credit." One minute.
