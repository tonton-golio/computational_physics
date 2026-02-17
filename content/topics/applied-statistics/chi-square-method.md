# Chi-Square Method

## Chi-Square Method

The chi-square method incorporates measurement uncertainties directly into the fit:

$$
\chi^2(\theta) = \sum_i^N \frac{(y_i - f(x_i,\theta))^2}{\sigma_i^2}
$$

The number of **degrees of freedom** is:

$$
N_\text{dof} = N_\text{samples} - N_\text{fit parameters}
$$

We can compute the **p-value** from the $\chi^2$ value and $N_\text{dof}$:

$$
\text{Prob}(\chi^2 = 65.7, N_\text{dof}=42) = 0.011
$$

If errors are large, different models yield fits of similar quality. With small errors, there is a large discrepancy in fit quality. Note: the weighted mean is a special case of chi-squared (fitting a constant).

If we plot the distribution of residuals $\frac{y_i-f(x_i, \theta)}{\sigma_i}$, they should be Gaussian. This gives a way to evaluate uncertainty estimates.

```python
chi2_prob = stats.chi2.sf(chi2_value, N_dof)
```

### Chi-Square for Binned Data

For large datasets, we can bin the data first:

$$
\chi^2 = \sum_{i\in \text{bins}} \frac{(O_i-E_i)^2}{E_i}
$$

Empty bins should be excluded from the sum (they would cause division by zero), at the cost of some loss of fidelity. Tools like `iminuit` handle this automatically.

### Why Chi-Square Is Powerful

Stable fits have parabolic minima near the best-fit point. This provides an easily assessable local curvature, giving us confidence intervals on the fitted parameters.

### Uncertainties in $x$

For data with uncertainties in both $x$ and $y$, fit without $x$ errors first, then update:

$$
\sigma_{y_i}^{\text{new}} = \sqrt{\sigma_{y_i}^2 + \left( \frac{\partial y}{\partial x}\bigg|_{x_i} \sigma_{x_i} \right)^2}
$$

This is iterative but converges quickly.

### Reporting Errors

Report results with both statistical and systematic uncertainties:

$$
a = (0.24 \pm 0.05_\text{stat} \pm 0.07_\text{syst}) \times 10^4 \; \text{kg}
$$
