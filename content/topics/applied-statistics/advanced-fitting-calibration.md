# Advanced Fitting and Calibration

## Beyond Simple Fitting

Standard least-squares fitting assumes a single functional form with normally distributed residuals. In practice, experimental data often requires more sophisticated approaches: models with multiple components, correlated parameters, systematic uncertainties, or the need to compare competing models quantitatively.

**Advanced fitting** extends the basic framework by addressing parameter correlations, goodness-of-fit assessment, and model comparison. **Calibration** connects raw instrument readings to physical quantities through reference standards and control measurements.

[[simulation applied-stats-sim-3]]

## Multi-Component Models

Many physical measurements involve a **signal plus background** decomposition. The observed data follow a model

$$
f(x; \boldsymbol{\theta}) = S(x; \boldsymbol{\theta}_S) + B(x; \boldsymbol{\theta}_B),
$$

where $S$ describes the signal of interest and $B$ accounts for background contributions. The total parameter vector $\boldsymbol{\theta} = (\boldsymbol{\theta}_S, \boldsymbol{\theta}_B)$ is estimated simultaneously.

When the background shape is known from control measurements or simulation, its parameters may be **constrained** by adding penalty terms to the objective function. This is equivalent to Bayesian fitting with informative priors on the background parameters.

## Profile Likelihood

For models with many parameters, the **profile likelihood** provides confidence intervals that account for correlations. Given parameters of interest $\boldsymbol{\psi}$ and nuisance parameters $\boldsymbol{\lambda}$, the profile likelihood is

$$
L_p(\boldsymbol{\psi}) = \max_{\boldsymbol{\lambda}} L(\boldsymbol{\psi}, \boldsymbol{\lambda}).
$$

At each value of $\boldsymbol{\psi}$, the nuisance parameters are re-optimized. A confidence region at level $\alpha$ is defined by

$$
-2 \ln \frac{L_p(\boldsymbol{\psi})}{L_p(\hat{\boldsymbol{\psi}})} \leq \chi^2_{k, \alpha},
$$

where $k = \dim(\boldsymbol{\psi})$. This correctly propagates parameter correlations into the uncertainty.

## Goodness of Fit

After fitting, we must assess whether the model adequately describes the data.

The **chi-squared statistic** $\chi^2 = \sum_i (y_i - f(x_i; \hat{\boldsymbol{\theta}}))^2 / \sigma_i^2$ should follow a $\chi^2$ distribution with $n - p$ degrees of freedom if the model is correct. The **reduced chi-squared** $\chi^2_\nu = \chi^2 / (n - p)$ should be approximately 1. Values much greater than 1 indicate a poor fit or underestimated uncertainties; values much less than 1 suggest overestimated uncertainties.

The **p-value** gives the probability of obtaining a $\chi^2$ at least as large as observed, assuming the model is correct. Small p-values (typically $< 0.05$) suggest the model is inadequate.

## Model Comparison

When multiple models could describe the data, we need principled criteria for selection.

The **likelihood ratio test** compares nested models (where one is a special case of the other). The test statistic $\Lambda = -2\ln(L_0 / L_1)$ follows a $\chi^2$ distribution with degrees of freedom equal to the difference in number of parameters.

For non-nested models, **information criteria** penalize model complexity:

$$
\text{AIC} = -2\ln L + 2p, \qquad \text{BIC} = -2\ln L + p\ln n,
$$

where $p$ is the number of parameters and $n$ the number of data points. Lower values indicate a better balance of fit quality and parsimony. BIC penalizes complexity more heavily and tends to favor simpler models.

## Calibration

**Calibration** establishes the relationship between a measured quantity (instrument response) and a known reference standard. The calibration function $R = g(S; \boldsymbol{\theta})$ maps the true physical signal $S$ to the instrument response $R$.

A typical calibration procedure involves:

1. **Reference measurements**: Measure known standards spanning the range of interest.
2. **Fit the calibration curve**: Determine $g$ and its parameters from the reference data.
3. **Invert for unknowns**: Given a new measurement $R_{\text{obs}}$, solve $R_{\text{obs}} = g(S; \hat{\boldsymbol{\theta}})$ for $S$.
4. **Propagate uncertainties**: Include both statistical uncertainty from the measurement and systematic uncertainty from the calibration curve.

## Control Channels

**Control channels** (or sidebands) are data regions where the signal is absent but the background is present. They provide an independent constraint on background parameters and reduce the uncertainty on the signal.

In a simultaneous fit, the likelihood combines the signal region and control region:

$$
L_{\text{total}} = L_{\text{signal}}(\boldsymbol{\theta}_S, \boldsymbol{\theta}_B) \times L_{\text{control}}(\boldsymbol{\theta}_B).
$$

The control channel pins down $\boldsymbol{\theta}_B$, effectively reducing the number of free parameters in the signal region. This technique is standard in particle physics (sidebands), astrophysics (off-source regions), and medical imaging (baseline measurements).

## Systematic Uncertainties

Systematic uncertainties arise from imperfect knowledge of the experimental setup: detector response, energy scale, efficiency corrections, and theoretical modeling. They are handled by introducing **nuisance parameters** $\boldsymbol{\lambda}$ with constraint terms:

$$
-2\ln L_{\text{constrained}} = -2\ln L(\boldsymbol{\theta}, \boldsymbol{\lambda}) + \sum_k \frac{(\lambda_k - \hat{\lambda}_k)^2}{\sigma_{\lambda_k}^2}.
$$

Each nuisance parameter is profiled out during the fit. The resulting uncertainty on parameters of interest automatically includes systematic contributions.

[[simulation applied-stats-sim-8]]
