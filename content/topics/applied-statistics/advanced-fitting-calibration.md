# Advanced Fitting and Calibration

## Beyond Simple Fitting

Systematic errors are like ghosts in the machine. You can't see them directly, but they haunt your results. Your detector isn't perfectly calibrated. Your energy scale drifts. Your efficiency varies across the measurement range. And lurking behind your signal of interest is a background that must be modeled before you can claim to have seen anything.

In lesson 5, you fit models with a handful of parameters and Gaussian errors. Real experimental data is messier: multiple overlapping signals, correlated parameters, systematic uncertainties from the instrument itself, and competing models that all look plausible. This section builds on that foundation to handle the complications that arise in serious data analysis.

## Multi-Component Models

Many physical measurements involve a **signal plus background** decomposition. You're looking for a small peak sitting on top of a large, slowly-varying background:

$$
f(x; \boldsymbol{\theta}) = S(x; \boldsymbol{\theta}_S) + B(x; \boldsymbol{\theta}_B),
$$

where $S$ describes the signal of interest and $B$ accounts for background contributions. The total parameter vector $\boldsymbol{\theta} = (\boldsymbol{\theta}_S, \boldsymbol{\theta}_B)$ is estimated simultaneously.

Think of it as trying to hear a conversation in a noisy room. You need to model both the conversation (signal) and the room noise (background) at the same time. If you model the background poorly, it leaks into your signal estimate and distorts your conclusions.

When the background shape is known from control measurements or simulation, its parameters may be **constrained** by adding penalty terms to the objective function. This is equivalent to Bayesian fitting with informative priors on the background parameters (the connection to lesson 11 is direct and deliberate).

## Profile Likelihood

In a model with many parameters, some are interesting (the signal strength, a physical constant) and others are nuisance parameters — things you must account for but don't actually care about. The nuisance parameters are ghosts you need to catch so they don't contaminate what you're measuring. You want confidence intervals on the interesting parameters that correctly account for uncertainty in the nuisance ones.

The **profile likelihood** does this by optimizing over the nuisance parameters at every point:

$$
L_p(\boldsymbol{\psi}) = \max_{\boldsymbol{\lambda}} L(\boldsymbol{\psi}, \boldsymbol{\lambda}).
$$

Think of it as asking: "For this particular value of the interesting parameter, what is the *best* the model can do if I freely adjust everything else?" You trace out a curve of best-case likelihoods, and that curve gives you confidence intervals.

A confidence region at level $\alpha$ is:

$$
-2 \ln \frac{L_p(\boldsymbol{\psi})}{L_p(\hat{\boldsymbol{\psi}})} \leq \chi^2_{k, \alpha},
$$

where $k = \dim(\boldsymbol{\psi})$. This correctly propagates parameter correlations into the uncertainty — something that simply reading off the diagonal of the covariance matrix would miss if parameters are correlated.

## Goodness of Fit

After fitting, you must assess whether the model actually describes the data. A beautiful fit to a wrong model is worse than no fit at all.

The **chi-squared statistic** $\chi^2 = \sum_i (y_i - f(x_i; \hat{\boldsymbol{\theta}}))^2 / \sigma_i^2$ should follow a $\chi^2$ distribution with $n - p$ degrees of freedom if the model is correct. The **reduced chi-squared** $\chi^2_\nu = \chi^2 / (n - p)$ should be approximately 1. This is the same diagnostic from lesson 5, now applied to more complex models.

What do the numbers mean?

- $\chi^2_\nu \gg 1$: the model doesn't describe the data, or the uncertainties are underestimated. Something is wrong.
- $\chi^2_\nu \approx 1$: the model describes the data within the quoted uncertainties. Good.
- $\chi^2_\nu \ll 1$: the fit is *too* good — the uncertainties are probably overestimated. Suspicious in a different way.

The **p-value** gives the probability of obtaining a $\chi^2$ at least as large as observed, assuming the model is correct. Small p-values (typically $< 0.05$) suggest the model is inadequate.

## Model Comparison

When multiple models could describe the data, you need principled criteria for deciding which one to use. The temptation is to pick the model with the lowest $\chi^2$, but a model with more parameters will *always* fit at least as well. You need to penalize complexity.

The **likelihood ratio test** (from lesson 6) compares nested models. The test statistic $\Lambda = -2\ln(L_0 / L_1)$ follows a $\chi^2$ distribution with degrees of freedom equal to the difference in number of parameters. It asks: does the extra parameter buy enough improvement to justify its inclusion?

For non-nested models, **information criteria** balance fit quality against complexity:

$$
\text{AIC} = -2\ln L + 2p, \qquad \text{BIC} = -2\ln L + p\ln n,
$$

where $p$ is the number of parameters and $n$ the number of data points. Lower values are better. BIC penalizes complexity more heavily than AIC and tends to favor simpler models, especially for large datasets.

## Calibration

All the fitting methods above assume you know the relationship between what your instrument reads and the physical quantity you care about. **Calibration** establishes that relationship.

A typical calibration procedure:

1. **Reference measurements**: Measure known standards spanning the range of interest. These are your anchor points.
2. **Fit the calibration curve**: Determine $R = g(S; \boldsymbol{\theta})$ mapping the true physical signal $S$ to the instrument response $R$.
3. **Invert for unknowns**: Given a new measurement $R_{\text{obs}}$, solve $R_{\text{obs}} = g(S; \hat{\boldsymbol{\theta}})$ for the physical quantity.
4. **Propagate uncertainties**: Include both the statistical uncertainty from the measurement and the systematic uncertainty from the calibration curve itself (using the error propagation from lesson 3).

## Control Channels

How do you pin down the background model without contaminating the signal region? **Control channels** (or sidebands) are data regions where the signal is absent but the background is present.

In a simultaneous fit, the likelihood combines signal and control regions:

$$
L_{\text{total}} = L_{\text{signal}}(\boldsymbol{\theta}_S, \boldsymbol{\theta}_B) \times L_{\text{control}}(\boldsymbol{\theta}_B).
$$

The control channel pins down $\boldsymbol{\theta}_B$, reducing the number of free parameters in the signal region. It's like learning the room noise by recording in an empty room, then using that knowledge when you try to hear the conversation. This technique is standard in particle physics (sidebands), astrophysics (off-source regions), and medical imaging (baseline measurements).

## Systematic Uncertainties

Systematic uncertainties arise from imperfect knowledge of the experimental setup: detector response, energy scale, efficiency corrections, theoretical modeling. Unlike statistical uncertainties, they don't shrink with more data — they reflect fundamental limitations of the measurement.

The modern approach handles them by introducing **nuisance parameters** $\boldsymbol{\lambda}$ with constraint terms:

$$
-2\ln L_{\text{constrained}} = -2\ln L(\boldsymbol{\theta}, \boldsymbol{\lambda}) + \sum_k \frac{(\lambda_k - \hat{\lambda}_k)^2}{\sigma_{\lambda_k}^2}.
$$

Each nuisance parameter is profiled out during the fit. The resulting uncertainty on parameters of interest automatically includes systematic contributions — they propagate naturally through the fit rather than being added in quadrature after the fact.

Notice something: those constraint terms $(\lambda_k - \hat{\lambda}_k)^2 / \sigma_{\lambda_k}^2$ are Gaussian priors on the nuisance parameters. This is Bayesian reasoning (lesson 11) applied within a frequentist fitting framework. The two approaches converge when it matters most.

---

**What we just learned, and why it matters.** Real data analysis requires multi-component models, systematic uncertainty handling, and principled model comparison. The profile likelihood lets you marginalize over nuisance parameters. AIC and BIC help you choose between competing models. Control channels pin down backgrounds independently. And nuisance parameters with constraint terms — Bayesian priors in disguise — let systematic uncertainties propagate naturally into your final result. This section synthesizes ideas from lessons 2, 5, 6, and 11 into a complete analysis framework.

> **Challenge.** Explain the difference between statistical and systematic uncertainty to a friend. Statistical uncertainty is like the wobble in your measurements — take more measurements, and it shrinks. Systematic uncertainty is like a ruler that's slightly wrong — no matter how many times you measure, the bias stays. One minute.
