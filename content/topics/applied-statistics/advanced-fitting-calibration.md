# Advanced Fitting and Calibration

## Beyond Simple Fitting

Systematic errors are ghosts in the machine. Your detector drifts, your energy scale shifts, your efficiency varies. And behind your signal lurks a background you must model before claiming to have seen anything.

In [chi-square fitting](./chi-square-method), you fit models with a few parameters and Gaussian errors. Real data is messier: overlapping signals, correlated parameters, systematic uncertainties, competing models. This section handles those complications.

## Multi-Component Models

Many measurements involve **signal plus background**:

$$
f(x; \boldsymbol{\theta}) = S(x; \boldsymbol{\theta}_S) + B(x; \boldsymbol{\theta}_B)
$$

Think of hearing a conversation in a noisy room. Model both the conversation and the noise simultaneously. Model the noise poorly, and it leaks into your signal estimate.

When background shape is known from control measurements, its parameters can be **constrained** by adding penalty terms. This is Bayesian fitting with informative priors on background parameters -- the connection to [Bayesian statistics](./bayesian-statistics) is deliberate.

## Profile Likelihood

Some parameters are interesting (signal strength, a physical constant). Others are nuisance -- things you must account for but don't care about. You want confidence intervals on the interesting ones that honestly reflect uncertainty in the nuisance ones.

The **profile likelihood** handles this. For each value of the interesting parameter, ask: "what's the *best* the model can do if I freely adjust everything else?" Trace out a curve of best-case likelihoods. That curve gives you confidence intervals:

$$
L_p(\boldsymbol{\psi}) = \max_{\boldsymbol{\lambda}} L(\boldsymbol{\psi}, \boldsymbol{\lambda})
$$

[[simulation profile-likelihood]]

A confidence region at level $\alpha$:

$$
-2 \ln \frac{L_p(\boldsymbol{\psi})}{L_p(\hat{\boldsymbol{\psi}})} \leq \chi^2_{k, \alpha}
$$

This correctly propagates parameter correlations into the uncertainty -- reading off the diagonal of a covariance matrix would miss correlated parameters entirely.

## Goodness of Fit

After fitting, ask: does the model actually describe the data? The reduced chi-squared $\chi^2_\nu = \chi^2/(n-p)$ should be around 1. Much larger means wrong model or underestimated errors. Much smaller means overestimated errors.

## Model Comparison

Multiple models could fit. More parameters *always* fit better, so you penalize complexity:

The **likelihood ratio test** compares nested models: $\Lambda = -2\ln(L_0/L_1)$ follows $\chi^2$ with degrees of freedom = difference in parameters.

For non-nested models, **information criteria**:

$$
\text{AIC} = -2\ln L + 2p, \qquad \text{BIC} = -2\ln L + p\ln n
$$

Lower is better. BIC penalizes complexity harder and favors simpler models for large datasets.

## Calibration

All fitting assumes you know the relationship between instrument reading and physical quantity. **Calibration** establishes it:

1. Measure known standards spanning the range.
2. Fit the calibration curve: $R = g(S; \boldsymbol{\theta})$.
3. Invert for unknowns.
4. Propagate uncertainties from both measurement and calibration.

## Control Channels

How do you pin down the background without contaminating the signal? Use data regions where signal is absent but background is present:

$$
L_{\text{total}} = L_{\text{signal}}(\boldsymbol{\theta}_S, \boldsymbol{\theta}_B) \times L_{\text{control}}(\boldsymbol{\theta}_B)
$$

It's like recording room noise in an empty room, then using that when you try to hear the conversation. Standard in particle physics, astrophysics, medical imaging.

## Systematic Uncertainties

Systematics don't shrink with more data -- they reflect fundamental limitations. The modern approach introduces **nuisance parameters** with constraint terms:

$$
-2\ln L_{\text{constrained}} = -2\ln L(\boldsymbol{\theta}, \boldsymbol{\lambda}) + \sum_k \frac{(\lambda_k - \hat{\lambda}_k)^2}{\sigma_{\lambda_k}^2}
$$

Each nuisance parameter is profiled out. The resulting uncertainty automatically includes systematic contributions.

And here's the punchline: those constraint terms $(\lambda_k - \hat{\lambda}_k)^2/\sigma_{\lambda_k}^2$ are Gaussian priors on the nuisance parameters. This is [Bayesian reasoning](./bayesian-statistics) inside a frequentist framework. The two philosophies converge where it matters most. Nuisance parameters *are* priors in disguise -- and understanding this makes the bridge between frequentist and Bayesian fitting feel less like two separate worlds and more like old friends who speak different dialects.

> **Challenge.** Explain statistical vs. systematic uncertainty. Statistical: the wobble in your measurements -- take more, it shrinks. Systematic: a ruler that's slightly wrong -- no matter how many times you measure, the bias stays. One minute.

## Big Ideas

* The profile likelihood honestly reports uncertainty after letting everything else float -- the covariance diagonal misses parameter correlations.
* AIC and BIC penalize complexity to prevent the trivial truth that more parameters always fit better.
* Constraint terms on nuisance parameters are Gaussian priors in disguise -- frequentist and Bayesian converge here.
* Control channels pin down the background using signal-free data. Not optional in serious analysis.

## What Comes Next

Advanced fitting synthesizes the entire toolkit: distributions, likelihood, chi-square, testing, and Bayesian priors. The final lesson shows how these ideas reappear inside machine learning -- often unrecognized. Regularization is Bayesian inference. Cross-validation is model comparison. Fisher's discriminant is ANOVA in disguise. Having learned the statistics, you're positioned to see what's really happening inside the algorithms.

## Check Your Understanding

1. Model with 5 parameters, 2 nuisance. Why does the covariance diagonal give wrong uncertainty on your parameter of interest? What does the profile likelihood do differently?
2. Model A: 3 parameters, $\ln L = -120$. Model B: 5 parameters, $\ln L = -118$. Compute AIC for both. Which wins? Does BIC with $n = 200$ agree?
3. $\chi^2_\nu = 0.6$. Good? What might cause reduced chi-squared significantly below 1?

## Challenge

You're searching for a narrow signal peak in a spectrum with a polynomial background. Signal: Gaussian with known width, unknown amplitude and position. Background: second-degree polynomial. Describe: (a) likelihood setup and signal-background decomposition, (b) sideband constraints on the background, (c) profile likelihood for signal amplitude and uncertainty, (d) significance via likelihood ratio test. What would $\chi^2_\nu \gg 1$ say about the polynomial background?
