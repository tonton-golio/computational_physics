# Longitudinal Data and Repeated Measures

## What Makes Longitudinal Data Special

Same subjects, measured repeatedly over time. You follow 200 patients for five years, measuring blood pressure every six months. Some drop out. Some miss visits. Now what?

Two fundamental challenges:

* **Within-subject correlation**: a patient with high blood pressure in January probably still has it in March.
* **Time structure**: today's measurement correlates more with yesterday's than with last month's.

Ignore these and you treat each measurement as if it came from a different person -- wrong standard errors, misleading p-values.

## Repeated Measures ANOVA

The classical extension of ANOVA for within-subject factors:

$$
Y_{ij} = \mu + \pi_i + \tau_j + \varepsilon_{ij}
$$

**Sphericity assumption**: the variances of all pairwise differences between time points must be equal. Mauchly's test checks this. When it fails (common):

* **Greenhouse-Geisser**: reduces degrees of freedom. Conservative.
* **Huynh-Feldt**: less conservative alternative.

**Limitation**: requires complete data and equal spacing. Real subjects drop out and miss visits. That's where mixed models take over.

## Linear Mixed Models for Longitudinal Data

LMMs model each subject's trajectory directly:

$$
Y_{ij} = \beta_0 + \beta_1 t_{ij} + u_{0i} + u_{1i} t_{ij} + \varepsilon_{ij}
$$

where $u_{0i}$ is a **random intercept** (each person starts at a different level) and $u_{1i}$ is a **random slope** (each person changes at a different rate).

$$
\begin{pmatrix} u_{0i} \\ u_{1i} \end{pmatrix} \sim \mathcal{N}\!\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} \sigma_0^2 & \rho\sigma_0\sigma_1 \\ \rho\sigma_0\sigma_1 & \sigma_1^2 \end{pmatrix}\right)
$$

Why LMMs beat repeated measures ANOVA:

* Handle missing data naturally (use all available observations).
* Accommodate unequal time spacing.
* Model individual trajectories, not just group means.

[[simulation spaghetti-trajectory]]

## Growth Curve Models

When trajectories are nonlinear -- growth spurts, learning curves, disease progression:

$$
Y_{ij} = \beta_0 + \beta_1 t_{ij} + \beta_2 t_{ij}^2 + u_{0i} + u_{1i} t_{ij} + \varepsilon_{ij}
$$

Fixed effects describe the population-average curve. Random effects capture individual deviations. Some patients improve faster. Some plateau early. The model handles all of it.

## Autocorrelation

If today is warm, tomorrow probably is too. Weather has *memory*. So do blood pressure readings, reaction times, stock prices.

$$
R(\tau) = \frac{1}{N} \sum_{t=1}^{N-\tau} (x_t - \bar{x})(x_{t+\tau} - \bar{x})
$$

The correlation structure of residuals matters. Here's the menu:

* **Compound symmetry**: constant correlation between all pairs. Assumes measurements 1 day apart are equally correlated as 1 year apart. Often too simple.
* **AR(1)**: correlation decays exponentially with lag: $\phi^{|j-k|}$. Usually the most realistic for equally-spaced data.
* **Unstructured**: separate correlation for each pair. Most flexible but parameter-hungry -- only feasible with few time points.

Choose using AIC or BIC (the information criteria from [advanced fitting](./advanced-fitting-calibration)).

## Handling Missing Data

Patients drop out, miss appointments, move away. It matters enormously *why*.

**MCAR** (missing completely at random): missingness unrelated to anything. Your remaining data is a random subset. Standard methods work, just with less power.

**MAR** (missing at random): missingness depends on observed data but not unobserved values. A patient drops out because of a recorded side effect, not their unrecorded future. LMMs handle this correctly.

**MNAR** (missing not at random): patients drop out *because* they're getting worse, and you never see how much worse. This biases your results -- remaining data makes treatment look better than it is. Requires specialized models.

[[simulation missingness-mechanisms]]

## Time Series Analysis

When temporal patterns are the primary interest: **moving averages** smooth noise to reveal trends, **autoregressive models** predict from past values, and the **Fourier transform** decomposes signals into frequencies.

Throughout this course you've built a likelihood-based, frequentist toolkit. But there's a fundamentally different way to think about probability -- where it represents *belief* rather than long-run frequency, and you formally incorporate what you knew before seeing data. That's Bayesian statistics, next.

## Big Ideas

* Longitudinal data is grouped data with a time axis -- random effects handle it once you account for within-subject correlation and time ordering.
* The *mechanism* of missingness (MCAR, MAR, MNAR) determines which analyses are valid. This is not optional to think about.
* Correlation structure is a modeling decision, not a default setting -- inform it with data and evaluate with information criteria.
* Repeated measures ANOVA is a special case of LMMs that requires complete, equally-spaced data. LMMs handle messy reality.

## What Comes Next

You've built a frequentist toolkit: distributions, likelihood, chi-square, tests, ANOVA, design, mixed models, longitudinal analysis. All treat probability as long-run frequency. Bayesian statistics offers a complementary view: probability as degree of belief. Start with what you knew, update with the likelihood, get a posterior. The likelihood you've been using since PDFs is the bridge -- same engine, different direction.

## Check Your Understanding

1. A patient drops out because their condition worsens. MCAR, MAR, or MNAR? What are the consequences for a linear mixed model?
2. Compound symmetry vs. AR(1): what does each assume? Give a real-world example where each is appropriate.
3. Repeated measures ANOVA gives significance but Mauchly's test rejects sphericity. What should you do?

## Challenge

Five years of blood pressure data, 200 patients, measured every six months. By year 3, ~40 have dropped out. Describe: (a) how you'd characterize the dropout mechanism, (b) which model and why, (c) how you'd choose the correlation structure, (d) what you'd do if you suspected MNAR.
