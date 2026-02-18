# Longitudinal Data and Repeated Measures

## What Makes Longitudinal Data Special

In the previous section, you handled grouped data where measurements within a cluster are correlated. Remember those random effects — the random intercepts and the ICC? Here's why they save you in real life.

**Longitudinal data** is a specific and important case: the same subjects are measured repeatedly *over time*. Suppose you follow 200 patients for five years, measuring their blood pressure every six months. Some drop out. Some miss visits. Now what?

This creates two fundamental challenges:

- **Within-subject correlation**: measurements from the same person are not independent. A patient with high blood pressure in January will probably still have high blood pressure in March.
- **Time structure**: the spacing and ordering of observations carry information. A measurement today is more correlated with yesterday's than with last month's.

Ignoring these features and applying standard methods treats each measurement as if it came from a different person — a mistake that produces incorrect standard errors and misleading p-values.

Picture one patient's trajectory over time — their blood pressure goes up, dips, then rises again. That individual curve deviates from the population average in a way that's consistent and personal. The mixed model captures exactly this: the population trend plus each person's individual deviation from it.

## Repeated Measures ANOVA

The classical approach extends ANOVA to handle within-subject factors. For a single within-subject factor with $k$ time points:

$$
Y_{ij} = \mu + \pi_i + \tau_j + \varepsilon_{ij},
$$

where $\pi_i$ is the subject effect and $\tau_j$ is the time effect.

**Sphericity assumption**: the variances of all pairwise differences between time points must be equal. In plain terms, the change from time 1 to time 2 should be as variable as the change from time 2 to time 3. Mauchly's test checks this. When sphericity is violated (which is common):

- **Greenhouse-Geisser correction**: reduces the degrees of freedom to control type I error. Conservative — it may miss real effects.
- **Huynh-Feldt correction**: less conservative alternative.

**Limitation**: repeated measures ANOVA requires complete data (no missing time points) and equally spaced measurements. In practice, subjects drop out, miss visits, or show up on irregular schedules. This is where mixed models take over.

## Linear Mixed Models for Longitudinal Data

**Linear mixed models** (LMMs) overcome these limitations by modeling each subject's trajectory directly. This builds directly on the mixed model framework from the previous section — the random intercept model you already know, now with the time dimension added:

$$
Y_{ij} = \beta_0 + \beta_1 t_{ij} + u_{0i} + u_{1i} t_{ij} + \varepsilon_{ij},
$$

where $u_{0i}$ is a **random intercept** (each subject starts at a different level) and $u_{1i}$ is a **random slope** (each subject changes at a different rate).

The random effects are assumed multivariate normal:

$$
\begin{pmatrix} u_{0i} \\ u_{1i} \end{pmatrix} \sim \mathcal{N}\!\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} \sigma_0^2 & \rho\sigma_0\sigma_1 \\ \rho\sigma_0\sigma_1 & \sigma_1^2 \end{pmatrix}\right).
$$

**Advantages over repeated measures ANOVA**:

- Handles missing data naturally (uses all available observations, not just complete cases).
- Accommodates unequally spaced time points.
- Models individual trajectories, not just group means.
- Allows complex correlation structures.

## Growth Curve Models

When the outcome follows a nonlinear trajectory over time — growth spurts, learning curves, disease progression — polynomial or nonlinear growth curves can be fit within the mixed-model framework:

$$
Y_{ij} = \beta_0 + \beta_1 t_{ij} + \beta_2 t_{ij}^2 + u_{0i} + u_{1i} t_{ij} + \varepsilon_{ij}.
$$

The fixed effects $(\beta_0, \beta_1, \beta_2)$ describe the population-average trajectory — the "typical" curve. The random effects $(u_{0i}, u_{1i})$ capture how each individual deviates from that average. Some patients improve faster. Some start higher but plateau earlier. The model captures all of this.

## Autocorrelation

Here's an everyday example. If today is warm, tomorrow is probably warm too. Weather doesn't jump randomly from day to day — it has *memory*. The same is true of many measurement sequences: blood pressure readings, reaction times across trials, stock prices.

**Autocorrelation** quantifies how a signal correlates with itself at different time lags:

$$
R(\tau) = \frac{1}{N} \sum_{t=1}^{N-\tau} (x_t - \bar{x})(x_{t+\tau} - \bar{x}).
$$

In longitudinal data, residuals from adjacent time points are often more correlated than residuals from distant ones. The mixed model handles this through the correlation structure of the errors. Here's the menu of choices, ordered from simplest to most flexible — and here's when each one fails:

- **Compound symmetry**: Constant correlation between all pairs. Equivalent to a random intercept. Simple, but assumes that measurements 1 day apart are equally correlated as measurements 1 year apart. Fails when correlation decays over time.
- **AR(1)**: Correlation decays exponentially with time lag: $\text{Cor}(\varepsilon_{ij}, \varepsilon_{ik}) = \phi^{|j-k|}$. Often the most realistic choice for equally-spaced data. Fails for irregularly spaced observations.
- **Unstructured**: Separate correlation for each pair. The most flexible but parameter-intensive — only feasible when the number of time points is small (say, fewer than 10). Fails by overfitting when time points are many.

Model selection among correlation structures uses **AIC** or **BIC** (the information criteria we'll encounter again in lesson 12 on advanced fitting).

## Handling Missing Data

Suppose you follow patients for five years. Some drop out. Some miss appointments. Some move away. Now what? The surprising thing is: it matters enormously *why* the data are missing, because the mechanism determines which analyses are valid.

- **MCAR** (missing completely at random): Missingness is unrelated to any data. The data you have are a random subset of what you would have collected. Standard methods remain valid, though less powerful.
- **MAR** (missing at random): Missingness depends on observed data but not on the missing values themselves. A patient might drop out because of a recorded side effect, but not because of their unrecorded future outcome. LMMs under maximum likelihood provide valid inference.
- **MNAR** (missing not at random): Missingness depends on the unobserved values themselves. Patients drop out *because* they're getting worse, and you never see how much worse. This is the hardest case, requiring specialized models (selection models, pattern-mixture models).

Why the heck do you care whether the missing data is random or not? Because if patients who are getting worse are the ones who drop out, your remaining data makes the treatment look better than it is. The missingness *biases* your results. LMMs estimated by maximum likelihood or REML (the estimation method from lesson 9) are valid under MAR, making them the preferred approach for longitudinal analysis with incomplete data.

[[simulation applied-stats-sim-7]]

## Time Series Analysis

Many experiments produce sequential measurements where temporal patterns are the primary interest. **Time series analysis** provides tools specifically designed for this:

**Moving averages** smooth noisy signals by replacing each value with the average of its neighbors, revealing underlying trends. **Autoregressive (AR) models** predict the next value from a linear combination of previous values — formalizing the idea that the past carries information about the future. The **Fourier transform** (covered in the FFT topic) decomposes a signal into its frequency components, revealing periodic patterns that might be invisible in the raw time series.

Throughout this course, you've been building up a likelihood-based, frequentist toolkit. But there's an entirely different way of thinking about probability — one where probability represents *belief* rather than long-run frequency, and where you can formally incorporate what you knew before you saw the data. That's Bayesian statistics, and it's where we go next.
