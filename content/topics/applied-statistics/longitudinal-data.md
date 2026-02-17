# Longitudinal Data and Repeated Measures

## What makes longitudinal data special

**Longitudinal data** arise when the same subjects are measured repeatedly over time. This creates two fundamental challenges:

- **Within-subject correlation**: measurements from the same individual are not independent.
- **Time structure**: the spacing and ordering of observations carry information about dynamics.

Ignoring these features and applying standard methods leads to incorrect inference.

## Repeated measures ANOVA

The classical approach extends ANOVA to handle within-subject factors. For a single within-subject factor with $k$ time points:

$$
Y_{ij} = \mu + \pi_i + \tau_j + \varepsilon_{ij},
$$

where $\pi_i$ is the subject effect and $\tau_j$ is the time effect.

**Sphericity assumption**: the variances of all pairwise differences between time points must be equal. Mauchly's test checks this assumption. When sphericity is violated:

- **Greenhouse-Geisser correction**: reduces the degrees of freedom to control type I error (conservative).
- **Huynh-Feldt correction**: less conservative alternative.

**Limitation**: repeated measures ANOVA requires complete data (no missing time points) and equally spaced measurements.

## Linear mixed models for longitudinal data

**Linear mixed models** (LMMs) overcome the limitations of repeated measures ANOVA by modeling the correlation structure directly:

$$
Y_{ij} = \beta_0 + \beta_1 t_{ij} + u_{0i} + u_{1i} t_{ij} + \varepsilon_{ij},
$$

where $u_{0i}$ is a **random intercept** (each subject starts at a different level) and $u_{1i}$ is a **random slope** (each subject changes at a different rate).

The random effects are assumed multivariate normal:

$$
\begin{pmatrix} u_{0i} \\ u_{1i} \end{pmatrix} \sim \mathcal{N}\!\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} \sigma_0^2 & \rho\sigma_0\sigma_1 \\ \rho\sigma_0\sigma_1 & \sigma_1^2 \end{pmatrix}\right).
$$

**Advantages over repeated measures ANOVA**:

- Handles missing data naturally (uses all available observations).
- Accommodates unequally spaced time points.
- Models individual trajectories, not just group means.
- Allows complex correlation structures.

## Growth curve models

When the outcome follows a nonlinear trajectory over time, polynomial or nonlinear growth curves can be fit within the mixed-model framework:

$$
Y_{ij} = \beta_0 + \beta_1 t_{ij} + \beta_2 t_{ij}^2 + u_{0i} + u_{1i} t_{ij} + \varepsilon_{ij}.
$$

The fixed effects $(\beta_0, \beta_1, \beta_2)$ describe the population-average trajectory, while the random effects $(u_{0i}, u_{1i})$ capture individual deviations.

## Autocorrelation

In longitudinal data, residuals from adjacent time points are often more correlated than residuals from distant time points. Common correlation structures:

- **Compound symmetry**: constant correlation between all pairs (equivalent to random intercept).
- **AR(1)**: correlation decays exponentially with time lag: $\text{Cor}(\varepsilon_{ij}, \varepsilon_{ik}) = \phi^{|j-k|}$.
- **Unstructured**: separate correlation for each pair (most flexible but parameter-intensive).

Model selection among correlation structures uses **AIC** or **BIC**.

## Handling missing data

Longitudinal studies inevitably have missing observations. The mechanism matters:

- **MCAR** (missing completely at random): missingness is unrelated to any data. Standard methods remain valid.
- **MAR** (missing at random): missingness depends on observed data but not on the missing values themselves. LMMs under maximum likelihood provide valid inference.
- **MNAR** (missing not at random): missingness depends on the unobserved values. Requires specialized models (selection models, pattern-mixture models).

LMMs estimated by maximum likelihood or REML are valid under MAR, making them the preferred approach for longitudinal analysis with incomplete data.

[[simulation applied-stats-sim-7]]
