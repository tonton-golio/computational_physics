# Random Effects and Mixed Models

## Fixed vs random effects

In standard ANOVA and regression, all effects are **fixed**: they represent specific levels of interest (e.g., drug A vs drug B). When the levels in the study are a random sample from a larger population of possible levels, the effect is **random**.

- **Fixed effect**: the specific levels are of direct interest. Inference applies only to those levels.
- **Random effect**: the levels are sampled from a population. Inference generalizes to the entire population of levels.

Examples of random effects:

- Subjects in a repeated-measures study (each subject is one realization from a population).
- Classrooms in an educational study (each classroom is a cluster).
- Batches in an industrial process.

## The mixed-effects model

A **mixed model** contains both fixed and random effects. For a simple random-intercept model:

$$
Y_{ij} = \mu + \beta x_{ij} + u_i + \varepsilon_{ij},
$$

where $\beta$ is a fixed effect (e.g., treatment), $u_i \sim \mathcal{N}(0, \sigma_u^2)$ is a random intercept for group $i$, and $\varepsilon_{ij} \sim \mathcal{N}(0, \sigma_\varepsilon^2)$ is the residual error.

The random effect $u_i$ captures the correlation among observations within the same group: measurements from the same subject or cluster are more similar than measurements from different groups.

## Intraclass correlation coefficient (ICC)

The **ICC** quantifies the proportion of total variance attributable to the grouping factor:

$$
\text{ICC} = \frac{\sigma_u^2}{\sigma_u^2 + \sigma_\varepsilon^2}.
$$

- ICC $\approx 0$: observations within groups are no more similar than observations between groups.
- ICC $\approx 1$: nearly all variability is between groups; within-group observations are highly correlated.

The ICC determines whether ignoring the grouping structure is safe. When ICC is substantial, standard methods that assume independence will produce incorrect standard errors and p-values.

## Variance components

In a **variance components model**, all factors are random:

$$
Y_{ijk} = \mu + a_i + b_{j(i)} + \varepsilon_{ijk},
$$

where $a_i \sim \mathcal{N}(0, \sigma_a^2)$ and $b_{j(i)} \sim \mathcal{N}(0, \sigma_b^2)$ represent nested random effects.

**Restricted maximum likelihood** (REML) is the standard estimation method for variance components. Unlike ordinary maximum likelihood, REML accounts for the loss of degrees of freedom due to estimating fixed effects, producing unbiased variance estimates.

## Applications to clustered data

Mixed models are essential when data have a **hierarchical structure**:

- Students nested within classrooms nested within schools.
- Repeated measurements nested within patients nested within hospitals.
- Cells nested within wells nested within experimental plates.

Ignoring this structure and treating all observations as independent leads to:

- Underestimated standard errors (pseudo-replication).
- Inflated type I error rates.
- Misleading p-values.

The mixed model correctly partitions variability across levels of the hierarchy, producing valid inference even with unbalanced designs and missing data.

[[simulation applied-stats-sim-6]]
