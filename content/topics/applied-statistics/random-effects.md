# Random Effects and Mixed Models

## Fixed vs Random Effects

You're studying a new teaching method across 10 classrooms in 5 schools. The teaching method is a **fixed effect** -- you chose it, and your conclusions apply to it. But the classrooms and schools? You didn't pick them because they're special. They're a sample from a larger population. That's a **random effect**.

* **Fixed effect**: specific levels are of direct interest.
* **Random effect**: levels are sampled from a population. You estimate the *variance* across levels.

If you test a drug in several hospitals, the drug is fixed (you chose it) but the hospital is random (each is one sample from many). Different hospitals have different patients, equipment, practices -- variability that's not about the drug but about the context.

## The Mice-in-Cages Story

Here's a cautionary tale that has invalidated entire research programs. A researcher tests a treatment on mice. Four mice per cage, 5 cages per treatment. The researcher analyzes 20 mice per treatment as if they're independent. But mice in the same cage share food, temperature, social stress -- they're correlated. The real sample size is closer to 5 (cages), not 20 (mice). The "significant" treatment effect vanishes when cage is treated as a random effect.

This is **pseudo-replication** -- treating correlated observations as independent. It makes weak effects look real. And it's one of the most common mistakes in applied statistics.

The solution is a **mixed model**.

## The Mixed-Effects Model

A mixed model contains both fixed and random effects. The simplest version:

$$
Y_{ij} = \mu + \beta x_{ij} + u_i + \varepsilon_{ij},
$$

where $\beta$ is a fixed effect (the treatment), $u_i \sim \mathcal{N}(0, \sigma_u^2)$ is a random intercept for group $i$ (each group starts at a different baseline), and $\varepsilon_{ij} \sim \mathcal{N}(0, \sigma_\varepsilon^2)$ is residual error.

The random effect captures what makes each group different. It's like accounting for family traits in a height study -- members of the same family share genetic and environmental factors.

## Intraclass Correlation Coefficient (ICC)

How similar are observations within a group compared to across groups?

$$
\text{ICC} = \frac{\sigma_u^2}{\sigma_u^2 + \sigma_\varepsilon^2}
$$

* ICC $\approx 0$: grouping doesn't matter. Standard methods are fine.
* ICC $\approx 1$: nearly all variability is between groups.

When the ICC is above 0.05-0.10, ignoring the grouping produces incorrect standard errors and misleading p-values. Mixed models become necessary, not optional.

## Variance Components

When all factors are random:

$$
Y_{ijk} = \mu + a_i + b_{j(i)} + \varepsilon_{ijk},
$$

where $a_i \sim \mathcal{N}(0, \sigma_a^2)$ and $b_{j(i)} \sim \mathcal{N}(0, \sigma_b^2)$ are nested random effects.

### Estimating with REML

Ordinary maximum likelihood underestimates variance components -- the same issue Bessel's correction fixes for the sample variance. **REML** (Restricted Maximum Likelihood) is just Bessel's correction for mixed models. It separates the likelihood into two parts: one for fixed effects, one for variances. By estimating variances only from the second part, REML produces unbiased estimates.

For large datasets, ML and REML are nearly identical. For small datasets (where getting the variance right matters most), REML is noticeably better.

## Applications to Clustered Data

Mixed models are essential when data is hierarchical:

* Students in classrooms in schools.
* Patients in hospitals.
* Cells in wells in plates.

Ignoring the structure leads to underestimated standard errors, inflated type I error, and misleading p-values. The mixed model correctly partitions variability across hierarchy levels.

[[simulation applied-stats-sim-6]]

Now you know how to handle grouped data at a single time point. But what happens when you follow the same subjects over time? Each person becomes a cluster of their own repeated measurements, and time ordering carries information. That's longitudinal data, and random effects are exactly what makes it tractable.

## Big Ideas

* Fixed vs. random is about the science, not the math. Care about specific levels? Fixed. Levels sampled from a population? Random.
* Pseudo-replication inflates your sample size and makes weak effects look real. The mice-in-cages mistake has killed entire research programs.
* REML is Bessel's correction for mixed models -- fixing the bias that plain ML introduces.
* ICC above 0.05-0.10 means you can't safely ignore grouping structure.

## What Comes Next

You can now handle grouped data at a single time point. The natural extension: data where the same subjects are measured repeatedly over time. Each subject is their own cluster, time ordering carries information, and the within-subject correlation needs explicit modeling. That's longitudinal data.

## Check Your Understanding

1. A drug study uses 5 hospitals, 30 patients each. The 150 patients are analyzed as independent. Why is this wrong? What's the effective sample size if ICC = 0.20?
2. In what sense is REML doing the same thing as Bessel's correction?
3. You fit a random-intercept model and find $\sigma_u^2 \approx 0$. What does this tell you?

## Challenge

A cognitive training study: 8 participants, each completing the program twice (with and without feedback), 16 observations total. The researcher runs a paired t-test. But if the program runs in two cohorts (4 participants each) with substantial within-cohort correlation, how does this change things? Write the mixed model, identify fixed and random effects, and explain what the ICC would tell you.
