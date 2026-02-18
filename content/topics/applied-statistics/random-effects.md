# Random Effects and Mixed Models

## Fixed vs Random Effects

Imagine you're studying how a new teaching method affects math scores. You test it in 10 classrooms across 5 schools. The teaching method is a **fixed effect** — you chose it deliberately, and your conclusions apply specifically to it. But the classrooms and schools? You didn't pick *those* specific ones because they're special. They're a random sample from a larger population of classrooms and schools. That's a **random effect**.

- **Fixed effect**: the specific levels are of direct interest. Inference applies only to those levels.
- **Random effect**: the levels are sampled from a population. Inference generalizes to the entire population.

The distinction matters because it changes what you're estimating. With fixed effects, you estimate individual level means. With random effects, you estimate the *variance* across levels — how much variability exists in the population.

Think of it this way: if you test a drug in several hospitals, the drug effect is fixed (you chose that drug) but the hospital effect is random (each hospital is one sample from a population of hospitals). Different hospitals have different patient populations, equipment, and practices — this creates variability that is not about the drug itself but about the context. You need to account for it without treating each hospital as a separate, deliberate choice.

## A Running Example: Classrooms Within Schools

Let's follow a concrete story. Alex wants to know whether the new teaching method improves math scores. Alex collects data from students in classrooms nested within schools. Here's the problem: students in the same classroom tend to have more similar scores than students in different classrooms. Maybe one teacher is more experienced. Maybe one school has more resources. These similarities mean the observations are *not independent*.

If Alex ignores this grouping and runs a simple t-test, treating each student as an independent observation, the standard errors will be too small and the p-values will be too optimistic. Alex will "discover" effects that aren't real. This is **pseudo-replication** — one of the most common mistakes in applied statistics.

The solution is a **mixed model**.

## The Mixed-Effects Model

A **mixed model** contains both fixed and random effects. The simplest version is a random-intercept model:

$$
Y_{ij} = \mu + \beta x_{ij} + u_i + \varepsilon_{ij},
$$

where $\beta$ is a fixed effect (the teaching method), $u_i \sim \mathcal{N}(0, \sigma_u^2)$ is a random intercept for group $i$ (each classroom starts at a different baseline level), and $\varepsilon_{ij} \sim \mathcal{N}(0, \sigma_\varepsilon^2)$ is the residual error.

The random effect $u_i$ captures what makes each group different. In Alex's study, $u_i$ captures the fact that some classrooms have systematically higher scores — perhaps due to the teacher, the student demographics, or the time of day the class meets. It's like accounting for family traits in a height study: members of the same family share genetic and environmental factors that make them more similar to each other than to random strangers.

## Intraclass Correlation Coefficient (ICC)

How similar are observations within the same group compared to observations from different groups? The **ICC** quantifies this:

$$
\text{ICC} = \frac{\sigma_u^2}{\sigma_u^2 + \sigma_\varepsilon^2}.
$$

- ICC $\approx 0$: the grouping doesn't matter. Observations within a group are no more similar than observations from different groups. You could ignore the structure and use standard methods.
- ICC $\approx 1$: nearly all variability is between groups. Within-group observations are very similar.

The ICC is your diagnostic. When it is substantial (above 0.05-0.10), ignoring the grouping structure will produce incorrect standard errors and misleading p-values. This is where mixed models become necessary, not optional.

In Alex's study, suppose the ICC is 0.15. That means 15% of the total variability in math scores comes from classroom-level differences. Ignoring that would inflate the effective sample size and make the teaching method look significant when it might not be.

## Variance Components

In a **variance components model**, all factors are random:

$$
Y_{ijk} = \mu + a_i + b_{j(i)} + \varepsilon_{ijk},
$$

where $a_i \sim \mathcal{N}(0, \sigma_a^2)$ and $b_{j(i)} \sim \mathcal{N}(0, \sigma_b^2)$ represent nested random effects. The notation $b_{j(i)}$ means the $b$ levels are nested within the $a$ levels — classrooms within schools, or wells within experimental plates.

### Estimating Variance Components with REML

How do you estimate $\sigma_a^2$, $\sigma_b^2$, and $\sigma_\varepsilon^2$? Ordinary maximum likelihood (which we used for fitting distributions in lesson 2) has a bias problem here. It estimates the fixed effects first and then estimates variances from the residuals, but it doesn't account for the degrees of freedom lost in estimating those fixed effects. The result: ML systematically *underestimates* variance components — the same issue that Bessel's correction fixes for the sample variance (lesson 1).

**Restricted maximum likelihood** (REML) solves this. It separates the likelihood into two parts: one for the fixed effects and one for the variance components. By estimating variances only from the second part — which is free of the fixed effects — REML produces unbiased estimates. It is the standard method for mixed models.

The practical difference: for large datasets, ML and REML give nearly identical results. For small datasets (where getting the variance right matters most), REML is noticeably better.

## Applications to Clustered Data

Mixed models are essential when data have a **hierarchical structure**:

- Students nested within classrooms nested within schools.
- Repeated measurements nested within patients nested within hospitals.
- Cells nested within wells nested within experimental plates.

Ignoring this structure and treating all observations as independent leads to three problems:

- **Underestimated standard errors** (pseudo-replication). You think you have more independent information than you actually do.
- **Inflated type I error rates**. You reject $H_0$ too often.
- **Misleading p-values**. Effects look significant when they aren't.

The mixed model correctly partitions variability across levels of the hierarchy, producing valid inference even with unbalanced designs and missing data. It is the natural extension of ANOVA (lesson 7) to data with grouped or clustered structure.

> **Cautionary tale.** A researcher tests a new treatment on mice. There are 4 mice per cage, 5 cages per treatment. The researcher analyzes 20 mice per treatment as if they were independent. But mice in the same cage share food, temperature, and social stress — they're correlated. The real sample size is closer to 5 (cages), not 20 (mice). The "significant" treatment effect disappears when the cage is treated as a random effect. This mistake — ignoring the cage — has invalidated entire research programs.

[[simulation applied-stats-sim-6]]

Now you know how to handle grouped, hierarchical data at a single time point. But what happens when you follow the same subjects over time? The grouping gets richer — each person is a cluster of their own repeated measurements, and the time ordering carries information. That's longitudinal data, and the random effects you just learned are exactly the tool that makes it tractable. We continue this story in the next section.

---

**What we just learned, and why it matters.** When data have natural groupings — students in classrooms, patients in hospitals — ignoring the structure leads to false conclusions. Mixed models handle this by separating the variability due to individual groups (random effects) from the variability you care about (fixed effects). The ICC tells you whether the grouping matters. REML gives you unbiased variance estimates. And the whole framework generalizes ANOVA to the messy, hierarchical data structures you'll encounter in real research.
