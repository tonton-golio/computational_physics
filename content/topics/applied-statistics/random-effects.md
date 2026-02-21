# Random Effects and Mixed Models

## Fixed vs Random Effects

Imagine you're studying how a new teaching method affects math scores. You test it in 10 classrooms across 5 schools. The teaching method is a **fixed effect** — you chose it deliberately, and your conclusions apply specifically to it. But the classrooms and schools? You didn't pick *those* specific ones because they're special. They're a random sample from a larger population of classrooms and schools. That's a **random effect**.

* **Fixed effect**: the specific levels are of direct interest. Inference applies only to those levels.
* **Random effect**: the levels are sampled from a population. Inference generalizes to the entire population.

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

* ICC $\approx 0$: the grouping doesn't matter. Observations within a group are no more similar than observations from different groups. You could ignore the structure and use standard methods.
* ICC $\approx 1$: nearly all variability is between groups. Within-group observations are very similar.

The ICC is your diagnostic. When it is substantial (above 0.05-0.10), ignoring the grouping structure will produce incorrect standard errors and misleading p-values. This is where mixed models become necessary, not optional.

In Alex's study, suppose the ICC is 0.15. That means 15% of the total variability in math scores comes from classroom-level differences. Ignoring that would inflate the effective sample size and make the teaching method look significant when it might not be.

## Variance Components

In a **variance components model**, all factors are random:

$$
Y_{ijk} = \mu + a_i + b_{j(i)} + \varepsilon_{ijk},
$$

where $a_i \sim \mathcal{N}(0, \sigma_a^2)$ and $b_{j(i)} \sim \mathcal{N}(0, \sigma_b^2)$ represent nested random effects. The notation $b_{j(i)}$ means the $b$ levels are nested within the $a$ levels — classrooms within schools, or wells within experimental plates.

### Estimating Variance Components with REML

How do you estimate $\sigma_a^2$, $\sigma_b^2$, and $\sigma_\varepsilon^2$? Ordinary maximum likelihood (which we used for fitting distributions in [probability density functions](./probability-density-functions)) has a bias problem here. It estimates the fixed effects first and then estimates variances from the residuals, but it doesn't account for the degrees of freedom lost in estimating those fixed effects. The result: ML systematically *underestimates* variance components — the same issue that Bessel's correction fixes for the sample variance (from [introduction and concepts](./introduction-concepts)).

**Restricted maximum likelihood** (REML) solves this. It separates the likelihood into two parts: one for the fixed effects and one for the variance components. By estimating variances only from the second part — which is free of the fixed effects — REML produces unbiased estimates. It is the standard method for mixed models.

The practical difference: for large datasets, ML and REML give nearly identical results. For small datasets (where getting the variance right matters most), REML is noticeably better.

## Applications to Clustered Data

Mixed models are essential when data have a **hierarchical structure**:

* Students nested within classrooms nested within schools.
* Repeated measurements nested within patients nested within hospitals.
* Cells nested within wells nested within experimental plates.

Ignoring this structure and treating all observations as independent leads to three problems:

* **Underestimated standard errors** (pseudo-replication). You think you have more independent information than you actually do.
* **Inflated type I error rates**. You reject $H_0$ too often.
* **Misleading p-values**. Effects look significant when they aren't.

The mixed model correctly partitions variability across levels of the hierarchy, producing valid inference even with unbalanced designs and missing data. It is the natural extension of [ANOVA](./anova) to data with grouped or clustered structure.

**A cautionary tale.** A researcher tests a new treatment on mice. There are 4 mice per cage, 5 cages per treatment. The researcher analyzes 20 mice per treatment as if they were independent. But mice in the same cage share food, temperature, and social stress — they're correlated. The real sample size is closer to 5 (cages), not 20 (mice). The "significant" treatment effect disappears when the cage is treated as a random effect. This mistake — ignoring the cage — has invalidated entire research programs.

[[simulation applied-stats-sim-6]]

Now you know how to handle grouped, hierarchical data at a single time point. But what happens when you follow the same subjects over time? The grouping gets richer — each person is a cluster of their own repeated measurements, and the time ordering carries information. That's longitudinal data, and the random effects you just learned are exactly the tool that makes it tractable. We continue this story in the next section.

## Big Ideas

* The fixed/random distinction is not about the math — it is about your scientific question. If you care about the specific levels you tested, they are fixed. If the levels are a sample from a population you want to generalize to, they are random.
* Pseudo-replication is the mistake of treating correlated observations as if they were independent. It inflates your effective sample size and makes weak effects look significant. The mice-in-cages story is a warning that has invalidated entire research programs.
* REML is not just a computational detail — it corrects for the degrees of freedom consumed by estimating fixed effects, the same way Bessel's correction fixes the sample variance.
* The ICC is the diagnostic that tells you whether the grouping structure matters: an ICC above 0.05-0.10 means you cannot safely ignore it.

## What Comes Next

You now know how to handle grouped data at a single time point — students in classrooms, patients in hospitals. The random intercept captures baseline differences between groups, and the ICC tells you how much those differences matter.

The natural extension is data where the same subjects are measured repeatedly over time. Each subject becomes their own cluster of observations, the time ordering carries information, and the correlation structure within subjects needs to be modeled explicitly. That is the territory of longitudinal data.

## Check Your Understanding

1. A researcher studies the effect of a drug on blood pressure using 5 hospitals, with 30 patients per hospital. They analyze the 150 patients as if they were independent. Explain why this is wrong, and what the actual effective sample size might be if the ICC is 0.20.
2. In what sense is REML doing the same thing as Bessel's correction? What problem is each one solving, and why does the same issue arise in both contexts?
3. You fit a random-intercept model and find that the random effect variance $\sigma_u^2 \approx 0$. What does this tell you, and what simpler model would be appropriate?

## Challenge

A researcher runs an experiment testing whether a cognitive training program improves memory scores. They recruit 8 participants, each of whom completes the program twice under different conditions (with and without feedback), yielding 16 observations total. The researcher runs a standard paired t-test treating the 16 observations as 8 independent pairs. Now consider: if the training program is run in two cohorts (4 participants each), and cohort membership introduces substantial correlation within cohorts, how does this change the analysis? Write down the appropriate mixed model, identify which effects are fixed and which are random, and explain what the ICC would tell you about whether cohort matters.
