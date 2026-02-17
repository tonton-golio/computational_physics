# Analysis of Variance (ANOVA)

## The idea behind ANOVA

**Analysis of variance** tests whether the means of three or more groups differ significantly. Despite its name, ANOVA works by comparing variability *between* groups to variability *within* groups.

The key insight: if group means differ substantially, the between-group variance will be large relative to the within-group variance.

## One-way ANOVA

For $k$ groups with $n_i$ observations each, the model is:

$$
Y_{ij} = \mu + \alpha_i + \varepsilon_{ij}, \qquad \varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2),
$$

where $\mu$ is the grand mean, $\alpha_i$ is the effect of group $i$, and $\varepsilon_{ij}$ is random error.

The total sum of squares decomposes as:

$$
\text{SS}_{\text{total}} = \text{SS}_{\text{between}} + \text{SS}_{\text{within}}.
$$

The **F-statistic** is:

$$
F = \frac{\text{SS}_{\text{between}} / (k - 1)}{\text{SS}_{\text{within}} / (N - k)} = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}.
$$

Under $H_0: \alpha_1 = \cdots = \alpha_k = 0$, the statistic follows an $F(k-1, N-k)$ distribution.

**Assumptions**:

- Independence of observations.
- Normality within each group.
- Homogeneity of variances (**homoscedasticity**); checked with Levene's test or Bartlett's test.

[[simulation applied-stats-sim-1]]

## Two-way ANOVA

When two factors ($A$ and $B$) are varied simultaneously:

$$
Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}.
$$

This decomposes variability into:

- Main effect of $A$: do levels of factor $A$ differ on average?
- Main effect of $B$: do levels of factor $B$ differ on average?
- **Interaction** $A \times B$: does the effect of $A$ depend on the level of $B$?

Interaction effects are often the most scientifically interesting finding.

## Factorial designs

A **full factorial design** tests all combinations of factor levels. For factors with $a$ and $b$ levels, there are $a \times b$ treatment combinations. Factorial designs are efficient because every observation contributes information about every factor.

**Balanced designs** (equal sample sizes per cell) simplify the analysis and make the F-tests exact.

## Post-hoc tests

When ANOVA rejects $H_0$, we need to determine *which* groups differ. Multiple comparison procedures control the family-wise error rate:

- **Tukey's HSD**: compares all pairs of group means; controls the simultaneous confidence level.
- **Bonferroni correction**: divides $\alpha$ by the number of comparisons; conservative but general.
- **Scheffé's method**: allows arbitrary contrasts; most conservative.
- **Dunnett's test**: compares each treatment to a single control group.

## Non-parametric alternative: Kruskal-Wallis

When normality or homoscedasticity assumptions are violated, the **Kruskal-Wallis test** provides a non-parametric alternative to one-way ANOVA. It tests whether group medians differ by ranking all observations and comparing mean ranks across groups.

$$
H = \frac{12}{N(N+1)} \sum_{i=1}^{k} n_i (\bar{R}_i - \bar{R})^2,
$$

where $\bar{R}_i$ is the mean rank in group $i$. Under $H_0$, $H \sim \chi^2(k-1)$ approximately.

If significant, pairwise comparisons can be performed with the **Dunn test** (with Bonferroni correction).

[[simulation applied-stats-sim-2]]
