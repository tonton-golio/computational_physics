# Analysis of Variance (ANOVA)

Your fertilizer worked... or did it? You tested three different fertilizers on three different fields, and the yields look different. But here's the thing — fields vary naturally. Soil depth, sun exposure, drainage — all of it adds noise. So the real question isn't "are the numbers different?" (they always will be). The real question is: "are the numbers *more* different than you'd expect from noise alone?"

## The Idea Behind ANOVA

The t-test from the previous section compares two groups. But experiments often involve three, four, or a dozen groups. You *could* run t-tests on every pair, but this creates a dangerous multiple-testing problem — with enough comparisons, you will find "significant" differences by sheer chance.

**Analysis of variance** solves this with a single test: do *any* of the group means differ? Despite its name, ANOVA works by comparing variability *between* groups to variability *within* groups. The key insight: if group means differ substantially, the between-group variance will be large relative to the within-group variance. If all groups come from the same population, the two should be similar.

Think of it as a signal-to-noise ratio. The "signal" is how much the group means spread out. The "noise" is how much individual measurements scatter within each group. If the signal is big compared to the noise, something real is going on.

## One-Way ANOVA

For $k$ groups with $n_i$ observations each, the model is:

$$
Y_{ij} = \mu + \alpha_i + \varepsilon_{ij}, \qquad \varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2),
$$

where $\mu$ is the grand mean, $\alpha_i$ is the effect of group $i$, and $\varepsilon_{ij}$ is random error.

The total variability decomposes cleanly:

$$
\text{SS}_{\text{total}} = \text{SS}_{\text{between}} + \text{SS}_{\text{within}}.
$$

This decomposition is the heart of ANOVA — it splits the total variation into the part explained by group differences and the part left over as noise.

The **F-statistic** is the ratio:

$$
F = \frac{\text{SS}_{\text{between}} / (k - 1)}{\text{SS}_{\text{within}} / (N - k)} = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}.
$$

Under $H_0: \alpha_1 = \cdots = \alpha_k = 0$, this follows an $F(k-1, N-k)$ distribution. A large $F$ means the group differences are large relative to the noise — evidence that at least one group is different.

**Assumptions** (check these before trusting the result):

- Independence of observations.
- Normality within each group (check with Q-Q plots or the Shapiro-Wilk test).
- Homogeneity of variances (**homoscedasticity**); checked with Levene's test or Bartlett's test.

## Two-Way ANOVA

What if you're varying two factors simultaneously — say, fertilizer type *and* watering schedule? Two-way ANOVA handles this:

$$
Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}.
$$

This decomposes variability into three sources:

- Main effect of $A$: do levels of factor $A$ differ on average?
- Main effect of $B$: do levels of factor $B$ differ on average?
- **Interaction** $A \times B$: does the effect of $A$ depend on the level of $B$?

The interaction term is often the most scientifically interesting finding. A fertilizer might work brilliantly with daily watering but do nothing with weekly watering — that's an interaction. You might think the two factors work independently — but actually, they can amplify or cancel each other in ways neither factor alone would reveal.

## Factorial Designs

A **full factorial design** tests all combinations of factor levels. For factors with $a$ and $b$ levels, there are $a \times b$ treatment combinations. Factorial designs are efficient because every observation contributes information about every factor.

**Balanced designs** (equal sample sizes per cell) simplify the analysis and make the F-tests exact. When balance is lost (due to missing data or unequal groups), the sums of squares are no longer orthogonal, and you need to choose between Type I, II, or III sums of squares — a subtlety that trips up many practitioners.

## Post-Hoc Tests

ANOVA tells you *that* at least one group differs, but not *which* ones. To find the specific differences, you use post-hoc (after-the-fact) comparisons. Here's the menu, ordered from most to least commonly used:

- **Tukey's HSD**: Compares all pairs of group means. The go-to choice for pairwise comparisons. Controls the simultaneous confidence level.
- **Bonferroni correction**: Divides $\alpha$ by the number of comparisons. Conservative but general-purpose. Simple to explain and implement.
- **Scheffé's method**: Allows arbitrary contrasts (not just pairwise). Most conservative. Use it when you want to test combinations like "are groups 1 and 2 together different from group 3?"
- **Dunnett's test**: Compares each treatment to a single control group. Perfect when you have one reference condition.

All of these control the **family-wise error rate** — the probability of making *any* false discovery across all comparisons. Without this control, the more groups you compare, the more likely you are to find something "significant" that isn't real.

## What If the Assumptions Fail? Kruskal-Wallis

ANOVA relies on normality and equal variances. But real data is often skewed, heavy-tailed, or heteroscedastic. What then?

The **Kruskal-Wallis test** is a non-parametric alternative to one-way ANOVA. Instead of comparing means, it compares **ranks**: all observations are ranked together, and the test asks whether the mean ranks differ across groups.

$$
H = \frac{12}{N(N+1)} \sum_{i=1}^{k} n_i (\bar{R}_i - \bar{R})^2,
$$

where $\bar{R}_i$ is the mean rank in group $i$. Under $H_0$, $H \sim \chi^2(k-1)$ approximately.

Because it works with ranks rather than raw values, the Kruskal-Wallis test is robust to outliers, skewness, and non-constant variance. The trade-off is reduced power when the ANOVA assumptions *are* met — you pay a price for making fewer assumptions. If the Kruskal-Wallis test is significant, pairwise comparisons can be performed with the **Dunn test** (with Bonferroni correction).

Now you can compare groups and detect real differences. But all of this analysis is only as good as the data that went into it. A well-designed experiment with 50 observations can beat a sloppy one with 5000. That's the power of experimental design — and it's where we go next.

> **Challenge.** Explain the F-statistic to a friend using the signal-to-noise analogy. The "signal" is how different the group averages are; the "noise" is how much individuals scatter within each group. If the signal is big compared to the noise, something real is happening. One minute.
