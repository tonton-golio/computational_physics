# Analysis of Variance (ANOVA)

Your fertilizer worked... or did it? You tested three fertilizers on three fields, and the yields look different. But fields vary naturally -- soil, sun, drainage. So the real question isn't "are the numbers different?" (they always are). It's "are they *more* different than noise alone would produce?"

## The Idea Behind ANOVA

The t-test compares two groups. But experiments often involve three, four, a dozen groups. You *could* run t-tests on every pair, but with enough comparisons, you'll find "significant" differences by sheer luck. With 10 pairwise comparisons at $\alpha = 0.05$, you have nearly a 40% chance of a false positive.

**ANOVA** solves this with a single test: do *any* group means differ? Despite its name, it works by comparing variability *between* groups to variability *within* groups. Think signal-to-noise ratio. The "signal" is how much group means spread out. The "noise" is how much individuals scatter within each group. Big signal relative to noise? Something real is going on.

## One-Way ANOVA

For $k$ groups with $n_i$ observations:

$$
Y_{ij} = \mu + \alpha_i + \varepsilon_{ij}, \qquad \varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2)
$$

Total variability decomposes:

$$
\text{SS}_{\text{total}} = \text{SS}_{\text{between}} + \text{SS}_{\text{within}}
$$

The **F-statistic** is the ratio:

$$
F = \frac{\text{SS}_{\text{between}} / (k - 1)}{\text{SS}_{\text{within}} / (N - k)} = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}
$$

[[simulation variance-decomposition]]

Under $H_0$, $F$ follows an $F(k-1, N-k)$ distribution. Large $F$ means group differences are large relative to noise.

**Assumptions** (check before trusting):

* Independence.
* Normality within each group (Q-Q plots or Shapiro-Wilk).
* Equal variances (Levene's or Bartlett's test).

## Two-Way ANOVA

Two factors simultaneously -- say, fertilizer type *and* watering schedule:

$$
Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}
$$

Three sources of variability: main effect of $A$, main effect of $B$, and the **interaction** $A \times B$. The interaction is often the most interesting finding. A fertilizer that works brilliantly with daily watering but does nothing with weekly watering -- that's an interaction. The two factors can amplify or cancel each other in ways neither alone reveals.

## Factorial Designs

A **full factorial** tests all combinations. For factors with $a$ and $b$ levels: $a \times b$ cells. Efficient because every observation contributes to every factor.

**Balanced designs** (equal cell sizes) simplify everything and make F-tests exact. Lose balance, and you must choose between Type I, II, or III sums of squares -- a subtlety that trips up many practitioners.

## Post-Hoc Tests

ANOVA says *something* differs, not *what*. To find the specific differences:

* **Tukey's HSD**: All pairwise comparisons. Go-to choice.
* **Bonferroni**: Divides $\alpha$ by number of comparisons. Conservative but simple.
* **Scheffe's method**: Arbitrary contrasts, most conservative.
* **Dunnett's test**: Each treatment vs. a single control.

All control the **family-wise error rate** -- the probability of *any* false discovery across all comparisons.

## What If Assumptions Fail? Kruskal-Wallis

The **Kruskal-Wallis test** is the non-parametric alternative. Instead of comparing means, it compares *ranks*. Robust to outliers, skew, and unequal variances. The trade-off: less power when ANOVA's assumptions are actually met. If significant, follow up with the Dunn test for pairwise comparisons.

Now you can compare groups. But what happens when data has natural grouping you can't eliminate -- students in classrooms, patients in hospitals? Ignoring that structure inflates your sample size and leads to false discoveries. Random effects models fix this next.

> **Challenge.** Explain the F-statistic using signal-to-noise. The signal is how different the group averages are; the noise is how much individuals scatter within groups. Big signal relative to noise = something real. One minute.

## Big Ideas

* ANOVA tests whether between-group variance exceeds what you'd expect from within-group variance. The name describes the method, not the question.
* Multiple t-tests inflate false positives: 10 comparisons at $\alpha = 0.05$ gives ~40% chance of a false hit.
* The interaction in two-way ANOVA often contains the most interesting science.
* Kruskal-Wallis handles non-normal data by working with ranks -- trading some power for broader applicability.

## What Comes Next

ANOVA tells you which groups differ. But what happens when your data has nested structure -- students in classrooms in schools? Treating all observations as independent inflates sample sizes and produces false discoveries. Random effects models fix this by partitioning variability across hierarchy levels.

## Check Your Understanding

1. You test four fertilizers with six pairwise t-tests. Why is this problematic? What should you do instead?
2. One-way ANOVA: $F = 2.1$, $k = 4$, $N = 40$. What degrees of freedom do you need? How would you assess significance?
3. Significant interaction between fertilizer and watering. "So the best fertilizer is whichever has the highest average yield." Why is this potentially wrong?

## Challenge

You run one-way ANOVA on three groups (20 each) and find a significant F. Tukey's HSD identifies the differing pairs. Then Levene's test gives $p = 0.02$ -- unequal variances. What does this mean for validity? What alternative would you use? Would conclusions likely change with a non-parametric test? How would unequal sample sizes make things worse?
