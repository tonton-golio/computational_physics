# Hypothesis Testing and Limits

## Hypothesis Testing

We can perform either **one-tailed** or **two-tailed** tests depending on the alternative hypothesis.

### One-Sample Z Test

Usually not used in practice since the population standard deviation is typically unknown.

1. State the null hypothesis: $H_0: \mu = 100$.
2. State the alternative hypothesis: $H_1: \mu > 100$.
3. State the significance level: $\alpha = 0.05$.
4. Find the rejection region from the z-table. An area of 0.05 corresponds to a z-score of 1.645.
5. Calculate the test statistic:

$$
Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}
$$

6. If $Z$ exceeds the critical value from step 4, reject the null hypothesis.

### Student's t-Test

The **t-test** is more commonly used because it does not require knowing the population standard deviation.

A Student's t-test determines if there is a significant difference between the means of two groups. There are two types:

- **One-sample t-test**: Compares the mean of a single sample to a known or hypothesized population mean.
- **Two-sample t-test**: Compares the means of two independent samples.

The t-test assumes the data is approximately normally distributed and (for the two-sample version) that the variances of the two groups are equal. The test statistic is the difference between the sample mean and the population mean, divided by the standard error. This is compared to a $t$-distribution with the appropriate degrees of freedom.

[[simulation applied-stats-sim-4]]

### Kolmogorov-Smirnov Test

The **Kolmogorov-Smirnov test** (K-S test) is a non-parametric test that compares a sample with a reference probability distribution (one-sample), or compares two samples (two-sample). It returns a p-value indicating whether the samples are drawn from the same distribution.

### Runs Test

A non-parametric test that checks the randomness hypothesis for a two-valued data sequence. An example sequence is:

> $+ + + + - - - + + + - - + + + + + + - - - -$

## Confidence Intervals

A **confidence interval** is a range of values derived from sample data, used to estimate an unknown population parameter. The interval has an associated confidence level quantifying the probability that the true parameter lies within the interval.

For example, a 95% confidence interval means that if the sampling process were repeated many times, 95% of the calculated intervals would contain the true population parameter.

The interval is calculated from the point estimate (e.g., sample mean), the standard error, and the desired confidence level.

## Simpson's Paradox

**Simpson's paradox** occurs when a trend that appears in different groups of data disappears or reverses when the groups are combined. This happens when a lurking variable affects the relationship between the variables differently across groups.

For example, a study might find that men have a higher average salary than women in each department within a company. However, when all departments are combined, the overall average salary for women may be higher. This reversal is explained by the proportion of men and women in each department.

Simpson's paradox highlights the importance of considering all relevant factors when interpreting data.
