# Hypothesis Testing and Limits


KEY: description
Week 4 (Hypothesis Testing and limits):
* Dec 12: Hypothesis testing. Simple, Chi-Square, Kolmogorov, and runs tests.
* Dec 13: More hypothesis testing, limits, and confidence intervals. Testing your random (?) numbers.
  Project should been submitted by Wednesday the 14th of December at 22:00!
* Dec 16: Table Measurement solution discussion. Simpson's paradox.


KEY: Header 1
### Hypothesis testing
*We can either one- or two-tailed test.*

KEY: One Sample Z Test
###### One Sample Z Test
Usually not used as the population standard deviation is unknown.

1. State the Null hypothesis. $H_0: μ = 100$.
1. State the Alternate Hypothesis. $H_1: μ > 100$.
1. State the alpha level. $\alpha=0.05$
1. Find the rejection region area from the z-table. An area of .05 is equal to a z-score of 1.645.
1. Calculate Z
$$
  Z = \frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}
$$
1. Iff Step 6 is greater than Step 5, reject the null hypothesis. 

KEY: Header 2
#### Students t test
t-tests are more relaxed as they assume an exact uncertainty level.

A Student's t-test is a statistical test that is used to determine if there is a significant difference between the means of two groups. It is commonly used to compare the means of a sample to a known or hypothesized population mean.

There are two types of t-tests:

A one-sample t-test, which compares the mean of a single sample to a known population mean.
A two-sample t-test, which compares the means of two independent samples.
The t-test assumes that the data being analyzed is approximately normally distributed, and that the variances of the two groups being compared are equal (this assumption is known as the "equal variances assumption").

The test statistic is calculated as the difference between the sample mean and the population mean, divided by the standard error of the mean. The resulting value is then compared to a t-distribution with the appropriate degrees of freedom.


#### Kolmogorov
The Kolmogorov-Smirnov test (K-S test) is a non-parametric statistical test that compares a sample with a reference probability distribution (one-sample K-S test), or to compare two samples (two-sample K-S test). It can be used to determine whether two samples are drawn from the same distribution. The test returns a p-value.
#### runs tests
Non-parametric statistical test that checks a randomness hypothesis for a two-valued data sequence. An example sequence is:
> $+ + + + − − − + + + − − + + + + + + − − − −$


### confidence intervals
A confidence interval is a range of values, derived from a sample of data, that is used to estimate an unknown population parameter. The interval has an associated confidence level, which quantifies the level of confidence that the parameter lies in the interval.

For example, a 95% confidence interval for a mean would indicate that if the sampling process were repeated many times, the interval calculated from each sample would contain the true mean of the population 95% of the time.

It is calculated by taking a sample of data, estimating a statistic of interest (such as the mean or proportion), and then using that estimate to calculate the range of values that is likely to contain the true population parameter. The range is calculated by taking into account the standard error of the statistic, and the level of confidence desired.

A common way to report the interval is to give the point estimate (e.g. the sample mean) along with the lower and upper bounds of the interval.

### Simpson's paradox
Simpson's paradox refers to a phenomenon in which a trend that appears in different groups of data disappears or reverses when the groups are combined. It is a type of reversal paradox in statistics.

The paradox is named after Edward H. Simpson, who first described it in a 1951 paper. The phenomenon occurs when there are multiple groups being compared, and an apparent trend in the data disappears or reverses when the groups are combined. This can happen when there are lurking variables that affect the relationship between the two variables in different groups.

For example, suppose a study finds that men have a higher average salary than women in each of several different departments within a company. However, when the data for all departments are combined, the overall average salary for men is lower than that for women. This apparent reversal of the trend can be explained by a lurking variable such as the proportion of men and women in each department, which would affect the overall average.

Simpson's paradox highlights the importance of considering all relevant factors when interpreting data and making conclusions, and can occur in various fields such as social science, medical research, education and economics.