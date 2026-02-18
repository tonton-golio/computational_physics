# Hypothesis Testing and Limits

## The Logic of Hypothesis Testing

Think of hypothesis testing as a courtroom trial. The defendant (your null hypothesis) is presumed innocent until proven guilty. You gather evidence (data), and if the evidence is overwhelming enough, you reject the presumption of innocence. Notice the asymmetry: you never *prove* innocence — you either find enough evidence to convict, or you don't.

The **null hypothesis** $H_0$ is the boring explanation — nothing interesting is happening, there is no effect, the drug doesn't work. The **alternative hypothesis** $H_1$ says something real is going on. Your job is to ask: if $H_0$ were true, how surprising would my data be?

The answer is the **p-value**. And here is where most people get confused, so let's be precise.

The p-value is *not* the probability that your hypothesis is true. It's the probability you'd see data this extreme *if your hypothesis were true*. That's a huge difference — like the difference between "the suspect is probably guilty" and "an innocent person would almost never look this suspicious." The first is a statement about the suspect. The second is a statement about innocent people. They sound similar. They are not the same thing.

A small p-value means the data would be very surprising under $H_0$, so you reject it. The threshold for "surprising enough" is the **significance level** $\alpha$ (typically 0.05). You can perform **one-tailed** tests ("is the effect in *this* direction?") or **two-tailed** tests ("is there *any* effect at all?").

## The Testing Toolkit

With the logic in place, here's a toolkit of specific tests, each suited to different situations.

### One-Sample Z Test

The simplest test, but rarely used in practice because it requires knowing the population standard deviation — which you almost never do. Still, it illustrates the general framework clearly:

1. State the null hypothesis: $H_0: \mu = 100$.
2. State the alternative: $H_1: \mu > 100$.
3. Choose a significance level: $\alpha = 0.05$.
4. Find the rejection region. An area of 0.05 corresponds to $z = 1.645$.
5. Calculate:

$$
Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}
$$

6. If $Z$ exceeds the critical value, reject $H_0$.

Every test in this section follows this same skeleton. Learn it once, and you can apply it everywhere.

### Student's t-Test

In practice, you estimate the standard deviation from the data itself, which introduces extra uncertainty. The **t-test** accounts for this by using the heavier-tailed $t$-distribution (from lesson 2) instead of the Gaussian.

There are two flavors:

- **One-sample t-test**: Is this batch of lightbulbs lasting the claimed 1000 hours? Compares the mean of a sample to a known value.
- **Two-sample t-test**: Do patients on drug A recover faster than patients on drug B? Compares the means of two independent groups.

The test assumes approximately normal data and (for the two-sample version) equal variances. The test statistic is the difference between the sample mean and the hypothesized mean, divided by the standard error, compared against a $t$-distribution with the appropriate degrees of freedom.

[[simulation applied-stats-sim-4]]

### Non-Parametric Tests

What if your data is clearly not normal? Non-parametric tests make fewer assumptions about the underlying distribution. When the assumptions fail, you stop trusting means and start trusting ranks — nature doesn't care about your normality test.

The **Kolmogorov-Smirnov test** (K-S test) compares a sample with a reference distribution (one-sample) or compares two samples (two-sample). It works by measuring the maximum distance between the cumulative distribution functions.

The **Runs test** checks whether a sequence of two-valued data is random. Given a sequence like:

> $+ + + + - - - + + + - - + + + + + + - - - -$

it counts the number of "runs" (consecutive sequences of the same value). Too few runs suggest clustering; too many suggest alternation. Either way, the sequence is not random.

## Comparing Models: The Likelihood Ratio Test

The tests above compare data to a single hypothesis. But often you want to compare two competing models: does adding an extra parameter significantly improve the fit? This is where the **likelihood ratio test** connects hypothesis testing to the chi-square framework from lesson 5.

The test statistic is:

$$
d = -2\ln\frac{\mathcal{L}_\text{null}}{\mathcal{L}_\text{alt}} = -2\ln(\mathcal{L}_\text{null}) + 2\ln(\mathcal{L}_\text{alt})
$$

Under $H_0$, this follows a $\chi^2$ distribution with degrees of freedom equal to the difference in the number of parameters. A model with more parameters will *always* fit better (or at least as well), so the test asks: does the improvement justify the added complexity?

## Confidence Intervals: Ranges of Belief

A p-value gives a binary answer: reject or not. A **confidence interval** gives something richer — a range of plausible values for the parameter.

A 95% confidence interval means: if you repeated the entire experiment many times, 95% of the intervals you compute would contain the true parameter. It is *not* a statement about the probability that the true value lies in this particular interval — that's the Bayesian interpretation, which we'll see in lesson 11.

The interval is constructed from the point estimate, the standard error, and the desired confidence level. Wider intervals give more confidence but less precision — there is always a trade-off.

Confidence intervals connect directly to hypothesis testing: if a hypothesized value falls outside the 95% confidence interval, you would reject it at the 5% significance level.

## A Warning: Simpson's Paradox

Before moving on, here is a cautionary tale about what can go wrong when you test hypotheses without considering the full picture.

**Simpson's paradox** occurs when a trend that appears in separate groups reverses when the groups are combined. It isn't just a mathematical curiosity — it has fooled real researchers and real policymakers.

A famous example: in 1973, UC Berkeley was accused of gender bias in graduate admissions. The overall admission rate was significantly higher for men than for women. Case closed? Not quite. When researchers looked department by department, women were admitted at *equal or higher* rates in most departments. The paradox arose because women disproportionately applied to the most competitive departments (with low admission rates for everyone), while men applied to less competitive ones.

The lesson is fundamental: aggregated data can tell a completely different story than disaggregated data. Whenever you run a hypothesis test, ask yourself whether a lurking variable — one you haven't accounted for — could be confounding the results. Simpson's paradox is the strongest argument for careful experimental design, which is exactly where we go next. Your fertilizer worked... or did it? Three fields look different — but is it real or just soil noise? That's ANOVA.

---

**What we just learned, and why it matters.** Hypothesis testing gives you a formal framework for asking "is this real?" The p-value quantifies surprise under the null hypothesis. Confidence intervals give you a range of plausible values. The likelihood ratio test lets you compare competing models. And Simpson's paradox reminds you that the answer can depend entirely on how you slice the data. With these tools, you can now ask rigorous questions about your data — but so far, only for one or two groups at a time. ANOVA, up next, handles the case where you have many groups to compare simultaneously.
