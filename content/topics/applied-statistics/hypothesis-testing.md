# Hypothesis Testing and Limits

## The Logic of Hypothesis Testing

Think courtroom trial. The defendant (your null hypothesis) is presumed innocent until proven guilty. You gather evidence, and if it's overwhelming enough, you convict. Notice the asymmetry: you never *prove* innocence -- you either find enough evidence to convict, or you don't.

The **null hypothesis** $H_0$ is the boring explanation -- nothing happening, no effect, drug doesn't work. The **alternative** $H_1$ says something real is going on.

The **p-value** is the probability you'd see data this extreme *if the null were true*. Not the probability the null is true. Huge difference. It's like asking: "how surprising would an innocent person look if they had this much evidence against them?" Not "is this person guilty?"

A small p-value means the data would be shocking under $H_0$, so you reject it. The threshold is the **significance level** $\alpha$ (typically 0.05).

## The Testing Toolkit

### One-Sample Z Test

Rarely used in practice (requires knowing the population $\sigma$), but it illustrates the skeleton every test follows:

1. State $H_0$: $\mu = 100$.
2. State $H_1$: $\mu > 100$.
3. Choose $\alpha = 0.05$.
4. Compute:

$$
Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}
$$

5. If $Z$ exceeds the critical value, reject $H_0$.

Learn this skeleton once. Apply it everywhere.

### Student's t-Test

In practice, you estimate $\sigma$ from the data, which introduces extra uncertainty. The **t-test** uses the heavier-tailed $t$-distribution instead of the Gaussian.

Two flavors:

* **One-sample**: Is this batch of lightbulbs lasting the claimed 1000 hours?
* **Two-sample**: Do patients on drug A recover faster than drug B?

[[simulation applied-stats-sim-4]]

### Non-Parametric Tests

What if your data clearly aren't normal? Stop trusting means, start trusting ranks.

The **Kolmogorov-Smirnov test** compares a sample with a reference distribution by measuring the maximum distance between CDFs.

The **Runs test** checks whether a sequence is random. Given $+ + + + - - - + + + - - + + + + + + - - - -$, it counts "runs" of the same value. Too few runs = clustering. Too many = alternation. Either way, not random.

[[simulation residual-pattern]]

## Comparing Models: The Likelihood Ratio Test

Often you want to compare two models: does adding a parameter significantly improve the fit?

$$
d = -2\ln\frac{\mathcal{L}_\text{null}}{\mathcal{L}_\text{alt}} = -2\ln(\mathcal{L}_\text{null}) + 2\ln(\mathcal{L}_\text{alt})
$$

Under $H_0$, this follows a $\chi^2$ distribution with degrees of freedom equal to the difference in parameters. More parameters *always* fit better, so the test asks: is the improvement worth the added complexity?

## Confidence Intervals

A p-value gives binary: reject or not. A **confidence interval** gives something richer -- a range of plausible values.

A 95% confidence interval means: if you repeated the entire experiment many times, 95% of the intervals would contain the true value. It's *not* the probability that the truth lies in this particular interval -- that's the Bayesian interpretation (coming in [Bayesian statistics](./bayesian-statistics)).

The interval connects to testing: if a hypothesized value falls outside the 95% CI, you'd reject it at the 5% level.

## A Warning: Simpson's Paradox

Here's how smart people get fooled. In 1973, UC Berkeley was accused of gender bias -- overall admission rates were higher for men. But department by department, women were admitted at *equal or higher* rates. The paradox: women disproportionately applied to the most competitive departments.

The lesson: aggregated data can reverse the true trend. Whenever you test a hypothesis, ask whether a lurking variable could be confounding the results. This is the strongest argument for careful experimental design, which is next.

## Big Ideas

* A p-value is the probability of data this extreme if $H_0$ were true -- not the probability $H_0$ is true. Confusing them is one of the most common errors in published science.
* The t-test honestly admits you estimated $\sigma$ from the data; the z-test pretends you knew it.
* A confidence interval and a hypothesis test are the same thing in disguise.
* Simpson's paradox is a warning: aggregated data can reverse the real trend.

## What Comes Next

You now have a toolkit for testing hypotheses. But every test assumes the data was collected properly. Before analyzing multi-group experiments, you need to learn how to *collect* data -- randomize, replicate, block, and compute sample sizes so your tests actually answer the question you're asking.

## Check Your Understanding

1. You get $p = 0.04$. A colleague says "4% chance the null is true." Why is this wrong, and what's the correct interpretation?
2. You build a 95% CI from 12.3 to 18.7. "So the true value is almost certainly in there?" How do you answer correctly?
3. A study combining two hospitals finds Drug A beats Drug B. Broken down by severity, Drug B wins in both categories. What's happening?

## Challenge

You test a new fertilizer on 30 plots (15 treatment, 15 control). Mean yield difference: 8 kg/plot, pooled SD: 15 kg/plot. Set up the two-sample t-test, compute the t-statistic, and check significance at $\alpha = 0.05$. Then: you learn the 30 plots were split across two farms with very different soil. How might this affect your conclusion? What design feature would fix this?
