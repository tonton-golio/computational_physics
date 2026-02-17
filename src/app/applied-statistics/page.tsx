'use client';

import React from 'react';
import { MarkdownContent } from '@/components/content/MarkdownContent';

const content = `# Applied Statistics

## Introduction, General Concepts, ChiSquare Method

* Nov 21: Introduction to course and overview of curriculum. Mean and Standard Deviation. Correlations. Significant digits. Central limit theorem. 
* Nov 22: Error propagation (which is a science!). Estimate g measurement uncertainties.
* Nov 25: ChiSquare method, evaluation, and test. Formation of Project groups.

### Mean
Mean is a metric telling us about bulk magnitude-tendency of data.

#### Geometric mean
The root of the product;
$$
     \bar{x}_\text{geo} = \left( \prod_i^n x_i\right)^{1/n}
     =
     \exp\left(\frac{1}{n}\sum_{i}^n\ln x_i \right)
$$
*is equivalent to the arithmetic mean in logscale*

#### Arithmetic mean
The equal-weight center,
$$
     \hat{\mu} = \bar{x} = \left< x \right> = \frac{1}{N}\sum_i^N x_i.
$$

#### Median
The counting center of the data

#### Mode
The most typical data point

### STD

Standard deviation is a measure of how much datapoint deviates from the dataset mean, \`np.std(x)\`.

$$
     \hat{\sigma} = \sqrt{\frac{1}{N}\sum_i^N ( \left< x \right> -x_i)^2}
$$

In estimating this, we assume that we know the real mean. But in reality we dont really know this so we use:

$$
     \hat{\sigma} \approx \tilde{\sigma} = \sqrt{
     \frac{1}{N-1}\sum_i(x_i-\bar{x})^2
     }
$$
we subtract 1, because something something degrees of freedom...

### Normal Distribution

The normal (Gaussian) distribution is fundamental in statistics.

[[simulation applied-stats-sim-2]]

### Weighted mean

How to average data which has different uncertainties and what is the uncertainty on the average?

$$
    \begin{align*}
        \hat{\mu} = \frac{\sum x_i / \sigma_i^2}{\sum 1 / \sigma_i^2}
        , &&
        \hat{\sigma_\mu} = \sqrt{\frac{1}{\sum 1/\sigma_i^2}}
    \end{align*}
$$
Uncertainty descreases with the squares of the number of sampels;
$$
    \hat{\sigma_\mu} = \hat{\mu}/\sqrt{N}.
$$

### Correlation
Correlation speaks to whether a feature varies in concordance with another.

Normalizing by the widths gives the Pearson's (linear) correlation coefficient:

$$
\begin{align*}
     \rho_{xy} = \frac{V_{xy}}{\sigma_x\sigma_y} , && \text{i.e., } -1 < \rho_{xy} < 1 
\end{align*}
$$

Do bare in mind that we may get zero, but this just tells us that the correlation is not linear, so remember to plot 😉

### Linear Regression

Linear regression is a statistical method for modeling the relationship between a dependent variable and one or more independent variables.

The simple linear regression model is:

$$
y = \beta_0 + \beta_1 x + \epsilon
$$

Where $\beta_0$ is the intercept, $\beta_1$ is the slope, and $\epsilon$ is the error term.

[[simulation applied-stats-sim-1]]

### Central limit theorem intro
*The law of large numbers*

The central limit theorem answers the question: *why do statistics in the limit of large N tend toward a Gaussian?* 

If we roll sufficiently many dice; they will naturally find a mean around 3.5;

### Central limit theorem

The mean of many such extravagent rolls, will often tend towards a gaussian. In fact, *The distribution of numbers samples from a variety of distributions, is Gaussian given that they share variance and mean.*

[[simulation applied-stats-sim-3]]

### Error propagation
If we have a function $y(x_i)$ and we know the uncertainty on $\sigma(x_i)$, how do we find $\sigma(y(x_i))$?. Well obviously it depends on the gradient of $y$ with respect to $x$.

### ChiSquare method, evaluation, and test

Your typical least squares algorithm does **not** include uncertainties, the chi-square does.

$$
     \chi^2(\theta) = \sum_i^N \frac{(y_i - f(x_i,\theta))^2}{\sigma_i^2}
$$

## Probability Density Functions

### Binomial
N trials, p chance of succes, how many successes should you expect

### Poisson
if $N\\rightarrow \\infty$ and $p\\rightarrow 0$, but $Np\\rightarrow\\lambda$ i.e., some finite number. Then a binomioal approaches a Poisson.

### Gaussian
The normal normal distribution.

## Hypothesis Testing and Limits

### Hypothesis testing
*We can either one- or two-tailed test.*

#### Students t test
t-tests are more relaxed as they assume an exact uncertainty level.

A Student's t-test is a statistical test that is used to determine if there is a significant difference between the means of two groups. It is commonly used to compare the means of a sample to a known or hypothesized population mean.

There are two types of t-tests:

A one-sample t-test, which compares the mean of a single sample to a known population mean.
A two-sample t-test, which compares the means of two independent samples.
The t-test assumes that the data being analyzed is approximately normally distributed, and that the variances of the two groups being compared are equal (this assumption is known as the "equal variances assumption").

The test statistic is calculated as the difference between the sample mean and the population mean, divided by the standard error of the mean. The resulting value is then compared to a t-distribution with the appropriate degrees of freedom.

[[simulation applied-stats-sim-4]]

### Kolmogorov
The Kolmogorov-Smirnov test (K-S test) is a non-parametric statistical test that compares a sample with a reference probability distribution (one-sample K-S test), or to compare two samples (two-sample K-S test).

### confidence intervals
A confidence interval is a range of values, derived from a sample of data, that is used to estimate an unknown population parameter.

## Simulation and More Fitting

### Producing random numbers

For producing random number, we have to main approacheds: the *transformation method* and the *accept-reject method*.

## Machine Learning and Data Analysis

Machine learning algorithms can be split into two types, *supervised* and *unsupervised*. We may solve three tasks: clustering, classification & regression.

## Advanced Fitting and Calibration

* Jan 9: Advanced fitting with both functions and models.
* Jan 10: Calibration and use of control channels.
`;

export default function AppliedStatisticsPage() {
  return (
    <div className="min-h-screen bg-[#0a0a15] text-white">
      <div className="container mx-auto px-4 py-8">
        <MarkdownContent content={content} />
      </div>
    </div>
  );
}