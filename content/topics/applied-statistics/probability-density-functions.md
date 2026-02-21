# Probability Density Functions

In the previous section you summarized data that was already in hand — means, spreads, correlations. Now you ask a deeper question: what *process* generated that data? If you can describe the underlying mechanism with a mathematical function, you unlock the ability to predict, extrapolate, and quantify how uncertain you really are. That mathematical function is the **probability density function**.

## What is a PDF?

A **probability density function** describes a continuous random variable. The PDF itself is not a probability — it is a *density*. Think of it like mass density: the density of steel tells you how heavy a chunk will be, but the density *itself* isn't a mass. You need to integrate over a region to get an actual probability:

$$
P(a \leq X \leq b) = \int_{a}^{b} f(x) \, dx
$$

The **cumulative distribution function** (CDF) accumulates probability from $-\infty$:

$$
F_X(x) = \int_{-\infty}^{x} f(x') \, dx'.
$$

The CDF is always non-decreasing, starts at 0, and ends at 1. It answers a simple question: what fraction of outcomes fall below $x$?

## Common Distributions

Different physical processes give rise to different distributions. Here are the ones you'll encounter most often, organized from discrete counting processes to the continuous workhorse of statistics. Each one arises naturally from a specific kind of situation — learn the situation, and the distribution follows.

### Binomial

Suppose you flip a coin $N$ times, each with probability $p$ of landing heads. How many heads should you expect? The binomial distribution answers this:

$$
\begin{aligned}
f(n;N,p) &= \frac{N!}{n!(N-n)!}p^n(1-p)^{N-n}\\
\langle f(n;N,p)\rangle &= Np \\
\sigma^2 &= Np(1-p)
\end{aligned}
$$

This is a **discrete** distribution — it counts whole events (0 heads, 1 head, 2 heads, ...). Any time you have a fixed number of independent yes/no trials, each with the same probability, the binomial is your distribution.

### Poisson

Now imagine you have a huge number of trials ($N\rightarrow \infty$), each with a tiny probability of success ($p\rightarrow 0$), but the *expected count* stays finite ($Np\rightarrow\lambda$). The binomial simplifies to the **Poisson** distribution:

$$
f(n, \lambda) = \frac{\lambda^n}{n!}e^{-\lambda}
$$

**Example**: A large number of people go into traffic every day ($N\rightarrow\infty$), the probability of any one person being killed is tiny ($p\rightarrow 0$), but some number of fatalities occur every year ($\lambda\neq 0$).

The Poisson distribution has a remarkable property: its mean and variance are both equal to $\lambda$. This means the error on a Poisson count is simply the square root of that count — a fact you'll use constantly when working with histograms. If a histogram bin contains $N$ events, its statistical uncertainty is $\sqrt{N}$, provided the count is large enough ($N \gtrsim 5{-}20$) for the Gaussian approximation to hold.

The sum of independent Poissons is itself a Poisson: $\lambda = \lambda_a + \lambda_b$. And when $\lambda$ is large ($\lambda \gtrsim 20$), the Poisson approaches a Gaussian. Nature keeps converging on that bell curve.

### Gaussian

The **normal distribution** is the central character of statistics:

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left[-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right]
$$

Why does it appear everywhere? The central limit theorem (covered in the next section) gives the answer: whenever you average many independent contributions, the result tends toward a Gaussian regardless of what the individual contributions look like. This is why measurement errors, which arise from the sum of many small disturbances, are almost always Gaussian. It is nature's favorite trick.

[[simulation applied-stats-sim-2]]

### Student's t-Distribution

The Gaussian works beautifully when you have plenty of data. But with small samples, the estimated mean and standard deviation are themselves uncertain, and the Gaussian underestimates the tails. You might think the Gaussian is always good enough — but actually, for small samples, the tails matter a lot.

The **Student's t-distribution** accounts for this extra uncertainty by having heavier tails. As the sample size grows, the $t$-distribution converges to the Gaussian. For **low statistics** work, always prefer the $t$-distribution.

## Maximum Likelihood Estimation

You've now met the major distributions. A natural question follows: given observed data, how do you figure out *which* distribution (and which parameter values) generated it?

Think of it as a detective story. The data are the clues left at the scene. The parameters are the culprit. Your job is to ask: which culprit makes these clues least surprising?

### The Detective's Question

Suppose you flip a coin 10 times and get 7 heads. Which value of $p$ (the probability of heads) makes this outcome most plausible?

If $p = 0.5$, getting 7 heads is possible but not the most likely result. If $p = 0.7$, it is. If $p = 0.99$, getting *only* 7 heads is actually surprising — you'd expect more. So the answer is somewhere around 0.7. **Maximum likelihood estimation** (MLE) formalizes this intuition: find the parameter values that make the observed data as probable as possible.

> **Challenge.** Grab a coin and flip it 10 times. Write down how many heads you got. What value of $p$ would make your result least surprising? If you got 6 heads, the MLE is $\hat{p} = 0.6$. Simple — but that simplicity hides a powerful principle.

Mathematically, the **likelihood** is the probability of the data as a function of the parameters $\theta$:

$$
\mathcal{L}(\theta) = \prod_i f(x_i; \theta)
$$

This is just the joint probability of all observations, but treated as a function of $\theta$ rather than of the data. Multiplying many small probabilities causes numerical problems, so in practice you maximize the **log-likelihood** instead:

$$
\ln\mathcal{L}(\theta) = \sum_i \ln f(x_i; \theta)
$$

For Gaussian-distributed data, minimizing $-2\ln\mathcal{L}$ is equivalent to minimizing the familiar $\chi^2$ statistic — a connection we will exploit heavily in [the chi-square method](./chi-square-method).

The likelihood $\mathcal{L}(\theta)$ is the bridge between frequentist and Bayesian statistics. In the [Bayesian framework](./bayesian-statistics), you multiply the likelihood by a *prior* — your beliefs about $\theta$ before seeing the data — to get a *posterior* distribution. The likelihood is doing the same job in both worlds: telling you what the data have to say.

### Why MLE Works Well

MLE estimators have three desirable properties. First, they are **consistent**: as you collect more data, the estimate homes in on the true value, like an archer getting closer to the bullseye with every shot. Second, they are **asymptotically normal**: for large samples, the distribution of the estimator becomes Gaussian, making confidence intervals straightforward. Third, they are **efficient**: among all consistent estimators, MLE achieves the smallest possible variance — the **Cramér-Rao bound**. No other method can do better in the large-sample limit.

### The Fitting Procedure

Putting MLE into practice follows a clear recipe:

1. **Choose a model**: Decide which PDF describes the data (Gaussian, Poisson, exponential, ...).
2. **Compute the likelihood**: Write out $\mathcal{L}(\theta)$ for your observed data.
3. **Optimize**: Find the $\theta$ that maximizes $\mathcal{L}$ (or equivalently, minimizes $-\ln\mathcal{L}$). This is typically done numerically.
4. **Assess the fit**: Compare the model predictions to the data. Does the model actually describe what you see?

There are different flavors depending on whether you use individual data points (**unbinned** likelihood) or group the data into histogram bins (**binned** likelihood). The unbinned approach uses all available information but can be computationally expensive; the binned approach is faster but loses some resolution.

Now you have distributions and a method for fitting them. But why do those Gaussian errors keep showing up everywhere? That's the mystery the central limit theorem solves — and it's where we go next.

> **Challenge.** Explain maximum likelihood to a friend using only the coin example. No equations — just the idea of "which value of $p$ makes my data least surprising?" You have one minute.

## Big Ideas

* A PDF is a density, not a probability — you must integrate over a region to get an actual probability, just as you integrate mass density to get mass.
* The Poisson distribution has a magical property: its mean and variance are the same number, so the statistical uncertainty on any count is simply its square root.
* Maximum likelihood estimation is just asking "which parameter values make my actual data least surprising?" — the same question a good detective asks about a suspect.
* MLE is optimal in a precise sense: no consistent estimator can do better in the large-sample limit (the Cramér-Rao bound), so there is no point looking for something cleverer.

## What Comes Next

You have distributions and a method for finding the parameters that best describe data. But a nagging question remains: why does the Gaussian appear *everywhere* in measurement science, even when the underlying process has nothing to do with a bell curve?

The answer is the central limit theorem — and once you have it, you can start asking how uncertainties in your raw measurements flow through to uncertainties in derived quantities. That is error propagation, and together with the CLT it forms the theoretical foundation for almost everything that follows.

## Check Your Understanding

1. The Poisson distribution has equal mean and variance. If you count 25 photons in one second, what is your best estimate of the true rate, and what is the statistical uncertainty on that count?
2. A friend fits a Gaussian to a small dataset of 8 points and reports the result as definitive. What distribution should they have used instead, and why does it matter for the tails?
3. You flip a coin 20 times and get 14 heads. Without computing anything formally, roughly what value of $p$ maximizes the likelihood? What would change if you had gotten 14 heads out of 200 flips?

## Challenge

You measure a physical quantity and collect 50 data points. Plotting a histogram, you notice the distribution looks neither Gaussian nor Poisson — it appears to have two peaks. Describe a strategy for fitting this dataset using maximum likelihood. What model would you try first? How would you decide between a single-component model and a two-component mixture? What would you look at in the residuals to confirm that your chosen model actually describes the data?
