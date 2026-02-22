# Rare Events and the Poisson Fingerprint

> *The universe is not just stranger than we suppose; when it comes to cells, it is stranger than we can suppose -- until we write the equations.*

## Where we are headed

Those smooth differential equations from last time? They describe the average beautifully. But life isn't average. Every molecular event is random, and randomness shows up first and most dramatically in DNA replication. DNA polymerase makes fewer than one error per billion bases copied -- that's absurdly precise. And when errors *do* slip through, the statistics of their appearance have a distinctive shape. The distributions we develop here -- **binomial** and **Poisson** -- will be our toolkit for understanding randomness throughout this course.

## The statistics of rare events

Suppose you watch cells divide and count how many carry a new mutation. Each division is an independent "trial" with a tiny probability $p$ of producing a mutant. After $N$ divisions, how many mutants? This is exactly the **binomial distribution**.

## The binomial distribution

If each of $N$ independent trials has probability $p$ of success, the probability of exactly $k$ successes is:

$$
P_N(k) = \binom{N}{k} p^k (1-p)^{N-k}.
$$

> *Choose which $k$ trials succeed (the binomial coefficient), multiply by the probability of $k$ successes and $N-k$ failures.*

The mean is $\mu = Np$ and the variance is $\sigma^2 = Np(1-p)$.

[[simulation binomial-distribution]]

Start with $N = 20$ and $p = 0.5$ -- like flipping a fair coin 20 times. Nice bell curve centered at 10. Now drop $p$ to $0.05$. The distribution piles up near zero and gets skewed. That's what mutation counts look like.

## The Poisson limit

In biology, $N$ is enormous (millions of divisions) and $p$ is tiny (one error per billion bases), but the product $m = Np$ is a modest number. In this limit, the binomial simplifies to the **Poisson distribution**:

> **Key Equation -- The Poisson Distribution**
> $$
> P_m(k) = \frac{m^k}{k!} e^{-m}
> $$
> When $N$ is large and $p$ is small (with $m = Np$ moderate), the binomial simplifies to the Poisson: the universal distribution for counting rare, independent events.

And here's the gorgeous part: the Poisson's mean and variance are *both* equal to $m$. This gives us the **Fano factor** $F = \text{Var}/\text{mean} = 1$. A Poisson process has $F = 1$. When we see $F > 1$, something more interesting is going on -- bursts, correlations, or feedback.

[[simulation poisson-distribution]]

Set $m = 3$ and compare the Poisson to the binomial with $N = 1000$, $p = 0.003$. Nearly identical -- the Poisson limit in action. Crank $m$ up to 20 and watch it become bell-shaped. Even the Poisson looks Gaussian when the mean is large enough.

## The Luria-Delbruck experiment: jackpot cultures

One of the most beautiful experiments in genetics. In 1943, Luria and Delbruck asked: do mutations arise *before* or *after* exposure to a selective agent?

If mutations are *directed* (arising after exposure), every independent culture should have roughly the same number of resistant mutants -- a Poisson distribution with low variance. But if mutations are *spontaneous* (arising during normal growth), timing matters enormously. A mutation that happens early produces a **jackpot** of descendants. A mutation that happens late produces only a few. The result: enormous variation between cultures.

Luria and Delbruck found the jackpot pattern -- variance far larger than the mean, ruling out directed mutation. Mutations are random, spontaneous events. Nobel Prize. Foundation of molecular biology.

> *The jackpot distribution has a Fano factor far greater than 1 -- the same statistical fingerprint that will tell us about transcriptional bursting in the noise lesson.*

[[simulation luria-delbruck-comparison]]

Compare the two models side by side. Left: directed mutations (Poisson, low variance). Right: spontaneous mutations (jackpots, enormous variance). Play with the mutation rate and watch the Fano factor distinguish the two.

## Check your understanding

* You plate 10 cultures on antibiotic plates. Nine have 0--5 colonies, but one has 200. Consistent with spontaneous mutation? Why?
* If the mutation rate is $10^{-9}$ per base per replication and a gene is 1000 bases long, what's the probability of a mutation in that gene in one replication?

## Challenge

A gene is 1,200 bases long, mutation rate $10^{-9}$ per base per replication. You grow from one cell to $10^9$ cells (~30 generations). What's the expected number of mutants? Now: suppose one mutation happened in the very first generation. How many mutant descendants by the time the culture hits $10^9$? Compare to a mutation in the last generation. That's the Luria-Delbruck jackpot.

## Big ideas

* The Poisson distribution emerges when you count many rare, independent events -- it's nature's fingerprint for randomness.
* Its Fano factor ($F = 1$) is the baseline; $F > 1$ signals bursts or correlations.
* The Luria-Delbruck jackpot pattern proves mutations are spontaneous, and the same high-variance signature will reappear when we study transcriptional noise.

## What comes next

You now own the Poisson distribution and the Fano factor. Next, we discover that the noise in gene expression isn't a nuisance -- it's a feature that lets genetically identical cells make wildly different decisions.
