# Probability for Mutational Analysis

## Where we are headed

In the last two lessons you learned to write differential equations for production and degradation, and you saw how mRNA and protein form a coupled two-timescale system. Those equations describe the average beautifully — but life is not average. Every molecular event is random, and randomness shows up first and most dramatically in DNA replication. Today we zoom in on **mutations**: DNA polymerase is one of the most astonishing proofreaders in nature, making fewer than one error per billion bases copied. How does it achieve that? And when errors *do* slip through, what does the statistics of their appearance look like? The probability distributions we develop here — **binomial** and **Poisson** — will become our toolkit for understanding randomness throughout this course, from rare mutational events to the noise in gene expression that we tackle in the very next lesson.

## What causes mutations?

There are two broad categories:

**Spontaneous mutations** arise from chemical mistakes during DNA replication. Even under perfect conditions, about 5,000 potentially mutagenic events happen per day in a human cell:
* **Depurination**: an adenine or guanine base simply falls off the sugar backbone.
* **Deamination**: cytosine spontaneously converts to uracil.

**Induced mutations** are caused by external agents — UV light, ionizing radiation, or chemical mutagens. These account for roughly 100 additional DNA lesions per day in a human cell.

## The fidelity of the central dogma

Not every step in gene expression is equally accurate. The error rates tell a striking story:

* **DNA replication**: $\sim 1$ error per $10^9$ bases — almost unbelievably precise.
* **Transcription**: $\sim 1$ error per $10^6$ bases — good, but a thousand times worse.
* **Translation**: $\sim 1$ error per $10^4$ bases — the least accurate step.

> *Think about what this means: evolution has invested enormous machinery into keeping DNA replication faithful, because errors there are permanent and heritable. Errors in transcription and translation are temporary — the bad mRNA or protein will be degraded soon enough.*

## How does DNA polymerase proofread so well?

Imagine a typist who makes a mistake every few hundred keystrokes. That is roughly the error rate of base-pairing alone — the free energy difference between a correct and incorrect base pair (4--13 kJ/mol) is only enough to give an error rate of about $10^{-2}$. So how does the cell get from $10^{-2}$ down to $10^{-9}$?

The answer is **two extra layers of proofreading**, each multiplying the fidelity:

**Editing by DNA polymerase** — DNA polymerase does not just add bases; it also checks the last base it added. If the fit is wrong, the polymerase backs up and removes it. This is like a typist who pauses after every keystroke to check and erase mistakes before moving on.

**Strand-directed mismatch repair** — After replication, dedicated repair enzymes scan the new DNA strand for mismatches. They can tell which strand is new (in bacteria, by its methylation pattern) and recruit DNA polymerase to fix the error. This is like a second proofreader who reads the whole manuscript after the typist is done.

Each layer improves fidelity by roughly a factor of 100--1000, and together they achieve the extraordinary $10^{-9}$ error rate.

## Recombination

During meiosis (in sexually reproducing organisms), chromosomes exchange segments through **crossing-over**. This is not an error — it is a deliberate reshuffling that generates genetic diversity, the raw material for evolution.

## Working with mutants

Bacteria are a geneticist's dream: you can grow a billion of them overnight in a single milliliter. To find mutants in this ocean of cells, we use:

* **Genetic selection**: design conditions where only mutants survive. For example, grow bacteria on a plate containing an antibiotic — only resistant mutants form colonies.
* **Genetic screens**: check every colony individually for the phenotype of interest. More labor-intensive, but necessary when there is no way to select directly.

## The statistics of rare events

Now suppose you watch cells divide and count how many carry a new mutation. Each cell division is an independent "trial" with a tiny probability $p$ of producing a mutant. After $N$ divisions, how many mutants do you expect? This is exactly the setting for the **binomial distribution**.

## The binomial distribution

If each of $N$ independent trials has probability $p$ of success, the probability of getting exactly $k$ successes is:

$$
P_N(k) = \binom{N}{k} p^k (1-p)^{N-k}.
$$

> *In words: choose which $k$ trials are the "successes" (the binomial coefficient), multiply by the probability of $k$ successes and $N-k$ failures.*

The mean is $\mu = Np$ and the variance is $\sigma^2 = Np(1-p)$.

## The Poisson limit

In biology, $N$ is typically enormous (millions of cell divisions) and $p$ is tiny (one error per billion bases), but the product $m = Np$ is a modest number. In this limit, the binomial distribution simplifies beautifully to the **Poisson distribution**:

$$
P_m(k) = \frac{m^k}{k!} e^{-m}.
$$

> *The Poisson distribution has a remarkable property: its mean and variance are both equal to $m$. This is exactly where the Fano factor $F = 1$ comes from — a Poisson process has $F = \text{Var}/\text{mean} = 1$.*

This connects directly to what we learned about noise: if gene expression events (transcription, translation) are independent Poisson events, the Fano factor is 1. When we see $F > 1$, something more interesting is going on — bursts, correlations, or feedback.

[[simulation binomial-distribution]]

Start with $N = 20$ trials and $p = 0.5$ — this is like flipping a fair coin 20 times. The distribution looks like a nice bell curve centered at 10. Now keep $N = 20$ but reduce $p$ to $0.05$. The distribution becomes skewed and piles up near zero. This is what mutation counts look like: rare events in many trials.

[[simulation poisson-distribution]]

Set the mean $m = 3$ and compare the Poisson distribution to the binomial with $N = 1000$ and $p = 0.003$. They should be nearly identical — this is the Poisson limit in action. Now increase $m$ to 10 or 20 and watch the distribution become more and more bell-shaped. Even the Poisson distribution looks Gaussian when the mean is large enough.

## The Luria-Delbruck experiment: jackpot cultures

One of the most beautiful experiments in genetics was performed by Salvador Luria and Max Delbruck in 1943. They asked a seemingly simple question: do mutations in bacteria arise *before* or *after* exposure to a selective agent (a virus called phage T1)?

If mutations arise *after* exposure (directed by the selection), you would expect each independent culture to have roughly the same number of resistant mutants — a Poisson distribution with low variance.

If mutations arise *before* exposure (spontaneously, during normal growth), then the timing matters enormously. A mutation that happens early in the growth of a culture produces many resistant descendants — a **jackpot**. A mutation that happens late produces only a few. The result: enormous variation between cultures, with a few "jackpot" cultures containing vastly more mutants than average.

Luria and Delbruck found the jackpot pattern — the variance was far larger than the mean, ruling out the directed-mutation hypothesis. Mutations are random, spontaneous events. This experiment won them the Nobel Prize and established one of the foundations of molecular biology.

[[figure jackpot-cultures]]

[[figure lineage-tree]]

> *The jackpot distribution (now called the Luria-Delbruck distribution) has a variance much larger than its mean — a Fano factor far greater than 1. Sound familiar? The same statistical fingerprint that tells us about transcriptional bursting in the noise lesson.*

## Check your understanding

* Why is DNA replication so much more accurate than translation? What would happen to the organism if the error rates were reversed?
* You plate 10 independent cultures of bacteria on antibiotic plates. Nine have 0--5 colonies, but one has 200. Is this consistent with random spontaneous mutation? Why?
* If the mutation rate is $10^{-9}$ per base per replication, and a gene is 1000 bases long, what is the probability of a mutation in that gene in one replication?

## Challenge

A bacterial gene is 1,200 bases long. The mutation rate is $10^{-9}$ per base per replication. You grow a culture from a single cell to $10^9$ cells (about 30 generations). What is the expected number of mutants in the final population? Now here is the twist: assume one of those mutations happened in the very first generation. How many mutant descendants does that one early mutant produce by the time the culture reaches $10^9$? Compare that to a mutation in the last generation. This is the Luria-Delbruck jackpot in action.

## Big ideas

* **DNA replication achieves extraordinary fidelity** ($10^{-9}$ errors per base) through three layers: base-pairing selectivity, polymerase proofreading, and mismatch repair.
* **The Poisson distribution** emerges naturally when many rare, independent events are counted — it is the universal distribution for rare events in biology.
* **The Luria-Delbruck experiment** proved that mutations arise spontaneously, and the "jackpot" pattern of high variance is the statistical signature of random events occurring at different times during growth.

## What comes next

You now own the Poisson distribution, the Fano factor, and the concept of "rare events in many trials." These are exactly the tools we need for the next lesson, where we discover that the noise in gene expression is not a nuisance but a feature. Genetically identical cells can have wildly different protein levels — and the same Poisson statistics and Fano factor you just learned will tell us whether that noise comes from random molecular firing or from cell-to-cell variation in shared resources. Get ready: the steady state you learned to love in lesson one is about to become a lie.
