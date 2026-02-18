# Quantifying Noise in Gene Expression

## Where we are headed

In the last lesson you learned to write differential equations for production and degradation, and you saw that the system settles into a nice, clean steady state. Beautiful. But here is the twist: that steady state is a *lie*. Or rather, it is the average of something much messier. Inside a real cell, molecules are made and destroyed one at a time by random collisions, and when the numbers are small — tens or hundreds of molecules, not trillions — the randomness matters enormously. Today we find out just how noisy gene expression really is, and why that noise is not just a nuisance but a tool that cells use to make life-or-death decisions.

## The central dogma, revisited

You already know the central dogma:

$$
\text{DNA} \xrightarrow{\text{transcription}} \text{mRNA} \xrightarrow{\text{translation}} \text{Protein}
$$

Two molecular machines drive this process: **RNA polymerase** transcribes DNA into mRNA, and the **ribosome** translates mRNA into protein. In our deterministic model, these machines hum along at constant rates. But in reality, every molecular encounter — polymerase finding the promoter, ribosome latching onto the mRNA — is a random event driven by diffusion. The molecules are doing a drunken walk through the cytoplasm, and sometimes they find their target quickly, sometimes slowly.

## Measuring the noise

To see noise in action, biologists fuse a **reporter protein** like GFP (green fluorescent protein) to a gene of interest. The brighter the cell glows, the more protein it has.

- **Bulk measurement** (a tube full of billions of cells): you see the population average. Smooth. Boring.
- **Single-cell measurement** (microscopy or flow cytometry): you see the full distribution. And it is *wide*. Genetically identical cells, grown in the same conditions, can have two-fold or even ten-fold differences in protein level.

This is not measurement error. It is real, biological noise.

## Why is gene expression noisy?

The fundamental reason is **small numbers**. A typical *E. coli* cell has one or two copies of each gene, perhaps a handful of mRNA molecules, and maybe a few hundred to a few thousand proteins from each gene. When you are working with numbers this small, random fluctuations are unavoidable — it is the same reason that flipping a coin 10 times gives a much noisier fraction of heads than flipping it 10,000 times.

This noise has real biological consequences:

- **Competence in *B. subtilis***: noisy expression of the master regulator *comK* causes a small fraction of cells to stochastically flip into a state where they can take up DNA from the environment — a survival strategy under stress.
- **Persister cells**: in a genetically uniform bacterial population, noise creates rare cells with low metabolic activity that survive antibiotic treatment, even without resistance mutations.

## Quantifying noise: the coefficient of variation

The **coefficient of variation** (CV) measures noise as the ratio of the standard deviation to the mean:

$$
\eta = \frac{\sigma}{\langle N \rangle} = \frac{\sqrt{\langle (N - \langle N \rangle)^2 \rangle}}{\langle N \rangle},
$$

where $N$ is the protein copy number in a single cell, and the angle brackets denote the average over the population.

> *In words: the CV tells you how wide the distribution is relative to its center. A CV of 0.5 means the typical fluctuation is about half the mean — that is very noisy.*

## Decomposing noise: the Elowitz experiment

Here is one of the most clever experiments in modern biology. In 2002, Michael Elowitz and colleagues asked: where does the noise come from? Is it because individual gene copies fire randomly (**intrinsic noise**), or because the whole cell's environment fluctuates — varying numbers of ribosomes, polymerases, and metabolites from cell to cell (**extrinsic noise**)?

The trick: put two *different-colored* fluorescent reporters (CFP and YFP) under the control of *identical* promoters in the same cell. If the noise is all extrinsic (upstream fluctuations shared by both genes), the two colors go up and down together — every cell lands on the diagonal of a CFP-vs-YFP plot. If the noise is intrinsic (independent random firing), the colors fluctuate independently — cells scatter off the diagonal.

The math makes this precise. Imagine two glasses of water, one blue and one red, and you are trying to figure out why the water levels fluctuate. One glass might be noisy because the faucet drips unpredictably (intrinsic — each glass has its own random drip). The other source of variation is that some days you leave the tap on longer (extrinsic — both glasses get more or less water together).

**Extrinsic noise** (correlated fluctuations):

$$
\eta_{\mathrm{ext}}^2 = \frac{\langle N^{(1)} N^{(2)} \rangle - \langle N^{(1)} \rangle \langle N^{(2)} \rangle}{\langle N^{(1)} \rangle \langle N^{(2)} \rangle}.
$$

**Intrinsic noise** (uncorrelated fluctuations):

$$
\eta_{\mathrm{int}}^2 = \frac{\langle (N^{(1)} - N^{(2)})^2 \rangle}{2\,\langle N^{(1)} \rangle \langle N^{(2)} \rangle}.
$$

And the total noise decomposes exactly:

$$
\eta_{\mathrm{total}}^2 = \eta_{\mathrm{int}}^2 + \eta_{\mathrm{ext}}^2.
$$

> *The beauty of this decomposition is that you can measure it directly from the two-color data, without knowing anything about the underlying mechanism.*

[[simulation gene-expression-noise]]

> **Try this**: Start with a high production rate and watch the protein trace — it should look smooth. Now reduce the production rate by a factor of 10 while keeping the same mean (by reducing degradation too). The trace gets wild and jagged. That is the small-number effect in action. Next, try running two identical genes side by side and watch whether their fluctuations are correlated (extrinsic noise) or independent (intrinsic noise).

## The Fano factor

An alternative way to characterize noise is the **Fano factor**:

$$
F = \frac{\text{Var}[N]}{\langle N \rangle}.
$$

> *For a Poisson process (completely random, independent events), $F = 1$. If $F > 1$, the noise is "burstier" than Poisson — transcription happens in bursts, with several mRNAs made in quick succession followed by silence. If $F < 1$, the noise is suppressed below Poisson, which can happen when negative feedback clamps down on fluctuations.*

The Fano factor is a fingerprint: it tells you something about the underlying mechanism of gene expression just from measuring the variance and mean of protein levels across cells.

## Stochastic simulation: the Gillespie algorithm

Our deterministic equations give us the average, but they miss the noise entirely. To simulate what actually happens inside a single cell, molecule by molecule, we use the **Gillespie algorithm** (also called the stochastic simulation algorithm). Here is how it works:

1. List all possible reactions and compute their **propensities** $a_i$ — how likely each reaction is to happen right now.
2. Draw the time to the next reaction from an exponential distribution with total rate $a_0 = \sum_i a_i$.
3. Choose *which* reaction fires with probability proportional to its propensity: $a_i / a_0$.
4. Update the molecule counts and repeat.

> *This gives you exact sample trajectories of the chemical master equation. It is the computational workhorse for studying noise in gene expression, and you will use it throughout this course.*

Play with the simulation above — try changing the production and degradation rates and watch how the noise changes.

## Why does nature do it this way?

You might think noise is a problem cells would want to eliminate. And sometimes it is — negative autoregulation (which we will see in a later lesson) exists partly to suppress noise. But noise can also be *useful*. A genetically identical population with noisy gene expression is a population that hedges its bets: if the environment suddenly changes, some cells will already be in the right state to survive, purely by chance. This is a form of **bet-hedging**, and it is one of evolution's most elegant strategies.

## Check your understanding

- If you increase the mean protein level while keeping the variance fixed, what happens to the CV? What does this tell you about noise in highly expressed genes?
- In the Elowitz two-color experiment, what would the CFP-vs-YFP scatter plot look like if noise were *entirely* intrinsic?
- A gene has Fano factor $F = 5$. Is transcription happening in bursts, or is it smooth and Poisson-like?

## Challenge

Imagine a gene with a mean protein level of $\langle N \rangle = 100$ and variance $\text{Var}[N] = 400$. Compute the CV and the Fano factor. Is this gene's expression burstier than Poisson? Now suppose you add negative feedback that halves the variance without changing the mean. What are the new CV and Fano factor? Has the feedback pushed the system below the Poisson limit?

## Big ideas

- **Gene expression is inherently noisy** because it involves small numbers of molecules undergoing random reactions.
- **Intrinsic noise** comes from the randomness of individual molecular events; **extrinsic noise** comes from cell-to-cell variation in shared factors.
- **The Fano factor** ($F = \text{Var}/\text{mean}$) is a fingerprint that reveals the underlying mechanism — $F = 1$ is Poisson, $F > 1$ is bursty.

## What comes next

Noise in gene expression and mutations in DNA replication are two faces of the same coin: randomness at the molecular level. In the next lesson, we dive into the statistics of rare events — how DNA polymerase achieves an error rate of one in a billion, and what the distribution of mutations across a population looks like. The probability distributions you will meet there (binomial, Poisson) are exactly the same ones that underpin everything we just said about noise.
