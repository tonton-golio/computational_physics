# Quantifying Noise in Gene Expression

> *The universe is not just stranger than we suppose; when it comes to cells, it is stranger than we can suppose -- until we write the equations.*

## Where we are headed

Remember the bathtub equation's clean, smooth steady state? That steady state is a *lie*. Or rather, it's the average of something much messier. Inside a real cell, molecules are made and destroyed one at a time by random collisions. When the numbers are small -- tens or hundreds of molecules, not trillions -- the randomness matters enormously. Today we find out just how noisy gene expression really is, and why that noise isn't a nuisance but a tool cells use to make life-or-death decisions.

## The central dogma, revisited

$$
\text{DNA} \xrightarrow{\text{transcription}} \text{mRNA} \xrightarrow{\text{translation}} \text{Protein}
$$

In our deterministic model, the molecular machines hum along at constant rates. But in reality, every encounter -- polymerase finding the promoter, ribosome latching onto mRNA -- is a random event. The molecules are doing a drunken walk through the cytoplasm.

## Why is gene expression noisy?

The fundamental reason: **small numbers**. A typical *E. coli* cell has one or two gene copies, a handful of mRNAs, and maybe a few hundred to a few thousand proteins from each gene. With numbers this small, random fluctuations are unavoidable -- same reason flipping a coin 10 times gives a much noisier fraction of heads than flipping it 10,000 times.

This noise has real consequences. In *B. subtilis*, noisy expression of the master regulator *comK* causes a small fraction of cells to stochastically flip into competence -- a survival strategy. In any bacterial population, noise creates rare persister cells with low metabolic activity that survive antibiotics without resistance mutations.

## Stochastic simulation: the Gillespie algorithm

Our deterministic equations give the average but miss the noise entirely. To simulate what actually happens molecule by molecule, we use the **Gillespie algorithm**: compute each reaction's propensity, draw a random waiting time, pick which reaction fires, update counts, repeat. This gives you exact sample trajectories of the stochastic process.

[[simulation gillespie-trajectory]]

Compare the white ODE mean to the colored stochastic trajectories. With high production rate, traces cluster tightly around the mean. Drop $k$ to 5 and watch individual trajectories wander far from the deterministic prediction. That's the small-number effect: when molecules are few, noise dominates.

## Measuring the noise

To see noise in action, biologists fuse a **reporter protein** like GFP to a gene of interest. The brighter the cell, the more protein.

* **Bulk measurement** (billions of cells): you see the average. Smooth. Boring.
* **Single-cell measurement** (microscopy or flow cytometry): you see the full distribution. And it's *wide*. Genetically identical cells can differ two-fold or even ten-fold in protein level.

This isn't measurement error. It's real biological noise.

## Quantifying noise: the coefficient of variation

The **coefficient of variation** (CV) measures noise as the ratio of standard deviation to mean:

$$
\eta = \frac{\sigma}{\langle N \rangle} = \frac{\sqrt{\langle (N - \langle N \rangle)^2 \rangle}}{\langle N \rangle}.
$$

> *A CV of 0.5 means the typical fluctuation is about half the mean -- that's very noisy.*

## The Fano factor

An alternative noise measure:

$$
F = \frac{\text{Var}[N]}{\langle N \rangle}.
$$

> *For a Poisson process, $F = 1$. If $F > 1$, transcription is bursty. If $F < 1$, negative feedback is clamping down fluctuations.* The Fano factor is a fingerprint: it tells you about the underlying mechanism from just the variance and mean.

## Decomposing noise: the Elowitz experiment

Here's one of the cleverest experiments in modern biology. In 2002, Elowitz and colleagues asked: is the noise from individual gene copies firing randomly (**intrinsic**), or from cell-to-cell variation in shared resources like ribosomes (**extrinsic**)?

The trick: put two *different-colored* reporters (CFP and YFP) under *identical* promoters in the same cell. If noise is all extrinsic, both colors go up and down together -- every cell lands on the diagonal. If noise is intrinsic, the colors fluctuate independently -- cells scatter off the diagonal.

> **Key Equation -- Noise Decomposition**
> $$
> \eta_{\mathrm{total}}^2 = \eta_{\mathrm{int}}^2 + \eta_{\mathrm{ext}}^2
> $$
> Total noise decomposes exactly into intrinsic noise (independent random firing) and extrinsic noise (shared fluctuations across the cell).

**Extrinsic noise** (correlated):

$$
\eta_{\mathrm{ext}}^2 = \frac{\langle N^{(1)} N^{(2)} \rangle - \langle N^{(1)} \rangle \langle N^{(2)} \rangle}{\langle N^{(1)} \rangle \langle N^{(2)} \rangle}.
$$

**Intrinsic noise** (uncorrelated):

$$
\eta_{\mathrm{int}}^2 = \frac{\langle (N^{(1)} - N^{(2)})^2 \rangle}{2\,\langle N^{(1)} \rangle \langle N^{(2)} \rangle}.
$$

> *The beauty: you can measure this directly from two-color data, without knowing anything about the underlying mechanism.*

[[simulation gene-expression-noise]]

Start with high production rate -- smooth trace. Reduce it by 10x (keeping the same mean by adjusting degradation). The trace gets wild and jagged. That's the small-number effect. Run two identical genes side by side and watch whether fluctuations are correlated (extrinsic) or independent (intrinsic).

## Binomial partitioning at division

Another major noise source: when a cell splits, molecules are randomly distributed between daughters. If a cell has only a handful of molecules, one daughter may get most while the other gets nearly none.

[[simulation bacterial-lineage-tree]]

Watch molecular counts diverge across the lineage. Start with 100 molecules -- modest variation. Drop to 10 and the leaves show dramatic differences, purely from partitioning randomness.

## Why does nature do it this way?

Noise seems like a problem, and cells do suppress it when precision matters (negative autoregulation, as you'll see in [feedback loops](./feedback-loops)). But noise can also be *useful*: a population with noisy gene expression hedges its bets, so some cells are already in the right state when the environment suddenly changes.

## Check your understanding

* If you increase mean protein level while keeping variance fixed, what happens to the CV?
* In the Elowitz two-color experiment, what would the scatter plot look like if noise were *entirely* intrinsic?
* A gene has Fano factor $F = 5$. Bursty or Poisson-like?

## Challenge

A gene has mean $\langle N \rangle = 100$ and $\text{Var}[N] = 400$. Compute the CV and Fano factor. Burstier than Poisson? Now add negative feedback that halves the variance without changing the mean. New CV and Fano factor? Has feedback pushed below the Poisson limit?

## Big ideas

* Gene expression is inherently noisy because small numbers of molecules undergo random reactions.
* Intrinsic noise = random molecular firing; extrinsic noise = cell-to-cell variation in shared factors. They add in quadrature.
* The Fano factor ($F = \text{Var}/\text{mean}$) is a fingerprint: $F = 1$ is Poisson, $F > 1$ is bursty.

## What comes next

So far we've treated the transcription rate as a fixed number. But real genes have a knob -- a repressor that shuts them down, an activator that cranks them up. Next, we derive the Hill function and watch nature turn a gentle dimmer into a sharp on-off switch.
