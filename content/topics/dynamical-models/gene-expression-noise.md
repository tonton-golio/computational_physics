# Quantifying Noise in Gene Expression

## Cellular identity and the central dogma

In multicellular organisms, all cells carry the same DNA yet display enormous diversity. The answer lies in **gene expression**: depending on cell type, different genes are transcribed and translated into proteins.

The **central dogma** describes the flow of genetic information:

$$
\text{DNA} \xrightarrow{\text{transcription}} \text{mRNA} \xrightarrow{\text{translation}} \text{Protein}
$$

Two key molecular machines drive this process:

- **RNA polymerase**: transcribes DNA into mRNA.
- **Ribosome**: translates mRNA into protein.

## Transcriptional regulation

Gene expression is controlled at the promoter, a region of DNA upstream of the gene:

- **Strong promoter**: attracts RNA polymerase efficiently; the gene is on by default unless a repressor binds.
- **Weak promoter**: attracts RNA polymerase poorly; the gene is off by default unless an activator recruits polymerase.
- Classic example: the *lac* promoter in *E. coli*, which is repressed by LacI and activated by CAP.

## Measuring gene expression

To measure expression, a **reporter protein** such as GFP is fused to the target gene. Fluorescence intensity then reports protein abundance.

- **Bulk measurement**: total fluorescence from billions of cells in a tube gives the population average.
- **Single-cell measurement**: flow cytometry or microscopy reveals the full distribution, showing that gene expression is inherently noisy.

## Why is gene expression noisy?

Low copy numbers of key molecules, particularly transcription factors and gene copies, make reactions stochastic. Chemical reactions inside the cell are driven by **diffusion** (confirmed by FRAP and single-molecule tracking), so each molecular encounter is a random event.

Noise in gene expression has biological consequences:

- **Bistability** in *comK* expression drives competence in *B. subtilis*.
- **Persister cells** in bacterial populations survive antibiotic treatment through stochastic switching.

## Defining total noise

The **coefficient of variation** (CV) quantifies noise as the ratio of standard deviation to mean:

$$
\eta(t) = \frac{\sqrt{\langle (N(t) - \langle N(t) \rangle)^2 \rangle}}{\langle N(t) \rangle}
$$

where $N_j(t)$ is the protein copy number in cell $j$, the population mean is

$$
\langle N(t) \rangle = \frac{1}{n} \sum_{j=1}^{n} N_j(t),
$$

and the variance is

$$
\text{Var}[N(t)] = \frac{1}{n} \sum_{j=1}^{n} \bigl(N_j(t) - \langle N(t) \rangle\bigr)^2.
$$

## Decomposing noise: intrinsic and extrinsic

Following the landmark experiment of Elowitz et al. (2002), total noise can be decomposed by expressing two distinguishable reporters (e.g., CFP and YFP) from identical promoters in the same cell.

**Extrinsic noise** captures correlated fluctuations (shared upstream factors):

$$
\eta_{\mathrm{ext}}^2 = \frac{\langle N^{(1)} N^{(2)} \rangle - \langle N^{(1)} \rangle \langle N^{(2)} \rangle}{\langle N^{(1)} \rangle \langle N^{(2)} \rangle}.
$$

**Intrinsic noise** captures uncorrelated fluctuations (independent birth-death events):

$$
\eta_{\mathrm{int}}^2 = \frac{\langle (N^{(1)} - N^{(2)})^2 \rangle}{2\,\langle N^{(1)} \rangle \langle N^{(2)} \rangle}.
$$

The total noise decomposes exactly:

$$
\eta_{\mathrm{total}}^2 = \eta_{\mathrm{int}}^2 + \eta_{\mathrm{ext}}^2.
$$

When both reporters are identically distributed ($N^{(1)} \stackrel{d}{=} N^{(2)}$), this reduces to the standard CV squared.

[[simulation gene-expression-noise]]

## The Fano factor

An alternative noise measure is the **Fano factor**:

$$
F = \frac{\text{Var}[N]}{\langle N \rangle}.
$$

- For a Poisson process, $F = 1$.
- $F > 1$ indicates super-Poissonian (bursty) noise, common in gene expression due to transcriptional bursting.
- $F < 1$ indicates sub-Poissonian noise, which can arise from negative autoregulation.

## Stochastic simulation

The **Gillespie algorithm** (stochastic simulation algorithm) provides exact trajectories of the chemical master equation. At each step:

1. Compute all reaction propensities $a_i$.
2. Draw the time to next reaction from an exponential distribution with rate $a_0 = \sum_i a_i$.
3. Choose which reaction fires with probability $a_i / a_0$.
4. Update molecule counts and repeat.

This algorithm is the computational workhorse for studying noise in gene expression at the single-cell level.
