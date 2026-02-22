# Differential Equations for Transcription and Translation

> *The universe is not just stranger than we suppose; when it comes to cells, it is stranger than we can suppose -- until we write the equations.*

## Where we are headed

Last time you saw how a single molecule type reaches steady state through production and degradation. Now we apply that same idea to the two-step process at the heart of every cell: DNA is transcribed into mRNA, and mRNA is translated into protein. The beautiful thing? These two steps operate on wildly different timescales -- and that difference has profound consequences.

## The equations

For **transcription**, mRNA follows its own bathtub equation -- produced at rate $k_\mathrm{m}$, degraded at rate $\Gamma_\mathrm{m}$:

> **Key Equation -- Coupled Transcription-Translation**
> $$
> \frac{\mathrm{d} n_\mathrm{m}}{\mathrm{d} t} = k_\mathrm{m} - \Gamma_\mathrm{m} \, n_\mathrm{m}, \qquad \frac{\mathrm{d} n_\mathrm{p}}{\mathrm{d} t} = k_\mathrm{p} \, n_\mathrm{m} - \Gamma_\mathrm{p} \, n_\mathrm{p}
> $$
> mRNA follows a bathtub equation; protein production depends on how much mRNA is present, coupling the two timescales.

For **translation**, protein is made at a rate proportional to the number of mRNA molecules (more mRNA = more ribosomes translating) and degraded at rate $\Gamma_\mathrm{p}$.

> *Notice the key difference: protein production depends on how much mRNA is around. These two equations are coupled.*

The parameters:
* $n_\mathrm{m}(t)$, $n_\mathrm{p}(t)$: mRNA and protein counts
* $k_\mathrm{m}$: transcription rate (mRNA/time)
* $k_\mathrm{p}$: translation rate (proteins per mRNA per time)
* $\Gamma_\mathrm{m}$, $\Gamma_\mathrm{p}$: degradation rates for mRNA and protein

## Two timescales, one system

Think of tuning a radio in a noisy room. The raw antenna signal jitters rapidly -- that's mRNA, made and destroyed on a timescale of minutes in bacteria. But what you actually hear through the speaker is smoothed out, because the speaker averages over those rapid fluctuations. That smooth output is protein: it changes slowly, following the mRNA signal with a delay.

This two-timescale structure isn't a bug -- it's a feature. Protein levels are naturally buffered against rapid transcriptional noise. Nature builds its own low-pass filter.

## Steady state

At steady state, mRNA reaches $n_\mathrm{m}^\mathrm{ss} = k_\mathrm{m} / \Gamma_\mathrm{m}$, just as before. Plug that into the protein equation:

$$
n_\mathrm{p}^\mathrm{ss} = \frac{k_\mathrm{p} \, k_\mathrm{m}}{\Gamma_\mathrm{p} \, \Gamma_\mathrm{m}}.
$$

> *Steady-state protein depends on all four parameters. Change any one, and the level shifts.*

## Number of molecules versus concentration

So far we've been counting molecules. But it's often more convenient to work with **concentrations** -- number per volume. If the cell has volume $V$, define $c_\mathrm{m}(t) = n_\mathrm{m}(t)/V$ and $c_\mathrm{p}(t) = n_\mathrm{p}(t)/V$. The equations keep the same form.

> *The switch from number to concentration is just a rescaling -- the physics doesn't change. But watch out: when cells grow and divide, volume changes, and dilution acts like extra degradation.*

## Scaling and dimensionless variables

Here's a trick that'll pay dividends throughout this course. By measuring everything in the right units -- time in units of the half-life ($\tilde{t} = \Gamma \, t$), concentration in units of the steady state ($\tilde{n} = n / n_\mathrm{ss}$) -- the bathtub equation becomes:

$$
\frac{\mathrm{d} \tilde{n}}{\mathrm{d} \tilde{t}} = 1 - \tilde{n}.
$$

All the parameters have disappeared. The solution $\tilde{n}(\tilde{t}) = 1 - e^{-\tilde{t}}$ is universal: *every* bathtub equation follows the same dimensionless curve. By measuring everything in the right units, suddenly every cell looks the same. That's the power of stripping away the numbers.

## Why does nature do it this way?

The two-step design adds two advantages beyond what degradation alone gives you: **amplification** (each mRNA is translated many times, turning one gene into thousands of proteins) and **independent control** (the cell can regulate transcription and translation separately -- two knobs instead of one).

## Check your understanding

* If $\Gamma_\mathrm{m} \gg \Gamma_\mathrm{p}$, what happens to the protein response time? Why?
* A gene is transcribed 10 times/min and each mRNA is translated 20 times before degradation. Roughly how many proteins per minute?
* Why is protein degradation in bacteria often dominated by dilution rather than active destruction?

## Challenge

In *E. coli*, a typical mRNA half-life is ~3 min, and the effective protein half-life (dominated by dilution) is ~20 min. Compute $\Gamma_\mathrm{m}$ and $\Gamma_\mathrm{p}$. With $k_\mathrm{m} = 0.5$ mRNA/min and $k_\mathrm{p} = 20$ proteins/mRNA/min, find the steady-state protein level. Now shut off transcription. How long for mRNA to halve? How long for protein?

## Big ideas

* Transcription and translation form a coupled two-step system, each described by a bathtub equation.
* mRNA is fast and jittery, protein is slow and smooth -- the two-timescale structure is nature's built-in noise filter.
* Steady-state protein = $(k_\mathrm{p} \, k_\mathrm{m}) / (\Gamma_\mathrm{p} \, \Gamma_\mathrm{m})$.

## What comes next

Every one of these molecular events -- polymerase finding the promoter, ribosome latching onto mRNA -- is fundamentally random. Next, we confront the statistics of rare events and discover that the Poisson distribution is nature's fingerprint for randomness.
