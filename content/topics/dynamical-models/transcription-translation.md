# Differential Equations for Transcription and Translation

## Where we are headed

In the first lesson we saw how a single molecule type reaches a steady state through the balance of production and degradation. Now it is time to apply that same idea to the two-step process at the heart of every cell: DNA is transcribed into mRNA, and mRNA is translated into protein. The beautiful thing is that these two steps operate on very different timescales — and that difference has profound consequences for how cells respond to change.

## Two timescales, one system

Imagine tuning a radio in a noisy room. The raw signal from the antenna jitters up and down rapidly — that is like mRNA, which is made and destroyed on a timescale of minutes in a bacterium. But what you actually hear through the speaker is smoothed out, because the speaker (like a capacitor in the circuit) averages over those rapid fluctuations. That smooth output is like the protein level: it changes slowly, following the mRNA signal with a delay.

This two-timescale structure is not a bug — it is a feature. It means protein levels are naturally buffered against rapid noise in transcription. Nature builds its own low-pass filter.

## The equations

For **transcription**, mRNA is produced at rate $k_\mathrm{m}$ and degraded at rate $\Gamma_\mathrm{m}$:

> **Key Equation — Coupled Transcription-Translation**
> $$
> \frac{\mathrm{d} n_\mathrm{m}}{\mathrm{d} t} = k_\mathrm{m} - \Gamma_\mathrm{m} \, n_\mathrm{m}, \qquad \frac{\mathrm{d} n_\mathrm{p}}{\mathrm{d} t} = k_\mathrm{p} \, n_\mathrm{m} - \Gamma_\mathrm{p} \, n_\mathrm{p}
> $$
> mRNA follows a bathtub equation; protein production depends on how much mRNA is present, coupling the two timescales.

> *The first equation is exactly the [bathtub equation](./differential-equations), applied to mRNA.*

For **translation**, protein is produced at a rate proportional to the number of mRNA molecules (more mRNA means more ribosomes translating) and degraded at rate $\Gamma_\mathrm{p}$.

> *Notice the key difference: the production of protein depends on how much mRNA is present. These two equations are coupled.*

The parameters:
* $n_\mathrm{m}(t)$: number of mRNA molecules at time $t$
* $n_\mathrm{p}(t)$: number of protein molecules at time $t$
* $k_\mathrm{m}$: transcription rate (mRNA molecules per unit time)
* $k_\mathrm{p}$: translation rate (proteins per mRNA per unit time)
* $\Gamma_\mathrm{m}$: mRNA degradation rate
* $\Gamma_\mathrm{p}$: protein degradation rate (often dominated by dilution through cell growth)

## Steady state

At steady state, mRNA reaches $n_\mathrm{m}^\mathrm{ss} = k_\mathrm{m} / \Gamma_\mathrm{m}$, just as before. Plugging this into the protein equation gives:

$$
n_\mathrm{p}^\mathrm{ss} = \frac{k_\mathrm{p} \, k_\mathrm{m}}{\Gamma_\mathrm{p} \, \Gamma_\mathrm{m}}.
$$

> *The steady-state protein level depends on all four parameters: both production rates and both degradation rates. Change any one, and the protein level shifts.*

## Number of molecules versus concentration

So far we have been counting molecules. But experimentally (and theoretically) it is often more convenient to work with **concentrations** — number per volume. If the cell has volume $V$, we define $c_\mathrm{m}(t) = n_\mathrm{m}(t)/V$ and $c_\mathrm{p}(t) = n_\mathrm{p}(t)/V$. The equations keep the same form:

$$
\frac{\mathrm{d} c_\mathrm{m}(t)}{\mathrm{d} t} = k_\mathrm{m} - \Gamma_\mathrm{m} \, c_\mathrm{m}(t),
$$

$$
\frac{\mathrm{d} c_\mathrm{p}(t)}{\mathrm{d} t} = k_\mathrm{p} \, c_\mathrm{m}(t) - \Gamma_\mathrm{p} \, c_\mathrm{p}(t).
$$

> *The switch from number to concentration is just a rescaling — the physics does not change. But be careful: when cells grow and divide, the volume changes, and dilution acts like an extra degradation term.*

## Scaling and dimensionless variables

There is one more rescaling trick that will pay dividends throughout this course. Instead of working with raw numbers or concentrations, measure time in units of the half-life ($\tilde{t} = \Gamma \, t$) and concentration in units of some characteristic scale — for example, the steady-state level $n_\mathrm{ss}$ itself. Define $\tilde{n} = n / n_\mathrm{ss}$. Then the bathtub equation becomes:

$$
\frac{\mathrm{d} \tilde{n}}{\mathrm{d} \tilde{t}} = 1 - \tilde{n}.
$$

All the parameters have disappeared. The solution $\tilde{n}(\tilde{t}) = 1 - e^{-\tilde{t}}$ is universal: *every* bathtub equation, regardless of its specific $k$ and $\Gamma$, follows the same dimensionless curve. This is the power of nondimensionalization — it strips away the details and reveals the universal behavior. When we later encounter the Hill function, the natural concentration scale will be the dissociation constant $K$, and measuring in units of $K$ will again collapse many different systems onto a single curve.

## Why does nature do it this way?

We saw in the [previous lesson](./differential-equations) that degradation lets a cell change its mind quickly. The two-step design adds two further advantages: **amplification** (each mRNA is translated many times, turning one gene into thousands of protein copies) and **independent control** (the cell can regulate transcription and translation separately — two knobs instead of one).

## Check your understanding

* If mRNA degradation becomes much faster ($\Gamma_\mathrm{m} \gg \Gamma_\mathrm{p}$), what happens to the protein response time? Why?
* A gene is transcribed 10 times per minute and each mRNA is translated 20 times before it is degraded. Roughly how many proteins does the cell make from this gene per minute?
* Why is protein degradation in bacteria often dominated by dilution (cell division) rather than active degradation?

## Challenge

In *E. coli*, a typical mRNA has a half-life of about 3 minutes, and a typical protein has an intrinsic half-life of about 1 hour, but in fast-growing *E. coli* the effective half-life is dominated by dilution through cell division (~20 minutes). Compute the degradation rates $\Gamma_\mathrm{m}$ and $\Gamma_\mathrm{p}$. If the transcription rate is $k_\mathrm{m} = 0.5$ mRNA per minute and the translation rate is $k_\mathrm{p} = 20$ proteins per mRNA per minute, what is the steady-state protein level? Now suddenly shut off transcription ($k_\mathrm{m} = 0$). How long does it take for the mRNA level to drop to half? How long for the protein level?

## Big ideas

* **Transcription and translation form a coupled two-step system**, each described by a bathtub equation.
* **mRNA is fast and jittery, protein is slow and smooth** — the two-timescale structure acts as a natural noise filter.
* **Steady-state protein level** = $(k_\mathrm{p} \, k_\mathrm{m}) / (\Gamma_\mathrm{p} \, \Gamma_\mathrm{m})$ — a product of all four rate parameters.

## What comes next

We now have the deterministic machinery: production, degradation, and the two-timescale coupling of mRNA and protein. But every one of these molecular events — polymerase finding the promoter, ribosome latching onto mRNA — is fundamentally random. In the next lesson, we confront the statistics of rare events head-on, starting with DNA replication itself. How does the polymerase achieve an error rate of one in a billion? And what does the distribution of mutations look like across a population? The probability tools we build there — binomial and Poisson distributions — will become the language we use to understand noise in gene expression.
