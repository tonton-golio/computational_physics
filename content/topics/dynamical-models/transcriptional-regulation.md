# Transcriptional Regulation and the Hill Function

## Where we are headed

Over the last four lessons you have built the foundations: differential equations for production and degradation, the two-timescale coupling of mRNA and protein, probability distributions for rare events, and the noisy reality of gene expression. Through all of that, we treated the transcription rate as a constant — the gene is either on or off, and we did not ask what controls it. But in a real cell, genes are regulated. A repressor protein sits on the promoter and blocks transcription; an activator protein recruits RNA polymerase and turns it on. Today we derive the mathematical function that describes this regulation — the **Hill function** — and you will see how nature turns a gentle, graded response into a sharp, switch-like one. This is one of the most important functions in all of quantitative biology.

## The committee picture

Before any math, think about it this way. Suppose a gene's promoter is like a conference room, and the gene only turns on when the room is occupied by a transcription factor. If only *one* molecule needs to sit down, then the gene turns on gradually as the concentration of that molecule increases — some rooms are occupied, some are not, and the fraction increases smoothly.

But now suppose the promoter requires a *committee*: four transcription factor molecules must all be present at the same time to turn the gene on. At low concentrations, the chance of all four showing up simultaneously is tiny. At high concentrations, it is almost certain. The transition from "almost never" to "almost always" happens over a narrow range of concentration. You have turned a gentle slope into a switch.

That is the Hill function. The Hill coefficient $n$ is the size of the committee.

## Deriving the Hill function from binding kinetics

### A single transcription factor

Consider a transcription factor (TF) that binds to and represses a promoter. The promoter can be in two states: **free** (gene on) or **occupied** (gene off). The binding rate depends on TF concentration, but the unbinding rate does not:

$$
\frac{\mathrm{d} P_\mathrm{free}}{\mathrm{d} t} = - v_\mathrm{bind} \, c_\mathrm{p} \, P_\mathrm{free} + k_\mathrm{unbind} \, (1 - P_\mathrm{free}).
$$

> *In words: free promoters get occupied at a rate proportional to TF concentration, and occupied promoters become free at a constant unbinding rate.*

We assume binding and unbinding are much faster than transcription itself, so the promoter reaches a quasi-steady state. Setting the derivative to zero:

$$
P_\mathrm{free} = \frac{1}{1 + c_\mathrm{p}/K}, \qquad \text{where } K = \frac{k_\mathrm{unbind}}{v_\mathrm{bind}}.
$$

> *$K$ is the **dissociation constant** — the concentration at which the promoter is occupied half the time. When $c_\mathrm{p} = K$, you get $P_\mathrm{free} = 1/2$. Below $K$, the gene is mostly on; above $K$, the gene is mostly off.*

### Dimers: the committee of two

Now suppose the transcription factor must form a **dimer** (a pair) before it can bind the promoter. The binding rate becomes proportional to $c_\mathrm{p}^2$ instead of $c_\mathrm{p}$:

$$
P_\mathrm{free} = \frac{1}{1 + (c_\mathrm{p}/K_2)^2}.
$$

> *The exponent of 2 makes the transition from "on" to "off" steeper. The gene does not care about individual molecules — it waits until the concentration is high enough that pairs are common.*

### The general Hill function

Because each extra bound repressor multiplies the probability of the next one binding (cooperativity), the fraction bound becomes $[c_\mathrm{p}/K]^n / (1 + [c_\mathrm{p}/K]^n)$. The gene is active only when the promoter is *empty*, so transcription rate is maximal rate times $(1 - \text{fraction bound})$ — giving us the Hill repression function you see everywhere in biology.

For a complex of $n$ molecules (Hill coefficient $n$), the pattern generalizes:

$$
P_\mathrm{free} = \frac{1}{1 + (c_\mathrm{p}/K)^n}.
$$

[[figure hill-function-overlay]]

Play with this in your head:
* **$n = 1$**: a gentle, hyperbolic curve. The gene gradually turns off as repressor concentration increases.
* **$n = 2$**: steeper. The transition sharpens.
* **$n = 4$**: very steep. The gene is either fully on or fully off, with a narrow transition zone around $c_\mathrm{p} = K$.

> *Look how nature turns a gentle slope into a switch! The higher the Hill coefficient, the more "all-or-nothing" the response. This is the mathematical heart of biological decision-making.*

[[simulation hill-function]]

Set the Hill coefficient to $n = 1$ and slowly increase the repressor concentration. Watch how gradually the gene turns off. Now crank $n$ up to 4 and repeat. See how the transition sharpens into a switch? Find the concentration where the gene is exactly at 50% — that is $K$, the dissociation constant. The cell has just invented a tunable amplifier.

## Repression

For a self-repressing gene, the transcription rate depends on the repressor (protein) concentration through the Hill function:

$$
\frac{\mathrm{d} c_\mathrm{m}}{\mathrm{d} t} = \frac{\alpha_\mathrm{m}}{1 + (c_\mathrm{p}/K)^n} - \Gamma_\mathrm{m} \, c_\mathrm{m},
$$

$$
\frac{\mathrm{d} c_\mathrm{p}}{\mathrm{d} t} = k_\mathrm{p} \, c_\mathrm{m} - \Gamma_\mathrm{p} \, c_\mathrm{p}.
$$

> *When protein is scarce ($c_\mathrm{p} \ll K$), the Hill function is close to 1 and mRNA is produced at full rate $\alpha_\mathrm{m}$. When protein is abundant ($c_\mathrm{p} \gg K$), the Hill function drops to nearly zero and transcription is shut off.*

More generally, if the repressor is a different protein (not the gene's own product), we replace $c_\mathrm{p}$ with the concentration of the regulating transcription factor $c_\mathrm{TF}$.

## Activation

For an activator, the gene needs the TF to be *bound* to produce mRNA. We use $P_\mathrm{occupied} = 1 - P_\mathrm{free}$:

$$
\frac{\mathrm{d} c_\mathrm{m}}{\mathrm{d} t} = \frac{\alpha_\mathrm{m} \, (c_\mathrm{TF}/K)^n}{1 + (c_\mathrm{TF}/K)^n} - \Gamma_\mathrm{m} \, c_\mathrm{m}.
$$

> *Now the gene is off when TF is absent, and turns on sharply as TF concentration rises past $K$. The Hill coefficient $n$ controls how sharp the switch is, just as before.*

## Regulation by small RNA

Nature has another elegant trick: **small RNAs** (sRNAs) — think of them as molecular Post-it notes that stick to mRNA and say "delete me" — that bind directly to mRNA and mark it for rapid degradation. The equations become:

$$
\frac{\mathrm{d} c_\mathrm{s}}{\mathrm{d} t} = \alpha_\mathrm{s} - \Gamma_\mathrm{s} \, c_\mathrm{s} - \delta \, c_\mathrm{m} \, c_\mathrm{s},
$$

$$
\frac{\mathrm{d} c_\mathrm{m}}{\mathrm{d} t} = \alpha_\mathrm{m} - \Gamma_\mathrm{m} \, c_\mathrm{m} - \delta \, c_\mathrm{m} \, c_\mathrm{s}.
$$

> *The key is the third term in each equation: $\delta \, c_\mathrm{m} \, c_\mathrm{s}$, where $\delta$ is the "mutual annihilation rate." When sRNA meets mRNA, they annihilate each other (form a complex that is quickly degraded). This mutual destruction creates a threshold-like response — the mRNA is only abundant when its production rate exceeds the sRNA production rate.*

## Why does nature do it this way?

Why not just make every gene constitutive — always on at a fixed rate? Because the environment changes. A bacterium swimming through your gut encounters shifting nutrient sources, toxins, and competitors. It needs to switch genes on and off rapidly and decisively. The Hill function, with its tunable steepness, gives evolution a knob to turn: make $n$ small for a graded response (useful for fine-tuning), or make $n$ large for an all-or-nothing switch (useful for committing to a decision).

## Check your understanding

* At what concentration of repressor is a gene with Hill coefficient $n$ repressed to exactly half its maximal rate?
* Why does dimerization (or higher-order multimerization) increase the effective Hill coefficient?
* An sRNA and an mRNA are produced at equal rates. What is the steady-state mRNA level? What happens if you double the sRNA production rate?

## Challenge

Plot the Hill function $f(c) = 1/(1 + (c/K)^n)$ for $K = 1$ and $n = 1, 2, 4, 8$. At what concentration range does the function transition from 90% activity to 10% activity? (Call this the "switching window.") How does the width of the switching window depend on $n$? Can you derive a formula? This tells you how sharp the cell's decision-making is.

## Big ideas

* **The Hill function** $1/(1 + (c/K)^n)$ is the universal model for cooperative binding and regulation in biology.
* **The Hill coefficient** $n$ controls the steepness of the response: $n = 1$ is gentle, $n \geq 4$ is switch-like.
* **The dissociation constant** $K$ sets the midpoint: the concentration at which the response is half-maximal.

## What comes next

We now have a knob that can turn genes up or down. But what happens when the gene product reaches back and turns its own knob? That is feedback, and it changes everything. In the next lesson, we close the loop and discover that negative feedback gives cells thermostats, positive feedback gives them memory, and rings of repression give them clocks. Welcome to the world of switches and oscillators.
