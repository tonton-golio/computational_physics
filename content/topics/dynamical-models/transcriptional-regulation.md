# Transcriptional Regulation and the Hill Function

> *The universe is not just stranger than we suppose; when it comes to cells, it is stranger than we can suppose -- until we write the equations.*

## Where we are headed

Through four lessons, we've treated the transcription rate as a constant -- the gene is either on or off, and we didn't ask what controls it. But in a real cell, genes are regulated. A repressor sits on the promoter and blocks transcription; an activator recruits RNA polymerase and cranks it up. Today we derive the **Hill function** -- and you'll see how nature turns a gentle, graded response into a sharp, switch-like one. This is one of the most important functions in all of quantitative biology.

## The committee picture

Before any math, think about it this way. A gene's promoter is like a conference room, and the gene only turns on when the room is occupied by a transcription factor. If only *one* molecule needs to sit down, the gene turns on gradually as concentration increases.

But now suppose the promoter requires a *committee*: four TF molecules must all show up simultaneously. At low concentrations, the chance of all four appearing is tiny. At high concentrations, it's almost certain. The transition from "almost never" to "almost always" happens over a narrow range. You've turned a gentle slope into a switch.

That's the Hill function. The Hill coefficient $n$ is the size of the committee.

## Deriving the Hill function from binding kinetics

### A single transcription factor

Consider a TF that binds to and represses a promoter. The promoter is either **free** (gene on) or **occupied** (gene off):

$$
\frac{\mathrm{d} P_\mathrm{free}}{\mathrm{d} t} = - v_\mathrm{bind} \, c_\mathrm{p} \, P_\mathrm{free} + k_\mathrm{unbind} \, (1 - P_\mathrm{free}).
$$

Binding and unbinding are much faster than transcription, so the promoter reaches quasi-steady state:

$$
P_\mathrm{free} = \frac{1}{1 + c_\mathrm{p}/K}, \qquad \text{where } K = \frac{k_\mathrm{unbind}}{v_\mathrm{bind}}.
$$

> *$K$ is the **dissociation constant** -- the concentration at which the promoter is occupied half the time. Below $K$, the gene is mostly on; above $K$, mostly off.*

### Dimers: the committee of two

If the TF must form a **dimer** before binding, the binding rate goes as $c_\mathrm{p}^2$:

$$
P_\mathrm{free} = \frac{1}{1 + (c_\mathrm{p}/K_2)^2}.
$$

> *The exponent of 2 makes the on-off transition steeper. The gene waits until concentration is high enough that pairs are common.*

### The general Hill function

For a complex of $n$ molecules:

> **Key Equation -- The Hill Function**
> $$
> P_\mathrm{free} = \frac{1}{1 + (c_\mathrm{p}/K)^n}
> $$
> The fraction of time the promoter is free decreases as repressor rises; the Hill coefficient $n$ controls how switch-like the transition is.

Play with this in your head:
* **$n = 1$**: gentle hyperbolic curve.
* **$n = 2$**: steeper transition.
* **$n = 4$**: nearly all-or-nothing, with a narrow switching zone around $c_\mathrm{p} = K$.

And here's the gorgeous part: suddenly a gentle slope becomes a switch. That's how a cell decides "on" or "off" with almost no in-between.

[[simulation hill-function]]

Set $n = 1$ and slowly increase repressor. Watch how gradually the gene dims. Now crank $n$ to 4 and repeat. See the transition sharpen into a switch? Find the 50% point -- that's $K$.

## Repression

For a self-repressing gene:

$$
\frac{\mathrm{d} c_\mathrm{m}}{\mathrm{d} t} = \frac{\alpha_\mathrm{m}}{1 + (c_\mathrm{p}/K)^n} - \Gamma_\mathrm{m} \, c_\mathrm{m},
$$

$$
\frac{\mathrm{d} c_\mathrm{p}}{\mathrm{d} t} = k_\mathrm{p} \, c_\mathrm{m} - \Gamma_\mathrm{p} \, c_\mathrm{p}.
$$

> *When protein is scarce, the Hill function is ~1 and mRNA cranks at full rate. When protein is abundant, the Hill function drops to ~0 and transcription shuts off.*

## Activation

For an activator, the gene needs the TF to be *bound*:

$$
\frac{\mathrm{d} c_\mathrm{m}}{\mathrm{d} t} = \frac{\alpha_\mathrm{m} \, (c_\mathrm{TF}/K)^n}{1 + (c_\mathrm{TF}/K)^n} - \Gamma_\mathrm{m} \, c_\mathrm{m}.
$$

> *The gene is off without the TF and turns on sharply as TF rises past $K$.*

## Regulation by small RNA

Nature has another trick: **small RNAs** that bind mRNA and mark it for destruction. The key idea is **mutual annihilation** -- when sRNA meets mRNA, both are rapidly degraded. This creates a threshold: mRNA is only abundant when its production rate exceeds the sRNA production rate.

## Why does nature do it this way?

Degradation lets a cell change levels quickly. The Hill function adds the ability to *switch* -- making $n$ small for graded control or large for all-or-nothing commitment.

## Check your understanding

* At what repressor concentration is a gene with Hill coefficient $n$ at exactly half its maximal rate?
* Why does dimerization increase the effective Hill coefficient?
* An sRNA and mRNA are produced at equal rates. Steady-state mRNA level?

## Challenge

Plot $f(c) = 1/(1 + (c/K)^n)$ for $K = 1$ and $n = 1, 2, 4, 8$. At what concentration range does the function go from 90% to 10% activity? How does the width of this "switching window" depend on $n$? Can you derive a formula?

## Big ideas

* The Hill function $1/(1 + (c/K)^n)$ is the universal model for cooperative regulation in biology.
* The Hill coefficient $n$ controls steepness: $n = 1$ is gentle, $n \geq 4$ is switch-like.
* The dissociation constant $K$ sets the midpoint: the concentration at which response is half-maximal.

## What comes next

We now have a knob that turns genes up or down. But what happens when the gene product reaches back and turns its *own* knob? That's feedback -- and it changes everything.
