# Bacterial Growth Physiology

## Where we are headed

This is the final lesson, and it is where everything comes together. We have modeled individual genes, noise, regulation, feedback, signaling, and networks. Now we zoom all the way out and ask the biggest question of all: how does a cell *grow*? How does a bacterium take in nutrients and convert them into more of itself, doubling in size and dividing every twenty minutes? The answer involves a stunning act of resource allocation — the cell must decide, moment by moment, how to divide its limited protein budget between making the machinery for growth (ribosomes) and making everything else. And it turns out that a single small molecule, ppGpp, acts as the cell's internal text message saying "I'm starving — stop making ribosomes!"

## The growth curve

Jacques Monod, working in the 1940s, carefully measured how bacterial populations grow over time and identified several distinct phases. We focus on the **exponential growth phase**, where bacteria have settled into a steady rhythm: every cell component doubles once per generation, the nutrient supply is constant, and the population increases exponentially.

As Frederick C. Neidhardt later wrote about encountering this simple exponential curve for the first time: it was a transforming experience. There is something profound about the fact that this enormously complex biochemical machine — thousands of genes, thousands of reactions — achieves such simple, predictable behavior.

## Defining steady-state growth

During balanced exponential growth, three conditions hold:

1. **Intrinsic parameters** of the cell (composition, ratios of components) remain constant.
2. **Extensive parameters** (total protein, total RNA, cell mass) all increase exponentially with precisely the same doubling time.
3. **Growth conditions** (temperature, nutrient concentrations) remain constant.

> *This is a remarkable state: the cell is a machine that is building a copy of itself, and during balanced growth it does so with perfect proportionality — everything doubles together.*

## Monod's growth law

Monod discovered that the growth rate $\lambda$ depends on the concentration of the limiting nutrient $S$ through a saturating function:

$$
\lambda = \lambda_\mathrm{max} \frac{S}{K_S + S}.
$$

> *At low nutrient concentration, growth rate increases linearly with $S$. At high concentration, the cell is growing as fast as it can and adding more nutrient makes no difference. The half-saturation constant $K_S$ is the concentration at which the cell grows at half its maximum rate.*

This is a **Michaelis-Menten** function — the same mathematical form we saw in the Hill function with $n = 1$. It appears here because nutrient uptake enzymes, like all enzymes, saturate when their substrate is abundant.

[[simulation michaelis-menten]]

> **Try this**: Vary the nutrient concentration $S$ and watch the growth rate $\lambda$ respond. At very low $S$, growth is nearly proportional to nutrients. At very high $S$, adding more makes no difference — the cell is growing as fast as its ribosomes allow. Find $K_S$: the concentration where growth is exactly half-maximal.

## Growth as protein production

Here is a key simplification: 55% of the total dry weight of a bacterium is protein. So to a first approximation, **bacterial growth is protein synthesis**. The equation for total protein mass $M$ is:

$$
\frac{\mathrm{d} M}{\mathrm{d} t} = \lambda \cdot M.
$$

> *This is just exponential growth: the more protein the cell has, the faster it makes more. This is possible because some of that protein is the very machinery (ribosomes) that makes protein.*

If protein synthesis is the rate-limiting step, we can be more specific:

$$
\frac{\mathrm{d} M}{\mathrm{d} t} = k \cdot N_R^A,
$$

where $k$ is the translation rate per ribosome and $N_R^A$ is the total number of actively translating ribosomes.

> *All the cell's growth ultimately depends on how many ribosomes it has and how fast they work. This is the central equation of bacterial growth physiology.*

## The proteome pie: three slices

We can divide the cell's protein into three functional categories:

- **Metabolic and transport proteins**: fraction $\phi_\mathrm{P} = M_\mathrm{P}/M$. These bring nutrients in and process them.
- **Ribosomes** (and associated factors): fraction $\phi_\mathrm{R} = M_\mathrm{R}/M$. These make protein.
- **Everything else** (housekeeping, DNA replication, etc.): fraction $\phi_\mathrm{Q} = M_\mathrm{Q}/M$.

Since these are the only categories:

$$
\phi_\mathrm{P} + \phi_\mathrm{R} + \phi_\mathrm{Q} = 1.
$$

> *The cell has a fixed budget. Every ribosome it makes is a metabolic enzyme it did not make, and vice versa. Growth physiology is fundamentally a problem of **resource allocation**.*

## The allocation dilemma

Here is the dilemma the cell faces. Metabolic enzymes ($\phi_\mathrm{P}$) bring nutrients into the cell. Ribosomes ($\phi_\mathrm{R}$) convert those nutrients into new protein. For the cell to grow efficiently, **nutrient influx must match nutrient usage**. Too many metabolic enzymes and not enough ribosomes? Nutrients pile up but cannot be used. Too many ribosomes and not enough metabolic enzymes? Ribosomes sit idle with nothing to translate.

The cell needs to balance the two — but how does it know when to adjust?

## ppGpp: the cell's "I'm starving" text message

The answer is a remarkable small molecule called **ppGpp** (guanosine pentaphosphate), sometimes called the "alarmone." Here is how it works:

When a ribosome encounters an uncharged tRNA (a tRNA without an amino acid attached), it means the cell is running low on that amino acid — demand exceeds supply. This triggers the enzyme **RelA** to synthesize ppGpp.

ppGpp then acts as a global signal: it **shuts down transcription of ribosomal RNA genes**. Fewer rRNA transcripts means fewer new ribosomes. The freed-up resources are redirected toward metabolic enzymes and other proteins that can fix the shortage.

> *In the default state, bacteria make ribosomes like crazy — ribosomal RNA promoters are among the strongest in the cell. ppGpp is the brake. When nutrients are plentiful, ppGpp levels are low and ribosomes are produced at full speed. When nutrients get scarce, ppGpp surges and ribosome production grinds to a halt. It is an elegant negative feedback loop: too few nutrients → ribosomes stall → ppGpp rises → fewer new ribosomes → resources redirected to metabolism → nutrient supply recovers.*

## Measuring the numbers

Combining our equations, we can connect growth rate to ribosome content. From $\mathrm{d}M/\mathrm{d}t = \lambda M$ and $\mathrm{d}M/\mathrm{d}t = k N_R^A$, we get:

$$
\lambda = k \cdot \frac{N_R^A}{M} \approx k \cdot \phi_\mathrm{R},
$$

where in the last step we used the fact that the ribosome fraction $\phi_\mathrm{R}$ is proportional to $N_R^A / M$.

> *This is a beautiful and testable prediction: the growth rate should be proportional to the ribosome fraction. Faster-growing cells should have more ribosomes. And indeed, this is exactly what experimentalists observe — it is one of the most robust quantitative relationships in all of microbiology.*

## Why does nature do it this way?

A bacterium lives in a feast-or-famine world. In the gut, nutrients arrive in bursts after the host eats, then disappear. The cell that can rapidly ramp up ribosome production when food is abundant — and rapidly shut it down when food runs out — will outcompete its neighbors. The ppGpp system gives bacteria this ability: a single molecule that coordinates the entire proteome in response to nutrient availability. It is a masterpiece of evolutionary engineering.

## Check your understanding

- If you grow *E. coli* in a rich medium where the doubling time is 20 minutes, and then switch to a poor medium where the doubling time is 60 minutes, what happens to the ribosome fraction $\phi_\mathrm{R}$?
- Why is it important that ppGpp targets ribosomal RNA *promoters* specifically, rather than slowing down all transcription equally?
- A cell devotes 50% of its protein to ribosomes and 30% to metabolic enzymes. Is it likely growing fast or slow? Why?

## Challenge

Suppose the housekeeping fraction is fixed at $\phi_\mathrm{Q} = 0.4$, so $\phi_\mathrm{P} + \phi_\mathrm{R} = 0.6$. Imagine that nutrient influx is proportional to $\phi_\mathrm{P}$ and growth rate is proportional to $\phi_\mathrm{R}$. What allocation maximizes growth? Now add a constraint: the cell needs *at least* $\phi_\mathrm{P} = 0.1$ to survive. How does the optimal $\phi_\mathrm{R}$ change? Plot growth rate versus $\phi_\mathrm{R}$ and see the tradeoff for yourself.

## Big ideas

- **Bacterial growth is fundamentally a resource allocation problem**: the cell must balance investment in ribosomes (growth machinery) against metabolic enzymes (nutrient acquisition).
- **ppGpp is the master regulator** that coordinates this balance — it senses nutrient scarcity through ribosome stalling and shuts down ribosome production.
- **Growth rate is proportional to ribosome fraction** — one of the most robust quantitative laws in microbiology, connecting molecular regulation to whole-cell physiology.

And that, in the end, is how a single cell turns physics into life.
