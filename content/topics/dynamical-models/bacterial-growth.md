# Bacterial Growth Physiology

> *The universe is not just stranger than we suppose; when it comes to cells, it is stranger than we can suppose -- until we write the equations.*

## Where we are headed

This is the final lesson, and it's where everything comes together. We've modeled individual genes, noise, regulation, feedback, signaling, and networks. Now zoom all the way out and ask: how does a cell *grow*? A bacterium takes in nutrients and converts them into more of itself, doubling every twenty minutes. The answer involves a stunning resource-allocation problem and a single small molecule -- ppGpp -- that acts as the cell's "I'm starving" alarm.

## The growth curve

Jacques Monod, in the 1940s, carefully measured how bacteria grow over time. We focus on the **exponential growth phase**: every component doubles once per generation, nutrient supply is constant, and the population increases exponentially.

There's something profound about it. This enormously complex biochemical machine -- thousands of genes, thousands of reactions -- achieves such simple, predictable behavior.

## Defining steady-state growth

During balanced exponential growth, three conditions hold:

1. **Intrinsic parameters** (composition, ratios) remain constant.
2. **Extensive parameters** (total protein, RNA, mass) increase exponentially with the same doubling time.
3. **Growth conditions** (temperature, nutrients) stay constant.

> *The cell is building a copy of itself, and during balanced growth it does so with perfect proportionality -- everything doubles together.*

## Monod's growth law

Growth rate $\lambda$ depends on the limiting nutrient $S$ through a saturating function:

$$
\lambda = \lambda_\mathrm{max} \frac{S}{K_S + S}.
$$

> *Low nutrient: growth is nearly proportional to $S$. High nutrient: adding more makes no difference. $K_S$ is where the cell grows at half max.*

This is **Michaelis-Menten** -- the same form as a Hill function with $n = 1$. It appears because nutrient-uptake enzymes saturate when substrate is abundant.

[[simulation michaelis-menten]]

Slide the nutrient concentration and watch growth rate respond. At very low $S$, growth tracks nutrients linearly. At high $S$, it plateaus. Find $K_S$: the half-maximal point.

## Growth as protein production

Key simplification: 55% of a bacterium's dry weight is protein. So to a first approximation, **growth = protein synthesis**:

$$
\frac{\mathrm{d} M}{\mathrm{d} t} = \lambda \cdot M.
$$

If protein synthesis is rate-limiting:

$$
\frac{\mathrm{d} M}{\mathrm{d} t} = k \cdot N_R^A,
$$

where $k$ is translation rate per ribosome and $N_R^A$ is the number of actively translating ribosomes. All growth ultimately depends on how many ribosomes the cell has and how fast they work.

## The proteome pie: three slices

Divide the cell's protein into three categories:

* **Metabolic enzymes** ($\phi_\mathrm{P}$): bring nutrients in and process them.
* **Ribosomes** ($\phi_\mathrm{R}$): make protein.
* **Housekeeping** ($\phi_\mathrm{Q}$): DNA replication, maintenance, everything else.

$$
\phi_\mathrm{P} + \phi_\mathrm{R} + \phi_\mathrm{Q} = 1.
$$

> *The cell has a fixed budget. Every ribosome it makes is a metabolic enzyme it didn't make. Growth physiology is fundamentally a problem of resource allocation.*

[[simulation proteome-allocation]]

Adjust nutrient quality and watch the cell redistribute its protein budget. The constraint $\phi_R + \phi_P + \phi_Q = 1$ means every extra ribosome costs a metabolic enzyme. Find the allocation that maximizes growth rate.

## The allocation dilemma

Here's the dilemma. Metabolic enzymes bring nutrients in. Ribosomes convert nutrients into protein. Too many enzymes and not enough ribosomes? Nutrients pile up unused. Too many ribosomes and not enough enzymes? Ribosomes sit idle. The cell must balance the two -- but how does it know when to adjust?

## ppGpp: the cell's "I'm starving" alarm

When a ribosome encounters an uncharged tRNA (no amino acid attached), the cell is running low on that amino acid. This triggers **RelA** to synthesize **ppGpp**.

ppGpp shuts down transcription of ribosomal RNA genes. Fewer rRNA transcripts = fewer new ribosomes. The freed resources go to metabolic enzymes instead.

> *In the default state, bacteria make ribosomes like crazy -- rRNA promoters are among the strongest in the cell. ppGpp is the brake. Plentiful nutrients = low ppGpp = full-speed ribosome production. Scarce nutrients = ppGpp surges = ribosome production halts. An elegant negative feedback loop.*

## The beautiful growth law

Combining our equations:

> **Key Equation -- Growth Rate is Proportional to Ribosome Fraction**
> $$
> \lambda = k \cdot \phi_\mathrm{R}
> $$
> A bacterium's growth rate equals translation speed per ribosome times the fraction of its protein that is ribosomes.

$$
\lambda = k \cdot \frac{N_R^A}{M} \approx k \cdot \phi_\mathrm{R}.
$$

Wait till you see this: the growth rate of a bacterium is *directly proportional* to the fraction of its protein that is ribosomes. Faster-growing cells are literally "more ribosome." This is one of the most robust quantitative relationships in all of microbiology.

## Why does nature do it this way?

The ppGpp system is the culmination of every principle in this course: degradation for speed, negative feedback for homeostasis, adaptation for responding to changes. A single small molecule coordinates the entire proteome, letting a bacterium rapidly reallocate resources in a feast-or-famine world.

And that's how a single cell turns physics into life.

## What Comes Next

This lesson completes a journey from a single differential equation to a whole cell making decisions in real time. The same structures -- feedback loops, noise, bistability, oscillations, network motifs -- reappear at every scale.

The framework extends far beyond bacteria. Cancer cells face their own resource-allocation problem. Immune cells use bistable switches for commitment. Neurons integrate signals through mechanisms resembling genetic circuits. Natural next steps: synthetic biology (designing circuits from scratch), systems biology (larger networks), or statistical mechanics of biological systems.

## Check your understanding

* You just moved from rich to poor medium. What do you do with your ribosome budget?
* Why does ppGpp target rRNA *promoters* specifically, rather than slowing all transcription?
* A cell devotes 50% of protein to ribosomes and 30% to metabolic enzymes. Growing fast or slow?

## Challenge

$\phi_\mathrm{Q} = 0.4$ is fixed, so $\phi_\mathrm{P} + \phi_\mathrm{R} = 0.6$. Nutrient influx $\propto \phi_\mathrm{P}$, growth rate $\propto \phi_\mathrm{R}$. What allocation maximizes growth? Add a constraint: the cell needs $\phi_\mathrm{P} \geq 0.1$ to survive. How does optimal $\phi_\mathrm{R}$ change? Plot growth rate vs $\phi_\mathrm{R}$.

## Big ideas

* Bacterial growth is a resource-allocation problem: ribosomes (growth machinery) vs. metabolic enzymes (nutrient acquisition), constrained by a fixed protein budget.
* ppGpp senses nutrient scarcity through ribosome stalling and shuts down ribosome production -- the master regulator of the proteome.
* Growth rate = $k \cdot \phi_R$ -- one of the most robust quantitative laws in microbiology.
