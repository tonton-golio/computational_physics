# Basic Design of Experiments

Here's a secret experienced researchers know: the quality of any analysis depends on how the data was gathered. A well-designed experiment answers your question with 50 observations. A poorly designed one might not answer it with 5000.

## Principles of Experimental Design

Three principles guide every good experiment:

* **Randomization**: Randomly assign units to treatments. Without it, apparent effects might just reflect pre-existing differences. Use a random number generator -- human "random" assignment is notoriously non-random.
* **Replication**: Repeat measurements to estimate variability. One measurement tells you nothing about reliability.
* **Blocking**: Group units by a known source of variability to reduce noise. Think of it as neutralizing a nuisance variable before it contaminates your results.

## Completely Randomized Design (CRD)

Assign all units to treatments purely at random. The model is one-way [ANOVA](./anova):

$$
Y_{ij} = \mu + \tau_i + \varepsilon_{ij}
$$

Simple to implement. But if units vary substantially (different ages, batches, instruments), within-group variance inflates and power drops. When you suspect heterogeneity, block on it.

## Randomized Block Design (RBD)

When a nuisance variable is known (batch, day, subject), blocking removes its effect from the error:

$$
Y_{ij} = \mu + \tau_i + \beta_j + \varepsilon_{ij}
$$

Each treatment appears once per block. The signal stays the same but the noise goes down.

[[simulation randomization-vs-blocking]]

**When in doubt, block.** If the blocking variable turns out unimportant, you lose one degree of freedom per block -- almost nothing. If it matters and you didn't block, you may have run an entire experiment that can't answer your question.

## Power and Sample Size

Before starting, answer the most practical question: *how many observations do I need?*

Alex is designing a clinical trial. The drug should lower blood pressure by about 5 mmHg, and past studies show $\sigma \approx 12$ mmHg. How many patients per group?

Four quantities form a connected system -- fix any three and the fourth is determined:

* **Effect size** ($\delta = 5$ mmHg): what you want to detect.
* **Sample size** ($n$): what you want to find.
* **Significance level** ($\alpha = 0.05$): your false-positive tolerance.
* **Variability** ($\sigma = 12$ mmHg): the noise level.

You need roughly **90 patients per group**. That's the number Alex takes to the funding agency. The reasoning: the effect is small compared to the noise ($d = 5/12 \approx 0.4$, a smallish effect size), so you need many patients to see it reliably above the scatter. Shrink the noise (better instruments, blocking) and you need fewer patients. Bigger effect? Fewer patients. No free lunch.

**Cohen's conventions** for $d = \delta/\sigma$: small ($0.2$), medium ($0.5$), large ($0.8$). These are rough benchmarks -- always think about what's scientifically meaningful in your context.

### Practical Tips

* Determine sample size *before* starting. Running until significance is a recipe for false positives.
* Use pilot studies (even 10-20 observations) to estimate $\sigma$.
* **Pre-register** your analysis plan to avoid p-hacking -- trying many analyses and reporting only the significant ones.

Think of it like a fishing trip. If you keep fishing until you catch something, you'll always catch something -- even in an empty lake. Pre-registration is saying beforehand: "I'm going to fish for exactly two hours in this specific spot." Now your catch means something.

[[simulation interaction-surface]]

## Simpson's Paradox Revisited

We met Simpson's paradox in [hypothesis testing](./hypothesis-testing), but it's fundamentally a *design* problem. Compare two hospitals: Hospital A has higher overall survival. But A mostly treats mild cases while B takes the severe ones. Within each severity category, B actually performs better. The aggregate reverses the truth.

Randomization prevents this. If patients were randomly assigned, severity would be balanced, and the paradox couldn't arise. When you can't randomize (observational studies), you must identify and adjust for confounders -- which often means recognizing hierarchical structure. That's where [mixed models](./random-effects) come in.

> **Challenge.** Explain why you need to decide sample size *before* running the experiment. Use the fishing-trip analogy: if you keep fishing until you catch something, you'll always catch something -- even in an empty lake. One minute.

## Big Ideas

* Blocking costs almost nothing if the variable is unimportant; failing to block when it matters can ruin an entire experiment.
* Power, sample size, effect size, and significance are linked -- fix three, the fourth is determined. No free lunch.
* Running until significance guarantees you'll find something, even in an empty lake. Pre-registration is the defense.
* Simpson's paradox is a design problem: randomization prevents it by balancing confounders.

## What Comes Next

You now know how to plan data collection: randomize, replicate, block, compute sample sizes. Next: analyzing multi-group data. ANOVA compares three or more means in a single test -- the signal-to-noise ratio of group differences vs. individual scatter.

## Check Your Understanding

1. An experimenter runs a CRD with 5 treatments and 10 reps each, then realizes the experiment ran over 5 different days. How should this change the analysis?
2. You want to detect a 10-unit difference with $\sigma = 25$. At $\alpha = 0.05$ and 80% power, roughly how many per group? What if the effect is only 5 units?
3. A pharma company analyzes data after every 50 patients, stopping when significance is reached. Why does this inflate type I error?

## Challenge

You're comparing three teaching methods across four schools. Each school has multiple classrooms, one method per classroom. Sketch a design that accounts for: (a) students within classrooms within schools, (b) balancing methods across schools, (c) detecting a medium effect with 80% power. Identify randomization, blocking, and the unit of replication. Why would treating individual students as independent replicates be a mistake?
