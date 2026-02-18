# Basic Design of Experiments

So far, you've developed tools for analyzing data *after* it's been collected. But here's a secret that experienced researchers know: the quality of any statistical analysis depends critically on how the data was gathered in the first place. A well-designed experiment can answer your question with 50 observations; a poorly designed one might not answer it with 5000.

Experimental design is where you invest thought *before* spending time and money on data collection. The payoff is enormous.

## Principles of Experimental Design

Three foundational principles guide every well-designed experiment:

- **Randomization**: Randomly assign experimental units to treatments to eliminate systematic bias. Without randomization, apparent treatment effects might simply reflect pre-existing differences between groups. **Practical tip**: use a random number generator, not convenience or judgment. Human "random" assignment is notoriously non-random.
- **Replication**: Repeat measurements to estimate variability and increase precision. A single measurement tells you nothing about reliability. **Practical tip**: determine how many replicates you need *before* starting — we'll see how in the power analysis section below.
- **Blocking**: Group experimental units by a known source of variability to reduce noise. If you know that batches, days, or operators introduce variability, account for it by design rather than hoping it averages out. Think of it as neutralizing a nuisance variable before it can contaminate your results.

## Completely Randomized Design (CRD)

The simplest design assigns all experimental units to treatments purely at random. The CRD is appropriate when units are homogeneous and no blocking variable is identified.

The model is identical to one-way ANOVA (lesson 7):

$$
Y_{ij} = \mu + \tau_i + \varepsilon_{ij},
$$

where $\tau_i$ is the treatment effect.

**Advantages**: simple to implement and analyze.
**Limitation**: if units vary substantially (different ages, batches, instruments), the within-group variance is inflated and power drops. When you suspect heterogeneity, don't ignore it — block on it.

## Randomized Block Design (RBD)

When a nuisance variable is known (batch, day, subject), blocking removes its effect from the error term:

$$
Y_{ij} = \mu + \tau_i + \beta_j + \varepsilon_{ij},
$$

where $\beta_j$ is the block effect. Each treatment appears exactly once in each block.

**Advantage**: removes block-to-block variability from the error term, increasing the F-statistic for the treatment effect. The signal stays the same but the noise goes down.

The **relative efficiency** of blocking compares the precision of RBD to CRD:

$$
\text{RE} = \frac{\text{MS}_{\text{blocks}} + (b-1)\,\text{MS}_{\text{error,RBD}}}{b\,\text{MS}_{\text{error,RBD}}},
$$

where $b$ is the number of blocks. Values greater than 1 mean blocking was beneficial. In practice, blocking almost always helps — it rarely hurts and often improves power substantially.

**Practical tip**: when in doubt, block. If the blocking variable turns out to be unimportant, you lose very little (just one degree of freedom per block). If it *is* important and you failed to block, you lose much more.

## Power and Sample Size

Before starting an experiment, you should answer the most practical question of all: *how many observations do I need?*

### The "How Many Patients Do I Really Need?" Story

Alex is designing a clinical trial. The new drug is expected to lower blood pressure by about 5 mmHg compared to the control. Alex knows from past studies that blood pressure measurements have a standard deviation of about 12 mmHg. How many patients per group does Alex need?

The answer depends on four quantities that form a connected system — changing one affects the others.

**Statistical power** is the probability of correctly rejecting $H_0$ when a true effect exists:

$$
\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_1 \text{ true}).
$$

Power depends on:

- **Effect size** ($\delta$): the magnitude of the difference you want to detect. Smaller effects need more data. Alex's 5 mmHg is the effect size.
- **Sample size** ($n$): more observations increase power. This is what Alex wants to find.
- **Significance level** ($\alpha$): relaxing $\alpha$ increases power at the cost of more false positives.
- **Variability** ($\sigma$): less noise increases power. This is where good experimental design (blocking, precise instruments) pays off. Alex's $\sigma = 12$ mmHg is the noise.

For a two-sample t-test, the required sample size per group to achieve power $1 - \beta$ at level $\alpha$ for detecting a difference $\delta$ is approximately:

$$
n \approx \frac{2(z_{\alpha/2} + z_\beta)^2 \sigma^2}{\delta^2}.
$$

Plugging in Alex's numbers ($\delta = 5$, $\sigma = 12$, $\alpha = 0.05$, power $= 0.80$): $n \approx 2(1.96 + 0.84)^2 \times 144 / 25 \approx 90$ patients per group. That's a real number Alex can take to the funding agency.

**Cohen's conventions** for effect size $d = \delta/\sigma$: small ($d = 0.2$), medium ($d = 0.5$), large ($d = 0.8$). These are rough benchmarks — always think about what effect size is scientifically meaningful in your specific context rather than blindly adopting conventions.

### Practical Tips for Power Analysis

- Always determine sample size *before* starting the experiment. Running until you get a significant result is a recipe for false positives.
- Use **pilot studies** to estimate $\sigma$ when it is unknown. Even a small pilot (10-20 observations) gives a rough variance estimate that makes your power calculation much more reliable than guessing.
- **Pre-register** the analysis plan to avoid p-hacking — the temptation to try many analyses and report only the significant ones.
- Consider **multiple testing corrections** when evaluating many endpoints. The more tests you run, the more likely one will be "significant" by chance.

## Simpson's Paradox Revisited

We met Simpson's paradox briefly in lesson 6, but it deserves special attention here because it's fundamentally a *design* problem. Here's how smart people get fooled by ignoring structure in their data.

You compare two hospitals. Hospital A has a higher overall survival rate. You conclude Hospital A is better. But Hospital A mostly treats mild cases, while Hospital B takes the severe ones. Within each severity category, Hospital B actually performs better. The aggregate reverses the trend because the groups were not comparable.

This is exactly the kind of confounding that randomization prevents. If patients were randomly assigned to hospitals, severity would be balanced across both, and Simpson's paradox couldn't arise. When you can't randomize (observational studies), you must identify and adjust for confounders — which often means recognizing that your data has a grouped or hierarchical structure. That's where mixed models (lesson 9) come in.

Now you know how to plan data collection. But what happens when the data you collect has a natural grouping structure — students in classrooms, patients in hospitals, cells in experimental plates? Ignoring that structure is one of the most common mistakes in applied statistics. Mixed models fix it, and that's where we go next.

> **Challenge.** Explain to a friend why you need to decide your sample size *before* running the experiment, not during it. Use the analogy of a fishing trip: if you keep fishing until you catch something, you'll always catch something — even in an empty lake. One minute.
