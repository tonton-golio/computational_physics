# Basic Design of Experiments

## Principles of experimental design

Good experimental design maximizes the information obtained while controlling for confounding factors. Three foundational principles guide every well-designed experiment:

- **Randomization**: randomly assign experimental units to treatments to eliminate systematic bias.
- **Replication**: repeat measurements to estimate variability and increase precision.
- **Blocking**: group experimental units by a known source of variability to reduce noise.

## Completely randomized design (CRD)

The simplest design assigns all experimental units to treatments purely at random. The CRD is appropriate when units are homogeneous and no blocking variable is identified.

The model is identical to one-way ANOVA:

$$
Y_{ij} = \mu + \tau_i + \varepsilon_{ij},
$$

where $\tau_i$ is the treatment effect.

**Advantages**: simple to implement and analyze.
**Limitation**: if units vary substantially, the within-group variance is inflated, reducing power.

## Randomized block design (RBD)

When a nuisance variable is known (e.g., batch, day, or subject), blocking removes its effect:

$$
Y_{ij} = \mu + \tau_i + \beta_j + \varepsilon_{ij},
$$

where $\beta_j$ is the block effect. Each treatment appears exactly once in each block.

**Advantage**: removes block-to-block variability from the error term, increasing the F-statistic for the treatment effect.

The **relative efficiency** of blocking compares the precision of RBD to CRD:

$$
\text{RE} = \frac{\text{MS}_{\text{blocks}} + (b-1)\,\text{MS}_{\text{error,RBD}}}{b\,\text{MS}_{\text{error,RBD}}},
$$

where $b$ is the number of blocks. Values greater than 1 indicate that blocking was beneficial.

## Power and sample size

**Statistical power** is the probability of correctly rejecting $H_0$ when a true effect exists:

$$
\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_1 \text{ true}).
$$

Power depends on:

- **Effect size** ($\delta$): the magnitude of the difference to detect.
- **Sample size** ($n$): more observations increase power.
- **Significance level** ($\alpha$): relaxing $\alpha$ increases power at the cost of more false positives.
- **Variability** ($\sigma$): less noise increases power.

For a two-sample t-test, the required sample size per group to achieve power $1 - \beta$ at level $\alpha$ for detecting a difference $\delta$ is approximately:

$$
n \approx \frac{2(z_{\alpha/2} + z_\beta)^2 \sigma^2}{\delta^2}.
$$

**Cohen's conventions** for effect size $d = \delta/\sigma$: small ($d = 0.2$), medium ($d = 0.5$), large ($d = 0.8$).

## Practical considerations

- Always determine sample size *before* starting the experiment.
- Pre-register the analysis plan to avoid p-hacking.
- Use pilot studies to estimate $\sigma$ when it is unknown.
- Consider multiple testing corrections when evaluating many endpoints.

[[simulation applied-stats-sim-5]]
