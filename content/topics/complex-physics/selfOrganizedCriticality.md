# Self-Organized Criticality

## The concept of SOC

**Self-organized criticality** (SOC) is a property of dynamical systems that spontaneously evolve toward a critical state without any external tuning of parameters. Unlike equilibrium phase transitions, where criticality requires fine-tuning the temperature to $T_c$, SOC systems naturally drive themselves to the critical point.

At criticality, the system displays **scale invariance** in both space and time: events of all sizes occur, with their frequency following power-law distributions.

Key signatures of SOC:

- Power-law distributed event sizes (avalanches).
- $1/f$ noise in temporal fluctuations.
- Fractal spatial structure.
- Separation of time scales between slow driving and fast relaxation.

## The sandpile model (BTW)

The **Bak-Tang-Wiesenfeld sandpile model** (1987) is the paradigmatic example of SOC.

On a 2D grid, each site $i$ has a height $z_i$. Sand is added one grain at a time at random sites. When $z_i$ exceeds a threshold $z_c$ (typically 4 for a square lattice), the site **topples**:

$$
z_i \to z_i - 4, \qquad z_j \to z_j + 1 \quad \text{for each neighbor } j.
$$

Toppling can trigger neighbors to topple in turn, creating an **avalanche**. Grains that topple off the boundary are lost (open boundary conditions).

After a transient, the system reaches a statistically steady state where the avalanche size distribution follows a power law:

$$
P(s) \sim s^{-\tau},
$$

with $\tau \approx 1.1$ in 2D. The distribution of avalanche areas, durations, and lifetimes also follow power laws.

[[simulation sandpile-model]]

## The Bak-Sneppen model

The **Bak-Sneppen model** (1993) applies SOC to biological evolution. Consider $N$ species arranged on a ring, each with a random fitness value $f_i \in [0, 1]$.

At each time step:

1. Find the species with the **lowest fitness**.
2. Replace its fitness and the fitness of its two neighbors with new random values from $[0, 1]$.

The model self-organizes to a critical state where most fitness values exceed a threshold $f_c \approx 0.667$ (in 1D). Below this threshold, species are unstable and trigger cascading replacements, which are the **avalanches** of the model.

Avalanches in the Bak-Sneppen model are correlated both **spatially** (consecutive replacements cluster in space) and **temporally** (avalanche durations are power-law distributed).

[[simulation bak-sneppen]]

## First-return times of random walks

The connection between SOC and random walks provides theoretical insight. Consider a 1D random walk starting at the origin. The **first-return time** $T$ is the number of steps until the walker returns to the origin.

For an unbiased random walk:

$$
P(T = 2n) \sim n^{-3/2},
$$

a power law with exponent $-3/2$. This result connects to SOC because avalanches in many SOC models can be mapped to random-walk first-return problems.

[[simulation random-walk-first-return]]

## Non-equilibrium steady states

SOC systems are inherently **out of equilibrium**: energy (or sand, or fitness) is continuously injected and dissipated. The critical state is a non-equilibrium steady state where the rate of injection balances the average rate of dissipation.

This distinguishes SOC from equilibrium critical phenomena:

- No detailed balance.
- No free energy or partition function.
- The steady state is maintained by the balance of driving and dissipation.
- Fluctuations (avalanches) are the mechanism of transport.

## Power laws in nature

SOC has been proposed as an explanation for power-law distributions observed in:

- **Earthquakes**: the Gutenberg-Richter law $\log N(M) = a - bM$ relates earthquake magnitude $M$ to frequency.
- **Forest fires**: fire size distributions in some models follow power laws.
- **Neuronal avalanches**: cascades of neural activity in cortical circuits show power-law size distributions.
- **Solar flares**: the energy distribution of solar flares follows a power law over several decades.

Not all power laws indicate SOC. Alternative mechanisms include preferential attachment, multiplicative processes, and finite-size effects. Careful statistical testing is needed to distinguish true power laws from other heavy-tailed distributions.
