# Agent-Based Models and Stochastic Simulation

## From equations to agents

In many systems, the relevant actors are discrete individuals, not continuous fields. **Agent-based models** (ABMs) simulate autonomous agents following local rules, and emergent collective behavior arises from their interactions.

ABMs are particularly powerful when:

- The population is **heterogeneous** (agents differ in attributes or behavior).
- Spatial structure matters (local interactions dominate over global averages).
- Stochasticity at the individual level drives macroscopic phenomena.

## Cellular automata: the Game of Life

**Conway's Game of Life** is the canonical deterministic agent-based model. On a 2D grid, each cell is alive or dead, and its state updates synchronously based on its eight neighbors:

- A live cell **survives** if it has exactly 2 or 3 live neighbors; otherwise it dies.
- A dead cell **is born** if it has exactly 3 live neighbors.

Despite these simple rules, the Game of Life produces remarkable emergent behavior: gliders that translate across the grid, oscillators with fixed periods, and self-replicating structures. It is Turing-complete, meaning it can in principle simulate any computation.

[[simulation game-of-life]]

## Stochastic simulation and the Gillespie algorithm

When reactions involve small numbers of molecules, deterministic rate equations (ODEs) fail to capture the inherent randomness. The **Gillespie algorithm** (stochastic simulation algorithm) provides exact trajectories from the chemical master equation.

At each step:

1. Compute all reaction **propensities** $a_i$ (rate $\times$ number of reactant combinations).
2. Compute the total propensity $a_0 = \sum_i a_i$.
3. Draw the **waiting time** to the next reaction: $\Delta t \sim \text{Exp}(a_0)$.
4. Select **which reaction** fires with probability $a_i / a_0$.
5. Update the state and repeat.

The algorithm generates sample paths that are statistically exact solutions of the master equation, at the cost of simulating one reaction at a time.

## Predator-prey dynamics

The **Lotka-Volterra model** describes the interaction between predators (foxes) and prey (rabbits). In the agent-based version:

- Rabbits reproduce with some probability at each step.
- Foxes eat nearby rabbits and reproduce; they die if they go too long without eating.
- Both species move randomly on a spatial grid.

The ODE (mean-field) version predicts sustained oscillations:

$$
\frac{dR}{dt} = \alpha R - \beta R F, \qquad \frac{dF}{dt} = \delta R F - \gamma F.
$$

The agent-based version reveals phenomena invisible to the ODE: spatial clustering, local extinctions, and stochastic fluctuations that can drive one species to global extinction.

## Random walks and diffusion

**Random walks** are the simplest stochastic models. A walker on a lattice takes steps in random directions at each time step.

Key results for an unbiased random walk in $d$ dimensions:

- Mean displacement: $\langle \mathbf{r}(t) \rangle = 0$.
- Mean-squared displacement: $\langle r^2(t) \rangle = 2d \, D \, t$, where $D$ is the diffusion coefficient.
- **Recurrence**: the walker returns to the origin with probability 1 in 1D and 2D, but not in 3D and higher (**Polya's theorem**).

The **Langevin equation** provides a continuous-time description:

$$
\frac{dx}{dt} = -\frac{\partial U}{\partial x} + \sqrt{2D} \, \xi(t),
$$

where $\xi(t)$ is Gaussian white noise with $\langle \xi(t) \xi(t') \rangle = \delta(t - t')$.

## Applications of agent-based models

- **Epidemics**: SIR models on contact networks, with heterogeneous transmission rates and super-spreaders.
- **Opinion dynamics**: voter models, bounded-confidence models, and polarization.
- **Flocking**: Vicsek model, where agents align their velocity with neighbors plus noise, producing collective motion.
- **Traffic flow**: Nagel-Schreckenberg model for traffic jams as emergent phenomena.
- **Social networks**: information cascades and the spread of content through heterogeneous networks.

The advantage of agent-based modeling is its flexibility: any mechanism can be incorporated at the individual level, and the macroscopic behavior is observed rather than assumed.

[[simulation lorenz-attractor]]
