# Agent-Based Models and Stochastic Simulation

No bird in a flock has a map of the formation. No driver on a highway knows the state of every other car. No neuron in your brain has a blueprint of your thoughts. And yet: the flock moves as one, traffic jams form and dissolve in waves, and your brain somehow thinks.

In all these cases, the amazing thing is the same: **no agent has a global plan, yet order emerges**. Simple local rules — follow your nearest neighbor, slow down if the car ahead is close, fire if enough of your neighbors fire — produce complex, coordinated global behavior. That is the central insight of agent-based modeling, and it connects directly to everything we have studied: the Ising spins following local rules that produce phase transitions, the sandpile grains following local rules that produce power-law avalanches, the network nodes following local rules that produce scale-free structure.

The power of "stupid" local rules producing "smart" global behavior is one of the deepest themes in complex physics.

## From equations to agents

In many systems, the relevant actors are discrete individuals, not continuous fields. **Agent-based models** (ABMs) simulate autonomous agents following local rules, and emergent collective behavior arises from their interactions.

ABMs are particularly powerful when:

- The population is **heterogeneous** (agents differ in attributes or behavior).
- Spatial structure matters (local interactions dominate over global averages).
- Stochasticity at the individual level drives macroscopic phenomena.

## Cellular automata: the Game of Life

**Conway's Game of Life** is the canonical deterministic agent-based model. On a 2D grid, each cell is alive or dead, and its state updates synchronously based on its eight neighbors:

- A live cell **survives** if it has exactly 2 or 3 live neighbors; otherwise it dies (of loneliness or overcrowding).
- A dead cell **is born** if it has exactly 3 live neighbors (just the right amount of community).

[[simulation game-of-life]]

Watch the simulation and something astonishing happens: from a random initial pattern, you see gliders that translate across the grid, oscillators that pulse with fixed periods, and complex structures that interact, collide, and sometimes produce entirely new structures. The Game of Life is actually Turing-complete — it can in principle simulate any computation. All from two rules on a grid.

Despite these simple rules, predicting the long-term behavior of a given initial configuration is in general impossible without actually running the simulation. This is the hallmark of complexity: simple rules, unpredictable outcomes.

## Stochastic simulation and the Gillespie algorithm

When reactions involve small numbers of molecules, deterministic rate equations (ODEs) fail to capture the inherent randomness. The **Gillespie algorithm** (stochastic simulation algorithm) provides exact trajectories from the chemical master equation.

At each step:

1. Compute all reaction **propensities** $a_i$ (rate $\times$ number of reactant combinations).
2. Compute the total propensity $a_0 = \sum_i a_i$.
3. Draw the **waiting time** to the next reaction: $\Delta t \sim \text{Exp}(a_0)$.
4. Select **which reaction** fires with probability $a_i / a_0$.
5. Update the state and repeat.

The algorithm generates sample paths that are statistically exact solutions of the master equation. The price is that you simulate one reaction at a time — but the payoff is that you capture fluctuations that deterministic equations completely miss. In small systems (a few dozen molecules in a cell), these fluctuations can drive qualitatively different behavior: switches, oscillations, extinctions.

## Predator-prey dynamics

The **Lotka-Volterra model** describes the interaction between predators (foxes) and prey (rabbits). The ODE (mean-field) version predicts sustained oscillations:

$$
\frac{dR}{dt} = \alpha R - \beta R F, \qquad \frac{dF}{dt} = \delta R F - \gamma F.
$$

But now put this on a spatial grid as an agent-based model:

- Rabbits reproduce with some probability at each step.
- Foxes eat nearby rabbits and reproduce; they die if they go too long without eating.
- Both species move randomly on a spatial grid.

The agent-based version reveals phenomena invisible to the ODEs: spatial clustering (predators chase prey in traveling waves), local extinctions (an island of rabbits gets wiped out even though the global population is fine), and stochastic fluctuations that can drive one species to *global* extinction — something the deterministic equations say is impossible.

This is a recurring lesson: mean-field equations tell you what happens on average, but agents live in a world of fluctuations, space, and individual histories. The average can be misleading.

## Random walks and diffusion

**Random walks** are the simplest stochastic models — and they keep showing up. We saw them in the first-return time analysis of SOC. Here they are again as the foundation of diffusion.

A walker on a lattice takes steps in random directions at each time step. Key results for an unbiased random walk in $d$ dimensions:

- Mean displacement: $\langle \mathbf{r}(t) \rangle = 0$ (no net drift).
- Mean-squared displacement: $\langle r^2(t) \rangle = 2d \, D \, t$ (spreads as $\sqrt{t}$).
- **Recurrence**: the walker returns to the origin with probability 1 in 1D and 2D, but not in 3D and higher (**Polya's theorem**). A drunk person will eventually find their way home on a 2D street grid, but a drunk bird in 3D space may wander forever.

The **Langevin equation** provides a continuous-time description:

$$
\frac{dx}{dt} = -\frac{\partial U}{\partial x} + \sqrt{2D} \, \xi(t),
$$

where $\xi(t)$ is Gaussian white noise with $\langle \xi(t) \xi(t') \rangle = \delta(t - t')$.

## Flocking, traffic, and the theme of emergence

Let us tie together the applications that make agent-based models so compelling:

- **Flocking** (Vicsek model): each bird aligns its velocity with its neighbors, plus some noise. No bird knows the global pattern, yet the flock moves as a coordinated whole. The transition from disordered to ordered motion is a phase transition — just like the Ising model, but for velocities instead of spins.

- **Traffic flow** (Nagel-Schreckenberg model): each driver follows simple rules — accelerate if there is space, brake if the car ahead is close, randomly slow down sometimes. No driver has a map of the whole highway, yet traffic jams emerge as traveling waves that propagate backward through the flow.

- **Epidemics**: each person can be susceptible, infected, or recovered (SIR). Transmission depends on local contacts. No one knows the global state of the epidemic, yet complex spatial patterns of infection emerge, especially on the scale-free networks we studied earlier.

In every case, the story is the same: local rules, no global plan, and yet coherent macroscopic patterns emerge. That is the power and the beauty of agent-based modeling.

[[simulation lorenz-attractor]]

This simulation shows the Lorenz attractor — another system where simple deterministic rules produce complex, unpredictable behavior. The trajectory never repeats, yet it stays confined to a beautiful butterfly-shaped structure. Deterministic chaos from three simple equations.

> **Key Intuition.** Agent-based models show that complex global behavior can emerge from simple local rules. No agent needs a plan; no one needs to see the big picture. Flocking, traffic jams, epidemics, and even the Game of Life all demonstrate the same principle: interactions between many simple agents produce patterns that no individual agent "intended." This is emergence — the central theme of complex physics.

> **Challenge.** Implement Conway's Game of Life on a small grid (say $30 \times 30$). Start with a random initial state where each cell is alive with probability $0.5$. Run it for 200 steps. Does the population stabilize? Try different initial densities ($0.1$, $0.3$, $0.7$). What fraction of the grid is alive in the long run? Does the initial density matter much?

---

*We have seen local rules produce global order in physical, biological, and social systems. For our final section, we turn to one of the most consequential complex systems of all: financial markets. The same power laws, the same heavy tails, the same avalanche-like dynamics show up in stock prices. Econophysics applies everything we have learned to understand why markets crash.*
