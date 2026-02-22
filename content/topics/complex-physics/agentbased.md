# Agent-Based Models and Stochastic Simulation

No bird in a flock has a map of the formation. No driver on a highway knows the state of every other car. No neuron in your brain has a blueprint of your thoughts. And yet: the flock moves as one, traffic jams form and dissolve in waves, and your brain somehow thinks.

The amazing thing is always the same: **no agent has a global plan, yet order emerges**. Simple local rules -- follow your nearest neighbor, slow down if the car ahead is close, fire if enough of your neighbors fire -- produce complex, coordinated global behavior. That connects directly to everything we've studied: Ising spins producing phase transitions, sandpile grains producing power-law avalanches, network nodes producing scale-free structure.

The power of "stupid" local rules producing "smart" global behavior is one of the deepest themes in complex physics.

## Big Ideas

* Emergence from simple local rules is the central theme. Agent-based models make it concrete: flocks, traffic, epidemics all show coherent macroscopic patterns from agents with no global plan.
* The Gillespie algorithm is exact stochastic simulation: it generates sample paths from the master equation one reaction at a time, capturing fluctuations that ODEs completely miss.
* Spatial structure matters: agent-based predator-prey produces traveling waves and local extinctions that mean-field Lotka-Volterra equations say are impossible.
* The Vicsek flocking model is an Ising model for velocities -- the transition from disordered motion to coordinated flocking is a genuine phase transition.

## Cellular Automata: the Game of Life

**Conway's Game of Life** is the canonical deterministic agent-based model. On a 2D grid, each cell is alive or dead, updating based on its eight neighbors:

* A live cell **survives** with exactly 2 or 3 live neighbors; otherwise it dies.
* A dead cell **is born** with exactly 3 live neighbors.

[[simulation game-of-life]]

From a random start, you see gliders translating across the grid, oscillators pulsing with fixed periods, complex structures that interact and sometimes produce entirely new structures. The Game of Life is Turing-complete -- it can simulate any computation. All from two rules on a grid.

Despite this simplicity, predicting long-term behavior is generally impossible without running it. Simple rules, unpredictable outcomes. That's complexity.

## Stochastic Simulation and the Gillespie Algorithm

When reactions involve small numbers of molecules, deterministic rate equations fail. The **Gillespie algorithm** provides exact trajectories from the chemical master equation:

1. Compute all reaction **propensities** $a_i$.
2. Total propensity $a_0 = \sum_i a_i$.
3. Draw **waiting time**: $\Delta t \sim \text{Exp}(a_0)$.
4. Select **which reaction** fires with probability $a_i / a_0$.
5. Update state and repeat.

You simulate one reaction at a time, but capture fluctuations that deterministic equations completely miss. In small systems (a few dozen molecules in a cell), these fluctuations can drive switches, oscillations, extinctions.

## Flocking: the Vicsek Model

Each bird aligns its velocity with its neighbors, plus noise. No bird knows the global pattern, yet the flock moves as a coordinated whole. The transition from disordered to ordered motion is a phase transition -- just like the Ising model, but for velocities instead of spins.

[[simulation vicsek-flocking]]

## Traffic Flow

Each driver follows simple rules -- accelerate if there's space, brake if the car ahead is close, randomly slow down sometimes. No driver has a map of the whole highway, yet traffic jams emerge as traveling waves propagating backward through the flow.

## Spatial Predator-Prey

The **Lotka-Volterra** ODEs predict sustained oscillations:

$$
\frac{dR}{dt} = \alpha R - \beta R F, \qquad \frac{dF}{dt} = \delta R F - \gamma F.
$$

Put this on a spatial grid as agents -- rabbits reproducing, foxes hunting nearby rabbits, both moving randomly -- and you see phenomena invisible to the ODEs: spatial clustering (predators chase prey in traveling waves), local extinctions, and stochastic fluctuations that can drive global extinction. Something the deterministic equations say is impossible.

Mean-field tells you what happens on average. Agents live in a world of fluctuations, space, and individual histories. The average can be deeply misleading.

## What Comes Next

Agent-based models distill complex systems to their purest form: agents, rules, interactions, and emergent behavior. In [Econophysics](econophysics), we apply this entire framework to financial markets -- one of the most consequential and least understood complex systems. Markets exhibit power-law events (crashes and rallies), fat tails, volatility clustering -- the same signatures we've seen in sandpiles, percolation clusters, and scale-free networks. The question: are markets just another self-organized critical system?

## Check Your Understanding

1. Conway's Game of Life is deterministic -- same initial state always produces the same trajectory. How can a deterministic system be unpredictable?
2. The Gillespie algorithm draws waiting times from an exponential distribution with rate $a_0$. Why exponential? What assumption about the underlying process makes this correct?
3. In the spatial predator-prey model, predators can wipe out local rabbit populations even when the global population is healthy. Why does spatial structure create this vulnerability that the mean-field ODE hides?

## Challenge

Implement the Vicsek model: $N$ point particles at constant speed $v_0$ on a 2D periodic domain. At each step, each particle updates its heading to the average heading within radius $r$, plus noise $\eta$. For $v_0 = 0.5$, $r = 1$, $N = 300$, vary $\eta$ from 0 to $2\pi$. Find $\eta_c$ where order-disorder transition occurs. Compute the order parameter (magnitude of mean velocity vector) as a function of $\eta$. Does the transition look continuous or discontinuous? Compare with the Ising transition.
