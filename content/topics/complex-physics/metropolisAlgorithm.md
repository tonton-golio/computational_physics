# Metropolis Algorithm

We just learned that the partition function $Z$ is the key to everything. But here's the problem: for an Ising model with $N$ spins, there are $2^N$ possible configurations. For a modest $10 \times 10$ lattice, that's about $10^{30}$ states. You couldn't sum over them all if you had every computer on Earth running until the heat death of the universe.

So what do we do? We cheat -- brilliantly. Instead of summing over all states, we let the computer wander through state space, visiting states with the correct Boltzmann probabilities. This is the Monte Carlo method, and the Metropolis algorithm is its most famous incarnation.

## Big Ideas

* The Metropolis algorithm converts the impossible task of summing $2^N$ states into a random walk through configuration space -- you never need to know $Z$, only energy differences.
* Detailed balance is the key: it's like making sure every doorway between two rooms is equally easy to walk through in both directions -- then the crowd settles into the most probable arrangement automatically.
* At each step, a spin "decides" whether to flip based on energy change and temperature -- the same competition between order and randomness that governs every real phase transition.
* Watching the Ising simulation cool through $T_c$ is worth more than a hundred equations: the moment correlated domains erupt from random noise is the moment you understand why phase transitions are surprising.

## Ising Model

Before we get to the algorithm, let's set up the playground. Ernst Ising introduced a beautifully simple model of ferromagnetism: put a spin on every site of a lattice, and let each spin point either up ($s_i = +1$) or down ($s_i = -1$). The energy is
$$
    \mathcal{H}
    =
    -J \sum_{\langle i j \rangle} s_i s_j - h \sum_i s_i.
$$
Here $\langle i j \rangle$ means we sum over nearest-neighbor pairs, $J > 0$ rewards neighboring spins for pointing the same way, and $h$ is an external magnetic field.

Think of it like peer pressure: every spin wants to agree with its neighbors (when $J > 0$). The magnetization is simply $m = \frac{1}{N} \sum_{i=1}^N s_i$.

Ising solved the 1D case exactly (no phase transition -- we'll see why in the Transfer Matrix section). Lars Onsager solved the 2D case in a celebrated tour de force. The 3D Ising model remains unsolved analytically. That's why we need computers.

## Monte Carlo Method

*Monte Carlo is a district of Monaco famous for its casinos. Nicholas Metropolis suggested the name -- sampling random numbers to solve physics problems felt a lot like gambling.*

The idea: estimate thermal averages without summing over all states. The statistical mechanical average of an observable $A$ is
$$
    \langle A \rangle
    =
    \frac{\sum_i A_i \exp(-\beta E_i)}{\sum_i \exp(-\beta E_i)}.
$$

If you can generate a sequence of states where state $i$ appears with probability $P_i \propto e^{-\beta E_i}$, then you just take a sample average:
$$
    \langle A \rangle
    \approx
    \frac{1}{M} \sum_{k=1}^{M} A(s^{(k)}).
$$

That's all Monte Carlo does: replace an impossible sum with a sample average.

## Detailed Balance

The Metropolis algorithm needs a rule for hopping between states. The key constraint is **detailed balance**: the flow between *every pair* of states must individually balance.
$$
    w_{ij}P_j = w_{ji}P_i.
$$

No net current between any two states. No loops. It's like a bar where every pair of tables must exchange customers at the same rate in both directions -- if two people per minute walk from table A to table B, then two per minute must walk back. If this holds for every pair, the bar is in equilibrium even though people are still moving around.

## Metropolis-Hastings Algorithm

Combine detailed balance with the Boltzmann distribution. In equilibrium, $P_i \propto e^{-\beta E_i}$, so
$$
    \frac{w_{ij}}{w_{ji}}
    =
    \frac{P_i}{P_j}
    =
    \exp\left(-\beta \Delta E_{ij}\right).
$$

The Metropolis algorithm turns this into a dead-simple recipe:

1. **Propose** a random move (e.g., flip a random spin).
2. **Calculate** the energy difference $\Delta E$.
3. **If** $\Delta E \leq 0$ (move lowers energy): accept. Always.
4. **If** $\Delta E > 0$ (move raises energy): accept with probability $e^{-\beta \Delta E}$. Draw $r \in [0,1]$; if $r < e^{-\beta \Delta E}$, accept. Otherwise keep the old state.

That's it. This simple accept/reject rule guarantees that after enough steps, the system samples states according to the Boltzmann distribution. You don't need $Z$. You don't need to enumerate all states. You just need energy differences, which are cheap.

## Watching the Simulation

Here's where things get exciting. Run the Metropolis algorithm on a 2D Ising model and watch the screen.

At **high temperature** the spins flip like mad -- thermal energy overwhelms the coupling, and you see pure noise, a salt-and-pepper mess. Magnetization bounces around zero.

Now **cool it down slowly**. At first, nothing dramatic. But as you approach $T_c \approx 2.27 \, J/k_\mathrm{B}$, something remarkable happens: **whole regions of aligned spins appear**. Domains form, grow, merge. The fluctuations become huge. The susceptibility shoots up.

Cool it further below $T_c$, and the entire system snaps into alignment. That sudden appearance of global order from local interactions is a **phase transition**, and you just watched it happen in real time.

## What Comes Next

The Metropolis simulation reveals something dramatic: as temperature drops through a critical value, the system snaps from a disordered mess into a coherent, magnetized state. But the simulation doesn't tell us *why* this happens, or how to predict *when* it will happen.

[Phase Transitions](phaseTransitions) builds the theoretical framework. The key concept is the **order parameter** -- a quantity that's zero in the disordered phase and nonzero in the ordered phase. We'll derive the mean-field Hamiltonian, which replaces the tangled many-body problem with a self-consistent single-body problem.

## Check Your Understanding

1. The Metropolis algorithm always accepts moves that lower energy, but only sometimes accepts moves that raise it. Why is it essential to sometimes accept uphill moves -- what goes wrong if you never do?
2. At high temperature the simulation looks like random noise, but at low temperature almost all spins align. Why does the system "choose" one direction rather than staying at zero magnetization?

## Challenge

The Metropolis algorithm requires computing $\Delta E$ when a single spin flips. Show that for the 2D nearest-neighbor Ising model, $\Delta E$ depends only on the four neighboring spins, not on the entire lattice. Use this to estimate how many floating-point operations per second a simple Metropolis implementation can perform on a modern CPU. How long would it take to simulate $10^9$ spin-flip attempts on a $100 \times 100$ lattice?
