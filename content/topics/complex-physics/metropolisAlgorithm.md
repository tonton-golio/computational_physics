# Metropolis Algorithm

We just learned that the partition function $Z$ is the key to everything. But here is the problem: for an Ising model with $N$ spins, there are $2^N$ possible configurations. For a modest $10 \times 10$ lattice, that is about $10^{30}$ states. You could not sum over them all if you had every computer on Earth running until the heat death of the universe.

So what do we do? We cheat — brilliantly. Instead of summing over all states, we let the computer wander through state space, visiting states with the correct Boltzmann probabilities. This is the Monte Carlo method, and the Metropolis algorithm is its most famous implementation.

## Ising Model

Before we get to the algorithm, let us set up the playground. Ernst Ising introduced a beautifully simple model of ferromagnetism: put a spin on every site of a lattice, and let each spin point either up ($s_i = +1$) or down ($s_i = -1$). The energy is
$$
    \mathcal{H}
    =
    -J \sum_{\langle i j \rangle} s_i s_j - h \sum_i s_i.
$$
Here $\langle i j \rangle$ means we sum over nearest-neighbor pairs (each pair counted once), $J > 0$ is the coupling strength that rewards neighboring spins for pointing the same way, and $h$ is an external magnetic field.

Think of it like peer pressure: every spin wants to agree with its neighbors (when $J > 0$). When $J < 0$ the interaction is antiferromagnetic — spins prefer to alternate. When $J$ varies from pair to pair, you get a spin glass, which is a whole other can of worms.

The magnetization is simply the average spin:
$$
    m = \frac{1}{N} \sum_{i=1}^N s_i.
$$

Ising himself solved the 1D case exactly (no phase transition — we will see why in the Transfer Matrix section). Lars Onsager solved the 2D case in a celebrated tour de force. The 3D Ising model remains unsolved analytically to this day. This is why we need computers.

We can spot the critical temperature of the 2D Ising phase transition by looking for a divergence in the susceptibility $\chi$, which we approximate as the variance of the magnetization:
$$
    \chi = \langle M^2 \rangle - \langle M \rangle^2.
$$

## Monte Carlo method

*Monte Carlo is a district of Monaco famous for its casinos. According to lore, Nicholas Metropolis suggested the name — sampling random numbers to solve physics problems felt a lot like gambling.*

The idea is to estimate thermal averages without summing over all states. The statistical mechanical average of an observable $A$ is
$$
\begin{align*}
    \langle A \rangle
    &=
    \sum_i A_i P_i
    \\&=
    \frac{\sum_i A_i \exp(-\beta E_i)}{\sum_i \exp(-\beta E_i)}.
\end{align*}
$$

If we could generate a sequence of states $\{s^{(1)}, s^{(2)}, \ldots\}$ where state $i$ appears with probability $P_i$, then we could approximate
$$
    \langle A \rangle
    \approx
    \frac{1}{M} \sum_{k=1}^{M} A(s^{(k)}).
$$

That is all Monte Carlo does: replace an impossible sum with a sample average. The Metropolis algorithm tells us *how* to generate those samples correctly.

## Markov process and master equation

The key assumption is that the system evolves as a **Markov process**: what happens next depends only on the current state, not on the history of how we got here. Under this assumption, the probability of being in state $i$ evolves according to the **master equation**:
$$
    \frac{\mathrm{d}P_i}{\mathrm{d}t}
    =
    \sum_j \left( w_{ij}P_j - w_{ji}P_i \right).
$$
Here $w_{ij}$ is the transition rate from state $j$ to state $i$. The first term is probability flowing *into* state $i$; the second is probability flowing *out*.

In steady state ($\mathrm{d}P_i/\mathrm{d}t = 0$), the total inflow equals total outflow:
$$
    \sum_j w_{ij}P_j = \sum_j w_{ji}P_i.
$$

But this is not enough. This condition still allows circular flows (probability sloshing around in loops), which would violate the spirit of thermal equilibrium.

## Detailed balance

To ensure true equilibrium, we impose a stronger condition called **detailed balance**: the flow between *every pair* of states must individually balance.
$$
    w_{ij}P_j = w_{ji}P_i.
$$

No net current between any two states. No loops. This is the condition we need.

## Metropolis-Hastings algorithm

Now we combine detailed balance with the Boltzmann distribution. In equilibrium, $P_i \propto e^{-\beta E_i}$, so
$$
    \frac{w_{ij}}{w_{ji}}
    =
    \frac{P_i}{P_j}
    =
    \exp\left(-\beta \Delta E_{ij}\right),
$$
where $\Delta E_{ij} = E_i - E_j$ is the energy change when transitioning from state $j$ to state $i$.

The Metropolis algorithm turns this into a simple recipe:

1. **Propose** a random move (e.g., flip a random spin).
2. **Calculate** the energy difference $\Delta E$.
3. **If** $\Delta E \leq 0$ (the move lowers the energy): accept it. Always.
4. **If** $\Delta E > 0$ (the move raises the energy): accept it with probability $e^{-\beta \Delta E}$. Draw a random number $r \in [0,1]$; if $r < e^{-\beta \Delta E}$, accept. Otherwise reject and keep the old state.

That is it. This simple accept/reject rule guarantees that after enough steps, the system samples states according to the Boltzmann distribution. You do not need to know $Z$. You do not need to enumerate all states. You just need to compute energy differences, which are cheap.

## Watching the simulation

Here is where things get exciting. Run the Metropolis algorithm on a 2D Ising model and watch the screen.

At **high temperature** the spins flip like mad — the thermal energy overwhelms the coupling, and you see pure noise, a salt-and-pepper mess of up and down spins. The magnetization bounces around zero.

Now **cool it down slowly**. At first, nothing dramatic — just a bit less noise. But then, as you approach the critical temperature $T_c \approx 2.27 \, J/k_\mathrm{B}$, something remarkable happens: **whole regions of aligned spins appear**. Domains of up-spins and down-spins form, grow, merge. The fluctuations become huge. The susceptibility $\chi$ shoots up.

Cool it further below $T_c$, and the entire system snaps into alignment — nearly all spins point the same way. That sudden appearance of global order from local interactions is a **phase transition**, and you just watched it happen in real time.

If we track the energy and magnetization as functions of temperature, we see the signatures: the energy drops, the magnetization jumps from zero to a finite value, and the susceptibility peaks sharply at $T_c$. That peak is the fingerprint of the transition, and it gets sharper as you increase the system size.

> **Key Intuition.** The Metropolis algorithm is nature's own decision-making process, implemented on a computer. At each step, a spin asks: "Would flipping lower the energy? If yes, I flip. If no, I might flip anyway if the temperature is high enough." The competition between energy-lowering and entropy-raising is what drives the entire simulation — and it is the same competition that drives every phase transition in nature.

> **Challenge.** Run a 2D Ising simulation (even a small $20 \times 20$ grid). Estimate $T_c$ by finding the temperature where $\chi$ peaks. Compare your answer with Onsager's exact result $T_c = 2J/(k_\mathrm{B} \ln(1+\sqrt{2})) \approx 2.269\, J/k_\mathrm{B}$. How close can you get?

---

*We have seen that something dramatic happens at a special temperature: the system suddenly orders. But what exactly is a phase transition? What is an order parameter? And can we build a theory that predicts when and how it happens? That is the subject of the next section.*
