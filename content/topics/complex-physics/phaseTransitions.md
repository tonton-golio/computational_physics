# Phase Transitions

Pick up a refrigerator magnet. It sticks to your fridge because trillions of tiny atomic magnets inside it have collectively decided to point the same direction. Now heat that magnet with a blowtorch. At first nothing much changes. But at a certain temperature — the Curie temperature — the magnet suddenly stops being a magnet. The atomic spins are still there, still interacting, but the thermal jiggling has overwhelmed their desire to align. Above that temperature it is just a lump of iron with no north or south pole. Below it, the whole thing points the same way.

That sudden appearance of a preferred direction is what we call an **order parameter** — a quantity that is zero in the disordered phase and nonzero in the ordered phase. For a magnet, the order parameter is the magnetization $m$. For a liquid-gas transition, it is the density difference. The order parameter is the signature that tells you something dramatic has happened.

## Ising Model Simulation

Watch the Ising model in action. At high temperature, the spins are a random mess — the magnetization is zero on average. As you cool toward $T_c$, domains of aligned spins start forming and growing. At $T_c$ itself, the fluctuations are enormous: the system cannot decide which way to point. Below $T_c$, one direction wins, and the magnetization becomes nonzero. That is the phase transition.

[[simulation ising-model]]

In this simulation, you are watching millions of spins make a collective decision — no leader, no blueprint, just nearest-neighbor interactions. The fact that global order emerges from purely local rules is one of the deepest surprises in physics.

## Mean-field Hamiltonian

Can we build a simple theory of this transition? The exact Hamiltonian of the Ising model couples every spin to its neighbors:
$$
    \mathcal{H}
    =
    -J \sum_{\langle i j \rangle} s_i s_j - h \sum_i s_i.
$$

The trouble is the $s_i s_j$ coupling — it ties every spin to its neighbors, and those neighbors to *their* neighbors, creating a tangled web of correlations.

The **mean-field approximation** cuts this knot with a beautifully simple idea: instead of feeling the actual fluctuating spins of its neighbors, each spin feels only the *average* field from all of them. Imagine being in a crowd where everyone is trying to face the same direction. You do not look at each individual — you just feel the general pull of the crowd.

## Mean-field Hamiltonian (derivation)

To make the mean-field idea precise, we decompose each spin into its mean and a fluctuation:
$$
    s_i = \langle s_i \rangle + \delta s_i = m + \delta s_i,
$$
where $m = \langle s_i \rangle$ is the magnetization per spin (all spins are equivalent by symmetry).

The product of two neighboring spins becomes:
$$
\begin{align*}
    s_i s_j
    &=
    (m + \delta s_i)(m + \delta s_j)
    \\&=
    m^2 + m \, \delta s_j + \delta s_i \, m + \delta s_i \, \delta s_j
    \\& \approx
    m^2 + m(s_j - m) + (s_i - m) m
    \\&=
    -m^2 + m(s_i + s_j).
\end{align*}
$$

We dropped the $\delta s_i \, \delta s_j$ term — this is the mean-field approximation. We are saying that the correlated fluctuations between two spins are small enough to ignore. (This is a good approximation when each spin has many neighbors, and a terrible one in low dimensions — but we will worry about that later.)

Substituting into the Hamiltonian and carefully handling the nearest-neighbor sums (each spin has $z$ neighbors, and each pair is counted once):

**First term:**
$$
    J m^2 \sum_{\langle i j \rangle} = \frac{J N z}{2} m^2.
$$

**Second term:**
$$
    * J \sum_{\langle i j \rangle} m(s_i + s_j) = - J z m \sum_{i} s_i.
$$

**Mean-field Hamiltonian:**
$$
\begin{align*}
    \mathcal{H}_\mathrm{MF}
    &=
    \frac{J N z}{2} m^2
    * (Jzm + h) \sum_i s_i.
\end{align*}
$$

Look at what happened: the spins have decoupled! Each spin $s_i$ now sees an effective field $(Jzm + h)$ that depends on the *average* magnetization $m$, not on the actual values of its neighbors. This makes the problem exactly solvable — each spin is independent, and we just need to find $m$ self-consistently.

## Big Ideas

* An order parameter is a number that is zero above the transition and nonzero below it — it is the system's way of "announcing" that it has chosen a preferred state.
* Phase transitions are collective phenomena: no single spin decides to align, yet the whole system snaps into order through purely local interactions.
* Mean-field theory tames the many-body problem by replacing all the messy neighbor-neighbor interactions with a single average field — the "wisdom of the crowd" acting on each individual.
* Dropping the fluctuation-fluctuation term $\delta s_i \, \delta s_j$ is the mean-field approximation: it works well when each spin has many neighbors, and fails badly in low dimensions where fluctuations dominate.

## What Comes Next

We have derived the mean-field Hamiltonian, but the real payoff comes from solving it. [Mean-Field Results](meanFieldResults) works through the partition function, the self-consistency equation for magnetization, and the Landau free energy expansion. The punchline is a family of **critical exponents** — the power laws that describe how the magnetization, susceptibility, and heat capacity behave as you approach $T_c$. These exponents will turn out to be the same for a huge class of systems, hinting at a deep universality that we will explain later.

## Check Your Understanding

1. The order parameter of a magnet is the magnetization $m$, which is zero above $T_c$ and nonzero below. Can you think of an order parameter for the liquid-gas phase transition? Why does it go to zero at the critical point?
2. The mean-field approximation drops the term $\delta s_i \, \delta s_j$ (correlated fluctuations). In a 1D chain where each spin has only two neighbors, why would you expect this approximation to be especially poor?
3. When a ferromagnet is cooled below $T_c$, it spontaneously picks one direction — all up or all down — even though both states have the same energy. What "breaks" this symmetry in a real experiment?

## Challenge

Imagine a mean-field model of opinion dynamics: $N$ people each hold an opinion $s_i = \pm 1$ (agree/disagree with some proposition). Each person is influenced by the average opinion of the whole group. Write down the self-consistency equation for the average opinion $m = \langle s_i \rangle$ as a function of the "social coupling" $J$ and the "temperature" $T$ (representing noise/uncertainty). Find the critical coupling $J_c$ at which a consensus can first emerge. What happens as $N \to \infty$? Does this match your intuition about how social consensus forms?
