# Phase Transitions

Pick up a refrigerator magnet. It sticks because trillions of tiny atomic magnets inside have collectively decided to point the same direction. Now heat it with a blowtorch. At the Curie temperature, the magnet suddenly stops being a magnet. The atomic spins are still there, still interacting, but thermal jiggling has overwhelmed their desire to align.

That sudden appearance of a preferred direction is what we call an **order parameter** -- a quantity that's zero in the disordered phase and nonzero in the ordered phase. For a magnet, it's the magnetization $m$. For a liquid-gas transition, it's the density difference. The order parameter is the signature that tells you something dramatic has happened.

## Big Ideas

* An order parameter is a number that's zero above the transition and nonzero below -- the system's way of "announcing" that it's chosen a preferred state.
* Phase transitions are fundamentally about symmetry breaking: the Hamiltonian treats "all up" and "all down" equally, yet the system spontaneously chooses one.
* Mean-field theory tames the many-body problem by replacing all the messy neighbor-neighbor interactions with a single average field -- the "wisdom of the crowd" acting on each individual.
* Dropping the fluctuation-fluctuation term $\delta s_i \, \delta s_j$ is the mean-field approximation: it works well when each spin has many neighbors, and fails badly in low dimensions where fluctuations dominate.

## Ising Model Simulation

Watch the Ising model in action. At high temperature, the spins are a random mess. As you cool toward $T_c$, domains of aligned spins start forming and growing. At $T_c$ itself, the system can't decide which way to point. Below $T_c$, one direction wins.

[[simulation ising-model]]

You're watching millions of spins make a collective decision -- no leader, no blueprint, just nearest-neighbor interactions. The fact that global order emerges from purely local rules is one of the deepest surprises in physics.

## Mean-Field Hamiltonian

Can we build a simple theory of this transition? The exact Hamiltonian couples every spin to its neighbors:
$$
    \mathcal{H}
    =
    -J \sum_{\langle i j \rangle} s_i s_j - h \sum_i s_i.
$$

The trouble is the $s_i s_j$ coupling -- it ties every spin to its neighbors, and those neighbors to *their* neighbors, creating a tangled web.

Instead of worrying about every neighbor, we say "each spin feels the average field created by all the others." That one trick turns the nightmare of $10^{23}$ interactions into a single self-consistent equation. Watch what happens.

Decompose each spin into its mean and a fluctuation: $s_i = m + \delta s_i$. The product of two neighbors becomes:
$$
\begin{align*}
    s_i s_j
    &=
    (m + \delta s_i)(m + \delta s_j)
    \\& \approx
    -m^2 + m(s_i + s_j).
\end{align*}
$$

We dropped the $\delta s_i \, \delta s_j$ term -- that's the mean-field approximation. We're saying correlated fluctuations between two spins are small enough to ignore. (Good when each spin has many neighbors. Terrible in low dimensions.)

Substituting into the Hamiltonian:

$$
\begin{align*}
    \mathcal{H}_\mathrm{MF}
    &=
    \frac{J N z}{2} m^2
    - (Jzm + h) \sum_i s_i.
\end{align*}
$$

Look at what happened: the spins have decoupled! Each spin now sees an effective field $(Jzm + h)$ that depends on the *average* magnetization, not on the actual values of its neighbors. This makes the problem exactly solvable.

## What Comes Next

We've derived the mean-field Hamiltonian, but the real payoff comes from solving it. [Mean-Field Results](meanFieldResults) works through the partition function, the self-consistency equation for magnetization, and the Landau free energy expansion. The punchline is a family of **critical exponents** -- power laws that describe how everything behaves as you approach $T_c$. These exponents turn out to be the same for a huge class of systems, hinting at a deep universality.

## Check Your Understanding

1. The order parameter of a magnet is the magnetization $m$. Can you think of an order parameter for the liquid-gas phase transition? Why does it go to zero at the critical point?
2. In a 1D chain where each spin has only two neighbors, why would you expect the mean-field approximation to be especially poor?
3. When a ferromagnet cools below $T_c$, it spontaneously picks one direction -- even though both have the same energy. What "breaks" this symmetry in a real experiment?

## Challenge

Consider a mean-field model where $N$ people each hold an opinion $s_i = \pm 1$, influenced by the average opinion of the group. Write down the self-consistency equation for the average opinion $m = \langle s_i \rangle$ as a function of "social coupling" $J$ and "temperature" $T$ (noise/uncertainty). Find the critical coupling $J_c$ at which consensus can first emerge. What happens as $N \to \infty$?
