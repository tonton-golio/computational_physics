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
    - J \sum_{\langle i j \rangle} m(s_i + s_j) = - J z m \sum_{i} s_i.
$$

**Mean-field Hamiltonian:**
$$
\begin{align*}
    \mathcal{H}_\mathrm{MF}
    &=
    \frac{J N z}{2} m^2
    - (Jzm + h) \sum_i s_i.
\end{align*}
$$

Look at what happened: the spins have decoupled! Each spin $s_i$ now sees an effective field $(Jzm + h)$ that depends on the *average* magnetization $m$, not on the actual values of its neighbors. This makes the problem exactly solvable — each spin is independent, and we just need to find $m$ self-consistently.

> **Key Intuition.** A phase transition is the moment when a system collectively chooses an ordered state. The order parameter measures the degree of this collective choice. Mean-field theory captures the essential physics by replacing the complicated many-body problem with a simpler one where each particle feels only the average effect of all the others.

> **Challenge.** Here is a thought experiment. Imagine 100 people in a room, each trying to decide whether to stand or sit. If they make the decision independently, roughly half will stand and half will sit. But now add a rule: each person looks at their two nearest neighbors and feels a slight urge to do the same thing. Can you convince yourself that below some "critical level of conformity," the crowd splits 50/50, but above it, nearly everyone stands (or sits)? That is a phase transition in a social system.

---

*We have the mean-field Hamiltonian, and it looks clean and solvable. But what actually comes out of it? What is the critical temperature? What does the free energy look like? And what goes wrong with the stability analysis near the transition? Let us find out in the mean-field results.*
