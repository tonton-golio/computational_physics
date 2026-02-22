# Statistical Mechanics

Suppose you've got a huge box full of air molecules bouncing around. Billions upon billions of them, all flying in random directions, colliding, swapping energy. Why is the temperature the same everywhere? Why doesn't all the energy suddenly rush to one corner, leaving the other side frozen? That's the miracle we're going to explain.

The answer is statistics. When you've got an enormous number of particles, the laws of probability take over -- and they're ruthlessly democratic. The system explores every possible arrangement of energy among its particles, and the most probable arrangement wins by an overwhelming margin. That's the core insight: you don't need to track every particle. You just need to count states.

## Big Ideas

* The partition function $Z$ is the Rosetta Stone of statistical mechanics: every thermodynamic quantity is a derivative of $\ln Z$.
* Entropy is just a count of microstates -- the universe drifts toward disorder because disordered states are overwhelmingly more numerous, not because disorder is "preferred."
* Temperature equality at equilibrium isn't an axiom but a theorem: it's simply the most probable way to divide energy between a system and a reservoir.
* Response functions (specific heat, susceptibility) equal fluctuations -- how much a system jitters tells you how much it reacts to a nudge.

## Microcanonical Ensemble

The central assumption is the principle of *equal a priori probabilities*: all microstates sharing a given total energy $E$ are equally likely, so $P_i = 1/\Omega$, where $\Omega$ is the total number of microstates at that energy. Boltzmann looked at that formula and realized something profound: entropy is nothing but the logarithm of the number of ways a system can arrange itself:
$$
    S = k_\mathrm{B} \ln \Omega.
$$

The universe drifts toward disorder simply because there are vastly more messy arrangements than tidy ones. From this single formula, temperature emerges as a derived quantity, $1/T = \partial S / \partial E$, measuring how fast the number of accessible states grows when you add energy. If adding a little energy opens up a huge number of new states, the system is cold; if the count barely changes, the system is hot.

## Canonical Ensemble

### Temperature equalizes in equilibrium

Imagine a teacup of water sitting in a room. The teacup is your "system" and the room is your "reservoir." They swap energy, but the total is fixed:
$$
    E_\mathrm{t} = E_\mathrm{s} + E_\mathrm{r}.
$$

The number of microstates factors:
$$
    \Omega(E_\mathrm{s}, E_\mathrm{r})
    = \Omega_\mathrm{s}(E_\mathrm{s}) \, \Omega_\mathrm{r}(E_\mathrm{r}),
$$
so total entropy is the sum $S_\mathrm{t} = S_\mathrm{s} + S_\mathrm{r}$, and the probability of a particular energy split is
$$
    P(E_\mathrm{s}, E_\mathrm{r})
    = \frac{\Omega_\mathrm{s}(E_\mathrm{s}) \, \Omega_\mathrm{r}(E_\mathrm{r})}{\Omega(E_\mathrm{t})}.
$$

Maximize this (set the derivative to zero) and you get a beautiful result:
$$
    \frac{1}{T_\mathrm{s}} = \frac{1}{T_\mathrm{r}}.
$$

That's it. In equilibrium, system and reservoir have the same temperature. We didn't assume this -- it *followed* from counting states. Temperature equality is the most probable outcome, and for large systems it's overwhelmingly probable.

### Boltzmann distribution

When the system sits in microstate $i$ with energy $E_i$, the reservoir holds $E_\mathrm{t} - E_i$. Taylor-expand the reservoir's entropy (it's enormous, so this works perfectly):
$$
\begin{align*}
    P_i
    &\propto
        \Omega_\mathrm{r}(E_\mathrm{t} - E_i)
    \\&\propto
        \exp \left(
        -\frac{E_i}{k_\mathrm{B}T}
        \right).
\end{align*}
$$

This is the **Boltzmann distribution**: the probability of state $i$ falls off exponentially with its energy. High-energy states are exponentially rare. Temperature controls how steep the falloff is.

### Partition function, free energy, and thermodynamic observables

Think of the partition function as a weighted catalog of every possible arrangement -- each one counted, but high-energy arrangements exponentially discounted. Once you've got this catalog, you can look up any thermodynamic quantity just by taking derivatives. It's the Rosetta Stone of statistical mechanics.

The **partition function** is the normalization constant:
$$
\begin{align*}
    Z &= \sum_i \exp \left(-\frac{E_i}{k_\mathrm{B}T}\right)
        \\&= \sum_i \exp \left(-\beta E_i\right).
\end{align*}
$$
Here $\beta = 1/(k_\mathrm{B}T)$. The probability of state $i$ is $P_i = e^{-\beta E_i}/Z$.

Don't underestimate this humble sum. Once you have $Z$, you can extract *every* thermodynamic quantity by taking derivatives. It's the master key.

The **(Helmholtz) free energy** is
$$
    F = -k_\mathrm{B}T \ln Z = -\frac{1}{\beta}\ln Z = \langle E \rangle - TS.
$$

The **average energy** follows from a $\beta$-derivative:
$$
    \langle E \rangle
    =
    - \frac{\partial}{\partial \beta} \ln Z.
$$

The **specific heat** (how much energy changes when you nudge the temperature) is
$$
    C
    =
    \frac{\partial \langle E \rangle}{\partial T}
    =
    \frac{1}{k_\mathrm{B} T^2}
    \frac{\partial^2}{\partial \beta^2} \ln Z.
$$

And here's where things get gorgeous. The specific heat equals the **variance** of the energy:
$$
    k_\mathrm{B} T^2 C
    =
    \langle E^2 \rangle - \langle E \rangle^2
    =
    (\Delta E)^2.
$$

Why does your coffee behave itself? The energy of $N$ particles scales as $\langle E \rangle \sim N k_\mathrm{B} T$, and the specific heat as $C \sim N k_\mathrm{B}$. So the fluctuation goes as $\Delta E \sim \sqrt{N}$. The *relative* fluctuation is $\Delta E / \langle E \rangle \sim 1/\sqrt{N}$. For a teaspoon of water ($N \sim 10^{23}$), that's about one part in $10^{11.5}$. That's why your coffee doesn't spontaneously boil in one half and freeze in the other. The law of large numbers isn't just a theorem -- it's the reason thermodynamics works.

From the free energy you can also extract:

**Entropy:**
$$
    S = - \left. \frac{\partial F}{\partial T} \right|_H.
$$

**Magnetization** (for magnetic systems):
$$
    M = - \left.\frac{\partial F}{\partial H} \right|_T.
$$

**Susceptibility** (how sensitive the magnetization is to the applied field):
$$
    \chi_T = - \left.\frac{\partial^2 F}{\partial H^2} \right|_T.
$$

[[simulation two-level-system]]

Just as the specific heat is the variance of energy, the susceptibility is the **variance of magnetization**:
$$
    k_\mathrm{B} T \chi_T
    =
    \langle M^2 \rangle - \langle M \rangle^2
    =
    (\Delta M)^2.
$$

This is a deep pattern: response functions (how much a system reacts to a small push) are always related to fluctuations (how much the system jiggles on its own). It's called the **fluctuation-dissipation theorem**, and it's one of the most beautiful results in all of physics.

## What Comes Next

The machinery of statistical mechanics is complete in principle: once you have $Z$, you have everything. But for an interacting system like a lattice of spins, computing $Z$ means summing $2^N$ terms -- a number that dwarfs the count of atoms in the observable universe for any macroscopic system.

The [Metropolis Algorithm](metropolisAlgorithm) is the clever escape from this impossibility. Rather than summing over all configurations, you let the computer wander through configuration space, visiting states with the correct Boltzmann probabilities. The trick is that you never need to know $Z$ at all -- only energy *differences* matter, and those are cheap to compute.

## Check Your Understanding

1. The specific heat equals the variance of energy: $k_\mathrm{B} T^2 C = \langle E^2 \rangle - \langle E \rangle^2$. What does it mean physically that a system with large energy fluctuations also has a large heat capacity?
2. Two systems at different temperatures are placed in thermal contact. Explain, in terms of counting microstates, why energy flows from the hotter system to the cooler one.

## Challenge

Estimate the relative energy fluctuation $\Delta E / \langle E \rangle$ for a glass of water at room temperature. Roughly how many water molecules are there? What is $1/\sqrt{N}$ numerically? Now imagine a hypothetical detector sensitive enough to measure this fluctuation -- what precision in energy measurement would it need? Compare this to the best calorimeters ever built, and reflect on why thermodynamics is so reliable.
