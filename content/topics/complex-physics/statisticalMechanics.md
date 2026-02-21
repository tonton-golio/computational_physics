# Statistical Mechanics

Suppose you have a huge box full of air molecules bouncing around. Billions upon billions of them, all flying in random directions, colliding, exchanging energy. Why is the temperature the same everywhere? Why doesn't all the energy suddenly rush to one corner, leaving the other side frozen? That is the miracle we are going to explain.

The answer is statistics. When you have an enormous number of particles, the laws of probability take over, and they are ruthlessly democratic. The system explores every possible arrangement of energy among its particles, and the most probable arrangement wins by an overwhelming margin. That is the core insight of statistical mechanics: we do not need to track every particle. We just need to count states.

## Microcanonical Ensemble

The central assumption of statistical mechanics is the principle of *equal a priori probabilities*: all microstates that share a given total energy $E$ are equally likely, so the probability of any one microstate is $P_i = 1/\Omega$, where $\Omega$ is the total number of microstates at that energy. Boltzmann's great insight was that the thermodynamic entropy is simply a measure of this count:
$$
    S = k_\mathrm{B} \ln \Omega.
$$
More microstates means more entropy, and a system left to itself will find its way to the macrostate with the most microstates, because that is simply the most probable outcome. The key conceptual point is that "disorder" wins not because it is preferred, but because disordered macrostates correspond to overwhelmingly more microstates. From this single formula, temperature emerges as a derived quantity, $1/T = \partial S / \partial E$, measuring how fast the number of accessible states grows when you add energy. If adding a little energy opens up a huge number of new states, the system is cold; if the count barely changes, the system is hot. For the classic worked example (counting states of an ideal gas via phase-space volume and Stirling's approximation to recover the Sackur-Tetrode equation), see Schroeder, *An Introduction to Thermal Physics*, Ch. 2, or Pathria & Beale, *Statistical Mechanics*, Ch. 1.

## Canonical Ensemble

### Temperature of system and reservoir become the same in equilibrium

Now let us zoom in on a small piece of a much larger system. Imagine a teacup of water sitting in a room. The teacup is our "system" and the room is our "reservoir" (or "heat bath"). They can exchange energy, but the total energy is fixed:
$$
    E_\mathrm{t} = E_\mathrm{s} + E_\mathrm{r}.
$$

The number of microstates for the whole arrangement factors into a product:
$$
    \Omega(E_\mathrm{s}, E_\mathrm{r})
    = \Omega_\mathrm{s}(E_\mathrm{s}) \, \Omega_\mathrm{r}(E_\mathrm{r}),
$$
so the total entropy is the sum:
$$
    S_\mathrm{t} = S_\mathrm{s} + S_\mathrm{r}.
$$

The probability of finding a particular split of energy is
$$
    P(E_\mathrm{s}, E_\mathrm{r})
    = \frac{\Omega_\mathrm{s}(E_\mathrm{s}) \, \Omega_\mathrm{r}(E_\mathrm{r})}{\Omega(E_\mathrm{t})}.
$$

The most probable state (thermodynamic equilibrium) maximizes this probability. Setting the derivative to zero:
$$
\begin{align*}
    0 &=
        \frac{\partial \ln P(E_\mathrm{s}, E_\mathrm{r})}{\partial E_\mathrm{s}}
        \\&=
        \frac{\partial}{\partial E_\mathrm{s}}
        \ln \Omega_\mathrm{s}(E_\mathrm{s})
        +
        \frac{\partial}{\partial E_\mathrm{r}}
        \frac{\partial E_\mathrm{r}}{\partial E_\mathrm{s}}
        \ln \Omega_\mathrm{r}(E_\mathrm{r})
        \\&=
        \frac{1}{k_\mathrm{B}}
        \frac{\partial S_\mathrm{s}}{\partial E_\mathrm{s}}
        -
        \frac{1}{k_\mathrm{B}}
        \frac{\partial S_\mathrm{r}}{\partial E_\mathrm{r}}
        \\&=
        \frac{1}{T_\mathrm{s}}
        -
        \frac{1}{T_\mathrm{r}}.
\end{align*}
$$

The conclusion is simple and beautiful: in equilibrium, the system and reservoir have the same temperature.
$$
    T_\mathrm{s} = T_\mathrm{r}
$$

This is not something we assumed. It *followed* from counting states. Temperature equality is the most probable outcome, and for large systems it is overwhelmingly probable.

### Boltzmann distribution

When the system is in a specific microstate $i$ with energy $E_i$, the reservoir has energy $E_\mathrm{t} - E_i$. Since the reservoir is enormous, we can Taylor-expand its entropy:
$$
\begin{align*}
    P_i
    &\propto
        \Omega_\mathrm{r}(E_\mathrm{t} - E_i)
    \\&=
        \exp
        \left[ \frac{1}{k_\mathrm{B}} S_r(E_\mathrm{t} - E_i) \right]
    \\&\approx
        \exp \left[ \frac{1}{k_\mathrm{B}}
        \left(S_\mathrm{r}(E_\mathrm{t})
        * \left.
        \frac{\mathrm{d}S_\mathrm{r}}{\mathrm{d}E}
        \right|_{E=E_\mathrm{t}}E_i
        \right)
        \right]
    \\&\propto
        \exp \left(
        -\frac{E_i}{k_\mathrm{B}T}
        \right).
\end{align*}
$$

This is the **Boltzmann distribution**: the probability of finding the system in state $i$ falls off exponentially with the energy of that state. High-energy states are exponentially rare. Low-energy states are exponentially favored. Temperature controls how steep the falloff is.

### Partition function, free energy, and thermodynamic observables

Imagine you have a huge library of every possible arrangement of energy among the particles in your system. Some arrangements have low energy, some high. The partition function is a single number that summarizes the entire library: it is a weighted catalog where each arrangement is counted, but high-energy arrangements are exponentially discounted. Once you have this catalog, you can look up any thermodynamic quantity — energy, entropy, heat capacity — just by taking derivatives. It is the Rosetta Stone of statistical mechanics.

The **partition function** is the normalization constant that makes probabilities add up to one:
$$
\begin{align*}
    Z &= \sum_i \exp \left(-\frac{E_i}{k_\mathrm{B}T}\right)
        \\&= \sum_i \exp \left(-\beta E_i\right).
\end{align*}
$$
Here $\beta = 1/(k_\mathrm{B}T)$. The probability of state $i$ is then
$$
    P_i = \frac{e^{-\beta E_i}}{Z}.
$$

Do not underestimate this humble sum. Once you have $Z$, you can extract *every* thermodynamic quantity by taking derivatives. It is the master key.

The **(Helmholtz) free energy** is
$$
    F = -k_\mathrm{B}T \ln Z = -\frac{1}{\beta}\ln Z = \langle E \rangle - TS.
$$

The **average energy** follows from a derivative with respect to $\beta$:
$$
\begin{align*}
    \langle E \rangle
    &=
    \sum_i E_i P_i
    \\&=
    -\frac{1}{Z} \frac{\partial Z}{\partial \beta}
    \\&=
    * \frac{\partial}{\partial \beta} \ln Z.
\end{align*}
$$

The **specific heat** (how much the energy changes when you nudge the temperature) is
$$
\begin{align*}
    C
    &=
    \frac{\partial \langle E \rangle}{\partial T}
    \\&=
    \frac{1}{k_\mathrm{B} T^2}
    \frac{\partial^2}{\partial \beta^2} \ln Z.
\end{align*}
$$

Here is where things get interesting. The specific heat turns out to equal the **variance** of the energy:
$$
\begin{align*}
    k_\mathrm{B} T^2 C
    &=
    \frac{\partial^2}{\partial \beta^2} \ln Z
    \\&=
    \langle E^2 \rangle - \langle E \rangle^2
    \\&=
    (\Delta E)^2.
\end{align*}
$$

Why does your coffee behave itself? The energy of $N$ particles scales as $\langle E \rangle \sim N k_\mathrm{B} T$, and the specific heat as $C \sim N k_\mathrm{B}$. So the fluctuation in energy goes as $\Delta E \sim \sqrt{N}$. The *relative* fluctuation is $\Delta E / \langle E \rangle \sim 1/\sqrt{N}$. For a teaspoon of water ($N \sim 10^{23}$) that is about one part in $10^{11.5}$. That is why your coffee does not spontaneously boil in one half and freeze in the other. The law of large numbers is not just a theorem — it is the reason thermodynamics works.

From the free energy we can also extract:

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
\begin{align*}
    k_\mathrm{B} T \chi_T
    &=
    \langle M^2 \rangle - \langle M \rangle^2
    \\&=
    (\Delta M)^2.
\end{align*}
$$

This is a deep pattern: response functions (how much a system reacts to a small push) are always related to fluctuations (how much the system jiggles on its own). It is called the **fluctuation-dissipation theorem**, and it is one of the most beautiful results in all of physics.

## Big Ideas

* The partition function $Z$ is the Rosetta Stone of statistical mechanics: every thermodynamic quantity is a derivative of $\ln Z$.
* Entropy is just a count of microstates — the universe tends toward disorder because disordered states are overwhelmingly more numerous, not because disorder is "preferred."
* Temperature equality at equilibrium is not an axiom but a theorem: it is simply the most probable way to divide energy between a system and a reservoir.
* Response functions (specific heat, susceptibility) equal fluctuations — how much a system jitters tells you how much it reacts to a nudge.

## What Comes Next

The machinery of statistical mechanics is complete in principle: once you have $Z$, you have everything. But for an interacting system like a lattice of spins, computing $Z$ means summing $2^N$ terms — a number that dwarfs the count of atoms in the observable universe for any macroscopic system.

The [Metropolis Algorithm](metropolisAlgorithm) is the clever escape from this impossibility. Rather than summing over all configurations, we let the computer wander through configuration space, visiting states with the correct Boltzmann probabilities. The trick is that you never need to know $Z$ at all — only energy *differences* matter, and those are cheap to compute.

## Check Your Understanding

1. The entropy formula $S = k_\mathrm{B} \ln \Omega$ implies that mixing two gases increases entropy. But if you mix two identical gases, entropy does not increase — the Gibbs paradox. Why does identical composition change the counting argument?
2. The specific heat equals the variance of energy: $k_\mathrm{B} T^2 C = \langle E^2 \rangle - \langle E \rangle^2$. What does it mean physically that a system with large energy fluctuations also has a large heat capacity?
3. Two systems at different temperatures are placed in thermal contact. Explain, in terms of counting microstates, why energy flows from the hotter system to the cooler one.

## Challenge

Estimate the relative energy fluctuation $\Delta E / \langle E \rangle$ for a glass of water at room temperature. Roughly how many water molecules are there? What is $1/\sqrt{N}$ numerically? Now imagine a hypothetical detector sensitive enough to measure this fluctuation — what precision in energy measurement would it need? Compare this to the best calorimeters ever built, and reflect on why thermodynamics is so reliable.
