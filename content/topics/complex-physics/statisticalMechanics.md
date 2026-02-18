# Statistical Mechanics

Suppose you have a huge box full of air molecules bouncing around. Billions upon billions of them, all flying in random directions, colliding, exchanging energy. Why is the temperature the same everywhere? Why doesn't all the energy suddenly rush to one corner, leaving the other side frozen? That is the miracle we are going to explain.

The answer is statistics. When you have an enormous number of particles, the laws of probability take over, and they are ruthlessly democratic. The system explores every possible arrangement of energy among its particles, and the most probable arrangement wins by an overwhelming margin. That is the core insight of statistical mechanics: we do not need to track every particle. We just need to count states.

## Microcanonical Ensemble

The central assumption of statistical mechanics is the principle of *equal a priori probabilities*: all microstates that share a given total energy are equally likely. There is no favoritism. Nature does not prefer one arrangement of molecular speeds over another, as long as the energy adds up.

With this assumption, the probability of finding the system in a particular microstate $i$ is simply
$$
    P_i = \frac{1}{\Omega},
$$
where $\Omega$ is the total number of microstates with energy $E$.

Think of it this way: if you have a jar with $\Omega$ lottery tickets and each one is equally likely, the chance of drawing any particular ticket is $1/\Omega$. That is all this equation says.

Ludwig Boltzmann connected this counting of states to thermodynamics through his famous entropy formula:
$$
    S = k_\mathrm{B} \ln \Omega.
$$
Here $k_\mathrm{B}$ is Boltzmann's constant. More microstates means more entropy. A system left to itself will find its way to the macrostate with the most microstates, because that is simply the most probable outcome.

Temperature then emerges as a statistical quantity:
$$
    \frac{1}{T}
    = \frac{\partial S}{\partial E}
    = k_\mathrm{B} \frac{\partial}{\partial E} \ln \Omega.
$$

Temperature tells you how fast the number of available states grows as you add energy. If adding a little energy opens up a huge number of new states, the system is cold. If the number of states barely changes, the system is hot.

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
        - \left.
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
    - \frac{\partial}{\partial \beta} \ln Z.
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

> **Aside — why your coffee behaves itself.** The energy of $N$ particles scales as $\langle E \rangle \sim N k_\mathrm{B} T$, and the specific heat as $C \sim N k_\mathrm{B}$. So the fluctuation in energy goes as $\Delta E \sim \sqrt{N}$. The *relative* fluctuation is $\Delta E / \langle E \rangle \sim 1/\sqrt{N}$. For a teaspoon of water ($N \sim 10^{23}$) that is about one part in $10^{11.5}$. That is why your coffee does not spontaneously boil in one half and freeze in the other. The law of large numbers is not just a theorem — it is the reason thermodynamics works.

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

> **Key Intuition.** The partition function $Z$ is the Rosetta Stone of statistical mechanics. Once you have it, every thermodynamic quantity is just a derivative away. Response functions (specific heat, susceptibility) measure fluctuations — and fluctuations shrink as $1/\sqrt{N}$, which is why macroscopic physics is predictable even though microscopic physics is random.

> **Challenge.** Estimate the relative energy fluctuation of a glass of water at room temperature. How many molecules are there? What is $1/\sqrt{N}$? Could you ever detect such a fluctuation with any instrument humans have built?

---

*We now have the machinery to compute anything — in principle. But in practice, summing over all microstates of an interacting system is impossibly hard. So how do we actually calculate things? We let the computer roll the dice. That is the Metropolis algorithm, and it is where we go next.*
