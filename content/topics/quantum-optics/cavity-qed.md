# Cavity QED

## When coupling wins the race

The Jaynes-Cummings model describes a perfect, isolated atom-cavity system. Real life isn't so kind. Photons leak out of the cavity, atoms decay into unwanted modes, and the question becomes: can the atom and field exchange energy *before* everything dissipates?

Three rates determine the answer:
* $g$: atom-cavity coupling strength (how fast they talk)
* $\kappa$: cavity decay rate (how fast photons leak out)
* $\gamma$: atomic decay rate (how fast the atom emits into non-cavity modes)

**Strong coupling** ($g \gg \kappa, \gamma$): the atom and field exchange excitations many times before either decays. You see reversible Rabi oscillations, vacuum Rabi splitting, and all the gorgeous JC physics.

**Weak coupling** ($g < \kappa$): the cavity dumps photons faster than the atom can reabsorb them. Instead of oscillations, you get enhanced irreversible decay -- the **Purcell effect**.

## The Purcell effect

Place an atom in a resonant cavity and its spontaneous emission rate changes. The **Purcell factor** gives the enhancement:

$$
F_P = \frac{3}{4\pi^2}\left(\frac{\lambda}{n}\right)^3\frac{Q}{V}
$$

High $Q$ (photons bounce many times) and small $V$ (tight confinement) mean the atom "sees" a much higher density of states in the cavity mode and decays faster into it: $\Gamma_\text{cav} = 4g^2/\kappa$.

Flip the setup -- detune the cavity far from the atomic transition -- and the emission is *inhibited*. The cavity suppresses the density of states at the transition frequency. Kleppner demonstrated this in 1981 with Rydberg atoms between conducting plates. Spontaneous emission isn't a fixed property of the atom. It's a property of the atom-plus-environment system. Change the environment, change the decay rate.

## Experimental platforms

**Microwave cavity QED** (Haroche and colleagues): Rydberg atoms fly through superconducting microwave cavities with photon lifetimes up to 0.1 seconds. Strong coupling with $g/2\pi \sim 50$ kHz. This is where collapse-revival, cat states, and quantum jumps were first observed.

**Optical cavity QED** (Kimble and colleagues): alkali atoms trapped in high-finesse Fabry-Perot cavities. Strong coupling at optical frequencies with $g/2\pi \sim 10$-$100$ MHz.

**Circuit QED**: superconducting qubits coupled to microwave transmission-line resonators. Here the "atom" is an engineered circuit with a dipole moment thousands of times larger than a natural atom, giving $g/2\pi \sim 100$ MHz. Strong coupling is easy. This is the dominant platform for quantum computing today (IBM, Google, and others).

[[simulation vacuum-rabi-oscillation]]

## Cat states in a cavity

A **Schrodinger cat state** is a superposition of two macroscopically distinct coherent states:

$$
\Ket{\text{cat}_\pm} = \mathcal{N}_\pm(\Ket{\alpha} \pm \Ket{-\alpha})
$$

The even cat contains only even photon numbers; the odd cat only odd.

In cavity QED, cat states are generated using the dispersive interaction. Start with an atom in a superposition $(|e\rangle + |g\rangle)/\sqrt{2}$ interacting dispersively with $\Ket{\alpha}$. At time $\chi t = \pi/2$, the atom and field become entangled. A subsequent $\pi/2$ pulse and measurement on the atom projects the cavity into a cat state.

Haroche's group created these states and watched them die. The decoherence rate is $\Gamma_\text{decoherence} = 2\kappa|\alpha|^2$ -- bigger cats die faster, consistent with why you never see quantum superpositions of basketballs.

[[simulation wigner-cat-state]]

## Decoherence and quantum jumps

The system dynamics with dissipation follow the **master equation**:

$$
\frac{d\hat{\rho}}{dt} = -\frac{i}{\hbar}[\hat{H}, \hat{\rho}] + \kappa\mathcal{D}[\hat{a}]\hat{\rho} + \gamma\mathcal{D}[\hat{\sigma}_-]\hat{\rho}
$$

where $\mathcal{D}[\hat{O}]\hat{\rho} = \hat{O}\hat{\rho}\hat{O}^\dag - \frac{1}{2}\{\hat{O}^\dag\hat{O}, \hat{\rho}\}$.

**Quantum jumps** are visible when you monitor the environment. Between jumps, the system evolves smoothly under a non-Hermitian Hamiltonian. Each jump -- a photon clicking in a detector -- abruptly changes the state. Haroche's group tracked photon number in a microwave cavity decaying from $n = 7$ to $n = 0$, one photon at a time. You literally watch the quantum state collapse, photon by photon.

## Big Ideas

* Cavity QED is defined by the race between coupling $g$, cavity decay $\kappa$, and atomic decay $\gamma$. Strong coupling is where the quantum magic happens.
* Spontaneous emission is not a fixed atomic property -- it's an atom-environment interaction. Change the cavity, change the decay rate (Purcell effect).
* Cat states can be created and observed, and their decoherence scales as $|\alpha|^2$ -- bigger cats die faster.
* Circuit QED extends this physics to superconducting chips, making strong coupling routine and powering modern quantum computing.

## Check Your Understanding

1. Why does small mode volume $V$ enhance the coupling? Why does high $Q$? What would a "perfect" Purcell cavity look like?
2. Cat state decoherence scales as $2\kappa|\alpha|^2$. Why do bigger superpositions die faster? Relate this to the distinguishability of $\Ket{\alpha}$ and $\Ket{-\alpha}$.

## Challenge

Write a Python script (using QuTiP or similar) that solves the master equation for the Jaynes-Cummings model with cavity decay. Start from $\Ket{\alpha}\otimes\Ket{e}$ and track $\langle\hat{n}(t)\rangle$ and $\langle\hat{\sigma}_z(t)\rangle$ as you sweep from the bad-cavity limit ($\kappa \gg g$) to the strong-coupling limit ($g \gg \kappa$). Show that in the bad-cavity limit, the atom decays at the Purcell rate $4g^2/\kappa$, while in strong coupling you see damped Rabi oscillations.
