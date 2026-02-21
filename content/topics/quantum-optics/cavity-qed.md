# Cavity QED

## Cavity Quantum Electrodynamics

**Cavity QED** studies the interaction between atoms and photons confined in a high-quality resonator. By trapping the electromagnetic field in a small volume, the atom-photon coupling strength $g$ can be made large relative to the dissipation rates, enabling the observation of fundamentally quantum phenomena.

The three key rates are:
* $g$: atom-cavity coupling strength.
* $\kappa$: cavity photon loss rate (inverse of cavity lifetime).
* $\gamma$: atomic spontaneous emission rate into non-cavity modes.

The **strong coupling regime** $g \gg \kappa, \gamma$ allows coherent exchange of excitations between atom and field before either decays.

## The Purcell Effect

When an atom is placed in a resonant cavity, its spontaneous emission rate is modified. The **Purcell factor** gives the enhancement:

$$
F_P = \frac{\Gamma_{\text{cav}}}{\Gamma_{\text{free}}} = \frac{3}{4\pi^2}\left(\frac{\lambda}{n}\right)^3 \frac{Q}{V},
$$

where $Q$ is the cavity quality factor and $V$ is the mode volume. High-$Q$, small-$V$ cavities dramatically enhance emission into the cavity mode.

In the weak coupling regime ($g < \kappa$), the atom still decays irreversibly but at the enhanced Purcell rate $\Gamma_{\text{cav}} = 4g^2/\kappa$. In the strong coupling regime, the decay is replaced by reversible Rabi oscillations described by the Jaynes-Cummings model.

Conversely, when the cavity is far detuned from the atomic transition, the density of states is suppressed and spontaneous emission is **inhibited**. This was first demonstrated by Kleppner (1981) using Rydberg atoms between conducting plates.

## Experimental Platforms

**Microwave cavity QED** uses Rydberg atoms (with transition frequencies in the GHz range) passing through superconducting microwave cavities. Pioneered by Haroche and colleagues, these experiments achieve:
* Photon lifetimes of $\sim 0.1$ seconds in superconducting cavities.
* Strong coupling with $g/2\pi \sim 50$ kHz.
* Single-atom, single-photon resolution.

**Optical cavity QED** uses alkali atoms trapped in high-finesse Fabry-Perot cavities. Pioneered by Kimble and colleagues, these experiments work at optical frequencies with:
* Small mode volumes ($\sim \lambda^3$).
* Strong coupling with $g/2\pi \sim 10{-}100$ MHz.
* Direct single-photon detection.

**Circuit QED** replaces atoms with superconducting qubits and cavities with microwave transmission line resonators. The coupling strength is orders of magnitude larger than in natural atoms:
* $g/2\pi \sim 100$ MHz (easily in strong coupling).
* Highly controllable fabrication.
* The dominant platform for quantum computing (IBM, Google).

## Cat-State Generation

A **Schrodinger cat state** is a superposition of two macroscopically distinct coherent states:

$$
|\text{cat}_\pm\rangle = \mathcal{N}_\pm(|\alpha\rangle \pm |-\alpha\rangle),
$$

where $\mathcal{N}_\pm$ is a normalization constant. The even cat $|+\rangle$ contains only even photon numbers, while the odd cat $|-\rangle$ contains only odd photon numbers.

In cavity QED, cat states are generated using the **dispersive interaction**. An atom in a superposition $(|e\rangle + |g\rangle)/\sqrt{2}$ interacting dispersively with a coherent state $|\alpha\rangle$ creates an entangled state:

$$
\frac{1}{\sqrt{2}}(|e\rangle|\alpha e^{i\chi t}\rangle + |g\rangle|\alpha e^{-i\chi t}\rangle).
$$

At $\chi t = \pi/2$, this becomes $\frac{1}{\sqrt{2}}(|e\rangle|i\alpha\rangle + |g\rangle|-i\alpha\rangle)$. A subsequent $\pi/2$ pulse and measurement on the atom projects the cavity into a cat state.

These experiments were performed by Haroche's group and provide direct evidence of quantum superpositions at the mesoscopic scale. The decoherence of cat states (monitored by Wigner function tomography) demonstrates the quantum-to-classical transition.

[[simulation vacuum-rabi-oscillation]]

[[simulation wigner-cat-state]]

## Decoherence and Quantum Jumps

Real cavities lose photons at rate $\kappa$, and atoms decay at rate $\gamma$. The system dynamics are described by a **master equation**:

$$
\frac{d\hat{\rho}}{dt} = -\frac{i}{\hbar}[\hat{H}, \hat{\rho}] + \kappa\mathcal{D}[\hat{a}]\hat{\rho} + \gamma\mathcal{D}[\hat{\sigma}_-]\hat{\rho},
$$

where $\mathcal{D}[\hat{O}]\hat{\rho} = \hat{O}\hat{\rho}\hat{O}^\dagger - \frac{1}{2}\{\hat{O}^\dagger\hat{O}, \hat{\rho}\}$ is the Lindblad dissipator.

For cat states, decoherence occurs at a rate $\Gamma_{\text{decoherence}} = 2\kappa|\alpha|^2$, proportional to the "size" of the superposition. Larger cats decohere faster, consistent with the difficulty of observing quantum effects at macroscopic scales.

**Quantum jump** monitoring (measuring the environment) reveals individual photon loss events in real time. Between jumps, the system evolves under a non-Hermitian effective Hamiltonian, and each jump projects the state. This was first observed by Nagourney, Sauter, and Dehmelt (1986) in ion traps.

## Big Ideas

* Cavity QED is defined by the competition between three rates: coupling $g$, cavity decay $\kappa$, and atomic decay $\gamma$. Strong coupling ($g \gg \kappa, \gamma$) is where the interesting quantum physics lives.
* The Purcell effect shows that spontaneous emission is not a fixed atomic property but an interaction between the atom and its electromagnetic environment — change the cavity, change the decay rate.
* Schrödinger cat states (superpositions of macroscopically distinct coherent states) can be created and observed in cavity QED, and their decoherence rate scales with $|\alpha|^2$ — bigger cats die faster.
* Circuit QED extends cavity QED to superconducting chips with coupling strengths so large that strong coupling is easy to achieve, making it the backbone of modern quantum computing.

## What Comes Next

Cavity QED shows us that measurement fundamentally shapes the quantum state — cat states collapse, photon jumps are observed, and every detection event changes the field. The final lesson takes this seriously and examines the full quantum theory of measurement: how do we extract information from a quantum field without destroying it, and what are the ultimate limits imposed by the back-action of measurement?

## Check Your Understanding

1. The Purcell factor $F_P = \frac{3}{4\pi^2}\left(\frac{\lambda}{n}\right)^3 \frac{Q}{V}$ grows with $Q/V$. Explain physically why a small mode volume $V$ enhances the atom-cavity coupling, and why a high quality factor $Q$ does so as well. What would a "perfect" Purcell cavity look like?
2. Cat state decoherence occurs at rate $\Gamma_\text{decoherence} = 2\kappa|\alpha|^2$, proportional to the "size" of the superposition. Why does a larger $|\alpha|$ (more macroscopic superposition) decohere faster? Relate this to the distinguishability of $|\alpha\rangle$ and $|-\alpha\rangle$ in phase space.
3. Circuit QED replaces the atom with a superconducting qubit and achieves $g/2\pi \sim 100$ MHz, far larger than natural atoms. What physical property of superconducting qubits gives them such large coupling, and why does this make it easy to reach the strong coupling regime?

## Challenge

The master equation for the Jaynes-Cummings system with cavity decay is $\dot{\hat{\rho}} = -\frac{i}{\hbar}[\hat{H}_\text{JC}, \hat{\rho}] + \kappa\mathcal{D}[\hat{a}]\hat{\rho}$. Starting from this equation, derive the time evolution of the mean photon number $\langle\hat{n}(t)\rangle$ and the atomic inversion $\langle\hat{\sigma}_z(t)\rangle$ in the bad-cavity limit $\kappa \gg g$. Show that in this limit the effective atomic decay rate is the Purcell rate $\Gamma_\text{cav} = 4g^2/\kappa$, confirming that the cavity mediates enhanced irreversible decay even though individual Rabi oscillations are not resolved.
