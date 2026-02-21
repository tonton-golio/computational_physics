# Jaynes-Cummings Model

## The Model

The **Jaynes-Cummings model** describes the simplest quantum interaction between a single two-level atom and a single mode of the electromagnetic field. The Hamiltonian is

$$
\hat{H} = \hbar\omega_c \hat{a}^\dagger \hat{a} + \frac{\hbar\omega_a}{2}\hat{\sigma}_z + \hbar g(\hat{a}^\dagger \hat{\sigma}_- + \hat{a}\hat{\sigma}_+),
$$

where $\omega_c$ is the cavity frequency, $\omega_a$ is the atomic transition frequency, $g$ is the coupling strength, and $\hat{\sigma}_\pm$ are the atomic raising/lowering operators.

The interaction term $\hat{a}^\dagger \hat{\sigma}_-$ describes the atom emitting a photon (going from excited to ground) while creating a cavity photon, and $\hat{a}\hat{\sigma}_+$ describes absorption. This is the **rotating wave approximation** (RWA), valid when $g \ll \omega_c, \omega_a$.

## Energy Levels and Dressed States

The total excitation number $\hat{N} = \hat{a}^\dagger\hat{a} + \hat{\sigma}_+\hat{\sigma}_-$ is conserved, so the Hilbert space splits into independent two-dimensional subspaces spanned by $\{|n, e\rangle, |n+1, g\rangle\}$ for each $n$.

Within each subspace, diagonalizing gives the **dressed states**:

$$
|+, n\rangle = \cos\theta_n |n, e\rangle + \sin\theta_n |n+1, g\rangle,
$$
$$
|-, n\rangle = -\sin\theta_n |n, e\rangle + \cos\theta_n |n+1, g\rangle,
$$

where $\tan(2\theta_n) = 2g\sqrt{n+1}/\Delta$ and $\Delta = \omega_a - \omega_c$ is the detuning. The dressed-state energies are

$$
E_{\pm, n} = \hbar\omega_c(n + \tfrac{1}{2}) \pm \frac{\hbar}{2}\sqrt{\Delta^2 + 4g^2(n+1)}.
$$

The splitting $\hbar\Omega_n = \hbar\sqrt{\Delta^2 + 4g^2(n+1)}$ between the dressed states is the **vacuum Rabi splitting** when $n = 0$.

## Rabi Oscillations

If the atom starts in the excited state with $n$ photons, $|\psi(0)\rangle = |n, e\rangle$, the state evolves as

$$
|\psi(t)\rangle = \cos(\Omega_n t/2)|n, e\rangle - i\sin(\Omega_n t/2)|n+1, g\rangle,
$$

at resonance ($\Delta = 0$), where $\Omega_n = 2g\sqrt{n+1}$ is the **$n$-photon Rabi frequency**. The atomic inversion oscillates:

$$
\langle \hat{\sigma}_z(t) \rangle = \cos(\Omega_n t).
$$

The $\sqrt{n+1}$ dependence is a purely quantum effect. For a coherent state input $|\alpha\rangle$ with $\bar{n} = |\alpha|^2$ photons, different Fock components oscillate at different frequencies $\Omega_n$, leading to **collapse and revival** of the Rabi oscillations. The initial oscillations collapse on a timescale $t_c \sim 1/(g\sqrt{\bar{n}})$ due to dephasing, then revive at $t_r \sim 2\pi\sqrt{\bar{n}}/g$ when the phases realign.

## Dispersive Regime

When the detuning is large ($|\Delta| \gg g\sqrt{n+1}$), the atom and field exchange only virtual excitations. Perturbation theory gives an effective Hamiltonian

$$
\hat{H}_{\text{disp}} \approx \hbar(\omega_c + \chi\hat{\sigma}_z)\hat{a}^\dagger\hat{a} + \frac{\hbar(\omega_a + \chi)}{2}\hat{\sigma}_z,
$$

where $\chi = g^2/\Delta$ is the **dispersive shift**. The cavity frequency shifts by $\pm\chi$ depending on the atomic state, and the atomic frequency shifts by $\chi$ per photon (**AC Stark shift** or **light shift**).

This regime enables **quantum non-demolition** (QND) measurement of photon number: measuring the atomic phase shift reveals $n$ without absorbing photons. It is the foundation of circuit QED readout in superconducting qubits.

## Spontaneous Emission

In free space, a two-level atom coupled to a continuum of modes decays irreversibly at the **Wightman-Weisskopf** rate

$$
\Gamma = \frac{\omega_a^3 d^2}{3\pi\epsilon_0\hbar c^3},
$$

where $d$ is the transition dipole moment. The excited-state population decays exponentially: $P_e(t) = e^{-\Gamma t}$.

In a cavity, spontaneous emission is modified by the density of states. When the cavity is resonant, the enhanced rate is $\Gamma_{\text{cav}} = 4g^2/\kappa$ (where $\kappa$ is the cavity decay rate), leading to the **Purcell effect** discussed in the Cavity QED topic.

## Big Ideas

* The Jaynes-Cummings model is exactly solvable because the total excitation number $\hat{N} = \hat{a}^\dagger\hat{a} + \hat{\sigma}_+\hat{\sigma}_-$ is conserved, breaking the full Hilbert space into independent two-dimensional subspaces.
* The $\sqrt{n+1}$ scaling of the Rabi frequency with photon number is a purely quantum effect: a classical field would drive oscillations at a fixed frequency independent of field strength at fixed amplitude, not this square-root dependence.
* Collapse and revival of Rabi oscillations with a coherent field input is the clearest signature of field quantization: different photon-number components oscillate at incommensurable frequencies, dephase, then rephase with quantum periodicity.
* The dispersive regime ($|\Delta| \gg g$) turns the atom into a phase meter for the cavity photon number, enabling quantum non-demolition measurements without ever absorbing the photons.

## What Comes Next

The Jaynes-Cummings model describes a perfect, isolated atom-cavity system. Real experiments face photon loss from the cavity and atomic decay into unwanted modes. The next lesson examines cavity QED — the regime where coherent coupling competes with dissipation — and shows how this competition determines whether you observe reversible Rabi oscillations or irreversible Purcell-enhanced decay.

## Check Your Understanding

1. The dressed states $|\pm, n\rangle$ are superpositions of $|n, e\rangle$ and $|n+1, g\rangle$ — entangled atom-field states. On resonance ($\Delta = 0$), what are the dressed states explicitly, and what does the splitting $2g\sqrt{n+1}$ between them tell you experimentally if you sweep a probe laser through the resonance?
2. For a coherent state input with mean photon number $\bar{n}$, the Rabi oscillations collapse on a timescale $t_c \sim 1/(g\sqrt{\bar{n}})$ and revive at $t_r \sim 2\pi\sqrt{\bar{n}}/g$. Show that $t_r/t_c \sim 2\pi\sqrt{\bar{n}}$, which grows with photon number. What does this imply about the possibility of observing revivals for very bright (large $\bar{n}$) coherent states?
3. In the dispersive regime, the atom shifts the cavity frequency by $\pm\chi = \pm g^2/\Delta$ depending on whether the atom is in $|e\rangle$ or $|g\rangle$. Explain how you would use this frequency shift to measure the atomic state without ever absorbing a photon from the cavity, and what makes this a quantum non-demolition measurement.

## Challenge

Starting from the Jaynes-Cummings Hamiltonian and the conserved total excitation number, derive the time evolution operator within each two-dimensional subspace $\{|n,e\rangle, |n+1,g\rangle\}$ at resonance. Use this to compute $\langle\hat{\sigma}_z(t)\rangle$ when the initial state is $|\psi(0)\rangle = |\alpha\rangle \otimes |e\rangle$ (coherent field, excited atom). Show that the result involves an incoherent sum over photon-number components, leading to the collapse and revival formula. Estimate the revival time $t_r$ for $g/2\pi = 50$ kHz and $\bar{n} = 25$ photons.
