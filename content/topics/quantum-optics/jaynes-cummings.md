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

[[simulation wigner-cat-state]]
