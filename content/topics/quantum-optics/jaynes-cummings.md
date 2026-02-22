# Jaynes-Cummings Model

## The exactly solvable atom-photon marriage

The **Jaynes-Cummings model** is the hydrogen atom of quantum optics -- the simplest nontrivial interaction between a single two-level atom and a single mode of light, and it's exactly solvable.

$$
\hat{H} = \hbar\omega_c\hat{a}^\dag\hat{a} + \frac{\hbar\omega_a}{2}\hat{\sigma}_z + \hbar g(\hat{a}^\dag\hat{\sigma}_- + \hat{a}\hat{\sigma}_+)
$$

Three ingredients: cavity energy, atomic energy, and the interaction. The interaction term $\hat{a}^\dag\hat{\sigma}_-$ means "atom drops down, cavity gains a photon." The term $\hat{a}\hat{\sigma}_+$ means "atom jumps up, cavity loses a photon." Energy sloshes back and forth, quantum mechanically.

## Why it's exactly solvable

The total excitation number $\hat{N} = \hat{a}^\dag\hat{a} + \hat{\sigma}_+\hat{\sigma}_-$ is conserved. This breaks the infinite-dimensional Hilbert space into independent $2\times 2$ blocks: $\{\Ket{n, e}, \Ket{n+1, g}\}$ for each $n$. Each block is just a $2\times 2$ matrix to diagonalize.

The **dressed states** are:

$$
\Ket{+, n} = \cos\theta_n\Ket{n, e} + \sin\theta_n\Ket{n+1, g}
$$
$$
\Ket{-, n} = -\sin\theta_n\Ket{n, e} + \cos\theta_n\Ket{n+1, g}
$$

with $\tan(2\theta_n) = 2g\sqrt{n+1}/\Delta$ (detuning $\Delta = \omega_a - \omega_c$). These are entangled atom-field states -- neither pure atom nor pure field. The splitting between them is the **vacuum Rabi splitting**: $\hbar\Omega_n = \hbar\sqrt{\Delta^2 + 4g^2(n+1)}$.

## Rabi oscillations with a quantum twist

Start with an excited atom and $n$ photons: $\Ket{\psi(0)} = \Ket{n, e}$. At resonance ($\Delta = 0$), the atom oscillates between excited and ground:

$$
\langle\hat{\sigma}_z(t)\rangle = \cos(\Omega_n t), \qquad \Omega_n = 2g\sqrt{n+1}
$$

The $\sqrt{n+1}$ is the quantum signature. A classical field drives oscillations at a rate proportional to amplitude. A quantum field drives them at a rate proportional to $\sqrt{\text{photon number} + 1}$. The "+1" is the vacuum contribution -- even with zero photons, the atom oscillates at frequency $2g$ (vacuum Rabi oscillations).

## Collapse and revival

Now feed the atom a coherent state $\Ket{\alpha}$ instead of a definite photon number. Different Fock components oscillate at different frequencies $\Omega_n = 2g\sqrt{n+1}$. They dephase, and the oscillations **collapse** on a timescale $t_c \sim 1/(g\sqrt{\bar{n}})$.

But wait. The frequencies aren't random -- they're spaced by the discrete $\sqrt{n+1}$ values. Eventually the phases realign, and the oscillations **revive** at $t_r \sim 2\pi\sqrt{\bar{n}}/g$. This periodic revival is impossible with a classical field. It's a direct signature of the discrete photon-number structure.

[[simulation jaynes-cummings-revival]]

> **Computational Note.** The collapse and revival are beautiful to simulate. Construct the JC Hamiltonian as a matrix in the truncated $\{\Ket{n, e}, \Ket{n+1, g}\}$ basis, exponentiate with `scipy.linalg.expm`, and plot $\langle\hat{\sigma}_z(t)\rangle$ for a coherent state with $\bar{n} = 25$. You'll see the collapse around $t \approx 1/(g\sqrt{25})$ and the first revival at $t \approx 2\pi\cdot 5/g$. About 30 lines of Python.

## The dispersive regime

When the detuning is large ($|\Delta| \gg g\sqrt{n+1}$), atom and field don't exchange real excitations -- only virtual ones. The effective Hamiltonian becomes:

$$
\hat{H}_\text{disp} \approx \hbar(\omega_c + \chi\hat{\sigma}_z)\hat{a}^\dag\hat{a} + \frac{\hbar(\omega_a + \chi)}{2}\hat{\sigma}_z
$$

where $\chi = g^2/\Delta$. The cavity frequency shifts by $\pm\chi$ depending on the atomic state. By measuring this frequency shift, you can determine the atom's state without absorbing any photons -- a **quantum non-demolition** measurement. This is exactly how superconducting qubits are read out in circuit QED.

## Big Ideas

* The JC model is exactly solvable because total excitation number is conserved, splitting the Hilbert space into $2\times 2$ blocks.
* The $\sqrt{n+1}$ Rabi frequency is a purely quantum effect -- a classical field can't produce it.
* Collapse and revival with a coherent field is the clearest fingerprint of field quantization: discrete photon numbers cause dephasing and rephasing that no classical model can explain.
* In the dispersive regime, the atom becomes a phase meter for cavity photons without absorbing them.

## Check Your Understanding

1. On resonance, what are the dressed states explicitly? What does the splitting $2g\sqrt{n+1}$ tell you if you sweep a probe laser through the resonance?
2. The ratio $t_r/t_c \sim 2\pi\sqrt{\bar{n}}$ grows with photon number. What does this mean for observing revivals with very bright coherent states?

## Challenge

Write a Python script that solves the Jaynes-Cummings model numerically. Construct the Hamiltonian matrix in the truncated Fock-space basis, compute $\langle\hat{\sigma}_z(t)\rangle$ for an initial coherent state $\Ket{\alpha}$ with $\bar{n} = 25$, and plot the collapse-and-revival dynamics. Add cavity decay (Lindblad term $\kappa\mathcal{D}[\hat{a}]$) using QuTiP's `mesolve` and show how the revivals die as $\kappa$ increases. Estimate the revival time for $g/2\pi = 50$ kHz and $\bar{n} = 25$.
