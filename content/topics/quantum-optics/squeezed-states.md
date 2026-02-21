# Squeezed States

> *Coherent states share the vacuum's noise equally between quadratures. Squeezed states break that symmetry — reducing noise below the vacuum in one quadrature at the expense of the other — and this redistribution is the key to [quantum-enhanced interferometry](./interferometry).*

## Quadrature Operators

The electromagnetic field quadratures are defined as

$$
\hat{X} = \frac{1}{2}(\hat{a} + \hat{a}^\dagger), \qquad \hat{P} = \frac{1}{2i}(\hat{a} - \hat{a}^\dagger),
$$

satisfying $[\hat{X}, \hat{P}] = i/2$. The Heisenberg uncertainty relation gives $\Delta X \cdot \Delta P \geq 1/4$. For vacuum and coherent states, the uncertainties are equal: $\Delta X = \Delta P = 1/2$ (minimum uncertainty states with symmetric noise).

**Squeezed states** redistribute the quantum noise between the two quadratures. One quadrature has reduced fluctuations below the vacuum level at the expense of increased fluctuations in the conjugate quadrature, while still satisfying the uncertainty relation.

## The Squeezing Operator

The single-mode **squeezing operator** is

$$
\hat{S}(\xi) = \exp\left[\frac{1}{2}(\xi^* \hat{a}^2 - \xi \hat{a}^{\dagger 2})\right],
$$

where $\xi = r e^{i\theta}$ is the complex squeezing parameter. The magnitude $r$ determines the degree of squeezing. The squeezed vacuum state is $|\xi\rangle = \hat{S}(\xi)|0\rangle$.

Under squeezing, the quadrature variances become

$$
(\Delta X_\theta)^2 = \frac{1}{4}e^{-2r}, \qquad (\Delta X_{\theta+\pi/2})^2 = \frac{1}{4}e^{+2r}.
$$

The noise reduction is characterized in decibels: squeezing of $r$ corresponds to $-10\log_{10}(e^{-2r}) \approx 8.69 r$ dB. Current experiments achieve over 15 dB of squeezing.

## Photon Statistics of Squeezed Vacuum

The squeezed vacuum contains only **even photon numbers**:

$$
|\xi\rangle = \frac{1}{\sqrt{\cosh r}} \sum_{n=0}^{\infty} \frac{(-e^{i\theta} \tanh r)^n \sqrt{(2n)!}}{2^n n!} |2n\rangle.
$$

The mean photon number is $\langle n \rangle = \sinh^2 r$, and the photon number distribution is super-Poissonian despite the sub-vacuum noise in one quadrature. The even-photon-number signature is a distinctive feature observable in photon-number-resolving measurements.

## Generation by Parametric Down-Conversion

The primary method for generating squeezed light is **optical parametric down-conversion** (PDC). A nonlinear crystal with $\chi^{(2)}$ nonlinearity is pumped by a strong laser at frequency $\omega_p$. The pump photons are converted into pairs of photons (signal and idler) satisfying energy and momentum conservation:

$$
\omega_p = \omega_s + \omega_i, \qquad \mathbf{k}_p = \mathbf{k}_s + \mathbf{k}_i.
$$

When the signal and idler are in the same mode (**degenerate** PDC), the output is a squeezed vacuum state. The Hamiltonian describing this process is

$$
\hat{H}_{\text{PDC}} = i\hbar\kappa(\hat{a}^{\dagger 2} - \hat{a}^2),
$$

where $\kappa$ is proportional to the pump amplitude and the nonlinear coefficient. This is exactly the squeezing Hamiltonian.

In an **optical parametric oscillator** (OPO), the nonlinear crystal is placed inside a cavity. Below threshold, the OPO produces continuous-wave squeezed light with narrow bandwidth determined by the cavity linewidth. Above threshold, it oscillates and produces a coherent output.

---

## Displaced Squeezed States

A **displaced squeezed state** is obtained by first squeezing the vacuum and then displacing it in phase space:

$$
|\alpha, \xi\rangle = \hat{D}(\alpha) \hat{S}(\xi) |0\rangle,
$$

where $\hat{D}(\alpha) = \exp(\alpha \hat{a}^\dagger - \alpha^* \hat{a})$ is the displacement operator. The ordering matters: $\hat{D}(\alpha)\hat{S}(\xi) \neq \hat{S}(\xi)\hat{D}(\alpha)$ in general. The result is a **minimum uncertainty state** centered at $\alpha$ in phase space with an elliptical noise distribution.

The mean values and variances are

$$
\langle \hat{X}_\phi \rangle = |\alpha|\cos(\phi - \phi_\alpha), \qquad (\Delta X_\phi)^2 = \frac{1}{4}(e^{-2r}\cos^2\psi + e^{2r}\sin^2\psi),
$$

where $\psi = \phi - \theta/2$ and $\phi_\alpha = \arg(\alpha)$.

### Amplitude and Phase Squeezed Light

The relative orientation of the squeezing ellipse and the coherent amplitude determines the type of squeezing:

**Amplitude squeezed light** has reduced intensity fluctuations. The squeezing axis is aligned with the coherent amplitude, so the radial (amplitude) quadrature has sub-vacuum noise: $(\Delta n)^2 < \langle n \rangle$ (sub-Poissonian). This is useful for precision intensity measurements.

**Phase squeezed light** has reduced phase fluctuations. The squeezing axis is perpendicular to the coherent amplitude: $\Delta\phi < 1/(2\sqrt{\langle n \rangle})$ (below shot noise). This is advantageous for interferometric measurements where phase sensitivity is the limiting factor.

The Wigner function of a displaced squeezed state is a Gaussian centered at $\alpha$ with principal axes tilted by $\theta/2$:

$$
W(x, p) = \frac{1}{\pi} \exp\left[-e^{2r}(x' - x_0')^2 - e^{-2r}(p' - p_0')^2\right],
$$

where $(x', p')$ are coordinates rotated by the squeezing angle.

[[simulation wigner-squeezed]]

---

## Two-Mode Squeezing and Entanglement

**Two-mode squeezing** correlates two distinct field modes and is the key resource for continuous-variable entanglement. The two-mode squeezing operator is

$$
\hat{S}_2(\xi) = \exp(\xi^* \hat{a}\hat{b} - \xi \hat{a}^\dagger \hat{b}^\dagger),
$$

which creates photon pairs: one in mode $a$ and one in mode $b$. The two-mode squeezed vacuum (TMSV) state is

$$
|\text{TMSV}\rangle = \frac{1}{\cosh r} \sum_{n=0}^{\infty} (-e^{i\theta}\tanh r)^n |n\rangle_a |n\rangle_b.
$$

The modes are perfectly correlated in photon number but individually each mode is a thermal state with $\langle n \rangle = \sinh^2 r$.

The sum and difference quadratures satisfy

$$
\Delta(X_a - X_b)^2 = \frac{1}{2}e^{-2r}, \qquad \Delta(P_a + P_b)^2 = \frac{1}{2}e^{-2r},
$$

both vanishing in the limit $r \to \infty$ (the original EPR state). The **Duan criterion** confirms entanglement when $\Delta(X_a - X_b)^2 + \Delta(P_a + P_b)^2 < 1$, which equals $e^{-2r} < 1$ for any $r > 0$.

[[simulation tmsv-entanglement]]

---

## Applications of Squeezed Light

* **Gravitational wave detection**: LIGO and Virgo inject squeezed vacuum into the interferometer's dark port, reducing shot noise. Frequency-dependent squeezing (using filter cavities) optimizes noise reduction across the detection bandwidth. See [interferometry](./interferometry) for how squeezed injection improves phase sensitivity.
* **Quantum teleportation**: The TMSV provides the shared entangled resource for continuous-variable teleportation of coherent states.
* **Quantum key distribution**: Squeezed states enable continuous-variable quantum cryptography where security is guaranteed by the uncertainty principle.
* **Cluster state generation**: Large entangled states for measurement-based quantum computation can be deterministically produced using squeezed light and linear optics.

---

## Big Ideas

* Squeezed states redistribute quantum noise: one quadrature drops below the vacuum level while the conjugate quadrature inflates, always satisfying $\Delta X \cdot \Delta P \geq \frac{1}{4}$.
* The squeezing operator $\hat{S}(\xi)$ creates and destroys photons in pairs — this is why squeezed vacuum contains only even photon numbers.
* Parametric down-conversion is the workhorse for generating squeezed light: a nonlinear crystal breaks pump photons into pairs, and that pairwise creation is exactly the squeezing Hamiltonian.
* A displaced squeezed state is the most general single-mode Gaussian state: whether it is amplitude-squeezed or phase-squeezed depends on how the noise ellipse is oriented relative to the coherent amplitude.
* Two-mode squeezing creates EPR-type entanglement: each mode looks thermal individually, yet their joint quadratures are correlated far beyond any classical limit.

## What Comes Next

We now have a full toolkit of quantum states of light — from Fock states and coherent states through squeezed and entangled Gaussian states. The question naturally arises: how does a quantum field interact with matter? The next lesson begins the atom-field interaction story, showing how a two-level atom couples to the quantized electromagnetic field through the dipole approximation.

## Check Your Understanding

1. The squeezing operator mixes $\hat{a}$ and $\hat{a}^\dagger$ via $\hat{S}^\dagger\hat{a}\hat{S} = \hat{a}\cosh r - \hat{a}^\dagger e^{i\theta}\sinh r$. Show from this that $\langle\hat{n}\rangle = \sinh^2 r$ for squeezed vacuum. Why does the squeezed vacuum contain photons even though it started from $|0\rangle$?
2. Amplitude-squeezed light has sub-Poissonian photon statistics ($Q < 0$), while phase-squeezed light has super-Poissonian statistics ($Q > 0$). Explain physically why reducing intensity noise necessarily inflates the phase variance and vice versa.
3. The two-mode squeezed vacuum has each individual mode in a thermal (mixed) state, yet the joint state is pure and entangled. How is this possible?

## Challenge

Work out the Bogoliubov transformation: show that $\hat{S}^\dagger(\xi)\hat{a}\hat{S}(\xi) = \hat{a}\cosh r - \hat{a}^\dagger e^{i\theta}\sinh r$. Use this to compute the quadrature variances for the squeezed vacuum and confirm $\frac{1}{4}e^{-2r}$ and $\frac{1}{4}e^{2r}$. Then derive the two-mode Duan criterion $\Delta(X_a - X_b)^2 + \Delta(P_a + P_b)^2 = e^{-2r}$, confirming entanglement for all $r > 0$.
