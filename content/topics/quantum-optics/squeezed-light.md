# Squeezed Light

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

## Applications of Squeezed Light

Squeezed light has practical applications in precision measurement:

* **Gravitational wave detection**: LIGO and Virgo inject squeezed vacuum into the interferometer's dark port, reducing shot noise and improving sensitivity. Frequency-dependent squeezing (using filter cavities) optimizes noise reduction across the detection bandwidth.

* **Quantum key distribution**: Squeezed states enable continuous-variable quantum cryptography protocols where security is guaranteed by the uncertainty principle.

* **Spectroscopy**: Sub-shot-noise measurements improve the sensitivity of absorption and phase-shift spectroscopy.

* **Quantum computing**: Squeezed states are resources for measurement-based quantum computation in the continuous-variable regime, where large cluster states can be deterministically generated.

## Big Ideas

* Squeezed states redistribute quantum noise: one quadrature drops below the vacuum level while the conjugate quadrature inflates, always satisfying $\Delta X \cdot \Delta P \geq \frac{1}{4}$.
* The squeezing operator $\hat{S}(\xi) = \exp[\frac{1}{2}(\xi^*\hat{a}^2 - \xi\hat{a}^{\dagger 2})]$ creates and destroys photons in pairs — this is why squeezed vacuum contains only even photon numbers.
* Parametric down-conversion is the workhorse for generating squeezed light: a nonlinear crystal breaks pump photons into pairs, and that pairwise creation is exactly the squeezing Hamiltonian.
* Squeezing is measured in decibels relative to the vacuum noise floor; LIGO uses squeezed light to hear gravitational waves that would otherwise be drowned out by quantum shot noise.

## What Comes Next

Squeezed states have reduced noise in one quadrature but no particular displacement in phase space. The next lesson combines squeezing with displacement to produce displaced squeezed states — the most general Gaussian states — and extends the idea to two-mode squeezing, which is the primary resource for continuous-variable entanglement and quantum teleportation.

## Check Your Understanding

1. The squeezing operator mixes $\hat{a}$ and $\hat{a}^\dagger$ with the transformation $\hat{S}^\dagger\hat{a}\hat{S} = \hat{a}\cosh r - \hat{a}^\dagger e^{i\theta}\sinh r$. Show from this that if $|\xi\rangle = \hat{S}(\xi)|0\rangle$, then $\langle\hat{n}\rangle = \sinh^2 r$. Interpret physically: why does the squeezed vacuum contain photons even though it started from $|0\rangle$?
2. The photon number distribution of squeezed vacuum has support only on even $|2n\rangle$ states. Explain this parity conservation from the structure of the squeezing operator: the Hamiltonian $\hat{H}_{\text{PDC}} = i\hbar\kappa(\hat{a}^{\dagger 2} - \hat{a}^2)$ creates and annihilates photons in pairs, so parity is conserved. What observable would you measure to directly confirm this even-photon-number distribution?
3. Squeezing of $r$ gives noise reduction of $e^{-2r}$ in one quadrature. Current experiments achieve $r \approx 1.7$ (about 15 dB). What fundamental physical mechanism limits how much squeezing can be achieved in practice, and why doesn't the uncertainty principle itself set the limit?

## Challenge

Work out the Bogoliubov transformation: show that $\hat{S}^\dagger(\xi)\hat{a}\hat{S}(\xi) = \hat{a}\cosh r - \hat{a}^\dagger e^{i\theta}\sinh r$. Use this to compute the quadrature variances $(\Delta X_\theta)^2$ and $(\Delta X_{\theta+\pi/2})^2$ for the squeezed vacuum and confirm the values $\frac{1}{4}e^{-2r}$ and $\frac{1}{4}e^{2r}$. Then compute the Wigner function of the squeezed vacuum and show it is a Gaussian ellipse with semi-axes $e^{-r}/2$ and $e^{r}/2$. Sketch how this ellipse evolves as $r$ increases from 0 (vacuum) to $r = 2$ (strong squeezing).
