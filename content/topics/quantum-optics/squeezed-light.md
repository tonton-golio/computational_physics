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

- **Gravitational wave detection**: LIGO and Virgo inject squeezed vacuum into the interferometer's dark port, reducing shot noise and improving sensitivity. Frequency-dependent squeezing (using filter cavities) optimizes noise reduction across the detection bandwidth.

- **Quantum key distribution**: Squeezed states enable continuous-variable quantum cryptography protocols where security is guaranteed by the uncertainty principle.

- **Spectroscopy**: Sub-shot-noise measurements improve the sensitivity of absorption and phase-shift spectroscopy.

- **Quantum computing**: Squeezed states are resources for measurement-based quantum computation in the continuous-variable regime, where large cluster states can be deterministically generated.
