# Interferometry

## Input-Output Relations

Optical networks of beam splitters and phase shifters are described by **input-output relations** that map input mode operators to output mode operators. For a network with $M$ modes, the transformation is

$$
\hat{a}_{\text{out}, i} = \sum_{j=1}^{M} U_{ij} \hat{a}_{\text{in}, j},
$$

where $U$ is a unitary matrix. Any unitary transformation on $M$ modes can be decomposed into a sequence of beam splitters and phase shifters (Reck decomposition). This universality makes linear optics a powerful platform for quantum information processing.

## Homodyne Detection

**Homodyne detection** measures a single quadrature of the electromagnetic field by interfering the signal with a strong local oscillator (LO) at the same frequency. The signal mode $\hat{a}$ and the LO mode $\hat{b} \approx |\beta|e^{i\theta}$ are combined on a 50:50 beam splitter, and the photocurrents from two detectors are subtracted:

$$
\hat{I}_- \propto \hat{a} e^{-i\theta} + \hat{a}^\dagger e^{i\theta} = \hat{X}_\theta,
$$

where $\hat{X}_\theta$ is the quadrature operator at phase angle $\theta$. By varying the LO phase $\theta$, any quadrature can be measured. This enables full reconstruction of the quantum state via **quantum state tomography**.

For a coherent state $|\alpha\rangle$, homodyne detection yields Gaussian noise centered at $2|\alpha|\cos(\theta - \phi_\alpha)$ with vacuum-level variance. For squeezed states, the variance is below vacuum for one quadrature.

## Heterodyne Detection

**Heterodyne detection** simultaneously measures both quadratures ($\hat{X}$ and $\hat{P}$) by using a local oscillator detuned from the signal frequency. This is equivalent to a joint measurement of non-commuting observables and necessarily adds half a quantum of noise to each quadrature (from the Heisenberg uncertainty principle):

$$
\Delta X \cdot \Delta P \geq \frac{1}{2}.
$$

Heterodyne detection projects onto coherent states and is equivalent to measuring the Husimi Q-function of the field.

## Mach-Zehnder Interferometer

The **Mach-Zehnder interferometer** (MZI) is the workhorse of optical interferometry. Two beam splitters enclose two paths with a relative phase shift $\phi$. The full unitary transformation for a balanced MZI is

$$
U_{\text{MZI}} = U_{\text{BS}} \cdot U_\phi \cdot U_{\text{BS}} = \begin{pmatrix} \cos(\phi/2) & i\sin(\phi/2) \\ i\sin(\phi/2) & \cos(\phi/2) \end{pmatrix},
$$

up to global phases. The output intensities oscillate sinusoidally with $\phi$, allowing precise phase measurements.

## Quantum-Enhanced Interferometry

Classical interferometry with coherent light achieves a phase sensitivity at the **shot-noise limit**:

$$
\Delta\phi_{\text{SNL}} = \frac{1}{\sqrt{\bar{n}}},
$$

where $\bar{n}$ is the mean photon number. This limit arises from the Poissonian photon statistics of coherent states.

Quantum states can beat this limit. The ultimate bound set by quantum mechanics is the **Heisenberg limit**:

$$
\Delta\phi_{\text{HL}} = \frac{1}{\bar{n}}.
$$

Strategies to approach the Heisenberg limit include:
* **Squeezed vacuum** injected into the unused port of the MZI reduces the noise in the measured quadrature. This was implemented in LIGO to improve gravitational wave sensitivity.
* **NOON states** $|N,0\rangle + |0,N\rangle$ achieve Heisenberg-limited sensitivity but are fragile and difficult to prepare for large $N$.
* **Twin-Fock states** $|N,N\rangle$ provide robustness against losses compared to NOON states.

## Sagnac Interferometer

The **Sagnac interferometer** uses a common path traversed in opposite directions. Rotation of the interferometer introduces a phase shift proportional to the enclosed area and angular velocity (Sagnac effect):

$$
\Delta\phi = \frac{8\pi A \Omega}{c\lambda},
$$

where $A$ is the enclosed area and $\Omega$ is the rotation rate. Fiber-optic gyroscopes and ring laser gyroscopes exploit this principle for navigation and geophysics.

## Big Ideas

* Any linear-optical network — no matter how complex — is described by a unitary matrix acting on the mode operators, and any unitary can be built from beam splitters and phase shifters alone.
* Homodyne detection projects the field onto a single quadrature by interfering it with a strong local oscillator; rotating the local oscillator phase rotates the measurement axis in phase space, enabling full state tomography.
* The shot-noise limit $\Delta\phi = 1/\sqrt{\bar{n}}$ is not fundamental — it is merely the limit for coherent-state inputs with Poissonian noise. Squeezing or entanglement can breach it.
* The Heisenberg limit $\Delta\phi = 1/\bar{n}$ is the fundamental quantum limit: beating it would require measuring more information than the uncertainty principle permits.

## What Comes Next

Phase measurement is the gateway to understanding why squeezed light matters. The next lesson introduces [squeezed states](./squeezed-states) — states where one quadrature has noise below the vacuum level — and shows how they are generated, characterized, and used to push interferometric sensitivity toward the Heisenberg limit. In particular, injecting squeezed vacuum into the dark port of a Mach-Zehnder interferometer improves the phase sensitivity from $\Delta\phi = 1/\sqrt{\bar{n}}$ to $\Delta\phi = e^{-r}/\sqrt{\bar{n}}$ — the principle behind LIGO's quantum noise reduction.

## Check Your Understanding

1. Homodyne detection measures $\hat{X}_\theta$ by interfering the signal with a strong local oscillator at phase $\theta$. By varying $\theta$, you sample different quadratures. Explain why you cannot measure $\hat{X}$ and $\hat{P}$ simultaneously with a single homodyne setup, and what the uncertainty principle has to say about it.
2. The shot-noise limit scales as $1/\sqrt{\bar{n}}$: doubling the photon number improves precision by $\sqrt{2}$. The Heisenberg limit scales as $1/\bar{n}$: doubling the photon number doubles the precision. What kind of quantum state achieves Heisenberg-limited sensitivity, and intuitively why do entangled photons do better than independent ones?
3. The Sagnac interferometer measures rotation using counter-propagating beams in a common path. Why does this configuration make it insensitive to vibrations, mirror imperfections, and slow thermal drifts that would wreck a Mach-Zehnder interferometer, while remaining sensitive to rotation?

## Challenge

For a Mach-Zehnder interferometer with coherent input $|\alpha\rangle$ in one port and vacuum in the other, propagate the state through both beam splitters and the phase shift $\phi$, and compute the photon number difference $\langle\hat{n}_c - \hat{n}_d\rangle$ and its variance at the output. Show that the phase sensitivity obtained from error propagation is $\Delta\phi = 1/|\alpha| = 1/\sqrt{\bar{n}}$ (shot-noise limit). Then repeat the analysis with a squeezed vacuum $|\xi\rangle$ (squeezing parameter $r$) injected into the normally empty port, and show that the sensitivity improves to $\Delta\phi = e^{-r}/\sqrt{\bar{n}}$. This is the principle behind the squeezed-light injection in LIGO.
