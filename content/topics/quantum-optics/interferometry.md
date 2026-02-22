# Interferometry and Single-Photon Experiments

## The beam splitter: quantum optics' favorite toy

A beam splitter does something deceptively simple: it partially reflects and partially transmits light. But feed it single photons and you get some of the most stunning demonstrations of quantum mechanics ever performed.

Quantum mechanically, a lossless beam splitter transforms the input mode operators as:

$$
\begin{pmatrix} \hat{a}_\text{out} \\ \hat{b}_\text{out} \end{pmatrix} = \begin{pmatrix} t & r \\ r' & t' \end{pmatrix} \begin{pmatrix} \hat{a}_\text{in} \\ \hat{b}_\text{in} \end{pmatrix}
$$

The matrix must be unitary (to preserve commutation relations), with $|t|^2 + |r|^2 = 1$.

## A single photon never splits

Send one photon $\Ket{1}_a\Ket{0}_b$ into a 50:50 beam splitter. The output is:

$$
\Ket{1}_a\Ket{0}_b \xrightarrow{\text{BS}} \frac{1}{\sqrt{2}}\left(\Ket{1}_c\Ket{0}_d + i\Ket{0}_c\Ket{1}_d\right)
$$

The photon exits through one port or the other -- **never both**. Put detectors at $c$ and $d$: you get a click here or a click there, but never simultaneous clicks. The anticorrelation parameter $\alpha = P_{cd}/(P_c P_d) = 0$ for a true single photon. That's a clean, direct proof that light comes in discrete quanta.

For a coherent state input? The outputs are uncorrelated product states -- perfectly classical behavior.

## Hong-Ou-Mandel: photons that refuse to part

Now send **two indistinguishable single photons** into a 50:50 beam splitter from opposite sides:

$$
\Ket{1}_a\Ket{1}_b \xrightarrow{\text{BS}} \frac{1}{\sqrt{2}}\left(\Ket{2}_c\Ket{0}_d - \Ket{0}_c\Ket{2}_d\right)
$$

Both photons always exit through the *same* port. The coincidence rate at the two outputs drops to zero -- the famous **Hong-Ou-Mandel dip**. No classical wave theory can explain this. The photons are bosons, and when they're perfectly indistinguishable, destructive interference kills the "one each way" outcome.

Any difference in frequency, polarization, or arrival time restores some coincidences. The HOM dip visibility is a direct measure of how indistinguishable your photons are -- and it's the basis of Bell-state measurements and linear-optical quantum computing.

## Homodyne detection: listening to one quadrature

**Homodyne detection** measures a single field quadrature by interfering the signal with a strong local oscillator (LO). Subtract the photocurrents from two detectors:

$$
\hat{I}_- \propto \hat{X}_\theta = \hat{a}e^{-i\theta} + \hat{a}^\dag e^{i\theta}
$$

Rotate the LO phase $\theta$ and you sample different quadratures. Collect enough quadrature histograms at different angles and you can reconstruct the entire quantum state via **tomography** -- the same inverse Radon transform used in medical CT scanning.

**Heterodyne detection** measures both quadratures simultaneously, at the cost of half a quantum of added noise per quadrature. It projects onto coherent states, effectively measuring the Husimi Q function.

## The Mach-Zehnder interferometer

The workhorse of precision measurement: two beam splitters enclosing two paths with a relative phase shift $\phi$. The output intensities oscillate as $\cos^2(\phi/2)$ and $\sin^2(\phi/2)$.

With classical (coherent) light, the best you can do is the **shot-noise limit**:

$$
\Delta\phi_\text{SNL} = \frac{1}{\sqrt{\bar{n}}}
$$

This comes from Poissonian photon statistics. Double your photon count, improve your precision by $\sqrt{2}$.

## Beating the shot-noise limit

Quantum mechanics allows you to do better. The ultimate bound is the **Heisenberg limit**:

$$
\Delta\phi_\text{HL} = \frac{1}{\bar{n}}
$$

Double your photons, *double* your precision. Strategies include:

- **Squeezed vacuum** injected into the dark port of the MZI. This is what LIGO actually does -- squeezed light reduces the noise in the measured quadrature, pushing sensitivity below the shot-noise limit.
- **NOON states** $\Ket{N,0} + \Ket{0,N}$: Heisenberg-limited but fragile.
- **Twin-Fock states** $\Ket{N,N}$: more robust against losses.

## Big Ideas

- A single photon at a beam splitter never splits: one detector clicks, never both. That's proof of quantization.
- The Hong-Ou-Mandel effect: two indistinguishable photons always exit together. Zero coincidences. No classical explanation.
- Homodyne detection projects onto a quadrature; rotating the LO phase enables full state tomography.
- The shot-noise limit $1/\sqrt{\bar{n}}$ is not fundamental -- squeezing or entanglement can breach it, approaching the Heisenberg limit $1/\bar{n}$.

## Check Your Understanding

1. Why can't you measure $\hat{X}$ and $\hat{P}$ simultaneously with a single homodyne setup?
2. What kind of quantum state achieves Heisenberg-limited sensitivity? Why do entangled photons beat independent ones?

## Challenge

Write a Python script that propagates a coherent state $\Ket{\alpha}$ through a Mach-Zehnder interferometer (two beam splitters + phase shift $\phi$) using matrix representations. Compute $\langle\hat{n}_c - \hat{n}_d\rangle$ and its variance at the output, and extract the phase sensitivity via error propagation. Verify $\Delta\phi = 1/\sqrt{\bar{n}}$ (shot-noise limit). Then inject squeezed vacuum into the unused port and show the sensitivity improves to $\Delta\phi = e^{-r}/\sqrt{\bar{n}}$.
