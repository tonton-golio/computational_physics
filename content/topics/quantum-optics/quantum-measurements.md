# Quantum Measurements

## Measurement in Quantum Mechanics

In quantum optics, measurement plays a fundamental role: the act of detection irreversibly alters the quantum state of light. The measurement formalism connects the abstract quantum state to experimentally observable quantities like photon counts, quadrature values, and correlation functions.

A general quantum measurement is described by a set of **measurement operators** $\{\hat{M}_m\}$ satisfying $\sum_m \hat{M}_m^\dagger \hat{M}_m = \hat{I}$. The probability of outcome $m$ given state $\hat{\rho}$ is

$$
p(m) = \operatorname{Tr}[\hat{M}_m^\dagger \hat{M}_m \hat{\rho}],
$$

and the post-measurement state is $\hat{\rho}_m = \hat{M}_m \hat{\rho} \hat{M}_m^\dagger / p(m)$.

## Projective vs. Generalized Measurements

**Projective (von Neumann) measurements** use orthogonal projectors $\hat{\Pi}_m = |m\rangle\langle m|$ satisfying $\hat{\Pi}_m \hat{\Pi}_n = \delta_{mn}\hat{\Pi}_m$. Photon number detection is a projective measurement in the Fock basis $\{|n\rangle\}$.

**Generalized measurements** (POVMs) allow non-orthogonal outcomes and describe realistic detectors. A **positive operator-valued measure** consists of positive operators $\hat{E}_m \geq 0$ with $\sum_m \hat{E}_m = \hat{I}$, without requiring orthogonality. Heterodyne detection is a POVM with elements $\hat{E}_\alpha = \frac{1}{\pi}|\alpha\rangle\langle\alpha|$ that projects onto coherent states.

## Photon Counting

Ideal **photon-number-resolving** (PNR) detectors measure the Fock state projectors. For a state $\hat{\rho}$, the probability of detecting $n$ photons is $p(n) = \langle n|\hat{\rho}|n\rangle$.

Real single-photon detectors (avalanche photodiodes, superconducting nanowire detectors) are typically **threshold detectors**: they distinguish "no click" from "one or more clicks" but cannot resolve the exact photon number. The POVM elements are

$$
\hat{E}_0 = |0\rangle\langle 0|, \qquad \hat{E}_{\text{click}} = \hat{I} - |0\rangle\langle 0|,
$$

with detector efficiency $\eta < 1$ modeled as a beam splitter loss before an ideal detector.

Transition-edge sensors and superconducting nanowire arrays can resolve photon numbers up to $\sim 10{-}20$, enabling direct measurement of photon statistics and Wigner function negativity.

## Quantum State Tomography

**Quantum state tomography** reconstructs the full density matrix $\hat{\rho}$ from a set of measurements on many identically prepared copies. For optical fields, this is achieved by:

1. **Homodyne tomography**: Measure the quadrature $\hat{X}_\theta$ for many phase angles $\theta$. The marginal distributions $p(x_\theta)$ are projections of the Wigner function. The state is reconstructed via the inverse Radon transform (same mathematics as medical CT scanning).

2. **Maximum likelihood estimation**: Numerically find the density matrix that maximizes the likelihood of the observed data, subject to the constraints that $\hat{\rho}$ is positive semidefinite with unit trace.

The Wigner function provides a complete phase-space representation:

$$
W(x, p) = \frac{1}{\pi\hbar}\int_{-\infty}^{\infty} \langle x + y|\hat{\rho}|x - y\rangle e^{-2ipy/\hbar} dy.
$$

Negative values of $W$ indicate non-classical states (e.g., Fock states, cat states).

## Quantum Non-Demolition Measurements

A **quantum non-demolition** (QND) measurement extracts information about an observable without disturbing it. For photon number, the key requirement is $[\hat{n}, \hat{H}_{\text{int}}] = 0$: the interaction used for measurement commutes with the measured observable.

In cavity QED, the dispersive interaction shifts the atomic phase by an amount proportional to $n$ without absorbing photons. By sending probe atoms through the cavity and measuring their phase shift, one can determine $n$ repeatedly and observe quantum jumps as individual photons are lost.

This was demonstrated by Haroche's group, who tracked the photon number in a microwave cavity decaying from $n = 7$ to $n = 0$ one photon at a time, directly observing the quantum trajectory of the field state.

## Back-Action and the Uncertainty Principle

Every measurement has **back-action**: gaining information about one observable increases uncertainty in conjugate observables. For homodyne detection of $\hat{X}$, the back-action appears as increased noise in $\hat{P}$.

The **standard quantum limit** (SQL) for monitoring the position of a free mass is

$$
\Delta x_{\text{SQL}} = \sqrt{\frac{\hbar t}{2m}},
$$

arising from the trade-off between measurement imprecision and radiation-pressure back-action. Beating the SQL requires correlating the measurement and back-action noise, achievable with squeezed light or back-action evasion techniques.

## Big Ideas

* Measurement in quantum optics is not passive reading — every detection event irreversibly alters the state of the field, and the post-measurement state is as important as the probability of each outcome.
* POVMs (positive operator-valued measures) capture the full range of realistic detectors: threshold detectors, homodyne receivers, and heterodyne detectors each correspond to a different set of POVM elements.
* Quantum state tomography reconstructs the full density matrix from quadrature histograms using the inverse Radon transform — the same mathematics that reconstructs a CT scan from X-ray projections.
* Back-action is unavoidable: learning about $\hat{X}$ necessarily disturbs $\hat{P}$, and beating the standard quantum limit requires cleverly correlating that disturbance rather than trying to eliminate it.

## What Comes Next

This is where the quantum optics journey culminates. The thread that began with a single field mode in a box has led us through vacuum fluctuations, coherent and squeezed states, photon statistics, atom-field coupling, and cavity QED — all converging on the fundamental question of how information is extracted from a quantum field. The ideas here — POVM measurement, back-action, tomography, quantum jumps — are the operating principles of quantum sensing, quantum communication, and quantum computing built on light.

## Check Your Understanding

1. A threshold detector has POVM elements $\hat{E}_0 = |0\rangle\langle 0|$ and $\hat{E}_\text{click} = \hat{I} - |0\rangle\langle 0|$. It cannot distinguish $|1\rangle$ from $|2\rangle$ from $|10\rangle$. What information is completely lost by using a threshold detector instead of a photon-number-resolving detector, and why does this matter for reconstructing the Wigner function?
2. Homodyne tomography measures the quadrature $\hat{X}_\theta$ for many values of $\theta$ and uses the inverse Radon transform to reconstruct $W(x, p)$. How many different phase angles $\theta$ do you need in principle, and what fundamentally limits the precision of the reconstruction for a finite number of measurement trials?
3. A quantum non-demolition measurement of photon number must satisfy $[\hat{n}, \hat{H}_\text{int}] = 0$. Explain why this condition guarantees that the photon number is undisturbed by the measurement, and give an example of a QND observable and a non-QND observable for the electromagnetic field.

## Challenge

Model a realistic photon detector with efficiency $\eta < 1$ as a beam splitter that mixes the signal with a vacuum mode, followed by an ideal detector. Derive the POVM elements for this lossy detector and compute the probability $p(n_\text{click})$ of registering $n_\text{click}$ clicks when the input state is a Fock state $|N\rangle$. Show that this gives a binomial distribution with success probability $\eta$. Then consider the post-measurement state: if $n_\text{click}$ clicks are registered from $|N\rangle$, what is the state of the field after detection? This calculation reveals the fundamental distinction between a destructive measurement and a QND measurement of the same photon number.
