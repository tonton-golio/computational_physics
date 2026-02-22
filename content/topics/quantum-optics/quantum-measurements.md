# Quantum Measurements

## Every click changes the state

In quantum optics, measurement isn't passive reading. Every photon detection event irreversibly alters the quantum state of light. The post-measurement state is just as important as the probability of each outcome. This lesson closes the loop: how do you actually extract information from a quantum field, and what does the extraction cost?

## The measurement formalism

A general quantum measurement is described by measurement operators $\{\hat{M}_m\}$ satisfying $\sum_m \hat{M}_m^\dag\hat{M}_m = \hat{I}$. The probability of outcome $m$ is:

$$
p(m) = \operatorname{Tr}[\hat{M}_m^\dag\hat{M}_m\hat{\rho}]
$$

and the post-measurement state is $\hat{\rho}_m = \hat{M}_m\hat{\rho}\hat{M}_m^\dag/p(m)$.

## Projective vs. generalized measurements

**Projective (von Neumann) measurements** use orthogonal projectors: $\hat{\Pi}_m = \Ket{m}\Bra{m}$. Photon counting is a projective measurement in the Fock basis.

**POVMs** (positive operator-valued measures) allow non-orthogonal outcomes. They describe realistic detectors. Heterodyne detection, for instance, is a POVM with elements $\hat{E}_\alpha = \frac{1}{\pi}\Ket{\alpha}\Bra{\alpha}$ that projects onto coherent states. You can't distinguish non-orthogonal states perfectly, but you can make a "best guess" measurement -- and POVMs are the mathematical framework for it.

## Photon counting in practice

Ideal photon-number-resolving (PNR) detectors measure Fock state projectors: $p(n) = \langle n|\hat{\rho}|n\rangle$.

Real single-photon detectors (avalanche photodiodes, superconducting nanowire detectors) are usually **threshold detectors**: they tell you "something clicked" or "nothing clicked," but can't tell you whether it was 1 photon or 10. The POVM elements are:

$$
\hat{E}_0 = \Ket{0}\Bra{0}, \qquad \hat{E}_\text{click} = \hat{I} - \Ket{0}\Bra{0}
$$

Detector inefficiency ($\eta < 1$) is modeled as a beam splitter that tosses away a fraction of the photons before they reach an ideal detector.

Modern transition-edge sensors and nanowire arrays can resolve photon numbers up to about 10-20, enabling direct measurements of photon statistics and Wigner function negativity.

## Quantum state tomography

**Tomography** reconstructs the full density matrix from measurements on many identical copies of the state.

For optical fields, the standard approach is **homodyne tomography**: measure the quadrature $\hat{X}_\theta$ at many phase angles $\theta$. Each angle gives a marginal distribution of the Wigner function -- a "shadow" of the phase-space blob from one direction. Collect enough shadows and you can reconstruct the full Wigner function via the **inverse Radon transform** -- the exact same mathematics that reconstructs a CT scan from X-ray projections.

Alternatively, use **maximum likelihood estimation** to find the density matrix that best explains all your data, subject to the constraint that $\hat{\rho}$ is positive semidefinite with unit trace.

[[simulation homodyne-tomography]]

> **Computational Note.** You can simulate homodyne tomography in Python: generate quadrature samples from a known Wigner function at random angles, then reconstruct the state using the inverse Radon transform (available in `scikit-image`). Start with a coherent state (Gaussian blob) and then try a Fock state $\Ket{1}$ (which has negative Wigner values). About 40 lines of code.

## Quantum non-demolition measurements

A **QND measurement** extracts information about an observable without disturbing it. The requirement: $[\hat{n}, \hat{H}_\text{int}] = 0$ -- the interaction commutes with the measured observable.

In cavity QED, the dispersive interaction shifts the atomic phase by an amount proportional to $n$ without absorbing photons. Send probe atoms through the cavity, measure their phase shifts, and you determine $n$ repeatedly. Haroche's group used this to track a microwave cavity field decaying from $n = 7$ to $n = 0$, one photon at a time, directly observing the quantum trajectory of the field.

## Back-action and the standard quantum limit

Every measurement has **back-action**. Learn about $\hat{X}$, and you necessarily increase the uncertainty in $\hat{P}$.

For continuously monitoring the position of a free mass, the **standard quantum limit** is:

$$
\Delta x_\text{SQL} = \sqrt{\frac{\hbar t}{2m}}
$$

This comes from the trade-off between measurement imprecision (not enough photons to see clearly) and radiation-pressure back-action (too many photons kick the mirror). Beating the SQL requires correlating these two noise sources -- achievable with squeezed light or back-action evasion techniques.

## Big Ideas

* Every detection event changes the quantum state. The post-measurement state matters as much as the outcome probability.
* POVMs capture realistic detectors: threshold detectors, homodyne, heterodyne -- each corresponds to different POVM elements.
* Quantum state tomography reconstructs the full density matrix from quadrature histograms via the inverse Radon transform -- same math as CT scanning.
* Back-action is unavoidable: learning about $\hat{X}$ disturbs $\hat{P}$. Beating the SQL requires correlating the disturbance, not eliminating it.

## Check Your Understanding

1. A threshold detector can't distinguish $\Ket{1}$ from $\Ket{10}$. What information is lost compared to a PNR detector, and why does this matter for Wigner function reconstruction?
2. Homodyne tomography uses the inverse Radon transform. How many phase angles do you need in principle, and what limits precision in practice?

## Challenge

Write a Python script that simulates homodyne tomography of a quantum state. Generate quadrature measurement samples at $M$ different phase angles $\theta_k = k\pi/M$ for a known state (start with a coherent state $\Ket{\alpha}$, then try $\Ket{1}$). Use the inverse Radon transform to reconstruct the Wigner function and compare to the exact result. Explore how the reconstruction quality depends on $M$ and on the number of samples per angle. Can you resolve the negative dip at the origin for $\Ket{1}$?
