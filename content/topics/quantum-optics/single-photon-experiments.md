# Single Photon Experiments

## The Beam Splitter

A **beam splitter** is the fundamental optical element for single-photon experiments. It partially reflects and partially transmits incoming light. Quantum mechanically, a lossless beam splitter with reflectivity $R$ and transmissivity $T = 1 - R$ transforms the input mode operators as

$$
\begin{pmatrix} \hat{a}_{\text{out}} \\ \hat{b}_{\text{out}} \end{pmatrix}
=
\begin{pmatrix} t & r \\ r' & t' \end{pmatrix}
\begin{pmatrix} \hat{a}_{\text{in}} \\ \hat{b}_{\text{in}} \end{pmatrix},
$$

where $|t|^2 + |r|^2 = 1$ and the matrix must be unitary to preserve the commutation relations $[\hat{a}, \hat{a}^\dagger] = 1$. For a symmetric beam splitter: $t = t' = \cos\theta$ and $r = -r'^* = i\sin\theta$.

## Single Photon at a Beam Splitter

When a single photon $|1\rangle_a |0\rangle_b$ enters one port of a 50:50 beam splitter, the output state is

$$
|1\rangle_a |0\rangle_b \xrightarrow{\text{BS}} \frac{1}{\sqrt{2}}\left(|1\rangle_c |0\rangle_d + i|0\rangle_c |1\rangle_d\right).
$$

The photon exits through one port or the other but is never split: a single detector click occurs in output $c$ or output $d$, never both simultaneously. This was demonstrated by Grangier, Roger, and Aspect (1986), confirming the particle nature of single photons.

The joint detection probability at both outputs is $P_{cd} = 0$ for a true single photon, while for a classical wave it would be $P_{cd} > 0$. The **anticorrelation parameter** $\alpha = P_{cd}/(P_c P_d)$ equals zero for a single photon, providing a clean test of the quantum nature of light.

## Coherent State at a Beam Splitter

For a coherent state input $|\alpha\rangle_a |0\rangle_b$, the output is a product of coherent states:

$$
|\alpha\rangle_a |0\rangle_b \xrightarrow{\text{BS}} |t\alpha\rangle_c |r\alpha\rangle_d.
$$

Unlike the single-photon case, the outputs are uncorrelated and each port independently contains a coherent state. This factorization property is unique to coherent states and reflects their classical nature.

A 50:50 beam splitter splits a coherent state $|\alpha\rangle$ into two coherent states $|\alpha/\sqrt{2}\rangle$, with the total mean photon number preserved: $|\alpha/\sqrt{2}|^2 + |\alpha/\sqrt{2}|^2 = |\alpha|^2$.

## Two-Photon Interference: Hong-Ou-Mandel Effect

When **two indistinguishable single photons** enter a 50:50 beam splitter from different ports, they always exit together through the same port:

$$
|1\rangle_a |1\rangle_b \xrightarrow{\text{BS}} \frac{1}{\sqrt{2}}\left(|2\rangle_c |0\rangle_d - |0\rangle_c |2\rangle_d\right).
$$

The coincidence rate at the two outputs drops to zero, known as the **Hong-Ou-Mandel (HOM) dip**. This two-photon quantum interference effect has no classical analogue and was first demonstrated by Hong, Ou, and Mandel (1987).

The dip visibility depends on the indistinguishability of the photons. If the photons differ in frequency, polarization, or arrival time, the dip becomes shallower. The HOM effect is the basis for **Bell-state measurements** and linear-optical quantum computing schemes.

## Mach-Zehnder Interferometer

A **Mach-Zehnder interferometer** consists of two beam splitters with a phase shift $\phi$ in one arm. For a single-photon input, the detection probabilities at the two outputs are

$$
P_c = \cos^2(\phi/2), \qquad P_d = \sin^2(\phi/2).
$$

This demonstrates single-photon interference: the photon interferes with itself as it traverses both paths simultaneously in superposition. The which-path information determines whether interference is observed, illustrating the complementarity principle.

For $N$-photon states or squeezed light inputs, the interferometer can achieve phase sensitivity beyond the **shot-noise limit** $\Delta\phi \sim 1/\sqrt{N}$, approaching the **Heisenberg limit** $\Delta\phi \sim 1/N$.
