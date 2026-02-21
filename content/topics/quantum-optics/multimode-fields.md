# Multimode Fields
__Topic 2 keywords__
* Multi-mode fields
* Thermal states-density of states
* Planck formula
* Density operators
* Lamb shift
* Casimir forces

## Multimode fields
In the textbook, they use the vector potential. 
We can do same things in different way.
Consider the Hamiltonian of multi-mode wave.
$$
\begin{aligned}
    \hat{H}
    &=
    \sum_{\vec{k}, s}
    \hbar \omega_k
    \left(
        \hat{a}^\dag_{\vec{k},s}
        \hat{a}_{\vec{k},s}
        +
        \frac{1}{2}
    \right)
    \\&=
    \sum_j
    \hbar \omega_j
    \left(
        \hat{n}_j
        +
        \frac{1}{2}
    \right)
\end{aligned}
$$
Here, paramters are follows.
* $\omega_k=kc$ is frequency
* $\vec{k}$ is the wave vetor
* $s$ is the polarization index

By using index $j$ we can simply write the eigenstate of this Hamiltonian.
The eigenstate would be product of number states.
$$
    \Ket{\left\{n_j\right\}}
    =
    \Ket{n_1, n_2, \cdots, n_j, \cdots}
$$
This state means how many photon in each mode.

We can also think the annihilation operator of multi-mode state.
$$
    \hat{a}_i
    \Ket{\left\{n_j\right\}}
    =
    \sqrt{n_i}
    \Ket{n_1, n_2, \cdots, n_i-1, \cdots}
$$

## Thermal fields
### Thermal light
So far we consider the zero-temperature box.
We can think interaction between photon and thermal wall.
In thermal equilibrium, the density matrix is
$$
    \hat{\rho}_\mathrm{th}
    =
    \frac
    {e^{ -\hat{H} / k_\mathrm{B}T}}
    {\operatorname{Tr}~e^{- \hat{H} / k_\mathrm{B}T}}
$$
Remember that operator on Napier number is symbolic representation of Maclaurin 
series.
$$
    e^{ -\hat{H} / k_\mathrm{B}T}
    =
    1 
    * 
    \frac{\hat{H}}{k_\mathrm{B}T}
    +
    \frac{1}{2}
    \left(
        \frac{\hat{H}}{k_\mathrm{B}T}
    \right)^2
    \cdots
$$
Probability of occupying $\Ket{n}$ is
$$
    P_n 
    =
    \Braket{
        n|
        \hat{\rho}_\mathrm{th}
        |n
    }
$$
Expectation number of photon is 
$$
    \boxed{
    \left< n \right>
    =
    \sum_n
    n P_n
    =
    \frac
    {1}
    {e^{\hbar \omega / k_\mathrm{B}T}  - 1 }
    }
$$
Let's check the order of this number.
Energy of visible light is 
$$
    \hbar \omega
    \sim
    1
    \mathrm{eV}
$$
Energy of room temperature is
$$
    k_\mathrm{B}T
    \sim
    \frac{1}{40}
    \mathrm{eV}
$$
By putting these number into equation, we see
$$
    \left< \hat{n} \right>
    \sim
    0
$$
So photon does not care temperature. This is different from quantum computer.
**Temperature does not play a role in quantum optics.**

Thermal fluctuation is
$$
\begin{aligned}
    \left< \left( \Delta n \right)^2 \right>
    &=
    \left< \hat{n}^2 \right>
    -
    \left< \hat{n} \right>^2
    \\&=
    \bar{n}
    +
    \bar{n}^2
\end{aligned}
$$
This is super-Poisonian distribution. Variance is larger than that of Poisonian.
### Planck's radiation law
Consider the light in the box with each length $L$ in thermal equilibrium.
From boundary condition,
$$
    e^{i k_i x_i}
    =
    e^{i k_i (x_i+L)}
$$
$$
    k_i 
    =
    \frac{2\pi}{L} 
    m_i
$$
The number of state is 
$$
\begin{aligned}
    \Delta m
    &=
    \Delta m_x
    \Delta m_y
    \Delta m_z
    \\&=
    2
    \frac{V}{2\pi}
    \Delta k_x
    \Delta k_y
    \Delta k_z
\end{aligned}
$$
Here multiplication of 2 is number of polarization.

We can think this in spherical coordinate.
$$
\begin{aligned}
    \mathrm{d}m
    &=
    \frac{V}{4\pi^3}
    \mathrm{d}^3 k
    \\&=
    \frac{V}{4\pi^3}
    k^2
    \mathrm{d} k
    \mathrm{d} \Omega
    \\&=
    \frac{V}{4\pi^3}
    \frac{\omega^2}{c^3}
    \mathrm{d} \omega
    \mathrm{d} \Omega
\end{aligned}
$$
Density of state is 
$$
    \rho(\omega)
    =
    \frac{\omega^2}{\pi^2 c^3}
$$
Dimension of density of state is 
$\frac{\text{\# of state}}{\text{frequency} \cdot \text{volume}}$

Energy density become
$$
\begin{aligned}
    \bar{U}
    &=
    \hbar \omega \bar{n} \rho (\omega)
    \\&=
    \frac
    {\hbar\omega}
    {e^{\hbar\omega/k_\mathrm{B}T} - 1}
    \frac
    {\omega^2}
    {\pi^2 c^3}
\end{aligned}
$$

## Big Ideas

* A realistic field is a superposition of infinitely many modes; quantizing each independently gives a multimode Fock state labeled by how many photons occupy each mode.
* Thermal light at room temperature contains essentially zero photons per visible mode — temperature is irrelevant for optical quantum optics in a way it is not for microwave quantum computing.
* Planck's blackbody law emerges directly from the quantum statistics of photons: each mode contributes its Bose-Einstein average $\bar{n} = 1/(e^{\hbar\omega/k_\mathrm{B}T}-1)$ weighted by the density of states.
* Thermal photon fluctuations are super-Poissonian, with variance $\bar{n} + \bar{n}^2$ — noisier than a coherent laser.

## What Comes Next

We have seen that the quantum vacuum is not empty: every mode carries zero-point energy $\frac{1}{2}\hbar\omega$. The next lesson takes this seriously and asks what observable consequences that vacuum energy has — because it turns out you can measure it directly in the lab, in phenomena ranging from spontaneous emission shifts to forces between uncharged metal plates.

## Check Your Understanding

1. At optical frequencies ($\hbar\omega \sim 1$ eV) and room temperature ($k_\mathrm{B}T \sim \frac{1}{40}$ eV), the mean photon number is essentially zero. What does this tell you about the usefulness of cooling an optical experiment versus a microwave experiment for eliminating thermal photons?
2. Thermal light has $\langle(\Delta n)^2\rangle = \bar{n} + \bar{n}^2$. Explain why the $\bar{n}^2$ term is a signature of photon bunching, and what it implies about the probability of detecting two photons close together in time.
3. The density of states $\rho(\omega) = \omega^2/(\pi^2 c^3)$ grows as $\omega^2$. What would happen to the total energy of blackbody radiation if the photon were not subject to Bose-Einstein statistics (suppose instead it obeyed Boltzmann statistics)? This is the ultraviolet catastrophe — why does quantization fix it?

## Challenge

Derive the Planck formula by computing the spectral energy density in a cubic cavity of side $L$, starting only from the allowed mode frequencies and the Bose-Einstein distribution. Then take the thermodynamic limit $L \to \infty$ and show you recover $\bar{U}(\omega) = \hbar\omega\rho(\omega)/({e^{\hbar\omega/k_\mathrm{B}T}-1})$. Finally, integrate over all frequencies to show that the total energy density is proportional to $T^4$ (Stefan-Boltzmann law), identifying the Stefan-Boltzmann constant in terms of fundamental constants.

