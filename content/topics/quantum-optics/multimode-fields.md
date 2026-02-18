# Multimode Fields
__Topic 2 keywords__
- Multi-mode fields
- Thermal states-density of states
- Planck formula
- Density operators
- Lamb shift
- Casimir forces

# Readings
Ch. 2.4-6, App. A.

# 2.4 Multimode fields
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
- $\omega_k=kc$ is frequency
- $\vec{k}$ is the wave vetor
- $s$ is the polarization index

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

# 2.5 Thermal fields
#### Thermal light
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
    - 
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
#### Planck's radiation law
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

