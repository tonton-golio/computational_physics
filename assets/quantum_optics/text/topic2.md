__Topic 2 keywords__
- Multi-mode fields
- thermal states-density of states
- Planck formula
- density operators
- Lamb shift
- Casimir forces

# __Readings__
Ch. 2.4-6, App. A.

# __2.4 Multimode fields__
In the textbook, they use the vector potential. 
We can do same things in different way.
Consider the Hamiltonian of multi-mode wave.
$$
    \hat{H}
    =
    \sum_{\vec{k}, s}
    \hbar \omega_k
    \left(
        \hat{a}^\dag_{\vec{k},s}
        \hat{a}_{\vec{k},s}
        +
        \frac{1}{2}
    \right)
    =
    \sum_j
    \hbar \omega_j
    \left(
        \hat{n}_j
        +
        \frac{1}{2}
    \right)
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
This state means how many photo in each mode.

We can also think the anihilation operator of multi-mode state.
$$
    \hat{a}_i
    \Ket{\left\{n_j\right\}}
    =
    \sqrt{n_i}
    \Ket{n_1, n_2, \cdots, n_i-1, \cdots}
$$

# __2.5 Thermal fields__
#### Thermal light
So far we consider the zero-temperature box.
We can think interaction between photon and thermal wall.
In thermal equilibrium, the density matrix is
$$
    \hat{\rho_\mathrm{th}}
    =
    \frac
    {e^{ -\hat{H} / k_\mathrm{B}T}}
    {\mathrm{Tr}~e^{- \hat{H} / k_\mathrm{B}T}}
$$
Remember that operator on napier number is symbolic representation of Mclaurin 
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
        \hat{\rho_\mathrm{th}}
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
    \left< \left( \Delta n \right)^2 \right>
    =
    \left< \hat{n}^2 \right>
    -
    \left< \hat{n} \right>^2
    =
    \bar{n}
    +
    \bar{n}^2
$$
This is super-Poisonian distribution. Variance is larger than that of Poisonian.
#### Planck's radiation law
Consider the light in the box with each length $L$ in termal equilibrium.
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
    \Delta m
    =
    \Delta m_x
    \Delta m_y
    \Delta m_z
    =
    2
    \frac{V}{2\pi}
    \Delta k_x
    \Delta k_y
    \Delta k_z
$$
Here multiplication of 2 is number of polarization.

We can think this in spherical coordinate.
$$
    \mathrm{d}m
    =
    \frac{V}{4\pi^3}
    \mathrm{d}^3 k
    =
    \frac{V}{4\pi^3}
    k^2
    \mathrm{d} k
    \mathrm{d} \Omega
    =
    \frac{V}{4\pi^3}
    \frac{\omega^2}{c^3}
    \mathrm{d} \omega
    \mathrm{d} \Omega
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
    \bar{U}
    =
    \hbar \omega \bar{n} \rho (\omega)
    =
    \frac
    {\hbar\omega}
    {e^{\hbar\omega/k_\mathrm{B}T} - 1}
    \frac
    {\omega^2}
    {\pi^2 c^3}
$$

# __2.6 Vacuum fluctuations and the zero-point energy__
Vacuum energy and fluctuations actually give rise to observable effects such as:
- spontaneous emission
- Lamb shift 
- Casimir effect
#### Lamb shift
The lamb shift is a discrepancy between experiment and the Dirac relativistic
theory of the hydrogen atom.
- The theory predicts that the $2^2S_{1/2}$ and $2^2P_{1/2}$ levels should be degenerate.
- Optical experiment suggest that these states were not degenerate.

This discrepancy explained by Bethe. Here we use the Welton's intuitive 
interpretation.

First the potential energy of electron of hydrogen atom is
$$
    V(r)
    =
    -
    \frac{e^2}{r}
    +
    V_\mathrm{vac}
$$
We added the small term $V_\mathrm{vac}$ to include vacuum energy.
Small displacement of potential energy is 
$$
    \Delta V 
    =
    \Delta \vec{r}
    \cdot
    \vec{\nabla} V
    +
    \sum_{i=1}^3
    \frac{1}{2}
    \left(
        \Delta x_i
    \right)^2
    \frac
    {\partial^2 V}
    {\partial x_I^2}
$$
We assume that fluctuation is uniform in all direction in sufficiently long time. 
So we set 
$$ 
    \left< \Delta \vec{r} \right> = 0 
$$
Also, we assume that the displacement is same in all direction. 
$$
    \left< 
        \left( 
            \Delta x_i 
        \right)^2 
    \right> 
    = 
    \frac{1}{3} 
    \left< 
        \left( 
            \Delta r
        \right)^2 
    \right>
$$
So the time-average potential energy displacement is 
$$
\begin{aligned}
    \left<
        \Delta V
    \right>
    &=
    \sum_{i=1}^3
    \frac{1}{2}
    \left<
        \left(
            \Delta x_i
        \right)^2
    \right>
    \frac
    {\partial^2 V}
    {\partial x_i^2}
    \\&=
    \sum_{i=1}^3
    \frac{1}{2}
    \frac{1}{3}
    \left<
        \left( 
            \Delta r
        \right)^2 
    \right>
    \frac
    {\partial^2 V}
    {\partial x_i^2}
    \\&=
    \frac{1}{6}
    \left<
        \left( 
            \Delta r
        \right)^2 
    \right>
    \sum_{i=1}^3
    \frac
    {\partial^2 V}
    {\partial x_i^2}
    \\&=
    \frac{1}{6}
    \left<
        \left( 
            \Delta r
        \right)^2 
    \right>
    \nabla^2 V
    \\&=
    \frac{1}{6}
    \left<
        \left( 
            \Delta r
        \right)^2 
    \right>
    4 \pi e^2 \delta(\vec{r})
\end{aligned}
$$
Here, $\delta(\vec{r})$ is Dirac delta function.

We can calculate quantum mechanical energy shift.
$$
\begin{aligned}
    \Delta E
    &=
    \Braket{nl|\Delta V|nl}
    \\&=
    \frac{4\pi e^2}{6}
    \left<
        \left( 
            \Delta r
        \right)^2 
    \right>
    \int
    \psi^*_{nl}(\vec{r})
    \delta(\vec{r})
    \psi_{nl}(\vec{r})
    \mathrm{d}\vec{r}
    \\&=
    \frac{2\pi e^2}{3}
    \left<
        \left( 
            \Delta r
        \right)^2 
    \right>
    \left|
        \psi_{nl}(\vec{r}=0)
    \right|^2
\end{aligned}
$$

We need to calculate 
$
    \left<
        \left( 
            \Delta r
        \right)^2 
    \right>
$.
To obtain this, we assume that the important field frequencies exceed the atomic
resonance frequencies.
The displacement $\Delta r_\nu$ induced with frequency between $\nu$ and 
$\nu+\mathrm{d}\nu$ is determined by
$$
    m
    \frac
    {\mathrm{d}^2 \Delta r_\nu}
    {\mathrm{d}t^2}
    =
    e E_\mathrm{vac} e^{2\pi i \nu t}
$$
Total vacuum energy is 
$$
\begin{aligned}
    \text{Total vacuum energy}
    &=
    \underbrace{
    \frac
    {8\pi \nu^2}
    {c^3}
    }_\text{density of state}
    \cdot 
    \overbrace{
    V
    }^\text{volume}
    \cdot 
    \overbrace{
    \frac{1}{2}
    h \nu
    }^\text{vacuum energy}
    \\&=
    \frac{1}{8\pi}
    \int
    \left(
        E_{\mathrm{vac}, \nu}^2
        +
        \overbrace{
            \cancel{
                B_{\mathrm{vac}, \nu}^2
            }
        }^{E_\nu \gg B_\nu}
    \right)
    \mathrm{d}V
\end{aligned}
$$
Thus energy difference become
$$
    \Delta E
    =
    \chi 
    \left|
        \psi_{nl}(\vec{r}=0)
    \right|^2
    \int_0^\infty 
    \mathrm{d} \nu
    \frac{1}{\nu}
    \longrightarrow
    \infty
$$

#### Casimir effect
From the vacuum electric field, we can show that two conducting plane attract 
each other.
Consider the box of dimension $L\times L \times d$. 
The total vacuum energy is 
$$
    E_0(d)
    =
    \sum_{lmn}
    2
    \frac{1}{2}
    \hbar \omega_{lmn}
$$
Due to two independent polarizations, we multiply two.
Here $\omega$ can be calculated from periodic boundary conditions.
$$
    \omega_{lmn}
    =
    \pi c
    \sqrt{
        \frac{l^2}{L^2}
        +
        \frac{m^2}{L^2}
        +
        \frac{n^2}{d^2}
    }
$$
We will conduct several approximations listed below:
- Calculate $E_0(d)$. We are interested in $L \gg d$, so we can replace the sums of $l$ and $m$ by integrals.
- Calculate $E(\infty)$. We assume that $d$ is arbitrarily large, so we can replace the sum by integral.
- Calculate $U(d)=E_0(d)-E_0(\infty)$, which is energy reuquired to brin gthe plates from infinity to a distance $d$.
- To transform $U(d)$ further, we need to introduce polar coordinates in the $x$-$y$ plane.
- To estimate the sum and integral, we use Euler-Maclaurin formulae. We keep the terms until third order.

From these intensive calculations, we can show 
$$
    U(d)
    =
    -
    \frac
    {\pi^2 \hbar c}
    {720}
    \frac
    {L^2}
    {d^3}
$$
which means there is an attractive force (Casimir force) between two plates.
