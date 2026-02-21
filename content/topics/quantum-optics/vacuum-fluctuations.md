# Vacuum Fluctuations and Observable Effects

## Vacuum fluctuations and the zero-point energy
Vacuum energy and fluctuations actually give rise to observable effects such as:
* Spontaneous emission
* Lamb shift
* Casimir effect
### Lamb shift
The lamb shift is a discrepancy between experiment and the Dirac relativistic
theory of the hydrogen atom.
* The theory predicts that the $2^2S_{1/2}$ and $2^2P_{1/2}$ levels should be degenerate.
* Optical experiment suggest that these states were not degenerate.

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

### Casimir effect
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
* Calculate $E_0(d)$. We are interested in $L \gg d$, so we can replace the sums of $l$ and $m$ by integrals.
* Calculate $E(\infty)$. We assume that $d$ is arbitrarily large, so we can replace the sum by integral.
* Calculate $U(d)=E_0(d)-E_0(\infty)$, which is energy required to bring the plates from infinity to a distance $d$.
* To transform $U(d)$ further, we need to introduce polar coordinates in the $x$-$y$ plane.
* To estimate the sum and integral, we use Euler-Maclaurin formulae. We keep the terms until third order.

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

## Big Ideas

* The vacuum is not nothing — zero-point fluctuations of the electromagnetic field jiggle electrons inside atoms, shifting energy levels in ways that Dirac's otherwise exact theory cannot predict.
* The Lamb shift tells us that the $2S_{1/2}$ and $2P_{1/2}$ levels of hydrogen, predicted to be degenerate, are split by roughly 1 GHz because the electron "trembles" in the vacuum field.
* Two neutral, uncharged conducting plates attract each other simply because the vacuum modes between them are restricted to a discrete set while the modes outside are continuous — energy imbalance produces a force.
* Observable effects of zero-point energy are always differences: absolute vacuum energies are not measurable, but relative changes (Lamb shift, Casimir force) are strikingly real.

## What Comes Next

Vacuum fluctuations show that the quantum field has irreducible noise even when it contains no photons. The natural next question is: what does the quietest possible field with actual photons in it look like? That brings us to coherent states — the states of the field that most closely mimic a classical electromagnetic wave, saturating the uncertainty relation and carrying Poissonian photon noise.

## Check Your Understanding

1. The Lamb shift calculation diverges when integrating over all vacuum mode frequencies, yet the physical shift is finite and well measured. What physical cutoff makes the integral converge in practice, and what does this tell you about the range of vacuum modes that actually matter for the electron in hydrogen?
2. The Casimir force between two parallel plates depends on plate separation as $U(d) \propto -d^{-3}$. What does the sign tell you (attractive or repulsive?), and how does the force scale with separation? Would you expect the effect to be easier or harder to measure as the plates are brought closer together?
3. Spontaneous emission occurs because the excited atom can interact with vacuum fluctuations, even in the absence of any real photons. If you could somehow block all electromagnetic modes at the atomic transition frequency (for example, by placing the atom in a photonic crystal bandgap), what would happen to the spontaneous emission rate?

## Challenge

The Casimir calculation uses the Euler-Maclaurin formula to regularize the difference between the discrete sum and the continuous integral. Work through this regularization explicitly: write the vacuum energy $E_0(d)$ as a sum over discrete $n$ and the free-space energy as an integral, take their difference, and apply the Euler-Maclaurin formula keeping terms through third order to extract the finite result $U(d) = -\pi^2\hbar c L^2/(720 d^3)$. At what step does the regularization feel like "throwing away infinity," and why is the physical result nonetheless unambiguous?
