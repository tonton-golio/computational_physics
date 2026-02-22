# Squeezed States

## Breaking the symmetry of the vacuum

Coherent states share the vacuum's noise equally between both quadratures -- a symmetric fuzzy circle in phase space. Squeezed states break that symmetry. They crush the noise in one direction below the vacuum level, at the expense of inflating it in the other. The total uncertainty still obeys Heisenberg, but the *redistribution* is the key to quantum-enhanced measurement.

## The squeezing operator

The single-mode **squeezing operator** is:

$$
\hat{S}(\xi) = \exp\left[\frac{1}{2}(\xi^*\hat{a}^2 - \xi\hat{a}^{\dag 2})\right]
$$

where $\xi = re^{i\theta}$ is the complex squeezing parameter. Apply it to the vacuum and you get the squeezed vacuum $\Ket{\xi} = \hat{S}(\xi)\Ket{0}$.

The quadrature variances become:

$$
(\Delta X_\theta)^2 = \frac{1}{4}e^{-2r}, \qquad (\Delta X_{\theta+\pi/2})^2 = \frac{1}{4}e^{+2r}
$$

One direction gets quieter, the other gets louder. In decibels: squeezing of $r$ corresponds to roughly $8.7r$ dB of noise reduction. Current experiments exceed 15 dB -- that's a factor of about 30 below the vacuum noise floor.

## Photon statistics: only even numbers

Here's a surprise. The squeezed vacuum contains **only even photon numbers**:

$$
\Ket{\xi} = \frac{1}{\sqrt{\cosh r}}\sum_{n=0}^\infty \frac{(-e^{i\theta}\tanh r)^n\sqrt{(2n)!}}{2^n\,n!}\Ket{2n}
$$

Why even? Because the squeezing operator creates photons in *pairs* ($\hat{a}^{\dag 2}$). Start from vacuum (zero photons), add pairs, and you only ever hit even numbers. The mean photon number is $\langle n\rangle = \sinh^2 r$.

## How to make squeezed light

**Optical parametric down-conversion** (PDC) is the workhorse. A strong pump laser hits a nonlinear crystal ($\chi^{(2)}$), and pump photons split into pairs satisfying energy and momentum conservation:

$$
\omega_p = \omega_s + \omega_i, \qquad \mathbf{k}_p = \mathbf{k}_s + \mathbf{k}_i
$$

When signal and idler are the same mode (degenerate PDC), the output is a squeezed vacuum. The effective Hamiltonian is $\hat{H}_\text{PDC} = i\hbar\kappa(\hat{a}^{\dag 2} - \hat{a}^2)$ -- exactly the squeezing Hamiltonian.

Put the crystal inside a cavity (an **optical parametric oscillator**) and you get continuous-wave squeezed light with narrow bandwidth. Below threshold: squeezing. Above threshold: coherent oscillation.

## Displaced squeezed states

First squeeze, then displace:

$$
\Ket{\alpha, \xi} = \hat{D}(\alpha)\hat{S}(\xi)\Ket{0}
$$

The result is a minimum-uncertainty state centered at $\alpha$ with an **elliptical** noise distribution. The orientation of the ellipse relative to the coherent amplitude determines the type:

* **Amplitude squeezed**: squeeze axis aligned with the amplitude direction. Reduced intensity noise, sub-Poissonian statistics ($Q < 0$). Great for precision intensity measurements.
* **Phase squeezed**: squeeze axis perpendicular. Reduced phase noise, below shot noise for phase measurements. This is what LIGO uses.

[[simulation wigner-squeezed]]

## Two-mode squeezing and entanglement

**Two-mode squeezing** correlates two distinct field modes via:

$$
\hat{S}_2(\xi) = \exp(\xi^*\hat{a}\hat{b} - \xi\hat{a}^\dag\hat{b}^\dag)
$$

The two-mode squeezed vacuum (TMSV) is:

$$
\Ket{\text{TMSV}} = \frac{1}{\cosh r}\sum_{n=0}^\infty(-e^{i\theta}\tanh r)^n\Ket{n}_a\Ket{n}_b
$$

Each mode individually looks thermal. But look at both together and they're perfectly correlated in photon number -- and their joint quadratures satisfy:

$$
\Delta(X_a - X_b)^2 = \frac{1}{2}e^{-2r}, \qquad \Delta(P_a + P_b)^2 = \frac{1}{2}e^{-2r}
$$

Both vanish as $r \to \infty$ -- that's the original EPR state. The Duan criterion confirms entanglement whenever $e^{-2r} < 1$, which holds for any $r > 0$. Any amount of two-mode squeezing produces entanglement.

[[simulation tmsv-entanglement]]

## Big Ideas

* Squeezed states redistribute quantum noise: one quadrature drops below vacuum while the other inflates, always satisfying $\Delta X \cdot \Delta P \ge 1/4$.
* The squeezing operator creates photons in pairs -- that's why squeezed vacuum has only even photon numbers.
* Parametric down-conversion is the workhorse: a nonlinear crystal breaks pump photons into pairs, and that pairwise creation *is* the squeezing Hamiltonian.
* Two-mode squeezing creates EPR-type entanglement: individually thermal, jointly correlated beyond any classical limit.

## Check Your Understanding

1. Use the Bogoliubov transformation $\hat{S}^\dag\hat{a}\hat{S} = \hat{a}\cosh r - \hat{a}^\dag e^{i\theta}\sinh r$ to show that $\langle\hat{n}\rangle = \sinh^2 r$ for squeezed vacuum. Why does the vacuum acquire photons after squeezing?
2. Reducing intensity noise necessarily inflates phase variance and vice versa. Why?

## Challenge

Write a Python script that computes the photon number distribution of a squeezed vacuum state for $r = 0.5, 1.0, 1.5$. Verify that only even photon numbers appear. Plot the quadrature variances as a function of $r$ and confirm $\frac{1}{4}e^{-2r}$ and $\frac{1}{4}e^{+2r}$. Compute the Mandel Q parameter and show that squeezed vacuum is super-Poissonian ($Q > 0$) despite having sub-vacuum noise in one quadrature.
