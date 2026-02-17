# Displaced Squeezed States

## Displaced Squeezed Vacuum

A **displaced squeezed state** is obtained by first squeezing the vacuum and then displacing it in phase space:

$$
|\alpha, \xi\rangle = \hat{D}(\alpha) \hat{S}(\xi) |0\rangle,
$$

where $\hat{D}(\alpha) = \exp(\alpha \hat{a}^\dagger - \alpha^* \hat{a})$ is the displacement operator and $\hat{S}(\xi) = \exp[\frac{1}{2}(\xi^* \hat{a}^2 - \xi \hat{a}^{\dagger 2})]$ is the squeezing operator with $\xi = re^{i\theta}$.

The ordering matters: $\hat{D}(\alpha)\hat{S}(\xi) \neq \hat{S}(\xi)\hat{D}(\alpha)$ in general. The state $|\alpha, \xi\rangle$ is a **minimum uncertainty state** centered at $\alpha$ in phase space with an elliptical noise distribution whose orientation and eccentricity are determined by $\xi$.

The mean values and variances are

$$
\langle \hat{X}_\phi \rangle = |\alpha|\cos(\phi - \phi_\alpha), \qquad (\Delta X_\phi)^2 = \frac{1}{4}(e^{-2r}\cos^2\psi + e^{2r}\sin^2\psi),
$$

where $\psi = \phi - \theta/2$ and $\phi_\alpha = \arg(\alpha)$.

## Amplitude and Phase Squeezed Light

The relative orientation of the squeezing ellipse and the coherent amplitude determines the type of squeezing:

**Amplitude squeezed light** has reduced intensity fluctuations. The squeezing axis is aligned with the coherent amplitude, so the radial (amplitude) quadrature has sub-vacuum noise:

$$
(\Delta n)^2 < \langle n \rangle \quad \text{(sub-Poissonian)}.
$$

This is useful for precision intensity measurements and direct detection experiments.

**Phase squeezed light** has reduced phase fluctuations. The squeezing axis is perpendicular to the coherent amplitude, compressing the tangential (phase) quadrature:

$$
\Delta\phi < \frac{1}{2\sqrt{\langle n \rangle}} \quad \text{(below shot noise)}.
$$

This is advantageous for interferometric measurements where phase sensitivity is the limiting factor.

The Wigner function of a displaced squeezed state is a Gaussian centered at $\alpha$ with principal axes tilted by $\theta/2$:

$$
W(x, p) = \frac{1}{\pi} \exp\left[-e^{2r}(x' - x_0')^2 - e^{-2r}(p' - p_0')^2\right],
$$

where $(x', p')$ are coordinates rotated by the squeezing angle.

[[simulation wigner-squeezed]]

## Two-Mode Squeezing

**Two-mode squeezing** correlates two distinct field modes and is the key resource for continuous-variable entanglement. The two-mode squeezing operator is

$$
\hat{S}_2(\xi) = \exp(\xi^* \hat{a}\hat{b} - \xi \hat{a}^\dagger \hat{b}^\dagger),
$$

which creates photon pairs: one in mode $a$ and one in mode $b$. The two-mode squeezed vacuum (TMSV) state is

$$
|\text{TMSV}\rangle = \frac{1}{\cosh r} \sum_{n=0}^{\infty} (-e^{i\theta}\tanh r)^n |n\rangle_a |n\rangle_b.
$$

The modes are perfectly correlated in photon number ($n_a = n_b$ in every term) but individually each mode is a thermal state with $\langle n \rangle = \sinh^2 r$.

## Continuous-Variable Entanglement

The TMSV state exhibits **EPR-type correlations** in the continuous variables. The sum and difference quadratures are

$$
\Delta(X_a - X_b)^2 = \frac{1}{2}e^{-2r}, \qquad \Delta(P_a + P_b)^2 = \frac{1}{2}e^{-2r}.
$$

Both vanish in the limit $r \to \infty$, corresponding to the original EPR state with perfect correlations. The entanglement is verified by the **Duan criterion**: the state is entangled if

$$
\Delta(X_a - X_b)^2 + \Delta(P_a + P_b)^2 < 1.
$$

For a TMSV state, this sum equals $e^{-2r} < 1$ for any $r > 0$, confirming entanglement.

## Applications

Displaced squeezed states and two-mode squeezing are central to:

- **Quantum teleportation**: The TMSV provides the shared entangled resource for continuous-variable teleportation of coherent states.
- **Quantum dense coding**: Entangled beams allow transmission of classical information at rates exceeding the classical channel capacity.
- **Quantum illumination**: Entangled signal-idler pairs improve target detection in noisy environments.
- **Cluster state generation**: Large entangled states for measurement-based quantum computation can be deterministically produced using squeezed light and linear optics.
