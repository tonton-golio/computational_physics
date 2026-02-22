# Coherence Functions

## How much does the wave remember its own past?

Coherence is about memory. Shine a laser through two pinholes and you get sharp interference fringes. Shine a light bulb through the same pinholes and the fringes wash out. The difference? The laser's electric field at time $t$ is strongly correlated with the field at time $t + \tau$ -- it remembers where it was in the oscillation cycle. The light bulb forgets almost immediately.

That's what the first-order correlation function measures.

## Classical coherence

The **degree of first-order coherence** is:

$$
g^{(1)}(\tau) = \frac{\langle E^*(t)\,E(t+\tau)\rangle}{\langle |E(t)|^2\rangle}
$$

Think of it as the field's autocorrelation. $|g^{(1)}(\tau)| = 1$ means perfect memory -- the field at time $t+\tau$ is perfectly predictable from the field at $t$. A monochromatic source achieves this for all $\tau$. A thermal source with spectral width $\Delta\nu$ loses memory on a timescale $\tau_c \sim 1/\Delta\nu$ -- the **coherence time**.

**Temporal coherence** = how well the field correlates with itself at different times (same point in space). **Spatial coherence** = correlations between different spatial points at the same time. The **Wiener-Khintchine theorem** connects $g^{(1)}(\tau)$ to the power spectrum through a Fourier transform: narrow linewidth means long coherence, broad spectrum means short coherence.

## Quantum coherence functions

In quantum optics, the field is an operator, and we need **normal ordering** (creation operators left, annihilation operators right):

$$
G^{(1)}(\mathbf{r}_1, t_1; \mathbf{r}_2, t_2) = \langle\hat{E}^{(-)}(\mathbf{r}_1, t_1)\,\hat{E}^{(+)}(\mathbf{r}_2, t_2)\rangle
$$

Different quantum states have different coherence:
* **Coherent state** $\Ket{\alpha}$: $|g^{(1)}(\tau)| = 1$ -- perfect first-order coherence, just like a classical wave.
* **Number state** $\Ket{n}$: also $|g^{(1)}(\tau)| = 1$ -- perfect coherence in first order!
* **Thermal state**: $|g^{(1)}(\tau)|$ decays with $\tau$.

Wait -- a number state and a coherent state look the same in first-order coherence? Yes. First-order coherence alone **cannot** distinguish quantum from classical light. The real quantum signatures show up at higher orders.

## Young's interference with quantum fields

Young's double slit illustrates first-order coherence perfectly. Two pinholes sample the field, and the intensity on the observation screen is:

$$
I(\mathbf{r}) \propto G^{(1)}(\mathbf{r}_1, \mathbf{r}_1) + G^{(1)}(\mathbf{r}_2, \mathbf{r}_2) + 2\operatorname{Re}[G^{(1)}(\mathbf{r}_1, \mathbf{r}_2)\,e^{i\phi}]
$$

The fringe visibility is simply $\mathcal{V} = |g^{(1)}(\mathbf{r}_1, \mathbf{r}_2)|$.

And yes, single photons produce interference fringes too -- detected one at a time, they build up the pattern statistically. Each photon interferes with itself.

## Photodetection theory

The probability of detecting a photon at $(\mathbf{r}, t)$ is proportional to $G^{(1)}(\mathbf{r}, t; \mathbf{r}, t)$ -- which is why normal ordering matters. If you naively computed $\langle\hat{E}^2\rangle$ without normal ordering, the vacuum would give a nonzero detection probability. Normal ordering ensures that "no photons" means "no clicks."

[[simulation wigner-coherent]]

## Big Ideas

* $g^{(1)}(\tau)$ measures how well the field remembers its own past -- it governs interference fringe visibility.
* The Wiener-Khintchine theorem ties time-domain coherence to spectral purity: narrow linewidth means long memory.
* First-order coherence can't distinguish quantum from classical light. The real quantum signatures show up at higher orders.

## Check Your Understanding

1. A thermal source and a laser both have $|g^{(1)}(0)| = 1$. How does $g^{(1)}(\tau)$ differ as $\tau$ increases?
2. Single photons make fringes in Young's experiment. How does a single particle "go through both slits"?

## Challenge

Write a Python script that models $g^{(1)}(\tau)$ for two source types: (a) a Lorentzian lineshape ($g^{(1)}(\tau) = e^{-\gamma|\tau|/2}$) and (b) a Gaussian lineshape ($g^{(1)}(\tau) = e^{-(\Delta\omega)^2\tau^2/2}$). Compute the power spectral density for each via FFT (verifying the Wiener-Khintchine theorem). Calculate the coherence length $l_c = c\tau_c$ for a source with $\Delta\nu = 1$ MHz at $\lambda = 633$ nm.
