# Coherence Functions

## Classical Coherence

**Coherence** describes the ability of light to produce interference. Classically, it is quantified by **correlation functions** of the electric field. The first-order correlation function is

$$
G^{(1)}(\mathbf{r}_1, t_1; \mathbf{r}_2, t_2) = \langle E^*(\mathbf{r}_1, t_1) E(\mathbf{r}_2, t_2) \rangle.
$$

The normalized version is the **degree of first-order coherence**:

$$
g^{(1)}(\tau) = \frac{\langle E^*(t) E(t+\tau) \rangle}{\langle |E(t)|^2 \rangle}.
$$

For a perfectly monochromatic source, $|g^{(1)}(\tau)| = 1$ for all $\tau$. For a thermal source with spectral width $\Delta\nu$, the coherence decays on a timescale $\tau_c \sim 1/\Delta\nu$, called the **coherence time**.

**Temporal coherence** measures how well the field correlates with itself at different times at the same point. **Spatial coherence** measures correlations between different spatial points at the same time. The **Wiener-Khintchine theorem** relates $g^{(1)}(\tau)$ to the power spectral density through a Fourier transform.

## Quantum Coherence Functions

In quantum optics, the electric field becomes an operator, and correlation functions involve **normal ordering** (creation operators to the left, annihilation operators to the right). The quantum first-order correlation function is

$$
G^{(1)}(\mathbf{r}_1, t_1; \mathbf{r}_2, t_2) = \langle \hat{E}^{(-)}(\mathbf{r}_1, t_1) \hat{E}^{(+)}(\mathbf{r}_2, t_2) \rangle,
$$

where $\hat{E}^{(+)}$ and $\hat{E}^{(-)}$ are the positive and negative frequency parts of the field operator.

For different quantum states, $g^{(1)}$ behaves differently:
- **Coherent state** $|\alpha\rangle$: $|g^{(1)}(\tau)| = 1$ (perfect first-order coherence, just like a classical field).
- **Number state** $|n\rangle$: $|g^{(1)}(\tau)| = 1$ (also perfectly coherent in first order).
- **Thermal state**: $|g^{(1)}(\tau)|$ decays with $\tau$, reflecting the broad spectral content.

First-order coherence alone cannot distinguish quantum from classical light. The differences emerge at higher orders.

## Young's Interference with Quantum Fields

Young's double-slit experiment illustrates first-order coherence. Two pinholes at positions $\mathbf{r}_1$ and $\mathbf{r}_2$ sample the field, and the intensity at the observation screen is

$$
I(\mathbf{r}) \propto G^{(1)}(\mathbf{r}_1, \mathbf{r}_1) + G^{(1)}(\mathbf{r}_2, \mathbf{r}_2) + 2\operatorname{Re}\left[G^{(1)}(\mathbf{r}_1, \mathbf{r}_2) e^{i\phi}\right],
$$

where $\phi$ is the phase difference due to path lengths. The **visibility** of the interference fringes is

$$
\mathcal{V} = \frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min}} = |g^{(1)}(\mathbf{r}_1, \mathbf{r}_2)|.
$$

Remarkably, single photons also produce interference fringes when detected one at a time over many trials. Each photon interferes with itself, building up the pattern statistically. This was demonstrated by Taylor (1909) and later with more controlled single-photon sources.

## Photodetection Theory

The quantum theory of **photodetection** connects correlation functions to measurable quantities. The probability of detecting a photon at position $\mathbf{r}$ and time $t$ is proportional to

$$
P_1 \propto \langle \hat{E}^{(-)}(\mathbf{r}, t) \hat{E}^{(+)}(\mathbf{r}, t) \rangle = G^{(1)}(\mathbf{r}, t; \mathbf{r}, t).
$$

The joint probability of detecting photons at two space-time points involves $G^{(2)}$, the second-order correlation function. Normal ordering ensures that these expressions give physically meaningful (non-negative) detection probabilities, consistent with the photoelectric effect.

[[simulation wigner-coherent]]
