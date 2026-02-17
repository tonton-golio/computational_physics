# Fast Fourier Transform

## The discrete Fourier transform

The **discrete Fourier transform** (DFT) converts a sequence of $N$ samples in the time domain into a sequence of $N$ complex coefficients in the frequency domain:

$$
X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i \, kn/N}, \qquad k = 0, 1, \ldots, N-1.
$$

The inverse transform recovers the original signal:

$$
x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \, e^{2\pi i \, kn/N}.
$$

Each coefficient $X_k$ represents the amplitude and phase of a sinusoidal component at frequency $k/N$ cycles per sample. The **power spectrum** $|X_k|^2$ reveals the dominant frequencies in the signal.

## The FFT algorithm

Computing the DFT directly requires $O(N^2)$ operations. The **fast Fourier transform** (Cooley-Tukey, 1965) reduces this to $O(N \log N)$ by exploiting the symmetry and periodicity of the complex exponentials.

The key idea for radix-2 FFT (when $N$ is a power of 2):

$$
X_k = \underbrace{\sum_{m=0}^{N/2-1} x_{2m} \, e^{-2\pi i \, k(2m)/N}}_{\text{even-indexed DFT}} + e^{-2\pi i \, k/N} \underbrace{\sum_{m=0}^{N/2-1} x_{2m+1} \, e^{-2\pi i \, k(2m)/N}}_{\text{odd-indexed DFT}}.
$$

This splits one $N$-point DFT into two $N/2$-point DFTs plus $O(N)$ multiplications. Applying this recursively yields the $O(N \log N)$ complexity.

**Practical impact**: for $N = 10^6$, the FFT is roughly $50{,}000$ times faster than the direct DFT.

## Applications in signal processing

The DFT and FFT are ubiquitous in scientific computing:

- **Spectral analysis**: identify periodic components in time series (e.g., tidal data, heart rhythms, seismic signals).
- **Filtering**: multiply the spectrum by a transfer function to remove noise or isolate frequency bands.
- **Interpolation and zero-padding**: increasing $N$ by appending zeros refines the frequency resolution.
- **Image processing**: the 2D DFT decomposes images into spatial frequencies for compression (JPEG) and enhancement.

## The convolution theorem

One of the most powerful properties of the DFT is the **convolution theorem**:

$$
\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\},
$$

where $*$ denotes convolution and $\cdot$ denotes pointwise multiplication. This means that convolution in the time domain becomes multiplication in the frequency domain.

**Practical consequence**: convolving two sequences of length $N$ via the FFT costs $O(N \log N)$, compared to $O(N^2)$ for direct convolution. This speedup is exploited in:

- Polynomial multiplication.
- Cross-correlation and matched filtering.
- Solving PDEs with periodic boundary conditions (spectral methods).

## Aliasing and the Nyquist frequency

The DFT assumes the signal is periodic with period $N$. If the signal contains frequencies above the **Nyquist frequency** $f_{\text{Nyq}} = f_s / 2$ (where $f_s$ is the sampling rate), those components are **aliased** into lower frequencies.

The **sampling theorem** (Shannon-Nyquist) states that a bandlimited signal can be perfectly reconstructed from its samples if and only if the sampling rate exceeds twice the highest frequency present.

## Windowing

Real signals are finite in duration. Truncation introduces spectral **leakage**, spreading energy from a true frequency into neighboring bins. **Window functions** (Hann, Hamming, Blackman) taper the signal at the edges to reduce leakage at the cost of slightly reduced frequency resolution.

[[simulation reaction-diffusion]]
