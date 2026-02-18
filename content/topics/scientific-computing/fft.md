# Fast Fourier Transform
*The machine that tells you which notes are in a song*

> *Eigenvalues revealed the personality of matrices. The FFT reveals the personality of signals — which frequencies are hiding inside any chunk of data. And it connects back to PDEs at the summit of our course.*

## Suppose you have a song

You hear a chord on a piano — three notes played together. Your ear somehow picks out the individual notes from the combined sound wave. The **Fourier transform** is the mathematical version of your ear: it takes a signal (the combined wave) and tells you exactly which frequencies (notes) are inside it, and how loud each one is.

The **Fast Fourier Transform** is the algorithm that makes this practical. Without it, analyzing a million-sample signal would take a trillion operations. With it, twenty million. That's the difference between "impossible" and "done before you blink."

## The discrete Fourier transform

The **discrete Fourier transform** (DFT) converts a sequence of $N$ samples in the time domain into a sequence of $N$ complex coefficients in the frequency domain:

$$
X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i \, kn/N}, \qquad k = 0, 1, \ldots, N-1.
$$

*This says: for each frequency $k$, multiply the signal by a complex sinusoid at that frequency and add everything up. The result tells you how much of that frequency is present.*

The inverse transform recovers the original signal:

$$
x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \, e^{2\pi i \, kn/N}.
$$

Each coefficient $X_k$ represents the amplitude and phase of a sinusoidal component at frequency $k/N$ cycles per sample. The **power spectrum** $|X_k|^2$ reveals the dominant frequencies in the signal.

> **You might be wondering...** "Why complex exponentials instead of sines and cosines?" Because $e^{i\theta} = \cos\theta + i\sin\theta$ packages both into one elegant expression. The real part gives you cosines, the imaginary part gives you sines, and the math becomes cleaner. It's like writing coordinates as $(x, y)$ instead of "x units east and y units north."

## The FFT algorithm

Computing the DFT directly requires $O(N^2)$ operations. The **fast Fourier transform** (Cooley-Tukey, 1965) reduces this to $O(N \log N)$ by exploiting the symmetry and periodicity of the complex exponentials.

Here's the trick — split the signal into even and odd samples:

$$
X_k = \underbrace{\sum_{m=0}^{N/2-1} x_{2m} \, e^{-2\pi i \, k(2m)/N}}_{\text{even-indexed DFT}} + e^{-2\pi i \, k/N} \underbrace{\sum_{m=0}^{N/2-1} x_{2m+1} \, e^{-2\pi i \, k(2m)/N}}_{\text{odd-indexed DFT}}.
$$

*This says: one big DFT of size $N$ equals two half-size DFTs (even samples and odd samples) plus $N$ multiplications to combine them. Apply this recursively and you go from $N^2$ to $N \log N$.*

This splits one $N$-point DFT into two $N/2$-point DFTs plus $O(N)$ multiplications. Applying this recursively yields the $O(N \log N)$ complexity.

**Practical impact**: for $N = 10^6$, the FFT is roughly $50{,}000$ times faster than the direct DFT.

> **You might be wondering...** "Is this really that big a deal?" Yes! The FFT is arguably one of the most important algorithms ever invented. It makes real-time audio processing, medical imaging (MRI), radio astronomy, and modern telecommunications possible. Without it, your phone couldn't decode a WiFi signal fast enough to be useful.

## The convolution theorem

One of the most powerful properties of the DFT is the **convolution theorem**:

$$
\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\},
$$

where $*$ denotes convolution and $\cdot$ denotes pointwise multiplication. This means that convolution in the time domain becomes multiplication in the frequency domain.

*Think of it like this: instead of stirring the whole pot for hours (direct convolution), transform both recipes into ingredient lists, multiply the ingredients together, and transform back. Same result, fraction of the time.*

**Practical consequence**: convolving two sequences of length $N$ via the FFT costs $O(N \log N)$, compared to $O(N^2)$ for direct convolution. This speedup is exploited in:

- Polynomial multiplication.
- Cross-correlation and matched filtering.
- Solving PDEs with periodic boundary conditions (spectral methods).

## Applications in signal processing

The DFT and FFT are ubiquitous in scientific computing:

- **Spectral analysis**: identify periodic components in time series (e.g., tidal data, heart rhythms, seismic signals).
- **Filtering**: multiply the spectrum by a transfer function to remove noise or isolate frequency bands.
- **Interpolation and zero-padding**: increasing $N$ by appending zeros refines the frequency resolution.
- **Image processing**: the 2D DFT decomposes images into spatial frequencies for compression (JPEG) and enhancement.

## Aliasing and the Nyquist frequency

The DFT assumes the signal is periodic with period $N$. If the signal contains frequencies above the **Nyquist frequency** $f_{\text{Nyq}} = f_s / 2$ (where $f_s$ is the sampling rate), those components are **aliased** into lower frequencies.

*This says: if you don't sample fast enough, high frequencies disguise themselves as low frequencies. It's like a spinning wheel that looks like it's going backwards in a movie — the camera wasn't sampling fast enough.*

The **sampling theorem** (Shannon-Nyquist) states that a bandlimited signal can be perfectly reconstructed from its samples if and only if the sampling rate exceeds twice the highest frequency present.

## Windowing

Real signals are finite in duration. Truncation introduces spectral **leakage**, spreading energy from a true frequency into neighboring bins. **Window functions** (Hann, Hamming, Blackman) taper the signal at the edges to reduce leakage at the cost of slightly reduced frequency resolution.

> **Challenge:** Generate a signal with two frequencies: `x = sin(2*pi*5*t) + 0.5*sin(2*pi*12*t)` sampled at 100 Hz for 1 second. Take the FFT with `np.fft.fft`, plot the power spectrum `|X|^2`, and find the two peaks at 5 Hz and 12 Hz. Now add noise and see if you can still find them.

---

**What we just learned in one sentence:** The FFT decomposes any signal into its frequency components in $O(N \log N)$ time, and the convolution theorem turns slow time-domain operations into fast frequency-domain multiplications.

**What's next and why it matters:** We've been solving algebraic equations (linear and nonlinear) and analyzing static properties (eigenvalues, frequencies). Now things start *moving* — we'll learn to march through time by solving ordinary differential equations, the language of everything that changes.
