# Fast Fourier Transform

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

Why complex exponentials instead of sines and cosines? Because $e^{i\theta} = \cos\theta + i\sin\theta$ packages both into one elegant expression. The real part gives you cosines, the imaginary part gives you sines, and the math becomes cleaner. It's like writing coordinates as $(x, y)$ instead of "x units east and y units north."

## The FFT algorithm

Computing the DFT directly requires $O(N^2)$ operations. The **fast Fourier transform** (the Cooley-Tukey algorithm) reduces this to $O(N \log N)$ by exploiting the symmetry and periodicity of the complex exponentials.

Here's the trick — split the signal into even and odd samples:

$$
X_k = \underbrace{\sum_{m=0}^{N/2-1} x_{2m} \, e^{-2\pi i \, k(2m)/N}}_{\text{even-indexed DFT}} + e^{-2\pi i \, k/N} \underbrace{\sum_{m=0}^{N/2-1} x_{2m+1} \, e^{-2\pi i \, k(2m)/N}}_{\text{odd-indexed DFT}}.
$$

*This says: one big DFT of size $N$ equals two half-size DFTs (even samples and odd samples) plus $N$ multiplications to combine them. Apply this recursively and you go from $N^2$ to $N \log N$.*

This splits one $N$-point DFT into two $N/2$-point DFTs plus $O(N)$ multiplications. Applying this recursively yields the $O(N \log N)$ complexity.

**Practical impact**: for $N = 10^6$, the FFT is roughly $50{,}000$ times faster than the direct DFT.

Is this really that big a deal? Yes! The FFT is arguably one of the most important algorithms ever invented. It makes real-time audio processing, medical imaging (MRI), radio astronomy, and modern telecommunications possible. Without it, your phone couldn't decode a WiFi signal fast enough to be useful.

## The convolution theorem

One of the most powerful properties of the DFT is the **convolution theorem**:

$$
\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\},
$$

where $*$ denotes convolution and $\cdot$ denotes pointwise multiplication. This means that convolution in the time domain becomes multiplication in the frequency domain.

*Think of it like this: instead of stirring the whole pot for hours (direct convolution), transform both recipes into ingredient lists, multiply the ingredients together, and transform back. Same result, fraction of the time.*

**Practical consequence**: convolving two sequences of length $N$ via the FFT costs $O(N \log N)$, compared to $O(N^2)$ for direct convolution. This speedup is exploited in:

* Polynomial multiplication.
* Cross-correlation and matched filtering.
* Solving PDEs with periodic boundary conditions (spectral methods).

## Applications in signal processing

The DFT and FFT are ubiquitous in scientific computing:

* **Spectral analysis**: identify periodic components in time series (e.g., tidal data, heart rhythms, seismic signals).
* **Filtering**: multiply the spectrum by a transfer function to remove noise or isolate frequency bands.
* **Interpolation and zero-padding**: increasing $N$ by appending zeros refines the frequency resolution.
* **Image processing**: the 2D DFT decomposes images into spatial frequencies for compression (JPEG) and enhancement.

## Aliasing and the Nyquist frequency

The DFT assumes the signal is periodic with period $N$. If the signal contains frequencies above the **Nyquist frequency** $f_{\text{Nyq}} = f_s / 2$ (where $f_s$ is the sampling rate), those components are **aliased** into lower frequencies.

*This says: if you don't sample fast enough, high frequencies disguise themselves as low frequencies. It's like a spinning wheel that looks like it's going backwards in a movie — the camera wasn't sampling fast enough.*

The **sampling theorem** (Shannon-Nyquist) states that a bandlimited signal can be perfectly reconstructed from its samples if and only if the sampling rate exceeds twice the highest frequency present.

## Windowing

Real signals are finite in duration. Truncation introduces spectral **leakage**, spreading energy from a true frequency into neighboring bins. **Window functions** (Hann, Hamming, Blackman) taper the signal at the edges to reduce leakage at the cost of slightly reduced frequency resolution.

> **Challenge.** Generate a signal with two frequencies: `x = sin(2*pi*5*t) + 0.5*sin(2*pi*12*t)` sampled at 100 Hz for 1 second. Take the FFT with `np.fft.fft`, plot the power spectrum `|X|^2`, and find the two peaks at 5 Hz and 12 Hz. Now add noise and see if you can still find them.

---

## Big Ideas

* The DFT is a change of basis: from the time domain to the frequency domain. Every signal is secretly a sum of sinusoids; the DFT finds the recipe.
* The FFT's $O(N \log N)$ trick is recursive splitting: one big transform equals two half-size transforms plus cheap combining. Applied log $N$ times, this transforms a trillion operations into twenty million.
* Convolution in time is multiplication in frequency — the convolution theorem is the reason spectral methods can compute derivatives and solve PDEs at the cost of a few FFTs.
* Aliasing is not a glitch but a mathematical necessity: if you sample too slowly, high frequencies have no choice but to masquerade as low ones. The Nyquist rate is the minimum sampling rate that prevents this.

## What Comes Next

The FFT is the last of the *static* tools: it analyzes a snapshot of a signal or a fixed spatial domain. The next topic is time itself — how do you march a solution forward step by step through an ordinary differential equation?

Everything learned so far feeds into this: error analysis tells you how to choose a step size, linear algebra appears inside implicit solvers, eigenvalues of the Jacobian determine whether a problem is stiff, and the FFT underlies the spectral methods that discretize the spatial part of PDEs before handing off to an ODE solver.

## Check Your Understanding

1. A signal is sampled at 1000 Hz. What is the highest frequency the DFT can represent without aliasing, and what happens to a 700 Hz sine wave in this signal?
2. The naive DFT costs $O(N^2)$ and the FFT costs $O(N \log N)$. For $N = 2^{20} \approx 10^6$, by what factor is the FFT faster?
3. You want to convolve two signals of length $N = 10^4$ using the FFT. Describe the steps and state the total operation count.

## Challenge

Generate a noisy signal consisting of three pure tones at 10 Hz, 35 Hz, and 80 Hz with amplitudes 1, 0.5, and 0.3, sampled at 500 Hz for 4 seconds. Add Gaussian noise with standard deviation 0.2. Use the FFT to recover the three frequencies and their amplitudes from the noisy data. Then apply a bandpass filter in the frequency domain to isolate only the 35 Hz component and reconstruct the filtered time-domain signal via the inverse FFT. Plot the original noisy signal, the power spectrum, and the filtered signal on a single figure.
