# Fast Fourier Transform

> *Eigenvalues revealed the personality of matrices. The FFT reveals the personality of signals — which frequencies are hiding inside any chunk of data. And it connects back to PDEs at the summit of our course.*

## Big Ideas

* The DFT is a change of basis: from time to frequency. Every signal is secretly a sum of sinusoids; the DFT finds the recipe — but it assumes periodicity over the observation window, so non-periodic signals leak energy across bins.
* The FFT's $O(N \log N)$ trick is recursive splitting: one big transform = two half-size transforms + cheap combining. A trillion operations become twenty million.
* Convolution in time is multiplication in frequency — the convolution theorem is why spectral methods can compute derivatives and solve PDEs at the cost of a few FFTs.
* Aliasing is a mathematical necessity: sample too slowly, and high frequencies masquerade as low ones. The Nyquist rate is the minimum that prevents this.

## Suppose You Have a Song

You hear a chord on a piano — three notes played together. Your ear somehow picks out the individual notes from the combined sound wave. The **Fourier transform** is the mathematical version of your ear: it takes a signal and tells you exactly which frequencies are inside it and how loud each one is.

The **Fast Fourier Transform** makes this practical. Without it, analyzing a million-sample signal would take a trillion operations. With it, twenty million. That's the difference between "impossible" and "done before you blink."

## The Discrete Fourier Transform

The DFT converts $N$ time-domain samples into $N$ frequency-domain coefficients:

$$
X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i \, kn/N}, \qquad k = 0, 1, \ldots, N-1.
$$

*For each frequency $k$, multiply the signal by a complex sinusoid at that frequency and add everything up. The result tells you how much of that frequency is present.*

The inverse recovers the original:

$$
x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \, e^{2\pi i \, kn/N}.
$$

The **power spectrum** $|X_k|^2$ reveals the dominant frequencies.

Why complex exponentials instead of sines and cosines? Because $e^{i\theta} = \cos\theta + i\sin\theta$ packages both into one expression. The math becomes cleaner — like writing coordinates as $(x, y)$ instead of "x units east and y units north."

## The FFT Algorithm

Computing the DFT directly: $O(N^2)$. The Cooley-Tukey algorithm: $O(N \log N)$. Here's the trick — split into even and odd samples:

$$
X_k = \underbrace{\sum_{m=0}^{N/2-1} x_{2m} \, e^{-2\pi i \, k(2m)/N}}_{\text{even-indexed DFT}} + e^{-2\pi i \, k/N} \underbrace{\sum_{m=0}^{N/2-1} x_{2m+1} \, e^{-2\pi i \, k(2m)/N}}_{\text{odd-indexed DFT}}.
$$

*One big DFT = two half-size DFTs + $N$ multiplications. Apply recursively: $N^2 \to N \log N$.*

For $N = 10^6$, the FFT is roughly $50{,}000\times$ faster than the direct DFT. This makes real-time audio, MRI, radio astronomy, and modern telecommunications possible. Without it, your phone couldn't decode a WiFi signal fast enough.

## The Convolution Theorem

$$
\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}
$$

*Instead of stirring the whole pot for hours (direct convolution), transform both recipes into ingredient lists, multiply them together, and transform back. Same result, fraction of the time.*

Convolving two length-$N$ sequences via FFT costs $O(N \log N)$ vs. $O(N^2)$ directly. This powers polynomial multiplication, cross-correlation, and spectral PDE methods.

## Applications

* **Spectral analysis**: periodic components in time series (tides, heartbeats, seismic signals)
* **Filtering**: multiply the spectrum by a transfer function to remove noise
* **Interpolation**: zero-padding refines frequency resolution
* **Image processing**: 2D DFT for compression (JPEG) and enhancement

## Aliasing and the Nyquist Frequency

If you don't sample fast enough, high frequencies disguise themselves as low frequencies. It's like a spinning wheel that looks like it's going backwards in a movie — the camera wasn't sampling fast enough.

The **Nyquist frequency** $f_{\text{Nyq}} = f_s / 2$. The **sampling theorem** says you can perfectly reconstruct a bandlimited signal only if you sample above twice its highest frequency.

(Drag the slider yourself — watch the wheel suddenly spin the wrong way. That's aliasing in action.)

[[simulation aliasing-slider]]

## Windowing

Real signals are finite. Truncation introduces spectral **leakage** — energy spreads from a true frequency into neighboring bins. **Window functions** (Hann, Hamming, Blackman) taper the signal at the edges to reduce leakage, at a slight cost to frequency resolution.

> **Challenge.** Generate a signal with two frequencies: `x = sin(2*pi*5*t) + 0.5*sin(2*pi*12*t)` sampled at 100 Hz for 1 second. Take the FFT, plot the power spectrum, and find the two peaks. Now add noise and see if you can still find them.

---

## What Comes Next

The FFT is the last of the *static* tools — it analyzes a snapshot of a signal or a fixed spatial domain. Next: time itself. How do you march a solution forward step by step through a differential equation? Everything feeds into this: error analysis for step sizes, linear algebra inside implicit solvers, eigenvalues for stiffness, and the FFT for spectral spatial discretization.

## Check Your Understanding

1. A signal sampled at 1000 Hz. What's the highest frequency the DFT can represent without aliasing, and what happens to a 700 Hz component?
2. For $N = 2^{20} \approx 10^6$, by what factor is the FFT faster than the naive DFT?
3. You want to convolve two length-$10^4$ signals via FFT. Describe the steps and state the operation count.

## Challenge

Generate a noisy signal: three tones at 10 Hz, 35 Hz, and 80 Hz (amplitudes 1, 0.5, 0.3), sampled at 500 Hz for 4 seconds, plus Gaussian noise (std = 0.2). Recover the three frequencies from the power spectrum, then isolate the 35 Hz component with a bandpass filter and reconstruct via inverse FFT.
