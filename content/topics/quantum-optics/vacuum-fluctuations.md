# Vacuum Fluctuations and Observable Effects

## From one mode to infinitely many

So far we've quantized a single mode -- one particular way light can wiggle in a box. But a real electromagnetic field is a superposition of infinitely many modes, each labeled by wave vector $\vec{k}$ and polarization $s$. The Hamiltonian is just a pile of independent harmonic oscillators:

$$
\hat{H} = \sum_{\vec{k},s} \hbar\omega_k\left(\hat{a}^\dag_{\vec{k},s}\hat{a}_{\vec{k},s} + \frac{1}{2}\right)
$$

Each mode has its own ladder operators and its own photon number. The eigenstate is a product of number states -- one per mode:

$$
\Ket{\{n_j\}} = \Ket{n_1, n_2, \ldots, n_j, \ldots}
$$

## Thermal fields and Planck's law

In thermal equilibrium at temperature $T$, the mean photon number per mode follows the Bose-Einstein distribution:

$$
\bar{n} = \frac{1}{e^{\hbar\omega/k_BT} - 1}
$$

At optical frequencies and room temperature, $\bar{n} \approx 0$. Temperature is essentially irrelevant for optical quantum optics -- a stark contrast with microwave quantum computing, where thermal photons are a real nuisance.

The number of ways light can wiggle in a box grows as frequency squared -- the density of states is $\rho(\omega) = \omega^2/(\pi^2 c^3)$. Multiply by the energy per mode and you get Planck's radiation law:

$$
\bar{U}(\omega) = \frac{\hbar\omega}{e^{\hbar\omega/k_BT} - 1}\cdot\frac{\omega^2}{\pi^2 c^3}
$$

The formula that started quantum mechanics.

## Observable effects of vacuum energy

Here's where it gets exciting. Those vacuum fluctuations aren't just theoretical bookkeeping -- they produce real, measurable effects.

### The Lamb shift

The electron in a hydrogen atom is constantly being jostled by vacuum fluctuations. It's jittering in a sea of invisible springs, and that tiny jitter shifts the energy levels by about one part in a million.

Specifically, Dirac's theory predicts the $2^2S_{1/2}$ and $2^2P_{1/2}$ levels should be degenerate. Experiments showed they're not. Bethe explained the discrepancy by accounting for vacuum fluctuations -- Welton gave the intuitive picture.

The vacuum-induced displacement modifies the Coulomb potential. Averaging over fluctuations:

$$
\langle\Delta V\rangle = \frac{1}{6}\langle(\Delta r)^2\rangle\,\nabla^2 V = \frac{1}{6}\langle(\Delta r)^2\rangle\,4\pi e^2\,\delta(\vec{r})
$$

The energy shift is:

$$
\Delta E = \frac{2\pi e^2}{3}\langle(\Delta r)^2\rangle\,|\psi_{nl}(0)|^2
$$

Only $s$-states (which have $|\psi(0)|^2 \neq 0$) feel the shift. The integral over vacuum mode frequencies diverges, but physical cutoffs make the measurable shift finite -- about 1 GHz for the $2S$-$2P$ splitting. The electron trembles in the vacuum sea, and that trembling is real.

### The Casimir effect

Two conducting plates attract each other, even in a perfect vacuum. Why? Because fewer vacuum wiggles fit between the plates than outside them.

Consider a box of dimensions $L \times L \times d$. The total vacuum energy inside depends on $d$ because the boundary conditions restrict which modes fit:

$$
\omega_{lmn} = \pi c\sqrt{\frac{l^2}{L^2} + \frac{m^2}{L^2} + \frac{n^2}{d^2}}
$$

The energy difference between plates at separation $d$ and plates at infinity works out to:

$$
U(d) = -\frac{\pi^2\hbar c}{720}\frac{L^2}{d^3}
$$

The minus sign means attraction. The force scales as $1/d^4$ -- it's tiny, but it's been measured with exquisite precision. The vacuum is literally pushing the plates together.

> **Computational Note.** You can verify the Casimir result numerically: sum the vacuum energies $\frac{1}{2}\hbar\omega_{lmn}$ over modes with a smooth exponential cutoff $e^{-\omega/\omega_c}$, subtract the continuum integral, and watch the $d^{-3}$ scaling emerge. About 20 lines of NumPy.

## Big Ideas

* A real field has infinitely many modes, each an independent quantum oscillator. Thermal light at room temperature has essentially zero photons per visible mode.
* The Lamb shift splits hydrogen's $2S$ and $2P$ levels by about 1 GHz because the electron trembles in the vacuum field.
* Two neutral conducting plates attract each other (Casimir effect) because the vacuum modes between them are restricted while modes outside are not -- the energy imbalance produces a measurable force.

## Check Your Understanding

1. The Lamb shift integral diverges over all frequencies, yet the physical shift is finite. What cutoff makes it converge, and what does this say about which vacuum modes actually matter?
2. The Casimir energy scales as $-d^{-3}$. What does the sign tell you? How does the force scale with separation? Is it easier or harder to measure as the plates get closer?

## Challenge

Write a Python script that computes the Casimir energy numerically. Sum $\frac{1}{2}\hbar\omega_{lmn}$ over modes $(l, m, n)$ in a box with a smooth UV cutoff, subtract the corresponding continuum integral, and extract the coefficient of $L^2/d^3$. Verify it converges to $-\pi^2\hbar c/720$ as the cutoff is removed. Plot the convergence as a function of cutoff frequency.
