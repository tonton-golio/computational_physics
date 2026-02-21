# Higher-Order Coherence

## Second-Order Correlation Function

The **second-order correlation function** measures intensity-intensity correlations and reveals the photon statistics of a light source. It is defined as

$$
g^{(2)}(\tau) = \frac{\langle \hat{a}^\dagger \hat{a}^\dagger \hat{a} \hat{a} \rangle}{\langle \hat{a}^\dagger \hat{a} \rangle^2} = \frac{\langle : \hat{n}(\hat{n}-1) : \rangle}{\langle \hat{n} \rangle^2},
$$

where the colons denote normal ordering. At zero delay, $g^{(2)}(0)$ characterizes the photon number fluctuations:

* **Coherent light** (Poissonian statistics): $g^{(2)}(0) = 1$.
* **Thermal light** (super-Poissonian): $g^{(2)}(0) = 2$.
* **Number state** $|n\rangle$ (sub-Poissonian): $g^{(2)}(0) = 1 - 1/n < 1$.
* **Single photon** $|1\rangle$: $g^{(2)}(0) = 0$.

The condition $g^{(2)}(0) < 1$ is a signature of **non-classical light** with no classical analogue. Classically, the Cauchy-Schwarz inequality requires $g^{(2)}(0) \geq 1$.

## The Hanbury Brown-Twiss Experiment

In 1956, Hanbury Brown and Twiss measured intensity correlations of starlight using two detectors. They observed **photon bunching**: photons from a thermal source tend to arrive in pairs, giving $g^{(2)}(0) > g^{(2)}(\tau)$ for $\tau > 0$.

The experimental setup splits the light beam with a beam splitter and sends it to two detectors. A correlator measures the coincidence rate as a function of the time delay $\tau$ between detections. For thermal light:

$$
g^{(2)}(\tau) = 1 + |g^{(1)}(\tau)|^2.
$$

At $\tau = 0$, this gives $g^{(2)}(0) = 2$, meaning photons are twice as likely to arrive together as independently. As $\tau$ increases beyond the coherence time, $g^{(2)}(\tau) \to 1$.

The HBT result was initially controversial because it seemed to imply that photons "attract" each other. The resolution is that bunching arises from the bosonic nature of photons and the statistical properties of thermal states, not from photon-photon interactions.

## Photon Bunching and Antibunching

**Photon bunching** ($g^{(2)}(0) > 1$) occurs for thermal and chaotic light sources. It is a classical phenomenon explainable by wave interference of many random emitters.

**Photon antibunching** ($g^{(2)}(0) < 1$) has no classical explanation and requires a quantum description. It means photons tend to arrive one at a time, with a suppressed probability of simultaneous detection. Antibunching was first observed by Kimble, Dagenais, and Mandel (1977) in the fluorescence of a single atom.

The key distinction is:
* Bunching: $g^{(2)}(\tau) < g^{(2)}(0)$ (correlations decrease with delay).
* Antibunching: $g^{(2)}(\tau) > g^{(2)}(0)$ (correlations increase from a minimum at $\tau = 0$).

## Sub-Poissonian Statistics

Closely related to antibunching is the concept of **sub-Poissonian** photon statistics, where the variance of the photon number is below the Poissonian value:

$$
\langle (\Delta n)^2 \rangle < \langle n \rangle.
$$

The **Mandel Q parameter** quantifies the deviation:

$$
Q = \frac{\langle (\Delta n)^2 \rangle - \langle n \rangle}{\langle n \rangle} = \langle n \rangle (g^{(2)}(0) - 1).
$$

$Q = 0$ for coherent light, $Q > 0$ for super-Poissonian (classical) light, and $Q < 0$ for sub-Poissonian (non-classical) light. Number states have $Q = -1$, the minimum possible.

## Higher-Order Correlations

The hierarchy extends to arbitrary order. The $n$-th order correlation function is

$$
g^{(n)}(\tau_1, \ldots, \tau_{n-1}) = \frac{\langle (\hat{a}^\dagger)^n \hat{a}^n \rangle}{\langle \hat{a}^\dagger \hat{a} \rangle^n}.
$$

For thermal light, $g^{(n)}(0) = n!$, while for coherent light, $g^{(n)}(0) = 1$ for all $n$. These higher-order functions provide increasingly stringent tests of non-classicality and are relevant for multi-photon experiments and quantum information protocols.

## Big Ideas

* The second-order correlation $g^{(2)}(0)$ is the single most powerful number for classifying a light source: $= 1$ for coherent, $= 2$ for thermal, $< 1$ for non-classical, and $= 0$ for a true single photon.
* The classical Cauchy-Schwarz inequality demands $g^{(2)}(0) \geq 1$; any measurement giving $g^{(2)}(0) < 1$ is direct proof that quantum mechanics is necessary to describe the light.
* Photon bunching (thermal light) is not because photons attract each other — it arises from the bosonic statistics of identical particles conspiring to favor joint detection.
* Antibunching ($g^{(2)}(0) < 1$) and sub-Poissonian statistics are related but distinct: antibunching is a statement about temporal correlations, sub-Poissonian is about the integrated photon number distribution.

## What Comes Next

We now have a complete statistical toolkit for characterizing any light source. The next lesson turns this into practice: single-photon experiments at the level of individual quanta, using beam splitters and coincidence detection to observe antibunching, Hong-Ou-Mandel interference, and the genuinely quantum behavior of light particle by particle.

## Check Your Understanding

1. Thermal light has $g^{(2)}(0) = 2$, meaning two photons are twice as likely to arrive simultaneously as a coherent beam with the same mean intensity. Where does this extra factor of 2 come from? What is physically happening in the thermal source that produces this bunching?
2. The condition $g^{(2)}(0) < 1$ cannot be satisfied by any classical field, yet a single atom emitting fluorescence photons achieves $g^{(2)}(0) = 0$. Explain physically why a single two-level atom cannot emit two photons at once, and what this tells you about the photon statistics immediately after a detection event.
3. The Mandel Q parameter is $Q = \langle n \rangle(g^{(2)}(0) - 1)$. A coherent state gives $Q = 0$, a thermal state gives $Q = \langle n \rangle > 0$, and a number state gives $Q = -1$. Interpret the sign of $Q$ in terms of whether the photon arrivals are more or less "regular" than random Poissonian events.

## Challenge

The Hanbury Brown-Twiss experiment with thermal light gives $g^{(2)}(\tau) = 1 + |g^{(1)}(\tau)|^2$. Starting from the moment theorem for Gaussian random variables (which applies to thermal light in the classical description), derive this Siegert relation. Then compute $g^{(2)}(\tau)$ explicitly for a Lorentzian-lineshape thermal source with $g^{(1)}(\tau) = e^{-\gamma|\tau|/2}$, and sketch the full curve. At what time delay does $g^{(2)}(\tau)$ fall to within 10% of its long-delay value of 1, and how does this relate to the coherence time?
