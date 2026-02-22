# Higher-Order Coherence

## The quantum fingerprint

First-order coherence tells you about interference. But it can't tell you whether your light is quantum or classical -- both coherent states and number states give $|g^{(1)}(\tau)| = 1$. To see the quantum, you need to look at **photon-photon correlations**: do photons tend to arrive in pairs, or do they space themselves out?

That's what the second-order correlation function measures.

## $g^{(2)}(0)$: the one number that classifies everything

$$
g^{(2)}(\tau) = \frac{\langle\hat{a}^\dag\hat{a}^\dag\hat{a}\hat{a}\rangle}{\langle\hat{a}^\dag\hat{a}\rangle^2}
$$

At zero delay, $g^{(2)}(0)$ tells you the whole story:

| Source | $g^{(2)}(0)$ | Character |
|---|---|---|
| Coherent light | $1$ | Poissonian |
| Thermal light | $2$ | Super-Poissonian (bunched) |
| Number state $\Ket{n}$ | $1 - 1/n$ | Sub-Poissonian |
| Single photon $\Ket{1}$ | $0$ | Perfectly anti-bunched |

The classical Cauchy-Schwarz inequality demands $g^{(2)}(0) \ge 1$. Any measurement giving $g^{(2)}(0) < 1$ is direct proof that the light is non-classical. Full stop. No classical wave theory can produce it.

## The Hanbury Brown-Twiss experiment

In 1956, Hanbury Brown and Twiss split starlight onto two detectors and measured coincidence rates. They found **photon bunching**: thermal photons arrive in pairs, with $g^{(2)}(0) = 2$ dropping to $g^{(2)} = 1$ for delays beyond the coherence time:

$$
g^{(2)}(\tau) = 1 + |g^{(1)}(\tau)|^2
$$

This initially seemed to imply photons "attract" each other. They don't. Bunching comes from bosonic statistics -- identical particles conspiring to favor joint detection, not from any photon-photon force.

## Antibunching: the quantum star

**Photon antibunching** -- $g^{(2)}(0) < 1$ -- has no classical explanation. It means photons actively avoid arriving together. First observed by Kimble, Dagenais, and Mandel (1977) in the fluorescence of a **single atom**.

Why does a single atom antibunch? Because after it emits one photon, it's in the ground state. It physically *cannot* emit a second photon until it's re-excited. So photons come one at a time, with $g^{(2)}(0) = 0$. That's a genuinely quantum effect.

## The Mandel Q parameter

The **Mandel Q parameter** links photon statistics to $g^{(2)}(0)$:

$$
Q = \frac{\langle(\Delta n)^2\rangle - \langle n\rangle}{\langle n\rangle} = \langle n\rangle(g^{(2)}(0) - 1)
$$

$Q = 0$ for coherent light, $Q > 0$ for classical (super-Poissonian) light, $Q < 0$ for non-classical (sub-Poissonian) light. Number states have $Q = -1$ -- the minimum possible.

## Higher-order correlations

The hierarchy extends to any order:

$$
g^{(n)}(0) = \frac{\langle(\hat{a}^\dag)^n\hat{a}^n\rangle}{\langle\hat{a}^\dag\hat{a}\rangle^n}
$$

For thermal light, $g^{(n)}(0) = n!$, while for coherent light, $g^{(n)}(0) = 1$ for all $n$. These provide increasingly stringent tests of non-classicality.

## Big Ideas

* $g^{(2)}(0)$ is the single most powerful number for classifying light: $= 1$ coherent, $= 2$ thermal, $< 1$ non-classical, $= 0$ single photon.
* Any measurement giving $g^{(2)}(0) < 1$ is direct proof that quantum mechanics is needed.
* Bunching is bosonic statistics, not photon-photon attraction. Antibunching is a single emitter that can't fire twice in a row.

## Check Your Understanding

1. A single two-level atom emits fluorescence with $g^{(2)}(0) = 0$. Why can't it emit two photons at once, and what does this tell you about the statistics right after a detection event?
2. The Mandel Q parameter is zero for coherent, positive for thermal, and $-1$ for number states. Interpret the sign: are the photon arrivals more or less "regular" than random Poissonian events?

## Challenge

Write a Python script that simulates the HBT experiment numerically. Generate thermal light as a sum of random-phase emitters, split it at a beam splitter, and compute the coincidence rate $g^{(2)}(\tau)$ as a function of delay. Verify $g^{(2)}(0) \approx 2$ and $g^{(2)}(\tau) \to 1$ for $\tau \gg \tau_c$. Then repeat for a coherent source (fixed phase) and confirm $g^{(2)}(\tau) = 1$ everywhere.
