# Phase-Space Representations

> *Three quasi-probability distributions — Q, P, and Wigner — each paint a different portrait of the same quantum state. Together they form the complete toolkit for visualizing quantum light in phase space.*

## Coherent and number states in phase space

Recall the quadrature operators:
$$
    \hat{X_1}
    =
    \frac{1}{2}
    \left(
        \hat{a} + \hat{a}^\dag
    \right), \qquad
    \hat{X_2}
    =
    \frac{1}{2i}
    \left(
        \hat{a} - \hat{a}^\dag
    \right).
$$

For a coherent state $|\alpha\rangle$, the quadrature expectations are $\langle\hat{X}_1\rangle = \text{Re}(\alpha)$ and $\langle\hat{X}_2\rangle = \text{Im}(\alpha)$, with equal variances $\langle(\Delta\hat{X}_{1,2})^2\rangle = \frac{1}{4}$. On the quadrature plane, a coherent state is a fuzzy dot of radius $\frac{1}{2}$ centered at $\alpha$, with distance from the origin equal to $\sqrt{\bar{n}}$.

A number state $|n\rangle$ has $\langle\hat{X}_{1,2}\rangle = 0$ and variances $\frac{1}{2}(n + \frac{1}{2})$ — a ring centered at the origin with no angular localization. The number state knows exactly how many photons it has but nothing about the oscillation phase.

[[simulation phase-space-states]]

As $|\alpha| \to \infty$ for a coherent state, the angular uncertainty $\Delta\theta \to 0$ — very bright coherent light has a well-defined phase, recovering classical behavior.

---

## The Husimi Q function

The simplest phase-space distribution is the **Husimi Q function**:

$$
Q(\alpha) = \frac{1}{\pi}\langle\alpha|\hat{\rho}|\alpha\rangle.
$$

The Q function is always non-negative (it is the overlap of the state with a coherent state, hence a probability). It gives a smoothed picture of the quantum state but never reveals sub-vacuum structure because the coherent-state projection acts as a Gaussian blur.

---

## The Glauber-Sudarshan P function

### Definition

The coherent states form an overcomplete basis, so any density operator can be expanded as

$$
    \hat{\rho}
    =
    \int \mathrm{d}^2 \alpha\,
    P(\alpha)
    |\alpha\rangle
    \langle\alpha|.
$$

The weight function $P(\alpha)$ is the **Glauber-Sudarshan P function**. It satisfies $\int P(\alpha)\,\mathrm{d}^2\alpha = 1$, so it looks like a probability distribution — but it need not be non-negative.

### Computing P via Fourier transform

Using the coherent-state overlap $\langle\beta|\alpha\rangle = e^{-|\alpha|^2/2 - |\beta|^2/2 + \beta^*\alpha}$, one obtains

$$
    P(\alpha)
    =
    \frac{e^{|\alpha|^2}}{\pi^2}
    \int \mathrm{d}^2 u\,
    e^{|u|^2}
    \langle -u|\hat{\rho}|u\rangle\,
    e^{u^*\alpha - \alpha^* u}.
$$

### Example: coherent state

For a pure coherent state $|\beta\rangle$, the P function is a Dirac delta: $P(\alpha) = \delta^2(\alpha - \beta)$. This is a single point in phase space — maximally classical.

### Non-classicality criterion

The P function is what separates classical from non-classical light. If $P(\alpha)$ can be interpreted as a genuine (non-negative, non-singular) probability distribution, the state is classical. If $P$ is negative or more singular than a delta function, the state has no classical analogue.

For a **number state** $|n\rangle$ with $n \geq 1$, the P function involves derivatives of the delta function and cannot be interpreted as a probability distribution. This mathematical pathology is the fingerprint of a genuinely quantum state: a number state has zero photon-number uncertainty, which is impossible for any classical or semi-classical light source. No mixture of coherent states — no matter how cleverly weighted — can reproduce the sharp, deterministic photon count of a Fock state.

### The optical equivalence theorem

For any normally ordered operator $\hat{G}^{(N)}(\hat{a}, \hat{a}^\dag) = \sum_{n,m} C_{nm}\,{\hat{a}^\dag}^n \hat{a}^m$:

$$
    \langle \hat{G}^{(N)} \rangle
    =
    \int P(\alpha)\,\mathrm{d}^2\alpha\,
    G^{(N)}(\alpha, \alpha^*).
$$

Replace operators with complex numbers, weight by $P(\alpha)$, and integrate — just as in classical statistical mechanics.

---

## The Wigner function

### Definition

The **Wigner function** is defined as

$$
    W(q, p)
    =
    \frac{1}{2\pi\hbar}
    \int_{-\infty}^\infty
    \mathrm{d}x\,
    \langle q + \tfrac{x}{2}|\hat{\rho}|q - \tfrac{x}{2}\rangle\,
    e^{-ipx/\hbar}.
$$

The Wigner function is always real, normalized to one, and its marginals give the correct position and momentum probability densities:

$$
\int W(q,p)\,\mathrm{d}p = |\psi(q)|^2, \qquad \int W(q,p)\,\mathrm{d}q = |\tilde{\psi}(p)|^2.
$$

### Negativity as non-classicality

Unlike the Q function, the Wigner function **can be negative**. Negative values are a direct signature of non-classicality:

* **Coherent states**: non-negative Gaussian $W$ — classical behavior.
* **Fock states** ($n \geq 1$): oscillating $W$ with $W(0,0) < 0$ — no classical analogue.
* **Cat states**: interference fringes between two coherent-state lobes — visible only in $W$, invisible in $Q$.

The Wigner function sits between the P function (too singular for non-classical states) and the Q function (always positive but smeared): it gives the sharpest accessible picture of quantum phase space.

[[simulation phase-space-explorer]]

---

## Big Ideas

* A coherent state $|\alpha\rangle$ is a fuzzy dot centered at $\alpha$ in the quadrature plane with uncertainty radius $\frac{1}{2}$; a number state $|n\rangle$ is a ring centered at the origin — it knows how many photons it has but not where in the oscillation cycle it is.
* The P function classifies light: if $P(\alpha) \geq 0$ everywhere (and is no more singular than a delta function), the state is classical; if $P$ goes negative or becomes hyper-singular, the state is genuinely quantum.
* The optical equivalence theorem reduces quantum expectation values to classical-looking integrals weighted by $P(\alpha)$, but the "probability" weights can fail to be probabilities — and that failure is exactly what makes quantum optics interesting.
* The Wigner function is the most informative quasi-probability distribution: its marginals are real probabilities, but its negativity is a smoking gun for non-classicality.
* The Q function is always non-negative but smooths away quantum features — it is the price of guaranteed positivity.

## What Comes Next

With the full family of phase-space distributions in hand — P, Q, and Wigner — we now have powerful tools for visualizing and classifying quantum states of light. The next lesson shifts focus from single-time snapshots to correlations in time and space, asking: when does light from two points interfere, and what determines the quality of that interference? This brings us to coherence functions.

## Check Your Understanding

1. The P function for a coherent state is $\delta^2(\alpha - \beta)$, a well-defined probability distribution concentrated at a single point. What must be true about the P function for a state to be considered "classical," and why does thermal light (which has $P(\alpha) = e^{-|\alpha|^2/\bar{n}}/(\pi\bar{n})$) qualify?
2. The Wigner function integrates to give correct position and momentum marginals, yet it can be negative. What exactly does the negativity tell you about the state, and why does it not lead to negative probabilities for any actual measurement?
3. The Q function is always non-negative by construction ($Q = \langle\alpha|\hat{\rho}|\alpha\rangle/\pi \geq 0$). Does this mean the Q function contains less information about the state than the Wigner function, and can you recover the full density matrix from $Q$ alone?

## Challenge

Compute the Wigner function for the number state $|n\rangle$ using the harmonic oscillator wavefunctions. Show that the result involves Laguerre polynomials: $W(q,p) \propto (-1)^n L_n(4(q^2+p^2))e^{-2(q^2+p^2)}$. Verify this for $n=0$ (positive Gaussian) and $n=1$ (negative at the origin). Then compute the P function for $|n=1\rangle$ and show it involves $\partial^2\delta^2(\alpha)/\partial\alpha\,\partial\alpha^*$, confirming it cannot be a probability distribution. What does this tell you about the impossibility of representing a single photon as a statistical mixture of coherent states?
