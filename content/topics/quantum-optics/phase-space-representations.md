# Phase-Space Representations

## Draw the vacuum blob

Before we define anything formally, picture this. Take the quadrature plane -- $\hat{X}_1$ horizontal, $\hat{X}_2$ vertical. The vacuum state is a fuzzy circle centered at the origin with radius $1/2$. A coherent state $\Ket{\alpha}$ is the same fuzzy circle, displaced to the point $(\text{Re}(\alpha), \text{Im}(\alpha))$. A number state $\Ket{n}$ is a ring -- it knows exactly how many photons it has, but its phase is completely random, smeared uniformly around a circle.

That's the picture. Now let's make it precise with three quasi-probability distributions: Q, P, and Wigner.

[[simulation phase-space-states]]

## The Husimi Q function

The simplest phase-space picture is the **Husimi Q function**:

$$
Q(\alpha) = \frac{1}{\pi}\langle\alpha|\hat{\rho}|\alpha\rangle
$$

It's always non-negative -- you're just asking "how much does my state overlap with a coherent state at point $\alpha$?" But that coherent-state projection acts like a Gaussian blur, smoothing away quantum features. The Q function is safe and pretty, but it hides the interesting stuff.

## The Glauber-Sudarshan P function

Since coherent states form an overcomplete basis, you can expand any density operator as:

$$
\hat{\rho} = \int d^2\alpha\,P(\alpha)\,\Ket{\alpha}\Bra{\alpha}
$$

The weight function $P(\alpha)$ is the **P function**. It satisfies $\int P(\alpha)\,d^2\alpha = 1$, so it *looks* like a probability distribution. But it doesn't have to be non-negative.

For a coherent state $\Ket{\beta}$, $P(\alpha) = \delta^2(\alpha - \beta)$: a single point in phase space. Maximally classical.

For a number state $\Ket{n}$ with $n \ge 1$, the P function involves derivatives of the delta function -- nature's way of saying you can't think of a single photon as a classical mixture of coherent states.

**This is the non-classicality test:** if $P(\alpha) \ge 0$ everywhere and is no worse than a delta function, the state is classical. If $P$ goes negative or hyper-singular, the state is genuinely quantum. That's the dividing line.

The **optical equivalence theorem** makes the P function enormously useful: for any normally ordered operator, replace operators with complex numbers, weight by $P(\alpha)$, and integrate. Just like classical statistical mechanics -- except the "probability" weights can fail to be probabilities.

## The Wigner function

The **Wigner function** is the Goldilocks distribution -- more informative than Q, less pathological than P:

$$
W(q, p) = \frac{1}{2\pi\hbar}\int_{-\infty}^\infty dx\,\langle q + \tfrac{x}{2}|\hat{\rho}|q - \tfrac{x}{2}\rangle\,e^{-ipx/\hbar}
$$

It's always real and normalized. Its marginals give the correct position and momentum distributions:

$$
\int W(q,p)\,dp = |\psi(q)|^2, \qquad \int W(q,p)\,dq = |\tilde{\psi}(p)|^2
$$

And here's the gorgeous part: **the Wigner function can go negative**. Negative values are a smoking gun for non-classicality:

* **Coherent states**: positive Gaussian. Classical behavior.
* **Fock states** ($n \ge 1$): oscillating $W$ with negative regions. No classical analogue.
* **Cat states**: interference fringes between two coherent blobs -- visible only in $W$, invisible in $Q$.

[[simulation phase-space-explorer]]

## Big Ideas

* A coherent state is a fuzzy dot in phase space; a number state is a ring with no angular localization -- it knows photon count but not phase.
* The P function is the classicality test: $P \ge 0$ and no worse than a delta means classical; anything else is genuinely quantum.
* The Wigner function is the most informative quasi-probability: its marginals are real probabilities, but its negativity is the clearest signature of non-classical light.

## Check Your Understanding

1. Thermal light has a smooth, non-negative P function ($P(\alpha) = e^{-|\alpha|^2/\bar{n}}/(\pi\bar{n})$). Why does this make thermal light "classical" even though it's noisy?
2. The Wigner function can be negative, yet marginals are always non-negative. How does this avoid producing negative probabilities for any actual measurement?

## Challenge

Write a Python script to compute and plot the Wigner function for Fock states $\Ket{0}$, $\Ket{1}$, and $\Ket{3}$ using the formula $W(q,p) \propto (-1)^n L_n(4(q^2+p^2))\,e^{-2(q^2+p^2)}$ (with Laguerre polynomials from `scipy.special`). Verify that $\Ket{0}$ is a positive Gaussian, $\Ket{1}$ is negative at the origin, and $\Ket{3}$ has three oscillating rings. Plot cross-sections through $p=0$.
