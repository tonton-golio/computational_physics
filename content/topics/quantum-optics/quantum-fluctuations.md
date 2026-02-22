# Quantum Fluctuations and Quadrature Operators

## The vacuum won't sit still

Strip away every photon from a cavity. Cool it, shield it, pump it down to the absolute ground state. And the electric field is *still fluctuating*. Not because of stray photons, not because of thermal noise, but because the uncertainty principle flat-out forbids the field from sitting perfectly still.

Let's prove it.

## The field averages to zero -- but it's not zero

Take a number state $\Ket{n}$ and compute the average electric field:

$$
\langle\hat{\vec{E}}\rangle = \Braket{n|\hat{\vec{E}}|n} = \sqrt{\frac{\hbar\omega}{\epsilon_0 V}}\sin(kz)\,\Braket{n|(\hat{a} + \hat{a}^\dag)|n} = 0
$$

Zero. The field averages to nothing. But now check the *variance*:

$$
\langle\hat{\vec{E}}^2\rangle = \frac{\hbar\omega}{\epsilon_0 V}\sin^2(kz)\left(n + \frac{1}{2}\right)
$$

Even when $n = 0$ -- zero photons, total vacuum -- the field is fluctuating with variance $\frac{\hbar\omega}{2\epsilon_0 V}$. The vacuum is buzzing. These aren't instrument artifacts. They're structural features of quantum mechanics.

## Quadrature operators

To track these fluctuations properly, we introduce the **quadrature operators** -- the quantum analogs of the two components of a classical oscillation:

$$
\hat{X}_1 = \frac{\hat{a} + \hat{a}^\dag}{2}, \qquad \hat{X}_2 = -i\frac{\hat{a} - \hat{a}^\dag}{2}
$$

They're proportional to the "position" $\hat{q}$ and "momentum" $\hat{p}$ of our field oscillator. Both are Hermitian (observable!), and they don't commute:

$$
[\hat{X}_1, \hat{X}_2] = \frac{i}{2}
$$

You can rewrite the electric field in terms of quadratures:

$$
\hat{E}_x = 2\mathcal{E}_0\sin(kz)\left[\hat{X}_1\cos\omega t + \hat{X}_2\sin\omega t\right]
$$

Think of $\hat{X}_1$ and $\hat{X}_2$ as the two "directions" the field can wiggle. Measuring one necessarily fuzzes up the other.

## The uncertainty bound

Because the quadratures don't commute, Heisenberg gives us a hard floor:

$$
\langle(\Delta\hat{X}_1)^2\rangle\,\langle(\Delta\hat{X}_2)^2\rangle \ge \frac{1}{16}
$$

This isn't a statement about your instruments being lousy. It's a structural feature of the quantum field. You *cannot* know both quadratures precisely, no matter how clever you are.

States that *saturate* this bound -- where the product equals exactly $1/16$ -- are called **minimum-uncertainty states**. The vacuum is one of them, with equal noise in both directions: $\langle(\Delta\hat{X}_1)^2\rangle = \langle(\Delta\hat{X}_2)^2\rangle = 1/4$. A perfect, symmetric fuzzy blob in phase space.

## Big Ideas

* Number states have zero average field but nonzero variance -- the field fluctuates even when its mean is silent.
* The quadrature operators $\hat{X}_1$ and $\hat{X}_2$ decompose the field into two oscillation components; measuring one unavoidably disturbs the other.
* The uncertainty product $\langle(\Delta\hat{X}_1)^2\rangle\langle(\Delta\hat{X}_2)^2\rangle \ge 1/16$ is not an instrumental limitation -- it's built into the structure of the quantum field.

## Check Your Understanding

1. For $\Ket{n}$, the mean field is zero but the mean-square field grows with $n$. What's physically fluctuating, and why do more photons produce *larger* fluctuations rather than a more definite field?
2. Why are $\hat{X}_1$ and $\hat{X}_2$ called "quadratures"? What does measuring one versus the other physically correspond to?

## Challenge

Write a Python script that computes $\langle(\Delta\hat{X}_1)^2\rangle$ and $\langle(\Delta\hat{X}_2)^2\rangle$ for the vacuum state $\Ket{0}$ and for superposition states $\Ket{\psi} = c_0\Ket{0} + c_1\Ket{1}$ using matrix representations in a truncated Fock space. Verify that the vacuum saturates the uncertainty bound. Then sweep over different $(c_0, c_1)$ values and find which superpositions are also minimum-uncertainty states.
