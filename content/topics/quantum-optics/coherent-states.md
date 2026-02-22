# Coherent States

## The closest thing to a classical wave

Number states are weird. They have a definite photon count but zero average electric field -- they don't look anything like a laser beam or a radio wave. So what quantum state *does* look classical? What state gives you a nice, oscillating electric field with minimum quantum fuzz?

The answer: **coherent states**. They're the quantum states that lasers actually produce, and they're the bridge between the quantum and classical descriptions of light.

## Definition: eigenstates of annihilation

A coherent state $\Ket{\alpha}$ is defined as an eigenstate of the annihilation operator:

$$
\hat{a}\Ket{\alpha} = \alpha\Ket{\alpha}
$$

Since $\hat{a}$ isn't Hermitian, the eigenvalue $\alpha$ is a complex number. That complex number encodes everything: $|\alpha|$ sets the amplitude, and $\arg(\alpha)$ sets the phase of the oscillating field.

There are three equivalent ways to think about coherent states:
- Eigenstates of $\hat{a}$
- States that minimize the uncertainty relation with equal noise in both quadratures
- Displaced vacuum states

We'll see all three.

## Expanding in the number basis

Insert the identity $\sum_n \Ket{n}\Bra{n}$ to expand a coherent state in the Fock basis. The eigenvalue equation gives a recursion for the coefficients, and with normalization you get:

$$
\Ket{\alpha} = e^{-|\alpha|^2/2}\sum_{n=0}^\infty \frac{\alpha^n}{\sqrt{n!}}\Ket{n}
$$

A coherent state is a specific superposition of *all* number states, weighted by a Poissonian distribution.

## Photon statistics: Poissonian

The mean photon number is $\bar{n} = |\alpha|^2$, and the variance is:

$$
\langle(\Delta n)^2\rangle_\alpha = |\alpha|^2 = \bar{n}
$$

Mean equals variance -- that's the hallmark of Poisson statistics. The probability of finding $n$ photons is:

$$
P_n = e^{-\bar{n}}\frac{\bar{n}^n}{n!}
$$

A coherent state with $\bar{n} = 100$ photons still has $\sqrt{100} = 10$ photons of uncertainty. You never know exactly how many photons are in a laser beam.

## The electric field behaves classically

The expectation value of the electric field in a coherent state is a perfect sine wave, with amplitude set by $|\alpha|$. That's exactly what you'd call "classical" light.

And here's the gorgeous part: the *variance* of the electric field is independent of $\alpha$:

$$
\langle(\Delta E_x)^2\rangle_\alpha = \frac{\hbar\omega}{2\epsilon_0 V}
$$

Same noise as the vacuum. A coherent state with a million photons has exactly the same noise floor as a coherent state with zero photons. The signal gets bigger but the noise stays fixed -- that's what makes coherent states "as classical as possible."

The quadrature variances confirm it:

$$
\langle(\Delta\hat{X}_1)^2\rangle_\alpha = \langle(\Delta\hat{X}_2)^2\rangle_\alpha = \frac{1}{4}
$$

Minimum uncertainty, equal in both directions. A symmetric fuzzy dot in phase space.

## The displacement operator

A coherent state is just the vacuum, kicked to a new location in phase space:

$$
\Ket{\alpha} = \hat{D}(\alpha)\Ket{0}, \qquad \hat{D}(\alpha) = e^{\alpha\hat{a}^\dag - \alpha^*\hat{a}}
$$

The displacement operator is a rigid translation -- it shifts the center without changing the shape or size of the uncertainty blob. That's why a coherent state has the same noise as the vacuum: you moved it but didn't squeeze it.

Physically, $\hat{D}(\alpha)$ corresponds to driving the cavity with a classical oscillating current. Turn on a classical source, and what you produce is a coherent state. Lasers are displacement operators.

## Non-orthogonality and overcompleteness

Unlike number states, coherent states are *not* orthogonal:

$$
|\langle\beta|\alpha\rangle|^2 = e^{-|\beta - \alpha|^2}
$$

Two coherent states are nearly orthogonal only when they're far apart in phase space. Despite this, they form an overcomplete set:

$$
\frac{1}{\pi}\int d^2\alpha\,\Ket{\alpha}\Bra{\alpha} = \hat{I}
$$

Overcomplete means you can expand any state in the coherent basis, but not uniquely. That's actually a feature, not a bug -- it's what makes the P function possible.

> **Computational Note.** Plot the Poissonian photon number distribution for coherent states with $\bar{n} = 1, 5, 25$ and compare to the thermal (geometric) distribution with the same mean. The visual difference is striking -- and it takes about 10 lines of Python with matplotlib.

## Big Ideas

- A coherent state is the eigenstate of $\hat{a}$, with complex eigenvalue $\alpha$ encoding both amplitude and phase.
- Photon statistics are Poissonian: $P_n = e^{-\bar{n}}\bar{n}^n/n!$, with mean equal to variance.
- The field variance is independent of $\alpha$ -- same noise as vacuum. Coherent states are as classical as quantum mechanics allows.
- Coherent states are non-orthogonal but overcomplete, forming the foundation of phase-space representations.

## Check Your Understanding

1. Why does it make physical sense that the eigenvalue $\alpha$ is complex? What do its modulus and argument represent?
2. Adding more photons doesn't reduce the noise. What *would* you have to do to get below-vacuum noise?
3. The displacement operator is a rigid translation in phase space. Without computing, predict whether $\langle(\Delta\hat{X}_2)^2\rangle_\alpha$ depends on $\alpha$.

## Challenge

Write a Python script that computes the photon-number distribution $P_n$ for a coherent state $\Ket{\alpha}$ and a thermal state with the same mean $\bar{n}$. Plot both distributions for $\bar{n} = 5$ and $\bar{n} = 25$. Compute the variance for each and verify $\langle(\Delta n)^2\rangle = \bar{n}$ (coherent) vs. $\langle(\Delta n)^2\rangle = \bar{n} + \bar{n}^2$ (thermal). What does the extra $\bar{n}^2$ term say physically about how thermal photons cluster?
