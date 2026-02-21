# Wigner Function and Characteristic Functions

### Wigner function
Actually there are three type of quasi-probability distributions.
* Glauber-Sudarshan $P$ function
* Husimi $Q$ function
* Wigner $W$ function


Let's see the Wigner function.
$$
    W \left( q, p \right)
    =
    \frac{1}{2 \pi \hbar}
    \int_{-\infty}^\infty
    \mathrm{d} x
    \cdot
    \Braket{
        q + \frac{1}{2}x|
        \hat{\rho}
        |q - \frac{1}{2}x
    }
    e^{-i \frac{p}{\hbar} x}
$$
Here, $\Ket{q \pm \frac{1}{2}x}$ are the eigenkets of the position operator.

In the case of the state in question is a pure state
$\hat{\rho} = \Ket{\psi} \Bra{\psi}$,
$$
\begin{aligned}
    W \left( q, p \right)
    &=
    \frac{1}{2 \pi \hbar}
    \int_{-\infty}^\infty
    \mathrm{d} x
    \cdot
    \Braket{q + \frac{1}{2}x | \psi}
    \Braket{\psi | q - \frac{1}{2}x}
    e^{-i \frac{p}{\hbar} x}
    \\&=
    \frac{1}{2 \pi \hbar}
    \int_{-\infty}^\infty
    \mathrm{d} x
    \cdot
    \psi^* \left(q - \frac{1}{2}x \right)
    \psi \left(q + \frac{1}{2}x \right)
    e^{-i \frac{p}{\hbar} x}
\end{aligned}
$$
where $\psi \left(q + \frac{1}{2}x \right) = \Braket{q + \frac{1}{2}x | \psi} $.

Integrating Wigner function over momentum, we see
$$
\begin{aligned}
    \int_{-\infty}^\infty
    \mathrm{d} p
    \cdot
    W \left( q, p \right)
    &=
    \frac{1}{2 \pi \hbar}
    \int_{-\infty}^\infty
    \int_{-\infty}^\infty
    \mathrm{d} x
    \mathrm{d} p
    \cdot
    \psi^* \left(q - \frac{1}{2}x \right)
    \psi \left(q + \frac{1}{2}x \right)
    e^{-i \frac{p}{\hbar} x}
    \\&=
    \int_{-\infty}^\infty
    \mathrm{d} x
    \cdot
    \psi^* \left(q - \frac{1}{2}x \right)
    \psi \left(q + \frac{1}{2}x \right)
    \cdot
    \int_{-\infty}^\infty
    \mathrm{d} p
    \cdot
    \frac{1}{2 \pi \hbar}
    e^{-i \frac{p}{\hbar} x}
    \cdot
    \\&=
    \int_{-\infty}^\infty
    \mathrm{d} x
    \cdot
    \psi^* \left(q - \frac{1}{2}x \right)
    \psi \left(q + \frac{1}{2}x \right)
    \cdot
    \delta (x)
    \cdot
    \\&=
    \left| \psi (q) \right|^2
\end{aligned}
$$

## Characteristic functions

## Big Ideas

* The Wigner function $W(q, p)$ is a quasi-probability distribution over phase space that is always real and normalizes to one, but — unlike a classical probability distribution — can be negative.
* Integrating $W(q,p)$ over momentum gives the correct position probability density $|\psi(q)|^2$; integrating over position gives the momentum density — the marginals are real, physical probabilities.
* Negative values of $W$ are a smoking gun for non-classicality: coherent states have non-negative Gaussian Wigner functions, but Fock states and cat states have regions where $W < 0$.
* The Wigner function is the middle ground between the P function (too singular for non-classical states) and the Q function (always positive but smeared): it gives the sharpest accessible picture of quantum phase space.

## What Comes Next

With the full family of phase-space distributions in hand — P, Q, and Wigner — we now have powerful tools for describing light. The next lesson shifts focus from single-time snapshots to correlations in time and space, asking: when does light from two points interfere, and what determines the quality of that interference? This brings us to coherence functions.

## Check Your Understanding

1. The Wigner function integrates to give correct position and momentum marginals, yet it can be negative. A genuine probability distribution cannot be negative. What exactly does the negativity of $W(q,p)$ tell you about the state, and why does it not lead to negative probabilities for any actual measurement outcome?
2. For the vacuum state $|0\rangle$, the Wigner function is a Gaussian centered at the origin with width $\frac{1}{2}$. For the number state $|1\rangle$, $W$ is negative near the origin. Without computing, sketch qualitatively what you would expect $W$ to look like for $|1\rangle$ and explain the physics of the central dip.
3. The Husimi Q function is defined as $Q(\alpha) = \frac{1}{\pi}\langle\alpha|\hat{\rho}|\alpha\rangle$ and is always non-negative. The Wigner function is can go negative. What does this imply about the information content of the Q function — does it have less, the same, or more information about the state than the Wigner function?

## Challenge

Compute the Wigner function for the number state $|n\rangle$ by evaluating the integral definition $W(q,p) = \frac{1}{2\pi\hbar}\int dx\, \psi^*(q-x/2)\psi(q+x/2)e^{-ipx/\hbar}$ using the harmonic oscillator wavefunctions. Show that $W$ for the vacuum ($n=0$) is a positive Gaussian, while for $n=1$ it is negative at the origin. The result involves Laguerre polynomials: $W(q,p) \propto (-1)^n L_n(4(q^2+p^2))e^{-2(q^2+p^2)}$. Verify this for $n=0$ and $n=1$, and interpret the sign of $W(0,0)$.
