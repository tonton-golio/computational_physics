# Wigner Function and Characteristic Functions

#### Wigner function
Actually there are three type of quasi-probability distributions.
- Glauber-Sudarshan $P$ function
- Husimi $Q$ function
- Wigner $W$ function


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

# 3.8 Characteristic functions
