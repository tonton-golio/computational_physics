# Quantum Fluctuations and Quadrature Operators

## Quantum fluctuations
Time average of electric field is
$$
\begin{aligned}
    \left<
        \hat{\vec{E}}
    \right>
    &=
    \Braket{n|\hat{\vec{E}}|n}
    \\&=
    \sqrt{
        \frac{\hbar \omega}{\epsilon_0 V}
    }
    \sin (kz)
    \Braket{n|\left(\hat{a}+\hat{a}^\dag\right)|n}
    \\&=
    0
\end{aligned}
$$
I see. How about variance?
$$
\begin{aligned}
    \left<
        \hat{\vec{E}}^2
    \right>
    &=
    \Braket{n|\hat{\vec{E}}^2|n}
    \\&=
    \frac{\hbar \omega}{\epsilon_0 V}
    \sin^2 (kz)
    \Braket{
        n|
        \left(
        \hat{a}^2 + {\hat{a}^\dag}^2 + \hat{a}\hat{a}^\dag + \hat{a}^\dag\hat{a}
        \right)
        |n
    }
    \\&=
    \frac{\hbar \omega}{\epsilon_0 V}
    \sin^2 (kz)
    \left(
        n+\frac{1}{2}
    \right)
\end{aligned}
$$
Wow! Even in the vacuum state ($n=0$), electric field fluctuates!!

## Quadrature Operators
Similar to $\hat{q}$ and $\hat{p}$ operators, we introduce quadrature operators
$\hat{X_1}$ and $\hat{X_2}$..
$$
    \hat{X_1}
    =
    \frac{\hat{a} + \hat{a}^\dag}{2}
    \propto
    \hat{q}(t)
$$
$$
    \hat{X_2}
    =
    -i
    \frac{\hat{a} - \hat{a}^\dag}{2}
    \propto
    \hat{p}(t)
$$
$$
    \left[
        \hat{X_1}, \hat{X_2}
    \right]
    =
    \frac{1}{2}
$$
Notice that quadrature operators are observable and homodyne technique is used for
that.
And these operators does not commute. This is because Heisenberg's uncertainty.

We can rewrite electric field with quadrature operators.
$$
\begin{aligned}
    \hat{E_x}
    &=
    \mathcal{E}_0
    \left[
        \hat{a}e^{-i\omega t}
        +
        \hat{a}^\dag e^{i\omega t}
    \right]
    \sin (kz)
    \\&=
    \mathcal{E}_0
    \sin (kz)
    \left[
        \left(
            \hat{a}
            +
            \hat{a}^\dag
        \right)
        \cos \omega t
        -
        i
        \left(
            \hat{a}
            -
            \hat{a}^\dag
        \right)
        \sin \omega t
    \right]
    \\&=
    2
    \mathcal{E}_0
    \sin (kz)
    \left[
        \hat{X_1}
        \cos \omega t
        +
        \hat{X_2}
        \sin \omega t
    \right]
\end{aligned}
$$
Important staff of quadrature operators is their uncertainty.
$$
    \left<
        \left(
            \Delta \hat{X}_1
        \right)^2
    \right>
    \left<
        \left(
            \Delta \hat{X}_2
        \right)^2
    \right>
    \ge
    \frac{1}{4}
    \left|
        \left<
            \left[
                \hat{X}_1, \hat{X}_2
            \right]
        \right>
    \right|^2
    \ge
    \frac{1}{16}
$$

## Big Ideas

* A number state has zero average electric field but non-zero variance — the field is fluctuating even when its mean is completely silent.
* The quadrature operators $\hat{X}_1$ and $\hat{X}_2$ decompose the field into two orthogonal oscillation components; measuring one unavoidably disturbs the other.
* The uncertainty product $\langle(\Delta\hat{X}_1)^2\rangle\langle(\Delta\hat{X}_2)^2\rangle \geq \frac{1}{16}$ is not a limitation of our instruments — it is a structural feature of the quantum field.

## What Comes Next

We now know that number states fluctuate in both quadratures, and that those fluctuations cannot be reduced below the uncertainty bound. The next lesson asks: are there states of the field that look as classical as possible — states that achieve the minimum uncertainty and whose average field behaves like a classical sine wave? The answer is yes, and they are called coherent states.

## Check Your Understanding

1. For a number state $|n\rangle$ the mean electric field is zero, yet the mean-square field grows with $n$. Explain physically what is fluctuating and why more photons produce larger fluctuations rather than a more definite field value.
2. The quadrature operators are defined as specific linear combinations of $\hat{a}$ and $\hat{a}^\dagger$. Why do we call them quadratures, and what is the physical interpretation of measuring $\hat{X}_1$ versus $\hat{X}_2$?
3. The Heisenberg uncertainty product $\langle(\Delta\hat{X}_1)^2\rangle\langle(\Delta\hat{X}_2)^2\rangle \geq \frac{1}{16}$ holds for all states. Which states saturate the inequality, and what special property does that saturation imply?

## Challenge

The vacuum state $|0\rangle$ satisfies $\hat{a}|0\rangle = 0$. Use this to prove directly that $\langle(\Delta\hat{X}_1)^2\rangle = \langle(\Delta\hat{X}_2)^2\rangle = \frac{1}{4}$ for the vacuum, confirming it is a minimum-uncertainty state with equal noise in both quadratures. Then construct any state of the form $|\psi\rangle = c_0|0\rangle + c_1|1\rangle$ (normalized) and compute both quadrature variances. Under what conditions on $c_0, c_1$ does the state achieve minimum uncertainty?
