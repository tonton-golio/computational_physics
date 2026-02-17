# Quantum Fluctuations and Quadrature Operators

# 2.2 Quantum fluctuations
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

# 2.3 Quadrature Operators
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
