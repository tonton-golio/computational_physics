# Phase-Space Pictures
__Topic 4 keywords__
- Quasi-probability distributions
- Characteristic functions
- Phase-space picture

# Readings
Ch. 3.6-8

# 3.6 Phase-space pictures of coherent states
#### Coherent state on a complex plane
Recall quadrature operators are
$$
    \hat{X_1}
    =
    \frac{1}{2}
    \left(
        \hat{a} + \hat{a}^\dag
    \right)
$$
$$
    \hat{X_2}
    =
    \frac{1}{2i}
    \left(
        \hat{a} - \hat{a}^\dag
    \right)
$$
As we learned in Topic 3.1, coherent state averages of quadrature operators are
$$
\begin{aligned}
    \Braket{\alpha|
    \hat{X_1}
    |\alpha}
    &=
    \frac{1}{2}
    \Braket{\alpha|
    \left(
        \hat{a} + \hat{a}^\dag
    \right)
    |\alpha}
    \\&=
    \frac{1}{2}
    \left(
        \alpha + \alpha^*
    \right)
    \\&=
    \mathrm{Re}(\alpha)
\end{aligned}
$$
$$
\begin{aligned}
    \Braket{\alpha|
    \hat{X_2}
    |\alpha}
    &=
    \frac{1}{2i}
    \Braket{\alpha|
    \left(
        \hat{a} - \hat{a}^\dag
    \right)
    |\alpha}
    \\&=
    \frac{1}{2i}
    \left(
        \alpha - \alpha^*
    \right)
    \\&=
    \mathrm{Im}(\alpha)
\end{aligned}
$$
Thus the coherent state $\Ket{\alpha}$ is centered at a $\alpha$ on a complex $\hat{X}_1$-$\hat{X}_2$ plane.

Remember that in Topic 3.1, we saw that that variance of quadrature operators are
$$
    \left<
        \left(
            \Delta \hat{X}_1
        \right)^2
    \right>_\alpha
    =
    \left<
        \left(
            \Delta \hat{X}_2
        \right)^2
    \right>_\alpha
    =
    \frac{1}{4}
$$
Thus the fluctuation of quadrature operators are both $\frac{1}{2}$.
On a complex $\hat{X}_1$-$\hat{X}_2$ plane, coherent state $\Ket{\alpha}$ is centered at a $\alpha$ with fluctuation of radius $\frac{1}{2}$.

As we learned in Topic 3.1, the average number of photon is $\bar{n}=|\alpha|^2$.
Thus the distance between coherent state and origin of complex plane is
square root of number of photon.

Notice when we consider $\alpha = \left|\alpha\right| e^{i\theta}$,
the uncertainty of theta $\Delta \theta$ become
maximum $2\pi$ when $\left|\alpha\right|=0$ and minimum $0$ when
$\left|\alpha\right| \rightarrow \infty$.

#### Number state on a complex plane
How about number state $\Ket{n}$?
Let's calculate!

[[simulation phase-space-states]]
$$
\begin{aligned}
    \Braket{n|
    \hat{X_1}
    |n}
    &=
    \frac{1}{2}
    \Braket{n|
    \left(
        \hat{a} + \hat{a}^\dag
    \right)
    |n}
    \\&=
    \frac{1}{2}
    \left(
        0 + 0
    \right)
    \\&=
    0
\end{aligned}
$$
$$
\begin{aligned}
    \Braket{n|
    \hat{X_2}
    |n}
    &=
    \frac{1}{2i}
    \Braket{n|
    \left(
        \hat{a} - \hat{a}^\dag
    \right)
    |n}
    \\&=
    \frac{1}{2i}
    \left(
        0 - 0
    \right)
    \\&=
    0
\end{aligned}
$$
Hmm. Number state is centered at origin.

How about fluctuation?
$$
\begin{aligned}
    \left<
        \left(
            \Delta \hat{X}_1
        \right)^2
    \right>_n
    &=
    \Braket{n|
            \hat{X}_1^2
    |n}
    -
    \Braket{n|
            \hat{X}_1
            |n}^2
    \\&=
    \Braket{n|
    \left(
        \frac{\hat{a} + \hat{a}^\dag}{2}
    \right)^2
    |n}
    -
    0
    \\&=
    \frac{1}{4}
    \Braket{n|
    \left(
        \hat{a}^2
        + {\hat{a}^\dag}^2
        + \hat{a} \hat{a}^\dag
        + \hat{a}^\dag \hat{a}
    \right)
    |n}
    \\&=
    \frac{1}{4}
    \Braket{n|
    \left(
        \hat{a}^2
        + {\hat{a}^\dag}^2
        + 1
        + \hat{a}^\dag \hat{a}
        + \hat{a}^\dag \hat{a}
    \right)
    |n}
    \\&=
    \frac{1}{4}
    \left(
        1
        +
        2n
    \right)
    \\&=
    \frac{1}{2}
    \left(
        \frac{1}{2} + n
    \right)
\end{aligned}
$$
$$
\begin{aligned}
    \left<
        \left(
            \Delta \hat{X}_2
        \right)^2
    \right>_n
    &=
    \Braket{n|
            \hat{X}_2^2
    |n}
    -
    \Braket{n|
            \hat{X}_2
            |n}^2
    \\&=
    \Braket{n|
    \left(
        \frac{\hat{a} - \hat{a}^\dag}{2i}
    \right)^2
    |n}
    -
    0
    \\&=
    -
    \frac{1}{4}
    \Braket{n|
    \left(
        \hat{a}^2
        + {\hat{a}^\dag}^2
        - \hat{a} \hat{a}^\dag
        - \hat{a}^\dag \hat{a}
    \right)
    |n}
    \\&=
    -
    \frac{1}{4}
    \Braket{n|
    \left(
        \hat{a}^2
        + {\hat{a}^\dag}^2
        - 1
        - \hat{a}^\dag \hat{a}
        - \hat{a}^\dag \hat{a}
    \right)
    |n}
    \\&=
    -
    \frac{1}{4}
    \left(
        - 1
        - 2n
    \right)
    \\&=
    \frac{1}{2}
    \left(
        \frac{1}{2} + n
    \right)
\end{aligned}
$$
When there is no photon, the fluctuation is $\frac{1}{2}$.
Therefore, the number state $\Ket{n=0}$ is centered at origin
and its uncertainty is circle with radius $\frac{1}{2}$.
This agree with that of coherent state $\Ket{\alpha=0}$.

When there is photon, the radius of fluctuation
$\sqrt{
    \frac{1}{2}
    \left(
        \frac{1}{2} + n
    \right)
}$
become larger than that of coherent state.
