# Phase-Space Pictures
__Topic 4 keywords__
* Quasi-probability distributions
* Characteristic functions
* Phase-space picture

## Phase-space pictures of coherent states
### Coherent state on a complex plane
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

### Number state on a complex plane
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
        * \hat{a} \hat{a}^\dag
        * \hat{a}^\dag \hat{a}
    \right)
    |n}
    \\&=
    -
    \frac{1}{4}
    \Braket{n|
    \left(
        \hat{a}^2
        + {\hat{a}^\dag}^2
        * 1
        * \hat{a}^\dag \hat{a}
        * \hat{a}^\dag \hat{a}
    \right)
    |n}
    \\&=
    -
    \frac{1}{4}
    \left(
        * 1
        * 2n
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

## Big Ideas

* A coherent state $|\alpha\rangle$ is a fuzzy dot centered at the complex number $\alpha$ in the quadrature plane, with uncertainty circle of radius $\frac{1}{2}$ — distance from origin equals $\sqrt{\bar{n}}$.
* A number state $|n\rangle$ has no well-defined phase and is represented as a ring centered at the origin with radius growing as $\sqrt{n+\frac{1}{2}}$ — it knows how many photons it has but not where in the oscillation cycle it is.
* Phase-space pictures make the Heisenberg uncertainty principle geometric: the uncertainty "blob" can be stretched and rotated but its area cannot be reduced below the vacuum value.
* The phase uncertainty of a coherent state shrinks as $1/|\alpha|$ for large amplitude — this is why strong laser fields behave classically with well-defined phase.

## What Comes Next

The phase-space picture invites a deeper question: can we assign a genuine probability distribution to points in this plane? The next lesson introduces the Glauber-Sudarshan P function, which is the closest thing to a phase-space probability density for quantum states — and which reveals the strange fact that some quantum states require a "probability" distribution that goes negative.

## Check Your Understanding

1. A coherent state $|\alpha\rangle$ is centered at $\alpha$ in the quadrature plane and has uncertainty radius $\frac{1}{2}$. A number state $|n=0\rangle$ (vacuum) is also centered at the origin with the same uncertainty radius. Why do these two descriptions represent physically different states if they look the same in the diagram?
2. The number state $|n\rangle$ is represented as a ring of radius $\sqrt{n + \frac{1}{2}}$, centered at the origin. What does it mean physically that the ring has no angular localization — that all phases are equally represented?
3. As $|\alpha| \to \infty$ for a coherent state, the angular uncertainty $\Delta\theta \to 0$. Explain why this means very bright coherent light has a well-defined phase, and connect this to the correspondence principle.

## Challenge

Consider a superposition of two coherent states $|\psi\rangle = \mathcal{N}(|\alpha\rangle + |-\alpha\rangle)$ where $\alpha$ is real and large. Compute the mean and variance of both quadrature operators $\hat{X}_1$ and $\hat{X}_2$ for this state. How does the picture in phase space differ from either a single coherent state or their classical mixture? This state is the simplest "Schrödinger cat" — sketch what its phase-space representation reveals that is invisible in either the number-state or coherent-state basis alone.
