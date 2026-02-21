# Coherent States: Field Averages and Photon Statistics

### Electric field from coherent state viewpoint
Let's consider the expectation value of electric field operator.
$$
    \hat{E}_x
    =
    i \mathcal{E}_0
    \left[
        \hat{a}
        e^{i
        \left( \vec{k}\cdot\vec{r} - \omega t\right)
        }
        -
        \text{Hermitian conjugate}
    \right]
$$
Coherent state average of electric field is
$$
    \Braket{\alpha | \hat{E}_x | \alpha}
    =
    i \mathcal{E}_0 \alpha
    e^{i
    \left( \vec{k}\cdot\vec{r} - \omega t\right)
    }
    +
    \text{complex conjugate}
$$
Coherent state average of square of electric field is
$$
\begin{aligned}
    \Braket{\alpha | \hat{E}_x^2 | \alpha}
    &=
    * \mathcal{E}_0^2
    \Braket{\alpha |
    \left(
        \hat{a}
        e^{i
        \left( \vec{k}\cdot\vec{r} - \omega t \right)
        }
        -
        \hat{a}^\dag
        e^{-i
        \left( \vec{k}\cdot\vec{r} - \omega t\right)
        }
    \right)^2
    | \alpha}
    \\&=
    * \mathcal{E}_0^2
    \Braket{\alpha |
    \left(
        \hat{a}^2
        e^{2i
        \left( \vec{k}\cdot\vec{r} - \omega t\right)
        }
        +
        {\hat{a}^\dag }^2
        e^{-2i
        \left( \vec{k}\cdot\vec{r} - \omega t\right)
        }
        -\hat{a}\hat{a}^\dag - \hat{a}^\dag\hat{a}
    \right)
    | \alpha}
    \\&=
    * \mathcal{E}_0^2
    \Braket{\alpha |
    \left(
        \hat{a}^2
        e^{2i
        \left( \vec{k}\cdot\vec{r} - \omega t\right)
        }
        +
        {\hat{a}^\dag }^2
        e^{-2i
        \left( \vec{k}\cdot\vec{r} - \omega t\right)
        }
        * (1 + \hat{a}^\dag\hat{a})
        * \hat{a}^\dag\hat{a}
    \right)
    | \alpha}
\end{aligned}
$$
Detail of calculation is left for readers ;).
By writing $\alpha = |\alpha|e^{i\theta}$, we can have sine wave which is **very classical**.

From these, the variance become
$$
    \left<
        \left( \Delta E_x \right)^2
    \right>_\alpha
    =
    \mathcal{E}_0
    =
    \frac{\hbar \omega}{2 \epsilon_0 V}
$$
which **does not depend on $\alpha$!**
And notice this is **identical to those for a vacuum state!!** (See section 2.2)

### Quadrature operators from coherent state viewpoint
We can show that fluctuation of quadrature operator also does not
depend on $\alpha$.
$$
    \hat{X}_1
    =
    \frac{\hat{a} + \hat{a}^\dag}{2}
$$
$$
    \hat{X}_2
    =
    \frac{\hat{a} - \hat{a}^\dag}{2i}
$$
$$
\begin{aligned}
    \left<
        \left(
            \Delta \hat{X}_1
        \right)^2
    \right>_\alpha
    &=
    \Braket{\alpha|
            \hat{X}_1^2
    |\alpha}
    -
    \Braket{\alpha|
            \hat{X}_1
            |\alpha}^2
    \\&=
    \Braket{\alpha|
    \left(
        \frac{\hat{a} + \hat{a}^\dag}{2}
    \right)^2
    |\alpha}
    -
    \Braket{\alpha|
        \frac{\hat{a} + \hat{a}^\dag}{2}
    |\alpha}^2
    \\&=
    \frac{1}{4}
    \Braket{\alpha|
    \left(
        \hat{a}^2
        + {\hat{a}^\dag}^2
        + \hat{a} \hat{a}^\dag
        + \hat{a}^\dag \hat{a}
    \right)
    |\alpha}
    -
    \frac{1}{4}
    \left(
        \alpha + \alpha^*
    \right)^2
    \\&=
    \frac{1}{4}
    \Braket{\alpha|
    \left(
        \hat{a}^2
        + {\hat{a}^\dag}^2
        + 1
        + \hat{a}^\dag \hat{a}
        + \hat{a}^\dag \hat{a}
    \right)
    |\alpha}
    -
    \frac{1}{4}
    \left(
        2\mathrm{Re}(\alpha)
    \right)^2
    \\&=
    \frac{1}{4}
    \left(
        \alpha^2
        + {\alpha^*}^2
        + 1
        + 2\alpha^* \alpha
    \right)
    -
    \left(
        \mathrm{Re}(\alpha)
    \right)^2
    \\&=
    \frac{1}{4}
    +
    \frac{1}{4}
    \left(
        \alpha + \alpha^*
    \right)^2
    -
    \left(
        \mathrm{Re}(\alpha)
    \right)^2
    \\&=
    \frac{1}{4}
\end{aligned}
$$
$$
\begin{aligned}
    \left<
        \left(
            \Delta \hat{X}_2
        \right)^2
    \right>_\alpha
    &=
    \Braket{\alpha|
            \hat{X}_2^2
    |\alpha}
    -
    \Braket{\alpha|
            \hat{X}_2
            |\alpha}^2
    \\&=
    \Braket{\alpha|
    \left(
        \frac{\hat{a} - \hat{a}^\dag}{2i}
    \right)^2
    |\alpha}
    -
    \Braket{\alpha|
        \frac{\hat{a} - \hat{a}^\dag}{2i}
    |\alpha}^2
    \\&=
    -
    \frac{1}{4}
    \Braket{\alpha|
    \left(
        \hat{a}^2
        + {\hat{a}^\dag}^2
        * \hat{a} \hat{a}^\dag
        * \hat{a}^\dag \hat{a}
    \right)
    |\alpha}
    +
    \frac{1}{4}
    \left(
        \alpha - \alpha^*
    \right)^2
    \\&=
    -
    \frac{1}{4}
    \Braket{\alpha|
    \left(
        \hat{a}^2
        + {\hat{a}^\dag}^2
        * 1
        * \hat{a}^\dag \hat{a}
        * \hat{a}^\dag \hat{a}
    \right)
    |\alpha}
    +
    \frac{1}{4}
    \left(
        2\mathrm{Im}(\alpha)
    \right)^2
    \\&=
    -
    \frac{1}{4}
    \left(
        \alpha^2
        + {\alpha^*}^2
        * 1
        * 2\alpha^* \alpha
    \right)
    +
    \left(
        \mathrm{Im}(\alpha)
    \right)^2
    \\&=
    \frac{1}{4}
    -
    \frac{1}{4}
    \left(
        \alpha - \alpha^*
    \right)^2
    +
    \left(
        \mathrm{Im}(\alpha)
    \right)^2
    \\&=
    \frac{1}{4}
\end{aligned}
$$
We need to use the commutation relation
$\left[ \hat{a}, \hat{a}^\dag \right] = \hat{a}\hat{a}^\dag - \hat{a}^\dag\hat{a}=1$

### Physical meaning of $\alpha$
From above the $|\alpha|$ is related to the amplitude of the field.
Thus we can show the relation of photon number and $\alpha$.
$$
\begin{aligned}
    \Braket{\alpha|\hat{n}|\alpha}
    &=
    \bar{n}
    \\&=
    \Braket{\alpha|\hat{a}^\dag \hat{a}|\alpha}
    \\&=
    \alpha^* \alpha
    \\&=
    |\alpha|^2
\end{aligned}
$$
$|\alpha|$ is average photon number of the field.

Let's calculate coherent state average of square of number operator.
$$
\begin{aligned}
    \Braket{\alpha|\hat{n}^2|\alpha}
    &=
    \Braket{\alpha| \hat{a}^\dag \hat{a} \hat{a}^\dag \hat{a} |\alpha}
    \\&=
    \Braket{\alpha| \hat{a}^\dag \left(1+\hat{a}^\dag\hat{a}\right) \hat{a} |\alpha}
    \\&=
    \Braket{\alpha| \hat{a}^\dag \hat{a} |\alpha}
    +
    \Braket{\alpha| {\hat{a}^\dag}^2 \hat{a}^2 |\alpha}
    \\&=
    |\alpha|^2
    +|\alpha|^4
\end{aligned}
$$
Thus, variance is
$$
\begin{aligned}
    \left<
        \left(
            \Delta n
        \right)^2
    \right>_\alpha
    &=
    \Braket{\alpha|\hat{n}^2|\alpha}
    -
    \Braket{\alpha|\hat{n}|\alpha}^2
    \\&=
    |\alpha|^2
    \\&=
    \bar{n}
\end{aligned}
$$
This is characteristic of Poisson process (average=variance).

This can be confirmed by Probability of detecting $n$ photons.
$$
\begin{aligned}
    P_n
    &=
    \left|
        \Braket{n|\alpha}
    \right|
    \\&=
    e^{-|\alpha|^2}
    \frac{|\alpha|^{2n}}{n!}
    \\&=
    e^{-\bar{n}}
    \frac{\bar{n}^{n}}{n!}
\end{aligned}
$$
Again this is Poisson distribution with a mean of $\bar{n}$.

## Big Ideas

* The expectation value of the electric field in a coherent state is a perfect sine wave with amplitude set by $|\alpha|$ — exactly what you would call "classical" light.
* The field fluctuations (variance) are independent of $\alpha$: a coherent state with a million photons has the same noise floor as the vacuum.
* The photon statistics are Poissonian: mean equals variance, just as if each photon arrived independently like a Geiger counter click from an uncorrelated source.
* A coherent state is simultaneously the closest thing to a classical wave and the most "boring" quantum state — it saturates the uncertainty principle and has no correlations between photons.

## What Comes Next

We know the coherent state behaves like a displaced vacuum, carrying the same intrinsic noise regardless of amplitude. The next lesson formalizes this intuition by introducing the displacement operator and showing that coherent states can be produced by driving the vacuum with a classical oscillating current. It also establishes the important algebraic properties — completeness and near-orthogonality — that make coherent states so useful as a basis.

## Check Your Understanding

1. For a coherent state $|\alpha\rangle$, the electric field variance $\langle(\Delta E_x)^2\rangle_\alpha$ equals the vacuum value and is independent of $\alpha$. Why does adding more photons (increasing $|\alpha|$) not reduce the noise, and what would you have to do to get below-vacuum noise?
2. The photon number distribution of a coherent state is Poissonian. In classical probability, a Poissonian count rate arises when events are independent and rare. What is the quantum mechanical analogue of this independence, and why do number states violate it so strongly?
3. The calculation shows $\langle(\Delta \hat{X}_1)^2\rangle_\alpha = \frac{1}{4}$ regardless of $\alpha$. Without computing, predict $\langle(\Delta \hat{X}_2)^2\rangle_\alpha$ and explain your reasoning.

## Challenge

Verify the claim that the photon-number variance of a coherent state equals the mean: $\langle(\Delta n)^2\rangle = \bar{n} = |\alpha|^2$. Then compute the same variance for a thermal state with density matrix $\hat{\rho}_\text{th}$, where the photon number distribution is geometric. Show that $\langle(\Delta n)^2\rangle_\text{th} = \bar{n} + \bar{n}^2$. What does the extra $\bar{n}^2$ term say physically about how thermal photons cluster compared to coherent photons?
