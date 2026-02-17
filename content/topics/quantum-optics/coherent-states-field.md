# Coherent States: Field Averages and Photon Statistics

#### Electric field from coherent state viewpoint
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
    - \mathcal{E}_0^2
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
    - \mathcal{E}_0^2
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
    - \mathcal{E}_0^2
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
        - (1 + \hat{a}^\dag\hat{a})
        - \hat{a}^\dag\hat{a}
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

#### Quadrature operators from coherent state viewpoint
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
        - \hat{a} \hat{a}^\dag
        - \hat{a}^\dag \hat{a}
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
        - 1
        - \hat{a}^\dag \hat{a}
        - \hat{a}^\dag \hat{a}
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
        - 1
        - 2\alpha^* \alpha
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

#### Physical meaning of $\alpha$
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
