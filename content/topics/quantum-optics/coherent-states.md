# Coherent States: Definition, Statistics, Displacement, and Generation

## Eigenstates of the annihilation operator and minimum uncertainty states
### Definition of coherent state
Coherent states are defined as eigenstate of annihilation operator.
$$
    \hat{a}
    \Ket{\alpha}
    =
    \alpha
    \Ket{\alpha}
$$
$$
    \hat{a}^\dag
    \Bra{\alpha}
    =
    \alpha^*
    \Bra{\alpha}
$$

> **Key Intuition.** A coherent state is defined as an eigenstate of the annihilation operator. Because the annihilation operator is non-Hermitian, the eigenvalue is a complex number, encoding both the amplitude and phase of the field.

Since $\hat{a}$ is non-Hermitian, the eigenvalue **$\alpha$ is complex
number**.
Note that $\hat{n}=\hat{a}^\dag\hat{a}$ is Hermitian.

Actually, there are three equivalent ways to define the coherent state:
* Right eigenstates of the annihilation operator
* States that minimize the uncertainty relation for the two orthogonal field quadratures
* Displaced vacuum state

We will see all three in this lesson.

### Expanding coherent state with number state
Let's expand coherent state with number state.
First, we multiply identity $\sum_{n=0}^\infty \Ket{n}\Bra{n}$ to alpha.
By this operation, we can project coherent state to number state.
$$
\begin{aligned}
    \Ket{\alpha}
    &=
    \sum_{n=0}^\infty
    \Ket{n}
    \Braket{n|\alpha}
    \\&=
    \sum_{n=0}^\infty
    c_n
    \Ket{n}
\end{aligned}
$$
We replaced the $\Braket{n|\alpha}=c_n$.

Apply operator to $\Ket{\alpha}$ resuts to
$$
\begin{aligned}
    \hat{a}
    \Ket{\alpha}
    &=
    \alpha
    \Ket{\alpha}
    \\&=
    \sum_{n=0}^\infty
    \alpha
    c_n
    \Ket{n}
\end{aligned}
$$
This follows directly from the definition.
Multiplying $\hat{a}$ from left gives
$$
\begin{aligned}
    \hat{a}
    \sum_{n=0}^\infty
    c_n
    \Ket{n}
    &=
    \sum_{n=1}^\infty
    \sqrt{n}
    c_n
    \Ket{n-1}
    \\&=
    \sum_{n=0}^\infty
    \sqrt{n+1}
    c_{n+1}
    \Ket{n}
\end{aligned}
$$
Coefficient of number state is equal.
$$
    \sqrt{n+1}
    c_{n+1}
    =
    \alpha
    c_n
$$
Thus, we can decide the $c_n$ recursively.
$$
\begin{aligned}
    c_n
    &=
    \frac
    {\alpha}
    {\sqrt{n}}
    c_{n-1}
    \\&=
    \frac
    {\alpha^2}
    {\sqrt{n(n-1)}}
    c_{n-2}
    \\&=
    \cdots
    \\&=
    \frac
    {\alpha^n}
    {\sqrt{n!}}
    c_{0}
\end{aligned}
$$
We almost complete expanding coherent state with number state.
$$
    \Ket{\alpha}
    =
    c_0
    \sum_{n=0}^\infty
    \frac
    {\alpha^n}
    {\sqrt{n!}}
    \Ket{n}
$$
We still need to decide $c_0$. We can do this from normalization condition.
$$
\begin{aligned}
    1
    &=
    \Braket{\alpha|\alpha}
    \\&=
    \left| c_0 \right|^2
    \sum_{n, n^\prime}
    \frac
    {\left| \alpha \right|^{2n}}
    {n!}
    \Braket{n|n^\prime}
    \left| c_0 \right|^2
    \\&=
    \sum_{n=0}^\infty
    \frac
    {\left| \alpha \right|^{2n}}
    {n!}
    \\&=
    \left| c_0 \right|^2
    e^
    {\left| \alpha \right|^{2}}
\end{aligned}
$$
Thus,
$$
    \left| c_0 \right|^2
    =
    e^
    {- \left| \alpha \right|^{2}}
$$
$$
    c_0
    =
    e^
    {- \left| \alpha \right|^{2}/2}
$$
Finally coherent coherent state of number state basis is
$$
    \Ket{\alpha}
    =
    e^
    {- \left| \alpha \right|^{2}/2}
    \sum_{n=0}^\infty
    \frac
    {\alpha^n}
    {\sqrt{n!}}
    \Ket{n}
$$

> **Key Intuition.** This expansion shows that a coherent state is a specific superposition of all number states, weighted by a Poissonian distribution. The prefactor ensures normalization, and the complex parameter $\alpha$ fully determines the state.

## Electric field averages and photon statistics

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

### Physical meaning of $\alpha$ and photon statistics
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

## Displacement operator and generation

### Displaced vacuum states
The displacement operator is defined as
$$
    \hat{D}(\alpha)
    =
    e^{\alpha \hat{a}^\dag - \alpha^* \hat{a}}
$$
And coherent state given as
$$
    \Ket{\alpha}
    =
    \hat{D}
    \Ket{0}
$$

The displacement operator is a rigid translation in phase space -- it shifts the center of the state without changing the shape or size of the uncertainty region. This is why a coherent state has exactly the same quadrature variances as the vacuum.

### Wave packets and time evolution
$$
    \hat{q}
    \Ket{q}
    =
    q
    \Ket{q}
$$
$$
    \left| \psi_\alpha (q) \right|
    =
    \left| \Braket{q|\alpha} \right|^2
$$
This is Gaussian.
Quantum fluctuation is constant with time.


### Generation of coherent states
Coherent state can be generated by classically oscillating thing.

## Algebraic properties of coherent states
### Review of number state
* Orthogonal $\Braket{n|n^\prime} = \delta_{nn^\prime}$
* Complete $\sum_{n=0}^\infty\Ket{n}\Bra{n} = 1$
### Properties of coherent state
##### Orthogonality
Let's check the orthogonality.
$$
\begin{aligned}
    \Braket{\beta|\alpha}
    &=
    e^{-\frac{|\alpha|^2}{2}}
    e^{-\frac{|\beta|^2}{2}}
    \sum_{n=0}^\infty
    \frac
    { {\beta^*}^n \alpha^n }
    {n!}
    \\&=
    e^{-\frac{|\alpha|^2}{2}}
    e^{-\frac{|\beta|^2}{2}}
    e^{
    {\beta^* \alpha}
    }
    \\&=
    e^{\frac{1}{2}(\beta^*\alpha - \beta\alpha^*)}
    e^{-\frac{1}{2}|\beta^2 - \alpha^2|}
    \\& \neq
    0
\end{aligned}
$$
So coherent states are not orthogonal.
If $\left| \beta - \alpha \right|$ is large, they are nearly orthogonal.
##### Completeness
$$
\begin{aligned}
    \int \Ket{\alpha} \Bra{\alpha} \mathrm{d}^2 \alpha
    &=
    \int \mathrm{d}^2 \alpha
    e^{-|\alpha|^2}
    \sum_{n, n^\prime}
    \frac{\alpha^n {\alpha^*}^n}{\sqrt{n!m!}}
    \Ket{n} \Bra{m}
    \\&=
    \sum_{n, m}
    \frac{\Ket{n} \Bra{m}}{\sqrt{n!m!}}
    \int \mathrm{d}^2 \alpha
    e^{-|\alpha|^2}
    \alpha^n {\alpha^*}^n
    \\&=
    \sum_{n, m}
    \frac{\Ket{n} \Bra{m}}{\sqrt{n!m!}}
    \int \int \mathrm{d}r \mathrm{d}\theta
    r
    e^{-r^2}
    r^{n+m}
    e^{i\theta(n-m)}
    \\&=
    \sum_{n}
    \frac{\Ket{n} \Bra{n}}{n!}
    \int \mathrm{d}r
    r^{2n+1}
    e^{-r^2}
    2\pi
    \\&=
    \sum_{n}
    \Ket{n} \Bra{n}
    \\&=
    \pi
\end{aligned}
$$
We changed the variable $\alpha=re^{i\theta}$,
so integration became $\mathrm{d}^2\alpha = r\mathrm{d}r\mathrm{d}\theta$.
Also, we used the Dirac delta function
$$
    \int_0^{2\pi}
    \mathrm{d}\theta
    e^{i\theta (n-m)}
    =
    2\pi
    \delta_{nm}
$$

Thus, coherent states is non-orthogonal states and it has a kind of overcomplete property.

## Big Ideas

* A coherent state is the eigenstate of the annihilation operator $\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$, and the complex eigenvalue $\alpha$ encodes both the field amplitude and phase.
* Expanding $|\alpha\rangle$ in the number-state basis reveals Poissonian photon statistics: the probability of finding $n$ photons follows $P_n = e^{-|\alpha|^2}|\alpha|^{2n}/n!$, with mean equal to variance.
* The expectation value of the electric field in a coherent state is a perfect sine wave with amplitude set by $|\alpha|$ -- exactly what you would call "classical" light.
* The field fluctuations (variance) are independent of $\alpha$: a coherent state with a million photons has the same noise floor as the vacuum.
* Coherent states can be generated from the vacuum by the displacement operator $\hat{D}(\alpha) = e^{\alpha\hat{a}^\dagger - \alpha^*\hat{a}}$ -- physically, this corresponds to driving the cavity with a classical oscillating current.
* Coherent states are not orthogonal: $|\langle\beta|\alpha\rangle|^2 = e^{-|\beta-\alpha|^2}$, so two coherent states are nearly orthogonal only when they are well separated in phase space.
* Despite the non-orthogonality, the coherent states form an overcomplete set with $\frac{1}{\pi}\int d^2\alpha\, |\alpha\rangle\langle\alpha| = \hat{I}$, which makes them enormously useful as a basis for calculations.

## What Comes Next

We now have a complete algebraic and geometric picture of coherent states. The natural next step is to visualize quantum states in general -- not just coherent states but thermal states, number states, and stranger beasts -- using phase-space pictures. The next lesson introduces the idea of representing a quantum state as a distribution over the complex phase-space plane, bringing us to the P function, the Wigner function, and the Husimi Q function.

## Check Your Understanding

1. The annihilation operator $\hat{a}$ is non-Hermitian, meaning its eigenvalues are complex. Why does this make physical sense for a coherent state, given that $\alpha$ encodes both the amplitude and phase of an oscillating field?
2. For a coherent state $|\alpha\rangle$, the electric field variance $\langle(\Delta E_x)^2\rangle_\alpha$ equals the vacuum value and is independent of $\alpha$. Why does adding more photons (increasing $|\alpha|$) not reduce the noise, and what would you have to do to get below-vacuum noise?
3. The calculation shows $\langle(\Delta \hat{X}_1)^2\rangle_\alpha = \frac{1}{4}$ regardless of $\alpha$. Without computing, predict $\langle(\Delta \hat{X}_2)^2\rangle_\alpha$ and explain your reasoning. (Hint: the displacement operator is a rigid translation in phase space -- it cannot change the shape or size of the uncertainty circle.)
4. Coherent states are overcomplete: the resolution of identity requires dividing by $\pi$, so $\frac{1}{\pi}\int d^2\alpha\, |\alpha\rangle\langle\alpha| = \hat{I}$. Why does overcompleteness mean you cannot expand an arbitrary state in the coherent basis in a unique way, and is this a problem in practice?

## Challenge

A number state $|n\rangle$ and a coherent state $|\alpha\rangle$ with $|\alpha|^2 = n$ both have the same mean photon number. Compare their photon number distributions. Show explicitly that the variance of the photon number is $n$ for the coherent state and $0$ for the number state. Then verify the claim that the photon-number variance of a coherent state equals the mean: $\langle(\Delta n)^2\rangle = \bar{n} = |\alpha|^2$, and compute the same variance for a thermal state with density matrix $\hat{\rho}_\text{th}$, where the photon number distribution is geometric. Show that $\langle(\Delta n)^2\rangle_\text{th} = \bar{n} + \bar{n}^2$. What does the extra $\bar{n}^2$ term say physically about how thermal photons cluster compared to coherent photons?
