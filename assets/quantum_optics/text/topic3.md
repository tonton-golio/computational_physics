__Topic 3 keywords__
- Coherent states
- Displacement operator
- Generation of coherent states

# __Readings__
Ch. 3.1-5

# __3.1 Eigenstates of the annihilation operator and minimum uncertainty states__
#### Definition of coherent state
Coherent states are defined as eigenstate of annihilation operator.
$$
    \boxed{
        \hat{a}
        \Ket{\alpha}
        =
        \alpha
        \Ket{\alpha}
    }
$$
$$
    \boxed{
        \hat{a}^\dag
        \Bra{\alpha}
        =
        \alpha^*
        \Bra{\alpha}
    }
$$
As you know that $\hat{a}$ is non-Hermitian so the eigenvalue **$\alpha$ is complex 
number**.
By the way $\hat{n}=\hat{a}^\dag\hat{a}$ is Hermitian.

#### Expanding coherent state with number state
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
You know, this is just a definition.
Multiplying $\hat{a}$ from left results
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
\boxed{
    \Ket{\alpha}
    =
    e^
    {- \left| \alpha \right|^{2}/2}
    \sum_{n=0}^\infty
    \frac
    {\alpha^n}
    {\sqrt{n!}}
    \Ket{n}
}
$$

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


# __3.2 Displaced vacuum states__
Actually, there are three ways to define the coherent state.
- Right eigenstates of the annihilation operator
- States that minimize the uncertainty relation for the two orthogonal field quadratures (which we didn't do)
- Displaced vacuum state

This displacing method is closely related to a mechanism for 
generating the coherent state from classical currents.

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

# __3.3 Wave packets and time evolution__
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


# __3.4 Generation of coherent states__
Coherent state can be generated by classically oscillating thing.

# __3.5 More on the properties of coherent states__
#### Review of number state
- Orthogonal $\Braket{n|n^\prime} = \delta_{nn^\prime}$
- Complete $\sum_{n=0}^\infty\Ket{n}\Bra{n} = 1$
#### Properties of coherent state
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
    {{\beta^*}^n \alpha^n}
    {n!}
    \\&= 
    e^{-\frac{|\alpha|^2}{2}}
    e^{-\frac{|\beta|^2}{2}}
    e^{
    {{\beta^*} \alpha}
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
    \sum_{n, m}
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

