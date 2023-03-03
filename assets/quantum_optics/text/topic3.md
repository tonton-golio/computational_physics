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
        \ket{\alpha}
        =
        \alpha
        \ket{\alpha}
    }
$$
$$
    \boxed{
        \hat{a}^\dag
        \bra{\alpha}
        =
        \alpha^*
        \bra{\alpha}
    }
$$
As you know that $\hat{a}$ is non-Hermitian so the eigenvalue **$\alpha$ is complex 
number**.
By the way $\hat{n}=\hat{a}^\dag\hat{a}$ is Hermitian.

#### Expanding coherent state with number state
Let's expand coherent state with number state.
First, we multiply identity $\sum_{n=0}^\infty \ket{n}\bra{n}$ to alpha.
By this operation, we can project coherent state to number state.
$$
\begin{align*}
    \ket{\alpha}
    &=
    \sum_{n=0}^\infty 
    \ket{n}
    \braket{n|\alpha}
    \\&=
    \sum_{n=0}^\infty 
    c_n
    \ket{n}
\end{align*}
$$
We replaced the $\braket{n|\alpha}=c_n$.

Apply operator to $\ket{\alpha}$ resuts to
$$
\begin{align*}
    \hat{a}
    \ket{\alpha}
    &=
    \alpha
    \ket{\alpha}
    \\&=
    \sum_{n=0}^\infty 
    \alpha
    c_n
    \ket{n}
\end{align*}
$$
You know, this is just a definition.
Multiplying $\hat{a}$ from left results
$$
\begin{align*}
    \hat{a}
    \sum_{n=0}^\infty 
    c_n
    \ket{n}
    &=
    \sum_{n=1}^\infty 
    \sqrt{n}
    c_n
    \ket{n-1}
    \\&=
    \sum_{n=0}^\infty 
    \sqrt{n+1}
    c_{n+1}
    \ket{n}
\end{align*}
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
\begin{align*}
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
\end{align*}
$$
We almost complete expanding coherent state with number state.
$$
    \ket{\alpha}
    =
    c_0
    \sum_{n=0}^\infty 
    \frac
    {\alpha^n}
    {\sqrt{n!}}
    \ket{n}
$$
We still need to decide $c_0$. We can do this from normalization condition.
$$
\begin{align*}
    1
    &=
    \braket{\alpha|\alpha}
    \\&=
    \left| c_0 \right|^2
    \sum_{n, n^\prime}
    \frac
    {\left| \alpha \right|^{2n}}
    {n!}
    \braket{n|n^\prime}
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
\end{align*}
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
    \ket{\alpha}
    =
    e^
    {- \left| \alpha \right|^{2}/2}
    \sum_{n=0}^\infty
    \frac
    {\alpha^n}
    {\sqrt{n!}}
    \ket{n}
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
    \braket{\alpha | \hat{E}_x | \alpha}
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
\begin{align*}
    \braket{\alpha | \hat{E}_x^2 | \alpha}
    &=
    - \mathcal{E}_0^2
    \braket{\alpha | 
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
    \braket{\alpha | 
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
    \braket{\alpha | 
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
\end{align*}
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
We can show that fluctuation of quadrature operator are non-zero.
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
    \left<
        \left(
            \Delta X_i
        \right)
    \right>_\alpha
    =
    \frac{1}{4},
    ~~~~ 
    i=1, 2
$$

#### Physical meaning of $\alpha$
From above the $|\alpha|$ is related to the amplitude of the field. 
Thus we can show the relation of photon number and $\alpha$.
$$
\begin{align*}
    \braket{\alpha|\hat{n}|\alpha}
    &=
    \braket{\alpha|\hat{a}^\dag \hat{a}|\alpha}
    \\&=
    \alpha^* \alpha
    \\&=
    |\alpha|^2
\end{align*}
$$
$|\alpha|$ is average photon number of the field.

Let's calculate coherent state average of square of number operator.
$$
\begin{align*}
    \braket{\alpha|\hat{n}^2|\alpha}
    &=
    \braket{\alpha| \hat{a}^\dag \hat{a} \hat{a}^\dag \hat{a} |\alpha}
    \\&=
    \braket{\alpha| \hat{a}^\dag \left(1+\hat{a}^\dag\hat{a}\right) \hat{a} |\alpha}
    \\&=
    \braket{\alpha| \hat{a}^\dag \hat{a} |\alpha}
    +
    \braket{\alpha| {\hat{a}^\dag}^2 \hat{a}^2 |\alpha}
    \\&=
    |\alpha|^2
    +|\alpha|^4
\end{align*}
$$
Thus, variance is 
$$
    \left<
        \left(
            \Delta n
        \right)^2
    \right>_\alpha
    =
    |\alpha|^2
    =\bar{n}
$$
This is characteristic of Poisson process (average=variance).

This can be confirmed by Probability of detecting $n$ photons.
$$
\begin{align*}
    P_n
    &=
    \left|
        \braket{n|\alpha}
    \right|
    \\&=
    e^{-|\alpha|^2}
    \frac{|\alpha|^{2n}}{n!}
    \\&=
    e^{-\bar{n}}
    \frac{\bar{n}^{n}}{n!}
\end{align*}
$$
Again this is Poisson distribution with a mean of $\bar{n}$.


# __3.2 Displaced vacuum states__
Actually, there are three ways to define the coherent state.
- Right eigenstates of the annihilation operator
- States that mize the undcertainty relation for the two orthogonal field quadratures (which we didn't do :P)
- Displaced vaccum state

This displacing method is closely related to a mechanism for 
generating the coherent state from classical currents.

The displacement operator is difined as 
$$
    \hat{D}(\alpha) 
    =
    e^{\alpha \hat{a}^\dag - \alpha^* \hat{a}}
$$
And coherent state given as 
$$
    \ket{\alpha}
    =
    \hat{D}
    \ket{0}
$$

# __3.3 Wave packets and time evolution__
$$
    \hat{q}
    \ket{q}
    =
    q
    \ket{q}
$$
$$
    \left| \psi_\alpha (q) \right|
    =
    \left| \braket{q|\alpha} \right|^2
$$
This is Gaussian. 
Quantum fluctuation is constant with time.


# __3.4 Generation of coherent states__
Coherent state can be generated by classically oscillating thing.

# __3.5 More on the properties of coherent states__
#### Review of number state
- Orthogonal $\braket{n|n^\prime} = \delta_{nn^\prime}$
- Complete $\sum_{n=0}^\infty\ket{n}\bra{n} = 1$
#### Properties of coherent state
##### Orthogonality
Let's check the orthogonality.
$$
\begin{align*}
    \braket{\beta|\alpha} 
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
    e^{\frac{1}{2}|\beta^*\alpha - \beta\alpha^*|}
    \\& \neq
    0
\end{align*}
$$
So coherent states are not orthogonal.
If $\left| \beta - \alpha \right|$ is large, they are nearly orthogonal.
##### Compelteness 
$$
\begin{align*}
    \int \ket{\alpha} \bra{\alpha} \mathrm{d}^2 \alpha
    &=
    \int \mathrm{d}^2 \alpha 
    e^{-|alpha|^2} 
    \sum_{n, m}
    \frac{\alpha^n {\alpha^*}^n}{\sqrt{n!m!}}
    \ket{n} \bra{m} 
    \\&=
    \sum_{n, m}
    \frac{\ket{n} \bra{m}}{\sqrt{n!m!}}
    \int \mathrm{d}^2 \alpha 
    e^{-|alpha|^2} 
    \alpha^n {\alpha^*}^n
    \\&=
    \sum_{n, m}
    \frac{\ket{n} \bra{m}}{\sqrt{n!m!}}
    \int \int \mathrm{d}r \mathrm{d}\theta
    r
    e^{-r^2}
    r^{n+m}
    e^{i\theta(n-m)}
    \\&=
    \sum_{n}
    \frac{\ket{n} \bra{n}}{n!}
    \int \mathrm{d}r 
    r^{2n+1}
    e^{-r^2}
    2\pi
    \\&=
    \sum_{n}
    \ket{n} \bra{n}
    \\&=
    \pi
\end{align*}
$$
We changed the variable $\alpha=re^{i\theta}$, 
so integradation became $\mathrm{d}^2\alpha = r\mathrm{d}r\mathrm{d}\theta$.
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

