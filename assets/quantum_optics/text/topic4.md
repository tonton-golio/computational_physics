__Topic 4 keywords__
- Quasi-probability distributions
- Characteristic functions
- phase-space picture

# __Readings__
Ch. 3.6-8

# __3.6 Phase-space pictures of coherent states__
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

# __3.7 Density operators and phase-space probability distributions__
#### $P$ function
As we learned in Topic 3.5, the *completeness* of coherent state is
$$
    \frac{1}{\pi}
    \int \mathrm{d}^2 \alpha
    \Ket{\alpha}
    \Bra{\alpha}
    =
    1
$$
Density operator can be represented by
$$
\begin{aligned}
    \hat{\rho}
    &=
    \left(
        \frac{1}{\pi}
        \int \mathrm{d}^2 \alpha^\prime
        \Ket{\alpha^\prime}
        \Bra{\alpha^\prime}
    \right)
    \hat{\rho}
    \left(
        \frac{1}{\pi}
        \int \mathrm{d}^2 \alpha
        \Ket{\alpha}
        \Bra{\alpha}
    \right)
    \\&=
    \int \mathrm{d}^2 \alpha
    \underbrace{
        \frac{1}{\pi^2}
        \int \mathrm{d}^2 \alpha^\prime
        \Ket{\alpha^\prime}
        \Bra{\alpha^\prime}
        \hat{\rho}
    }
    \Ket{\alpha}
    \Bra{\alpha}
\end{aligned}
$$
We want to replace the under braced part with something useful.
We introduce the Glauber-Sudarshan $P$ function.
$$
    \hat{\rho}
    =
    \int \mathrm{d}^2 \alpha
    P(\alpha)
    \Ket{\alpha}
    \Bra{\alpha}
$$

We are curious about the property of this density operator.
Let's check trace.
$$
\begin{aligned}
    1
    &=
    \mathrm{Tr} 
    \hat{\rho}
    \\&=
    \mathrm{Tr} 
    \left(
    \int \mathrm{d}^2 \alpha
    P(\alpha^\prime)
    \Ket{\alpha}
    \Bra{\alpha}
    \sum_i 
    \Ket{n}
    \Bra{n}
    \right)
    \\&=
    \int \mathrm{d}^2 \alpha
    \sum_i 
    P(\alpha^\prime)
    \Braket{n|\alpha}
    \Braket{\alpha|n}
    \\&=
    \int \mathrm{d}^2 \alpha
    P(\alpha^\prime)
    \sum_i 
    \Braket{\alpha|n}
    \Braket{n|\alpha}
    \\&=
    \int \mathrm{d}^2 \alpha
    P(\alpha^\prime)
    \Braket{\alpha|\alpha}
    \\&=
    \int \mathrm{d}^2 \alpha
    P(\alpha^\prime)
\end{aligned}
$$
Thus, $P$ function satisfies normalization condition. 
$P$ function is analogous to probability distribution function.

#### Calculate $P$ function
But how do we calculate $P(\alpha)$?
We need to use Fourier transform in the complex plane.
By using coherent states $\Ket{u}$ and $\Ket{-u}$, the density operator as $P$ 
function become
$$
\begin{aligned}
    \Braket{-u|\hat{\rho}|u}
    &=
    \int \mathrm{d}^2 \alpha
    P(\alpha)
    \Braket{-u|\alpha}
    \Braket{\alpha|u}
    \\&=
    \int \mathrm{d}^2 \alpha
    P(\alpha)
    \cdot
    e^{-\frac{|u|^2}{2}}
    e^{-\frac{|\alpha|^2}{2}}
    e^{
    {{-u^*} \alpha}
    }
    \cdot
    e^{-\frac{|\alpha|^2}{2}}
    e^{-\frac{|u|^2}{2}}
    e^{
    {{\alpha^*} u}
    }
    \\&=
    e^{-|u|^2}
    \int \mathrm{d}^2 \alpha
    P(\alpha)
    e^{-|\alpha|^2}
    e^{
    \alpha^* u
    -
    u^* \alpha
    }
\end{aligned}
$$
Remember, as we learned in Topic 3.5, 
we have kind of *orthogonal* relation of coherent states.
$$
    \Braket{\beta|\alpha} 
    =
    e^{-\frac{|\alpha|^2}{2}}
    e^{-\frac{|\beta|^2}{2}}
    e^{
    {{\beta^*} \alpha}
    }
$$
Notice that 
$$
    \alpha^* u - u^* \alpha
    =
    2i
    \left(
        \mathrm{Im}(\alpha)
        \mathrm{Re}(u)
        -
        \mathrm{Re}(\alpha)
        \mathrm{Im}(u)
    \right)
$$
Aha. We have a exponent of pure imaginary number. 
It is a time to do Fourier transform.

We define Fourier transforms in the complex plane:
$$
    g(u)
    =
    \int \mathrm{d}^2 \alpha
    \cdot
    f(\alpha) 
    e^{
    \alpha^* u
    -
    u^* \alpha
    }
$$
$$
    f(\alpha) 
    =
    \frac{1}{\pi^2}
    \int \mathrm{d}^2 u
    \cdot
    g(u)
    e^{
    u^* \alpha
    -
    \alpha^* u
    }
$$
Corresponding $f(\alpha)$ and $g(\alpha)$ are below.
$$
    g(u)
    =
    e^{\left| u \right|^2}
    \Braket{-u|\hat{\rho}|u}
$$
$$
    f(\alpha)
    =
    e^{-\left| \alpha \right|^2}
    P(\alpha)
$$
We need to do some transforms.
$$
\begin{aligned}
    f(\alpha)
    &=
    e^{-\left| \alpha \right|^2}
    P(\alpha)
    \\&=
    \frac{1}{\pi^2}
    \int \mathrm{d}^2 u
    \cdot
    g(u)
    e^{
    u^* \alpha
    -
    \alpha^* u
    }
    \\&=
    \frac{1}{\pi^2}
    \int \mathrm{d}^2 u
    \cdot
    e^{\left| u \right|^2}
    \Braket{-u|\hat{\rho}|u}
    e^{
    u^* \alpha
    -
    \alpha^* u
    }
\end{aligned}
$$
From this, 
$$
    P(\alpha)
    =
    \frac
    {e^{\left| \alpha \right|^2}}
    {\pi^2}
    \int \mathrm{d}^2 u
    \cdot
    e^{\left| u \right|^2}
    \Braket{-u|\hat{\rho}|u}
    e^{
    u^* \alpha
    -
    \alpha^* u
    }
$$

#### Example of $P$ function
Let's see the $P$ function of pure coherent state $\Ket{\beta}$.
$$
\begin{aligned}
    \Braket{-u|\hat{\rho}|u}
    &=
    \Braket{-u|\beta}
    \Braket{\beta|u}
    \\&=
    e^{-\frac{|u|^2}{2}}
    e^{-\frac{|\beta|^2}{2}}
    e^{
    {{-u^*} \beta}
    }
    \cdot
    e^{-\frac{|\beta|^2}{2}}
    e^{-\frac{|u|^2}{2}}
    e^{
    {{\beta^*} u}
    }
    \\&=
    e^{-|\beta|^2}
    e^{-|u|^2}
    e^{
    - u^* \beta
    + \beta^* u 
    }
\end{aligned}
$$
$$
\begin{aligned}
    P(\alpha)
    &=
    \frac
    {e^{\left| \alpha \right|^2}}
    {\pi^2}
    \int \mathrm{d}^2 u
    \cdot
    e^{\left| u \right|^2}
    \cdot
    e^{-|\beta|^2}
    e^{-|u|^2}
    e^{
    - u^* \beta
    + \beta^* u 
    }
    \cdot
    e^{
    u^* \alpha
    -
    \alpha^* u
    }
    \\&=
    e^{\left| \alpha \right|^2}
    e^{-\left| \beta \right|^2}
    \underbrace{
        \frac
        {1}
        {\pi^2}
        \int \mathrm{d}^2 u
        \cdot
        e^{
        u^* 
        \left( \alpha - \beta \right)
        - 
        u 
        \left( \alpha^* - \beta^* \right)
        }
    }
\end{aligned}
$$
Notice that underbraced part is Dirac delta function of complex form.
$$
\begin{aligned}
    \delta^2 \left( \alpha-\beta \right)
    &=
    \delta 
    \left[ 
        \mathrm{Re}\left(\alpha\right)
        -
        \mathrm{Re}\left(\beta\right)
    \right]
    \cdot
    \delta 
    \left[ 
        \mathrm{Im}\left(\alpha\right)
        -
        \mathrm{Im}\left(\beta\right)
    \right]
    \\&=
    \frac
    {1}
    {\pi^2}
    \int \mathrm{d}^2 u
    \cdot
    e^{
    u^* 
    \left( \alpha - \beta \right)
    - 
    u 
    \left( \alpha^* - \beta^* \right)
    }
\end{aligned}
$$
So, the $P$ function of pure coherent state $\Ket{\beta}$ is Dirac delta function.
$$
    P(\alpha)
    =
    \delta^2 \left( \alpha-\beta \right)
$$

#### Optical equivalence theorem
Suppose we have "normally ordered" function of the operators 
$\hat{a}$ and $\hat{a}^\dag$, $G^{(N)}(\hat{a}, \hat{a}^\dag)$ which is 
$$
    \hat{G}^{(N)}
    \left( \hat{a}, \hat{a}^\dag \right)
    =
    \sum_{n, m}
    C_{nm} 
    {\hat{a}^\dag}^n
    \hat{a}^m
$$
Let's have a look of coherent state average of this normally ordered operator.
$$
\begin{aligned}
    \left\langle 
        G^{(N)}\left(\hat{a}, \hat{a}^{\dagger}\right)
    \right\rangle 
    &=
    \operatorname{Tr}
    \left[
        \hat{G}^{(N)}
        \left(\hat{a}, \hat{a}^{\dagger}\right) 
        \cdot
        \hat{\rho}
    \right] 
    \\&=
    \operatorname{Tr} 
    \int P(\alpha) \mathrm{d}^2 \alpha 
    \cdot
    \sum_{n, m}
    C_{nm}
    {\hat{a}^{\dagger}}^n 
    \hat{a}^m
    \Ket{\alpha}
    \Bra{\alpha}
    \\&=
    \int P(\alpha) \mathrm{d}^2 \alpha 
    \cdot
    \sum_{n, m}
    C_{n m}
    \Braket{\alpha|
    {\hat{a}^\dag}^n 
    \hat{a}^m
    |\alpha}
    \\&=
    \int P(\alpha) \mathrm{d}^2 \alpha 
    \cdot
    \sum_{n, m}
    C_{n m} 
    \alpha^{*^n} \alpha^m 
    \\&=
    \int P(\alpha) \mathrm{d}^2 \alpha 
    \cdot
    G^{(\mathrm{N})} \left(\alpha, \alpha^*\right)
\end{aligned}
$$
We showed the **optical equivalence theorem** in the last line.
This is, interestingly, we can calculate average of 
normally ordered operator by just replacement 
$\hat{a} \rightarrow \alpha$ and $\hat{a}^\dag \rightarrow \alpha^*$


# __3.8 Characteristic functions__
