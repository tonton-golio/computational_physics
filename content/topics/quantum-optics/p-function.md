# Glauber-Sudarshan P Function

## Density operators and phase-space probability distributions
### $P$ function
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
    \operatorname{Tr}
    \hat{\rho}
    \\&=
    \operatorname{Tr}
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

### Calculate $P$ function
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
    {-u^* \alpha}
    }
    \cdot
    e^{-\frac{|\alpha|^2}{2}}
    e^{-\frac{|u|^2}{2}}
    e^{
    {\alpha^* u}
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
    {\beta^* \alpha}
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

### Example of $P$ function
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
    {-u^* \beta}
    }
    \cdot
    e^{-\frac{|\beta|^2}{2}}
    e^{-\frac{|u|^2}{2}}
    e^{
    {\beta^* u}
    }
    \\&=
    e^{-|\beta|^2}
    e^{-|u|^2}
    e^{
    * u^* \beta
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
    * u^* \beta
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

### Optical equivalence theorem
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
        \hat{G}^{(N)}
        \left(\hat{a}, \hat{a}^{\dagger}\right)
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

## Big Ideas

* The Glauber-Sudarshan P function represents any quantum state as a weighted average over coherent states — the weights $P(\alpha)$ play the role of a probability distribution on phase space.
* For a coherent state itself, $P(\alpha) = \delta^2(\alpha - \beta)$: a single point in phase space, the maximally classical picture.
* The optical equivalence theorem is a profound shortcut: expectation values of normally ordered operators can be computed by replacing operators with complex numbers and integrating against $P(\alpha)$, just as in classical statistical mechanics.
* The P function is what separates classical from non-classical light: if $P(\alpha)$ cannot be interpreted as a genuine probability distribution (i.e., it goes negative or is more singular than a delta function), the state has no classical analogue.

## What Comes Next

The P function is the most "classical-friendly" of the three phase-space distributions, but it becomes highly singular or negative for non-classical states like number states. The next lesson introduces the Wigner function, which is smoother and more symmetric, and which provides one of the most direct visual signatures of non-classicality: regions where the Wigner function is negative.

## Check Your Understanding

1. The P function for a coherent state is a delta function, which is a perfectly well-defined probability distribution (concentrated at one point). What must be true about the P function for a state to be considered "classical," and why does thermal light qualify?
2. The optical equivalence theorem says that expectation values of normally ordered operators can be computed by treating $\hat{a}$ as a complex number $\alpha$ weighted by $P(\alpha)$. Why does the normal ordering matter — what goes wrong if you try the same trick with an anti-normally ordered operator?
3. The P function is obtained from $\langle -u|\hat{\rho}|u\rangle$ by a Fourier-like transform. Why is it natural that the density matrix element between two coherent states (rather than, say, two Fock states) appears in this formula?

## Challenge

Compute the P function for a number state $|n\rangle$ with $n \geq 1$. You will find that the result involves derivatives of the delta function and is therefore more singular than a delta function itself. Show that this P function cannot be a probability distribution, and explain what this mathematical conclusion means physically about the photon statistics of a number state compared to any classical light source.
