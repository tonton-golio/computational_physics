# Mean-Field Results

Every spin feels an average field from all its neighbors, like a crowd where everyone faces the same way because they see everyone else doing it. We derived the mean-field Hamiltonian. Now let's see what comes out.

The punchline first:

## MF: Z, m, Tc, F & critical exponents

$$
\begin{align*}
    Z_{\mathrm{MF}}
    &=
    \exp
    \left(
        - \beta\frac{J N z}{2} m^2
    \right)
    \left[
    2\cosh
    \left(
        \beta J z m + \beta h
    \right)
    \right]^N
    \\
    m
    &= \langle s_i \rangle
    =
    \tanh
    \left(
        \beta J z m + \beta h
    \right)
    \\
    T_c &= \frac{Jz}{k_B}
    \\
    F_{\mathrm{MF}} &=
    \frac{JNz}{2}m^2
    - \frac{N}{\beta} \ln
    \left[
        2 \cosh \left(\beta Jzm + \beta h \right)
    \right]
\end{align*}
$$

Now let's earn these results.

## Big Ideas

* The self-consistency equation $m = \tanh(\beta J z m)$ is a fixed-point equation -- the magnetization is whatever value produces that same magnetization as an effective field.
* The free energy landscape changes shape at $T_c$: a single bowl becomes a double well, forcing the system to spontaneously break symmetry.
* Mean-field critical exponents ($\beta = 1/2$, $\gamma = 1$) are universal across all mean-field theories -- right above four dimensions, wrong below, because mean-field always ignores the correlations that matter most near the critical point.

## MF: Z, m, Tc, F & critical exponents (derivation)

### Mean-field partition function

The partition function sums over all $2^N$ spin configurations. Plug in the mean-field Hamiltonian and the crucial thing happens -- the sum factorizes:
$$
\begin{align*}
    Z_{\mathrm{MF}}
    &=
    \exp
    \left(
        - \beta\frac{J N z}{2} m^2
    \right)
    \left[
    2\cosh
    \left(
        \beta J z m + \beta h
    \right)
    \right]^N.
\end{align*}
$$

Each spin contributes a factor of $2\cosh(\beta(Jzm + h))$, and since they're all independent in mean-field, these factors multiply. That's why mean-field is solvable: the terrible many-body problem becomes $N$ copies of a one-body problem.

### Self-consistent equation of magnetization

The magnetization per spin works out to:
$$
    m
    =
    \tanh(\beta J z m + \beta h).
$$

This is the **self-consistency equation**: $m$ appears on both sides! With no external field ($h = 0$):
$$
    m = \tanh(\beta J z m).
$$

You can't solve this in closed form, but you can understand it graphically. Plot $y = m$ and $y = \tanh(\beta Jzm)$ and look for intersections:

* At **high temperature** ($\beta Jz < 1$), the $\tanh$ curve is too shallow -- it only crosses $y = m$ at $m = 0$. Zero magnetization. Disordered.
* At **low temperature** ($\beta Jz > 1$), the $\tanh$ curve is steep enough to cross at three points: $m = 0$ and two symmetric values $\pm m_0$. The system can be magnetized!

### Critical temperature

The transition happens when the slope of $\tanh(\beta Jzm)$ at $m = 0$ equals 1. Since $\tanh(x) \approx x$ for small $x$:
$$
    T_c = \frac{Jz}{k_\mathrm{B}}.
$$

Makes sense: stronger coupling $J$ means higher $T_c$ (harder to disorder), more neighbors $z$ means higher $T_c$ (more peer pressure). But mean-field predicts a phase transition in *every* dimension, including 1D -- and that's wrong. Mean-field overestimates the tendency to order because it ignores correlated fluctuations.

### Free energy

From the partition function:
$$
\begin{align*}
    F_\mathrm{MF}
    &=
    \frac{JNz}{2}m^2
    - \frac{N}{\beta} \ln
    \left[
        2 \cosh (\beta Jzm + \beta h)
    \right].
\end{align*}
$$

### Landau expansion and the free energy landscape

To see what happens near the transition, expand the free energy in powers of $m$. Introduce $\theta = T/T_c$ and the dimensionless free energy $f_\mathrm{MF} = F_\mathrm{MF}/(JzN)$:
$$
    f_\mathrm{MF}
    =
    \frac{1}{2}m^2\left(1 - \frac{1}{\theta}\right)
    + \frac{1}{12}\frac{m^4}{\theta^3}
    - \theta \ln 2
    + \mathcal{O}(m^6).
$$

Plot that little curve in your mind. Above $T_c$ it's a single well at $m=0$. Below $T_c$ it splits into two wells. That splitting is the birth of magnetism -- and you just saw it in algebra.

Setting $\partial f/\partial m = 0$ gives $m = 0$ always, plus (for $T < T_c$):
$$
    m^2 = 3(1 - \theta)\theta^2.
$$

Check the second derivative: above $T_c$, $m = 0$ is a minimum (stable). Below $T_c$, $m = 0$ becomes a maximum (unstable!) and two new minima appear at $\pm m_0$. The system *must* choose one -- that's **spontaneous symmetry breaking**.

[[simulation landau-free-energy]]

### Mean-field critical exponents

Near $T_c$:

**Order parameter exponent $\beta$:** From $m^2 \sim -3t$ we get $m \sim |t|^{1/2}$, so $\beta = 1/2$.

**Susceptibility exponent $\gamma$:** $\chi \sim |t|^{-1}$, so $\gamma = 1$.

**Heat capacity exponent $\alpha$:** Finite discontinuity at $T_c$, so $\alpha = 0$.

These mean-field exponents are the same for all systems -- because mean-field ignores spatial details. They're exact above the upper critical dimension $d_c = 4$, but only approximate below.

## What Comes Next

Mean-field theory is analytically tractable but ignores spatial fluctuations. For a reality check, consider the opposite limit: a one-dimensional chain where each spin talks only to its two nearest neighbors. The [Transfer Matrix](transferMatrix) method solves this exactly -- no approximations -- using a beautiful linear-algebra trick where the partition function becomes the trace of a matrix power. The exact solution reveals something mean-field gets completely wrong: the 1D Ising model has *no* phase transition at any finite temperature.

## Check Your Understanding

1. The self-consistency equation has the solution $m = 0$ for all temperatures, plus two nonzero solutions for $T < T_c$. Why is $m = 0$ unstable below $T_c$, even though it's always a valid solution?
2. The mean-field order parameter exponent is $\beta = 1/2$. The exact 2D Ising exponent is $\beta = 1/8$. What does this tell you about how fluctuations affect the onset of magnetization?

## Challenge

The Landau free energy is a fourth-order polynomial in $m$. Now imagine adding a sixth-order term: $f \sim a_2 t \, m^2 + a_4 m^4 + a_6 m^6$, but allowing $a_4$ to become negative. Show that when $a_4 < 0$, the transition becomes **first-order** (discontinuous jump in $m$) rather than second-order (continuous onset). Sketch the free energy landscape at temperatures above, at, and below the transition for both signs of $a_4$. What real physical systems exhibit first-order versus second-order transitions?
