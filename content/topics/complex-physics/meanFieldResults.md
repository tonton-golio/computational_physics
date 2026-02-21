# Mean-Field Results

Imagine every spin in our magnet feels an average field from all its neighbors, like a crowd where everyone is trying to face the same way because they see everyone else doing it. We derived the mean-field Hamiltonian in the previous section. Now let us see what comes out of it.

The punchline first — here are the key results:

## MF: Z, m, Tc, F & critical exponents

$$
\begin{align*}
    Z_{\mathrm{MF}}
    &=
    \exp
    \left(
        * \beta\frac{J N z}{2} m^2
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
    * \frac{N}{\beta} \ln
    \left[
        2 \cosh \left(\beta Jzm + \beta h \right)
    \right]
\end{align*}
$$

Now let us earn these results.

## MF: Z, m, Tc, F & critical exponents (derivation)

### Mean-field partition function

The partition function of the Ising model is a sum over all $2^N$ spin configurations:
$$
\begin{align*}
    Z
    &=
    \sum_{\{s_i\}}
    e^{-\beta \mathcal{H}(\{s_i\})}
    =
    \sum_{n}^{2^N}
    e^{-\beta E_n}.
\end{align*}
$$

Now we plug in the mean-field Hamiltonian. The crucial simplification is that the spins have decoupled — the sum over all configurations *factorizes*:
$$
\begin{align*}
    Z_{\mathrm{MF}}
    &=
    \sum_{s_1 = \pm 1} \cdots \sum_{s_N = \pm 1}
    \exp
    \left[
        * \beta\frac{J N z}{2} m^2
        + \beta \left( J z m + h \right) \sum_i s_i
    \right]
    \\&=
    \exp
    \left(
        * \beta\frac{J N z}{2} m^2
    \right)
    \prod_{i=1}^N
    \left[
    \sum_{s_i = \pm 1}
    \exp
    \left(
        \beta (J z m + h) s_i
    \right)
    \right]
    \\&=
    \exp
    \left(
        * \beta\frac{J N z}{2} m^2
    \right)
    \left[
    2\cosh
    \left(
        \beta J z m + \beta h
    \right)
    \right]^N.
\end{align*}
$$

Each spin contributes a factor of $2\cosh(\beta(Jzm + h))$, and since they are all independent in mean-field, these factors multiply. This is why mean-field theory is solvable: the terrible many-body problem has been reduced to $N$ copies of a one-body problem.

### Self-consistent equation of magnetization

The magnetization per spin $m = \langle s_i \rangle$ can be calculated directly:
$$
\begin{align*}
    m
    &= \langle s_i \rangle
    \\&=
    \frac{1}{Z_{\mathrm{MF}}}
    \sum_{\{s_i\}}
    s_i \, e^{-\beta \mathcal{H}_\mathrm{MF}}
    \\&=
    \frac
    {
    \displaystyle
    \sum_{s_i = \pm 1}
    s_i
    \exp\left[\beta (J z m + h) s_i\right]
    }
    {
    \displaystyle
    \sum_{s_i = \pm 1}
    \exp\left[\beta (J z m + h) s_i\right]
    }
    \\&=
    \frac{2\sinh(\beta J z m + \beta h)}{2\cosh(\beta J z m + \beta h)}
    \\&=
    \tanh(\beta J z m + \beta h).
\end{align*}
$$

This is the **self-consistency equation**: $m$ appears on both sides! The magnetization depends on itself through the mean field. With no external field ($h = 0$):
$$
    m = \tanh(\beta J z m).
$$

This equation cannot be solved in closed form, but we can understand it graphically. Plot $y = m$ and $y = \tanh(\beta Jzm)$ and look for intersections:

* At **high temperature** ($\beta Jz < 1$), the $\tanh$ curve is too shallow — it only crosses $y = m$ at $m = 0$. The only solution is zero magnetization. The system is disordered.
* At **low temperature** ($\beta Jz > 1$), the $\tanh$ curve is steep enough to cross $y = m$ at three points: $m = 0$ and two symmetric nonzero values $\pm m_0$. The system can be magnetized!

The transition between these two regimes is the **phase transition**.

### Critical temperature of mean-field approximation

The critical temperature is where the number of solutions changes from one to three. This happens when the slope of $\tanh(\beta Jzm)$ at $m = 0$ equals the slope of $y = m$ (which is 1).

Since $\tanh(x) \approx x$ for small $x$:
$$
    \left.\frac{\mathrm{d}}{\mathrm{d}m} \tanh(\beta Jzm)\right|_{m=0}
    = \beta Jz = 1.
$$

Solving for temperature:
$$
    T_c = \frac{Jz}{k_\mathrm{B}}.
$$

This makes physical sense: stronger coupling $J$ means higher $T_c$ (harder to disorder), and more neighbors $z$ means higher $T_c$ (more peer pressure to stay aligned). However, note that mean-field theory predicts a phase transition in *every* dimension, including 1D — and we know from the exact solution that there is no phase transition in the 1D Ising model. Mean-field theory overestimates the tendency to order because it ignores the correlated fluctuations that destroy order in low dimensions.

### Free energy of mean-field approximation

From the partition function:
$$
\begin{align*}
    F_\mathrm{MF}
    &=
    * \frac{1}{\beta} \ln Z_\mathrm{MF}
    \\&=
    \frac{JNz}{2}m^2
    * \frac{N}{\beta} \ln
    \left[
        2 \cosh (\beta Jzm + \beta h)
    \right].
\end{align*}
$$

As a check, we can recover the self-consistency equation by differentiating with respect to $h$:
$$
    m = -\frac{1}{N}\frac{\partial F_\mathrm{MF}}{\partial h}
    = \tanh(\beta Jzm + \beta h). \quad \checkmark
$$

### Critical exponents from Landau expansion

To understand the behavior near the transition, we expand the free energy in powers of $m$. Introduce the dimensionless temperature $\theta = T/T_c = 1/(\beta Jz)$ and dimensionless free energy $f_\mathrm{MF} = F_\mathrm{MF}/(JzN)$. For $h = 0$ and small $m/\theta$:

Using $\cosh(x) \approx 1 + x^2/2 + x^4/24 + \cdots$ and $\ln(1+x) \approx x - x^2/2 + \cdots$:
$$
\begin{align*}
    f_\mathrm{MF}
    &=
    \frac{1}{2}m^2\left(1 - \frac{1}{\theta}\right)
    + \frac{1}{12}\frac{m^4}{\theta^3}
    * \theta \ln 2
    + \mathcal{O}(m^6).
\end{align*}
$$

This is a **Landau free energy** — a polynomial in the order parameter $m$. The coefficient of $m^2$ changes sign at $\theta = 1$ (i.e., $T = T_c$). That sign change is the phase transition.

### Stability analysis: which extrema are minima?

Setting $\partial f_\mathrm{MF}/\partial m = 0$:
$$
    0 = m\left(1 - \frac{1}{\theta} + \frac{m^2}{3\theta^3}\right).
$$

This gives $m = 0$ always, plus (when $\theta < 1$, i.e., $T < T_c$):
$$
    m^2 = 3(1 - \theta)\theta^2 = -3t\,\theta^2,
$$
where $t = (T - T_c)/T_c = \theta - 1$ is the reduced temperature.

To determine which extrema are minima, we check the second derivative:
$$
    \frac{\partial^2 f_\mathrm{MF}}{\partial m^2}
    =
    \left(1 - \frac{1}{\theta}\right) + \frac{m^2}{\theta^3}
    =
    \frac{t}{1+t} + \frac{m^2}{(1+t)^3}.
$$

**Above $T_c$** ($t > 0$): The only extremum is $m = 0$, and the second derivative is $t/(1+t) > 0$. This is a minimum. The disordered state is stable.

**Below $T_c$** ($t < 0$): There are three extrema.
* At $m = 0$: the second derivative is $t/(1+t) < 0$. This is a **maximum** — the disordered state is *unstable*.
* At $m = \pm\sqrt{-3t}\,\theta$: substituting $m^2 = -3t\,\theta^2$,
$$
    \frac{\partial^2 f_\mathrm{MF}}{\partial m^2}\bigg|_{m \neq 0}
    = \frac{t}{1+t} + \frac{-3t\,\theta^2}{(1+t)^3}
    = \frac{t}{1+t}\left(1 - \frac{3\theta^2}{(1+t)^2}\right)
    = \frac{t}{1+t}(1 - 3)
    = \frac{-2t}{1+t}.
$$
Since $t < 0$, this gives $-2t/(1+t) > 0$. These are **minima**. The ordered states are stable.

So the picture is clear: above $T_c$, the free energy has a single minimum at $m = 0$ (disordered). Below $T_c$, the minimum at $m = 0$ becomes a maximum, and two new minima appear at $\pm m_0$ (ordered). The system must choose one — this is **spontaneous symmetry breaking**.

### Mean-field critical exponents

Near $T_c$ (where $|t| \ll 1$ and $\theta \approx 1$):

**Order parameter exponent $\beta$:** From $m^2 \sim -3t$ we get $m \sim |t|^{1/2}$, so $\beta = 1/2$.

**Susceptibility exponent $\gamma$:** Differentiating the self-consistency equation with respect to $h$ gives $\chi \sim |t|^{-1}$, so $\gamma = 1$.

**Heat capacity exponent $\alpha$:** The specific heat has a finite discontinuity at $T_c$, so $\alpha = 0$ (a jump, not a divergence).

These mean-field exponents ($\beta = 1/2$, $\gamma = 1$, $\alpha = 0$) are the same for all systems, regardless of dimension or lattice structure — because mean-field theory ignores all spatial details. They are exact above the upper critical dimension $d_c = 4$, but only approximate (and often wrong) in lower dimensions.

## Big Ideas

* The self-consistency equation $m = \tanh(\beta J z m)$ is a fixed-point equation — the magnetization is whatever value is consistent with itself producing that same magnetization as an effective field.
* The free energy landscape changes shape at $T_c$: a single bowl (one minimum at $m = 0$) becomes a double well (two minima at $\pm m_0$), forcing the system to spontaneously break symmetry.
* Mean-field critical exponents ($\beta = 1/2$, $\gamma = 1$) are universal across all mean-field theories — they are right above four dimensions and wrong below, for the same reason mean-field is always wrong in low dimensions.
* Every mean-field theory makes the same error: it ignores the correlations between fluctuations, which are largest precisely where the physics is most interesting — near the critical point.

## What Comes Next

Mean-field theory is analytically tractable but ignores spatial fluctuations. For a reality check, consider the opposite limit: a one-dimensional chain of spins where each spin interacts only with its two nearest neighbors. The [Transfer Matrix](transferMatrix) method solves this exactly — no approximations at all — using a beautiful linear-algebra trick where the partition function becomes the trace of a matrix power. The exact solution reveals something that mean-field theory gets completely wrong: the 1D Ising model has *no* phase transition at any finite temperature.

## Check Your Understanding

1. The self-consistency equation $m = \tanh(\beta J z m)$ has the solution $m = 0$ for all temperatures, plus two nonzero solutions for $T < T_c$. Why is the $m = 0$ solution unstable below $T_c$, even though it is always a valid solution of the equation?
2. Mean-field theory predicts $T_c = Jz/k_\mathrm{B}$. Why does having more neighbors $z$ raise $T_c$? What is the physical picture?
3. The mean-field order parameter exponent is $\beta = 1/2$, meaning $m \sim |t|^{1/2}$ near $T_c$. The exact 2D Ising exponent is $\beta = 1/8$. What does this tell you about the effect of fluctuations on how quickly the magnetization turns on?

## Challenge

The Landau free energy expansion $f \sim \frac{1}{2}(1 - 1/\theta)m^2 + \frac{1}{12\theta^3}m^4$ is a fourth-order polynomial in $m$. Now imagine adding a sixth-order term: $f \sim a_2 t \, m^2 + a_4 m^4 + a_6 m^6$, but allowing $a_4$ to become negative. Show that when $a_4 < 0$, the transition becomes **first-order** (discontinuous jump in $m$ at $T_c$) rather than second-order (continuous onset). Sketch the free energy landscape at temperatures above, at, and below the transition for both signs of $a_4$. What real physical systems exhibit first-order versus second-order transitions?
