# Initial Value Problems
*Teaching the computer to march through time without tripping*

> *The eigenvalues from Lesson 07-08 show up again here — the eigenvalues of the Jacobian determine whether your problem is stiff, and stiffness determines which solver you need.*

## Introduction

An **initial value problem** (IVP) asks us to find $y(t)$ satisfying an ordinary differential equation $y' = f(t, y)$ with a given initial condition $y(t_0) = y_0$. Most ODE systems arising in physics and engineering cannot be solved analytically, so we rely on numerical methods that advance the solution step by step through discrete time increments.

The key challenges are **accuracy** (how close the numerical solution tracks the true one), **stability** (whether errors grow or decay), and **efficiency** (how much computation is needed for a given accuracy). These three concerns drive the design of all IVP solvers.

## The Method of Lines — turning movies into photographs

Before we dive into time-stepping, here's the big picture of why ODE solvers matter beyond just ODEs.

Imagine you have a PDE — say, heat spreading across a metal plate. That's a function of both space and time. The **method of lines** says: discretize space (replace the smooth plate with a grid of points) but leave time continuous. Now instead of one PDE, you have a *system of ODEs* — one for each grid point.

*It's like taking a movie and turning each frame into a still photo you can process one by one. Each "photo" is a snapshot of the temperature at all grid points, and the ODE system tells you how to get from one frame to the next.*

$$
\frac{d\mathbf{u}}{dt} = \mathbf{F}(\mathbf{u}),
$$

This is why everything in this lesson matters for PDE solving too — it's not just about ODEs in isolation.

## Euler's Method

The simplest approach discretizes the derivative directly. Given a step size $h$, **forward Euler** updates the solution as

$$
y_{n+1} = y_n + h f(t_n, y_n).
$$

*This says: look at the slope where you are now, and walk in that direction for a step of size $h$. Hope you don't walk off a cliff.*

This is a first-order method: the local truncation error per step is $O(h^2)$, giving a global error of $O(h)$. While easy to implement, Euler's method requires very small step sizes for acceptable accuracy and is unstable for stiff problems.

**Backward Euler** replaces the right-hand side with the function evaluated at the new time:

$$
y_{n+1} = y_n + h f(t_{n+1}, y_{n+1}).
$$

*This says: look at the slope where you're going to end up, not where you are now. This requires solving an equation (since $y_{n+1}$ appears on both sides), but the payoff is much better stability.*

This is an **implicit** method since $y_{n+1}$ appears on both sides and generally requires solving a nonlinear equation at each step. The payoff is greatly improved stability for stiff systems.

> **You might be wondering...** "If backward Euler needs to solve a nonlinear equation at every step, isn't that expensive?" Yes, it is! You typically need Newton's method (Lesson 04) at each time step. But for stiff problems, forward Euler would need *billions* of tiny steps to stay stable, so solving one nonlinear equation per large step is a bargain.

## Runge-Kutta Methods

**Runge-Kutta methods** achieve higher accuracy by evaluating $f$ at intermediate points within each step. The classical fourth-order method (RK4) computes

$$
\begin{aligned}
k_1 &= f(t_n, y_n), \\
k_2 &= f(t_n + h/2, \; y_n + h k_1/2), \\
k_3 &= f(t_n + h/2, \; y_n + h k_2/2), \\
k_4 &= f(t_n + h, \; y_n + h k_3),
\end{aligned}
$$

and then advances with

$$
y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4).
$$

*This says: instead of looking at the slope at just one point (Euler), sample the slope at four different places within the step and take a weighted average. It's like getting four weather forecasts and combining them instead of relying on just one.*

RK4 has a local truncation error of $O(h^5)$ and a global error of $O(h^4)$, offering an excellent balance of accuracy and simplicity. It is the workhorse method for non-stiff problems.

A general $s$-stage Runge-Kutta method is defined by a **Butcher tableau** specifying the coefficients $a_{ij}$, $b_i$, and $c_i$. The method is explicit if $a_{ij} = 0$ for $j \geq i$ and implicit otherwise.

> **You might be wondering...** "Why is RK4 so popular instead of, say, RK8?" Diminishing returns. Going from order 1 to order 4 is a huge accuracy gain for moderate extra work. Going from order 4 to order 8 requires many more function evaluations per step, and you usually get better results by using RK4 with a smaller step size instead.

## Multistep Methods

Rather than using multiple evaluations within a single step, **multistep methods** use information from several previous steps. An $s$-step **Adams-Bashforth** method (explicit) takes the form

$$
y_{n+1} = y_n + h \sum_{j=0}^{s-1} \beta_j f(t_{n-j}, y_{n-j}).
$$

The two-step version ($s = 2$) is

$$
y_{n+1} = y_n + \frac{h}{2}\bigl(3f_n - f_{n-1}\bigr).
$$

*This says: use the slopes from the last two steps (not just the current one) to predict where to go next. You're using your memory — no extra function evaluations needed.*

**Adams-Moulton** methods are the implicit counterparts and include $f_{n+1}$ on the right-hand side. In practice, a **predictor-corrector** scheme uses Adams-Bashforth to predict and Adams-Moulton to correct, combining the efficiency of explicit methods with the improved stability of implicit ones.

## Stability Analysis

To study stability, we apply numerical methods to the **test equation** $y' = \lambda y$ where $\lambda \in \mathbb{C}$. The **stability region** of a method is the set of values $h\lambda$ for which the numerical solution does not grow without bound.

For forward Euler, the stability region is the disk $|1 + h\lambda| \leq 1$ in the complex plane. For backward Euler, the stability region is the complement of $|1 - h\lambda| < 1$, which includes the entire left half-plane. A method is **A-stable** if its stability region contains the entire left half-plane $\operatorname{Re}(h\lambda) \leq 0$.

*This says: stability regions tell you which step sizes are safe for a given problem. Explicit methods have small, bounded safe zones. Implicit methods have huge safe zones that cover the entire left half-plane. That's why implicit methods handle stiff problems.*

Explicit methods have bounded stability regions, so step sizes must satisfy $|h\lambda| < C$ for some constant $C$. For stiff problems (where eigenvalues of the Jacobian span many orders of magnitude), this restriction makes explicit methods impractical.

## Stiffness

A problem is **stiff** when it contains both fast-decaying and slow-varying components. The fast components force explicit methods to use tiny step sizes even when the solution is smooth.

A concrete example: in chemical reaction kinetics, a fast reaction (e.g., radical recombination with rate $k_1 = 10^9$) reaches equilibrium in nanoseconds, while a slow reaction (e.g., product formation with rate $k_2 = 1$) evolves over seconds. An explicit method must resolve the fast timescale ($h < 1/k_1 \sim 10^{-9}$) even long after the fast mode has decayed, wasting billions of steps tracking a nearly-constant component.

*It's like driving behind a car that suddenly brakes, then cruises at a steady speed for hours. An explicit method is a nervous driver who keeps slamming the brakes long after the car ahead has settled down. An implicit method relaxes and matches the cruise speed.*

This connects back to conditioning: the Jacobian of a stiff system has eigenvalues spanning many orders of magnitude, giving it a large condition number. The ratio $|\lambda_\text{max}/\lambda_\text{min}|$ of the Jacobian eigenvalues is a measure of stiffness and plays a role analogous to $\text{COND}(A)$ in linear systems.

Other classic examples include circuit simulations (fast capacitive transients vs. slow resistive dynamics) and discretized parabolic PDEs (where spatial refinement introduces increasingly stiff eigenvalues).

For stiff problems, **implicit methods** (backward Euler, implicit Runge-Kutta, BDF methods) are essential. The **backward differentiation formulas** (BDF) of order $s$ are

$$
\sum_{k=0}^{s} \alpha_k y_{n+1-k} = h \beta_0 f(t_{n+1}, y_{n+1}).
$$

BDF methods up to order 5 are A-stable or nearly so and form the basis of production codes like LSODA and SUNDIALS.

## Adaptive Step Size Control

In practice, a fixed step size is wasteful: the solution may be smooth in some regions and rapidly varying in others. **Adaptive methods** estimate the local error and adjust $h$ to maintain a user-specified tolerance.

A common approach uses an **embedded Runge-Kutta pair** where two methods of different orders share the same function evaluations. The **Dormand-Prince** method (used by MATLAB's `ode45` and SciPy's `solve_ivp`) pairs a fourth-order and fifth-order method. The difference between the two solutions estimates the local error:

$$
\text{err} \approx |y_{n+1}^{(5)} - y_{n+1}^{(4)}|.
$$

If the error exceeds the tolerance, the step is rejected and retried with a smaller $h$. If the error is well below the tolerance, $h$ is increased. A standard step-size update rule is

$$
h_{\text{new}} = h \left(\frac{\text{tol}}{\text{err}}\right)^{1/(p+1)}
$$

*This says: if the error is too big, shrink the step. If the error is much smaller than needed, grow the step. The method automatically takes big steps when the solution is smooth and small steps when it's changing fast.*

where $p$ is the order of the lower-order method.

## Systems of ODEs and Higher-Order Equations

Any higher-order ODE can be rewritten as a first-order system. For example, Newton's second law $m\ddot{x} = F(x, \dot{x}, t)$ becomes

$$
\frac{d}{dt}\begin{pmatrix} x \\ v \end{pmatrix} = \begin{pmatrix} v \\ F(x, v, t)/m \end{pmatrix}.
$$

All the methods above apply to vector-valued systems $\mathbf{y}' = \mathbf{f}(t, \mathbf{y})$ with no modifications other than replacing scalar operations with vector ones.

For Hamiltonian systems (where energy conservation matters), **symplectic integrators** like the Stormer-Verlet method preserve the geometric structure of phase space and give bounded energy errors over long times, unlike general-purpose methods which may exhibit secular energy drift.

> **Challenge:** Implement forward Euler and RK4. Solve the simple pendulum $\theta'' + \sin\theta = 0$ (rewrite as a 2D system). Integrate for 100 periods. Plot the energy $E = \frac{1}{2}\dot\theta^2 - \cos\theta$ over time. Watch Euler's energy drift grow while RK4 stays much more stable.

## Summary

| Method | Order | Type | Best for | Why it feels like magic |
|--------|-------|------|----------|------------------------|
| Forward Euler | 1 | Explicit | Prototyping, education | Simplest possible idea: follow the slope |
| Backward Euler | 1 | Implicit | Stiff problems, stability | Looks ahead to where you'll be |
| RK4 | 4 | Explicit | Non-stiff problems | Four slope samples give incredible accuracy |
| Dormand-Prince | 4(5) | Explicit, adaptive | General non-stiff | Automatically adjusts step size |
| BDF (1-5) | 1-5 | Implicit, multistep | Stiff problems | Uses memory to stay stable |
| Stormer-Verlet | 2 | Symplectic | Hamiltonian systems | Conserves energy by design |

---

**What we just learned in one sentence:** ODE solvers march through time step by step, and the choice between explicit (fast but fragile) and implicit (robust but expensive) depends on whether your problem's eigenvalues make it stiff.

**What's next and why it matters:** We've been solving ODEs — equations in time. At the summit of our mountain, we'll paint the whole sky with **partial differential equations** — equations in both space and time — using everything we've learned: finite differences, linear systems, eigenvalues, FFTs, and ODE solvers all come together.
