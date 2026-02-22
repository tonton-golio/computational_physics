# Initial Value Problems

> *The [eigenvalues](./eigen-systems) show up again here — the eigenvalues of the Jacobian determine whether your problem is stiff, and stiffness determines which solver you need.*

## Big Ideas

* Every ODE solver is a trade-off: explicit methods are cheap per step but require tiny steps near stiff eigenvalues; implicit methods pay per step to solve a nonlinear equation but can take arbitrarily large steps.
* Stiffness is a property of the equation *relative to the time scale you care about*. A problem is stiff when fast modes are irrelevant but still force your step size.
* RK4 samples the slope four times within each step and combines them with carefully chosen weights — it's not an improved guess, it's a weighted average of four local slopes.
* Adaptive step-size control is a real-time error estimator: embed two methods, measure their disagreement, and use that to set the next step.

## Introduction

An **initial value problem** asks: find $y(t)$ satisfying $y' = f(t, y)$ with $y(t_0) = y_0$. Most ODE systems in physics can't be solved analytically, so we march the solution forward step by step.

The three key concerns: **accuracy** (how close to truth), **stability** (do errors grow or decay), and **efficiency** (computation per unit accuracy).

## The Method of Lines

Here's the big picture of why ODE solvers matter beyond just ODEs.

You have a PDE — heat spreading across a metal plate. Discretize space (replace the plate with a grid) but leave time continuous. Now instead of one PDE, you have a *system of ODEs* — one per grid point.

*It's like taking a movie and turning it into photographs. Each photo is a snapshot of the temperature at all grid points, and the ODE system tells you how to get from one frame to the next.*

$$
\frac{d\mathbf{u}}{dt} = \mathbf{F}(\mathbf{u})
$$

This is why everything here matters for PDE solving too.

## Euler's Method

The simplest idea: look at the slope where you are, walk in that direction for step $h$.

$$
y_{n+1} = y_n + h f(t_n, y_n)
$$

First-order: local error $O(h^2)$, global error $O(h)$. Easy to implement, but needs tiny steps for accuracy and blows up on stiff problems.

**Backward Euler** looks at the slope where you're *going to end up*:

$$
y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})
$$

This is **implicit** — $y_{n+1}$ appears on both sides, so you need [Newton's method](./nonlinear-equations) at each step. The payoff: much better stability for stiff systems. Forward Euler might need *billions* of tiny steps to stay stable; backward Euler solves one nonlinear equation per large step. Bargain.

## Runge-Kutta Methods

Instead of sampling the slope at just one point (Euler), sample at four places within the step and take a weighted average. Like getting four weather forecasts and combining them.

$$
\begin{aligned}
k_1 &= f(t_n, y_n), \\
k_2 &= f(t_n + h/2, \; y_n + h k_1/2), \\
k_3 &= f(t_n + h/2, \; y_n + h k_2/2), \\
k_4 &= f(t_n + h, \; y_n + h k_3),
\end{aligned}
$$

$$
y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4).
$$

RK4: local error $O(h^5)$, global error $O(h^4)$. The workhorse for non-stiff problems.

Why not RK8? Diminishing returns. Going from order 1 to 4 is a huge accuracy gain. Going from 4 to 8 requires many more evaluations per step, and you usually do better with RK4 at a smaller step size.

## Multistep Methods

Instead of multiple evaluations within one step, use slopes from previous steps — your memory does the work.

Two-step Adams-Bashforth:
$$
y_{n+1} = y_n + \frac{h}{2}\bigl(3f_n - f_{n-1}\bigr)
$$

*Use slopes from the last two steps to predict where to go. No extra function evaluations needed.*

**Adams-Moulton** (implicit) plus Adams-Bashforth (explicit) gives **predictor-corrector** schemes: efficiency of explicit, stability of implicit.

## Stability Analysis

Apply any method to the test equation $y' = \lambda y$. The **stability region** is the set of $h\lambda$ values where the numerical solution doesn't blow up.

Forward Euler: the disk $|1 + h\lambda| \leq 1$. Backward Euler: the complement of $|1 - h\lambda| < 1$, covering the entire left half-plane. **A-stable** methods contain the entire left half-plane.

*Explicit methods have small safe zones. Implicit methods have huge ones. That's why implicit handles stiff problems.*

[[simulation stability-regions]]

## Stiffness

A problem is **stiff** when it has both fast-decaying and slow-varying components. The fast ones force tiny step sizes even when the solution is smooth.

*It's like driving behind a car that brakes suddenly, then cruises steadily for hours. An explicit method is a nervous driver who keeps braking long after the car ahead has settled. An implicit method relaxes and matches the cruise speed.*

The Lorenz system is a perfect example. Here's the magic — watch chaos emerge:

For stiff problems, use **implicit methods**: backward Euler, implicit Runge-Kutta, or BDF methods (the backbone of production codes like LSODA and SUNDIALS).

## Adaptive Step Size Control

Fixed step sizes are wasteful. **Adaptive methods** estimate local error and adjust $h$.

The **Dormand-Prince** method (behind `ode45` and `solve_ivp`) pairs a 4th-order and 5th-order method sharing the same evaluations. The difference estimates error:

$$
h_{\text{new}} = h \left(\frac{\text{tol}}{\text{err}}\right)^{1/(p+1)}
$$

*Error too big? Shrink the step. Error much smaller than needed? Grow the step. Automatically takes big steps when smooth, small steps when things change fast.*

## Systems of ODEs

Any higher-order ODE becomes a first-order system. Newton's second law $m\ddot{x} = F(x, \dot{x}, t)$ becomes:

$$
\frac{d}{dt}\begin{pmatrix} x \\ v \end{pmatrix} = \begin{pmatrix} v \\ F(x, v, t)/m \end{pmatrix}.
$$

All methods above work on vector systems with no changes.

For Hamiltonian systems, **symplectic integrators** (Stormer-Verlet) preserve phase-space geometry and give bounded energy errors over long times.

> **Challenge.** Here's a toy you can play with right now: implement forward Euler and RK4. Solve the simple pendulum $\theta'' + \sin\theta = 0$ (rewrite as a 2D system). Integrate for 100 periods. Plot the energy $E = \frac{1}{2}\dot\theta^2 - \cos\theta$ over time. Watch Euler's energy drift while RK4 stays stable.

## Summary

| Method | Order | Type | Best for |
|--------|-------|------|----------|
| Forward Euler | 1 | Explicit | Prototyping |
| Backward Euler | 1 | Implicit | Stiff problems |
| RK4 | 4 | Explicit | Non-stiff workhorse |
| Dormand-Prince | 4(5) | Adaptive | General non-stiff |
| BDF (1-5) | 1-5 | Implicit multistep | Stiff problems |
| Stormer-Verlet | 2 | Symplectic | Hamiltonian systems |

---

## What Comes Next

ODEs describe systems evolving at a single point. The final topic extends to *fields* — functions of both space and time. Partial differential equations require everything we've built: finite differences, linear algebra, eigenvalues, the FFT, and ODE integrators. The method of lines makes it explicit: discretize space to get ODEs, then apply these integrators.

## Check Your Understanding

1. RK4 has global error $O(h^4)$. If you halve $h$, by what factor does the error decrease, and how many more evaluations per unit time?
2. A problem has Jacobian eigenvalues from $-1$ to $-10^6$. Is it stiff? Which solver would you choose?
3. Forward Euler is unstable for $y' = -100y$ unless $h < 0.02$. Backward Euler is stable for any $h$. Why is paying the implicit cost worthwhile?

## Challenge

Implement the Lorenz system ($\sigma=10$, $\rho=28$, $\beta=8/3$) from $t=0$ to $t=50$ using (a) forward Euler with $h=0.01$, (b) RK4 with $h=0.01$, and (c) SciPy's adaptive solver. Plot the 3D butterfly trajectory. Identify the Lyapunov time from your data and explain why forward Euler's error diverges faster than the physical separation of nearby trajectories.
