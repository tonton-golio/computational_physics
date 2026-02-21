# Initial Value Problems

> *The [eigenvalues](./eigen-systems) show up again here — the eigenvalues of the Jacobian determine whether your problem is stiff, and stiffness determines which solver you need.*

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

If backward Euler needs to solve a nonlinear equation at every step, isn't that expensive? Yes, it is! You typically need [Newton's method](./nonlinear-equations) at each time step. But for stiff problems, forward Euler would need *billions* of tiny steps to stay stable, so solving one nonlinear equation per large step is a bargain.

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

Why is RK4 so popular instead of, say, RK8? Diminishing returns. Going from order 1 to order 4 is a huge accuracy gain for moderate extra work. Going from order 4 to order 8 requires many more function evaluations per step, and you usually get better results by using RK4 with a smaller step size instead.

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

> **Challenge.** Implement forward Euler and RK4. Solve the simple pendulum $\theta'' + \sin\theta = 0$ (rewrite as a 2D system). Integrate for 100 periods. Plot the energy $E = \frac{1}{2}\dot\theta^2 - \cos\theta$ over time. Watch Euler's energy drift grow while RK4 stays much more stable.

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

## Big Ideas

* Every ODE solver is a trade-off: explicit methods are cheap per step but require tiny steps near stiff eigenvalues; implicit methods pay per step to solve a nonlinear equation, but can take arbitrarily large steps.
* Stiffness is not a property of the equation alone — it is a property of the equation relative to the time scale you care about. A problem is stiff when the fast modes are irrelevant but still force your step size.
* RK4 achieves fourth-order accuracy by sampling the slope four times within each step and combining them with carefully chosen weights — it is the weighted average of four local slopes, not just an improved guess.
* Adaptive step-size control is simply a real-time error estimator: embed two methods of different orders, measure their disagreement, and use that disagreement to set the next step size.

## What Comes Next

ODEs describe how a system evolves in time when the state lives at a single point. The final topic extends this to systems where the state is a field — a function of both space and time. Partial differential equations require everything built so far: finite differences for derivatives, linear algebra for the implicit solves, eigenvalue analysis for stability, the FFT for spectral discretization, and ODE integrators for the time-marching.

The method of lines makes this explicit: discretize space to get a system of ODEs, then apply the integrators from this lesson. Stiffness, stability, and step-size control all reappear, but now the eigenvalues come from the spatial discretization, and making the grid finer makes the problem more stiff.

## Check Your Understanding

1. Forward Euler is unstable for $y' = -100y$ unless $h < 0.02$. Backward Euler is stable for any $h$. Why does backward Euler cost more per step, and in what sense is paying that cost worthwhile?
2. RK4 has global error $O(h^4)$. If you halve the step size, by what factor does the error decrease, and how many more function evaluations does this require per unit of simulated time?
3. A problem has Jacobian eigenvalues ranging from $-1$ to $-10^6$. Is it stiff? Which solver would you choose, and why?

## Challenge

Implement the Lorenz system $\dot{x} = \sigma(y-x)$, $\dot{y} = x(\rho - z) - y$, $\dot{z} = xy - \beta z$ with $\sigma=10$, $\rho=28$, $\beta=8/3$. Integrate from $t=0$ to $t=50$ using (a) forward Euler with $h=0.01$, (b) RK4 with $h=0.01$, and (c) an adaptive solver from SciPy. Start two trajectories with initial conditions differing by $10^{-8}$ and measure the divergence over time. Plot the 3D trajectories and the separation on a log scale. Identify the Lyapunov time from your data, and explain why the error in the forward Euler solution diverges faster than the physical separation of the two trajectories.
