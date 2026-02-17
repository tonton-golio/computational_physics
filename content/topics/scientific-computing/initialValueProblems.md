# Initial Value Problems

## Introduction

An **initial value problem** (IVP) asks us to find $y(t)$ satisfying an ordinary differential equation $y' = f(t, y)$ with a given initial condition $y(t_0) = y_0$. Most ODE systems arising in physics and engineering cannot be solved analytically, so we rely on numerical methods that advance the solution step by step through discrete time increments.

The key challenges are **accuracy** (how close the numerical solution tracks the true one), **stability** (whether errors grow or decay), and **efficiency** (how much computation is needed for a given accuracy). These three concerns drive the design of all IVP solvers.

## Euler's Method

The simplest approach discretizes the derivative directly. Given a step size $h$, **forward Euler** updates the solution as

$$
y_{n+1} = y_n + h f(t_n, y_n).
$$

This is a first-order method: the local truncation error per step is $O(h^2)$, giving a global error of $O(h)$. While easy to implement, Euler's method requires very small step sizes for acceptable accuracy and is unstable for stiff problems.

**Backward Euler** replaces the right-hand side with the function evaluated at the new time:

$$
y_{n+1} = y_n + h f(t_{n+1}, y_{n+1}).
$$

This is an **implicit** method since $y_{n+1}$ appears on both sides and generally requires solving a nonlinear equation at each step. The payoff is greatly improved stability for stiff systems.

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

RK4 has a local truncation error of $O(h^5)$ and a global error of $O(h^4)$, offering an excellent balance of accuracy and simplicity. It is the workhorse method for non-stiff problems.

A general $s$-stage Runge-Kutta method is defined by a **Butcher tableau** specifying the coefficients $a_{ij}$, $b_i$, and $c_i$. The method is explicit if $a_{ij} = 0$ for $j \geq i$ and implicit otherwise.

## Multistep Methods

Rather than using multiple evaluations within a single step, **multistep methods** use information from several previous steps. An $s$-step **Adams-Bashforth** method (explicit) takes the form

$$
y_{n+1} = y_n + h \sum_{j=0}^{s-1} \beta_j f(t_{n-j}, y_{n-j}).
$$

The two-step version ($s = 2$) is

$$
y_{n+1} = y_n + \frac{h}{2}\bigl(3f_n - f_{n-1}\bigr).
$$

**Adams-Moulton** methods are the implicit counterparts and include $f_{n+1}$ on the right-hand side. In practice, a **predictor-corrector** scheme uses Adams-Bashforth to predict and Adams-Moulton to correct, combining the efficiency of explicit methods with the improved stability of implicit ones.

## Stability Analysis

To study stability, we apply numerical methods to the **test equation** $y' = \lambda y$ where $\lambda \in \mathbb{C}$. The **stability region** of a method is the set of values $h\lambda$ for which the numerical solution does not grow without bound.

For forward Euler, the stability region is the disk $|1 + h\lambda| \leq 1$ in the complex plane. For backward Euler, the stability region is the complement of $|1 - h\lambda| < 1$, which includes the entire left half-plane. A method is **A-stable** if its stability region contains the entire left half-plane $\operatorname{Re}(h\lambda) \leq 0$.

Explicit methods have bounded stability regions, so step sizes must satisfy $|h\lambda| < C$ for some constant $C$. For stiff problems (where eigenvalues of the Jacobian span many orders of magnitude), this restriction makes explicit methods impractical.

## Stiffness

A problem is **stiff** when it contains both fast-decaying and slow-varying components. The fast components force explicit methods to use tiny step sizes even when the solution is smooth. Classic examples include chemical reaction kinetics, circuit simulations, and discretized parabolic PDEs.

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

where $p$ is the order of the lower-order method.

## Systems of ODEs and Higher-Order Equations

Any higher-order ODE can be rewritten as a first-order system. For example, Newton's second law $m\ddot{x} = F(x, \dot{x}, t)$ becomes

$$
\frac{d}{dt}\begin{pmatrix} x \\ v \end{pmatrix} = \begin{pmatrix} v \\ F(x, v, t)/m \end{pmatrix}.
$$

All the methods above apply to vector-valued systems $\mathbf{y}' = \mathbf{f}(t, \mathbf{y})$ with no modifications other than replacing scalar operations with vector ones.

For Hamiltonian systems (where energy conservation matters), **symplectic integrators** like the Stormer-Verlet method preserve the geometric structure of phase space and give bounded energy errors over long times, unlike general-purpose methods which may exhibit secular energy drift.

[[simulation reaction-diffusion]]

## Summary

| Method | Order | Type | Best for |
|--------|-------|------|----------|
| Forward Euler | 1 | Explicit | Prototyping, education |
| Backward Euler | 1 | Implicit | Stiff problems, stability |
| RK4 | 4 | Explicit | Non-stiff problems |
| Dormand-Prince | 4(5) | Explicit, adaptive | General non-stiff |
| BDF (1-5) | 1-5 | Implicit, multistep | Stiff problems |
| Stormer-Verlet | 2 | Symplectic | Hamiltonian systems |
