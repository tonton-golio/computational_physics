# Partial Differential Equations

> *This is the summit. Everything comes together here: finite differences ([error analysis](./bounding-errors)), [linear systems](./linear-equations), [eigenvalues](./eigen-systems) for stability, the [FFT](./fft) for spectral methods, and [ODE solvers](./initial-value-problems) via method of lines.*

## Classification

Partial differential equations (PDEs) are classified by the nature of their highest-order terms. A second-order linear PDE in two variables has the general form

$$
A u_{xx} + 2B u_{xy} + C u_{yy} + \text{lower order terms} = 0.
$$

The **discriminant** $B^2 - AC$ determines the type:

* **Elliptic** ($B^2 - AC < 0$): e.g., Laplace equation $\nabla^2 u = 0$. Describes equilibrium states.
* **Parabolic** ($B^2 - AC = 0$): e.g., heat equation $u_t = \alpha \nabla^2 u$. Describes diffusion processes.
* **Hyperbolic** ($B^2 - AC > 0$): e.g., wave equation $u_{tt} = c^2 \nabla^2 u$. Describes wave propagation.

*Think of it this way: elliptic PDEs describe things that have settled down (a rubber sheet draped over a frame), parabolic PDEs describe things that are settling down (heat spreading until everything is the same temperature), and hyperbolic PDEs describe things that are sloshing around (waves in a pond).*

Each type requires different numerical strategies and boundary conditions. Elliptic problems need boundary values on the entire domain boundary. Parabolic and hyperbolic problems need initial conditions plus boundary conditions.

## Finite Difference Methods

The fundamental idea is to replace continuous derivatives with **discrete approximations** on a grid. For a uniform grid with spacing $\Delta x$, the standard finite difference formulas are

$$
\frac{\partial u}{\partial x} \approx \frac{u_{i+1} - u_{i-1}}{2\Delta x} \quad \text{(central, } O(\Delta x^2)\text{)},
$$

$$
\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2} \quad \text{(} O(\Delta x^2)\text{)}.
$$

*This says: estimate derivatives by looking at neighboring grid points. First derivative = slope between neighbors. Second derivative = curvature: how much the value at a point deviates from the average of its neighbors.*

These replace the PDE at each interior grid point with an algebraic equation, producing a system of equations that can be solved by linear algebra techniques.

How fine does the grid need to be? That depends on the solution. Smooth solutions need fewer points; solutions with sharp gradients or boundary layers need more. The error is $O(\Delta x^2)$ for these central differences, so halving the grid spacing quarters the error — but also quadruples the number of equations.

## The Heat Equation

The one-dimensional heat equation $u_t = \alpha u_{xx}$ is the prototypical parabolic PDE. The **FTCS scheme** (Forward Time, Central Space) discretizes as

$$
\frac{u_i^{n+1} - u_i^n}{\Delta t} = \alpha \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{\Delta x^2},
$$

giving the explicit update

$$
u_i^{n+1} = u_i^n + r(u_{i+1}^n - 2u_i^n + u_{i-1}^n),
$$

where $r = \alpha \Delta t / \Delta x^2$. This scheme is **conditionally stable**: the solution blows up unless $r \leq 1/2$ (the **CFL condition** for this problem).

*This says: the ratio $r$ controls everything. Make the time step too big relative to the grid spacing and the solution explodes. It's a direct echo of the truncation-rounding tradeoff from [bounding errors](./bounding-errors) — you can't be reckless with step sizes.*

The **implicit (backward Euler)** scheme evaluates spatial derivatives at the new time level:

$$
u_i^{n+1} - r(u_{i+1}^{n+1} - 2u_i^{n+1} + u_{i-1}^{n+1}) = u_i^n.
$$

This is unconditionally stable but requires solving a tridiagonal linear system at each time step. The **Crank-Nicolson** scheme averages the explicit and implicit forms, achieving $O(\Delta t^2, \Delta x^2)$ accuracy while remaining unconditionally stable.

## The Wave Equation

The one-dimensional wave equation $u_{tt} = c^2 u_{xx}$ is discretized with central differences in both time and space:

$$
\frac{u_i^{n+1} - 2u_i^n + u_i^{n-1}}{\Delta t^2} = c^2 \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{\Delta x^2}.
$$

This leapfrog scheme is second-order accurate and stable when the **Courant number** satisfies $C = c\Delta t / \Delta x \leq 1$. The Courant condition ensures that the numerical domain of dependence contains the physical domain of dependence.

*This says: information in the wave equation travels at speed $c$. Your time step must be small enough that the numerical scheme can "keep up" with the wave. If the wave moves faster than one grid cell per time step, the scheme misses information and explodes.*

## Elliptic Problems

For Laplace's equation $\nabla^2 u = 0$ on a 2D grid, the five-point stencil gives

$$
u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j} = 0.
$$

*This says: at equilibrium, every point equals the average of its four neighbors. If it didn't, heat would flow and the system wouldn't be in equilibrium.*

This produces a large sparse linear system $A\mathbf{u} = \mathbf{b}$ where boundary values enter through the right-hand side. Direct solvers (LU decomposition) work for moderate grids, but **iterative methods** are preferred for large problems:

* **Jacobi iteration**: Update each point from its neighbors' previous values.
* **Gauss-Seidel**: Use updated values as soon as available (faster convergence).
* **Successive over-relaxation (SOR)**: Accelerates Gauss-Seidel with a relaxation parameter $\omega$.

For Poisson's equation $\nabla^2 u = f$, the same stencil applies with $f_{i,j}$ on the right-hand side.

How big can these linear systems get? For a 2D grid with $N$ points per side, you have $N^2$ unknowns. For $N = 1000$, that's a million unknowns and a million-by-million matrix. It's sparse (mostly zeros), which is the only reason it's tractable. This is exactly where the iterative methods and sparse solvers from [linear equations](./linear-equations) become essential.

## Boundary Conditions

The three standard types are:

* **Dirichlet**: $u = g$ on $\partial\Omega$. The boundary values are known and directly substituted.
* **Neumann**: $\partial u / \partial n = g$ on $\partial\Omega$. Discretized using one-sided or ghost-point differences.
* **Robin (mixed)**: $\alpha u + \beta \partial u / \partial n = g$. Combines the above.

Ghost points outside the domain are a clean way to implement Neumann conditions: introduce a fictitious point $u_{-1}$ and use the centered difference $u_1 - u_{-1} = 2\Delta x \cdot g$ to eliminate it.

## Spectral Methods

For problems with smooth solutions and periodic boundary conditions, **spectral methods** represent the solution in a basis of trigonometric functions. The spatial derivative of $u(x) = \sum_k \hat{u}_k e^{ikx}$ is computed exactly in Fourier space:

$$
\frac{\partial u}{\partial x} \longleftrightarrow ik\hat{u}_k.
$$

*This says: differentiation in physical space becomes multiplication in Fourier space. The [FFT](./fft) makes this practical — transform, multiply by $ik$, transform back.*

The **Fast Fourier Transform** (FFT) makes this approach computationally efficient with $O(N\log N)$ operations. Spectral methods achieve exponential convergence for smooth solutions, far outperforming finite differences.

For non-periodic domains, **Chebyshev spectral methods** use Chebyshev polynomials as the basis, with clustering of grid points near boundaries to handle boundary layers.

[[simulation reaction-diffusion]]

## Stability and Convergence

There's a beautiful theorem that says: if your scheme is consistent and stable, you're guaranteed to get the right answer. It's like saying: if you follow the recipe (consistency) and don't burn the house down (stability), dinner will taste right (convergence).

The **Lax equivalence theorem** states that for a consistent finite difference scheme applied to a well-posed linear PDE, **stability is equivalent to convergence**. This means we only need to verify stability (via von Neumann analysis or matrix methods) to guarantee that the numerical solution converges to the true solution as the grid is refined.

*This is one of the most important results in numerical analysis. It transforms the hard question "does my method converge?" into the easier question "is my method stable?" — and we have powerful tools for answering the stability question.*

**Von Neumann stability analysis** substitutes a Fourier mode $u_j^n = g^n e^{ij\theta}$ into the difference scheme and requires the **amplification factor** $|g| \leq 1$ for all wave numbers $\theta$. This is the standard tool for determining CFL-type restrictions on the time step.

> **Challenge.** Implement the FTCS scheme for the 1D heat equation with initial condition $u(x,0) = \sin(\pi x)$ on $[0,1]$. Run it with $r = 0.4$ (stable) and $r = 0.6$ (unstable). Watch the stable one smoothly decay to zero while the unstable one explodes into wild oscillations. The CFL condition $r \leq 0.5$ is the thin line between success and chaos.

## Summary

| PDE Type | Example | Stable Explicit Scheme | Implicit Alternative | Why it feels like magic |
|----------|---------|----------------------|---------------------|------------------------|
| Parabolic | Heat equation | FTCS ($r \leq 1/2$) | Crank-Nicolson | Heat smooths everything out, and so does the math |
| Hyperbolic | Wave equation | Leapfrog ($C \leq 1$) | Implicit leapfrog | Waves carry information at finite speed, and the CFL condition respects that |
| Elliptic | Laplace equation | N/A (no time) | Direct solve or SOR | Every point equals the average of its neighbors — equilibrium in one sentence |

---

## Big Ideas

* PDEs come in three flavors — elliptic (equilibrium), parabolic (diffusion), hyperbolic (waves) — and each flavor demands a different numerical strategy and a different relationship between time step and grid spacing.
* The CFL condition is not a numerical artifact but a physical constraint: the numerical scheme must be able to "keep up" with information propagating through the domain at physical speeds.
* Consistency plus stability equals convergence — the Lax equivalence theorem reduces the hard question of whether your answer is right to the tractable question of whether your scheme is stable.
* Spectral methods are exponentially accurate for smooth problems because they represent the solution in a global basis (Fourier modes), while finite differences accumulate polynomial errors from truncating the Taylor series locally.

## Wrapping Up

Every tool in this topic was built for this moment. Error analysis taught you to distrust your computer just enough to use it wisely. LU decomposition and least squares gave you the machinery to solve the large linear systems that arise when you discretize a PDE. Eigenvalue analysis determines stability. The FFT powers spectral methods. And ODE integrators time-march parabolic and hyperbolic PDEs via the method of lines.

PDEs are the language in which physics writes its deepest laws: heat, sound, light, quantum wavefunctions, fluid flow, and electromagnetic fields are all PDEs. The numerical methods here are not approximations to that language — they are the tools that let you speak it on a finite machine. You now have everything you need to simulate the physical world at whatever resolution your computer can sustain. The rest is physics, imagination, and compute time.

## Check Your Understanding

1. The FTCS scheme for the heat equation is conditionally stable with $r \leq 1/2$. If you halve the grid spacing $\Delta x$ (to get better spatial accuracy), what must happen to $\Delta t$ to maintain stability, and how does this affect the total number of time steps?
2. For Laplace's equation, the five-point stencil says that every interior point equals the average of its four neighbors. Use this fact to argue that the maximum of the solution over the domain must occur on the boundary.
3. A finite difference scheme is consistent (truncation error $\to 0$ as $\Delta x, \Delta t \to 0$) but unstable. Does the Lax theorem guarantee convergence? What does it guarantee?

## Challenge

Implement the 2D heat equation $u_t = \alpha(u_{xx} + u_{yy})$ on the unit square with Dirichlet boundary conditions $u = 0$ and initial condition $u(x,y,0) = \sin(\pi x)\sin(\pi y)$. The exact solution is $u(x,y,t) = e^{-2\pi^2\alpha t}\sin(\pi x)\sin(\pi y)$. Implement both the explicit FTCS scheme and the Crank-Nicolson scheme (which requires solving a linear system at each time step). Run both with the same $\Delta x$ and compare: (a) the stability limits, (b) the error at $t = 0.1$ as a function of $\Delta t$, and (c) the wall-clock time to reach a given accuracy. At what accuracy does Crank-Nicolson's implicit cost become worthwhile?
