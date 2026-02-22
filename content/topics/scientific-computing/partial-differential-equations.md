# Partial Differential Equations

> *This is the summit. Everything comes together here: finite differences ([error analysis](./bounding-errors)), [linear systems](./linear-equations), [eigenvalues](./eigen-systems) for stability, the [FFT](./fft) for spectral methods, and [ODE solvers](./initial-value-problems) via method of lines.*

## Big Ideas

* PDEs come in three flavors — elliptic (equilibrium), parabolic (diffusion), hyperbolic (waves) — and each demands a different numerical strategy.
* The CFL condition is not a numerical artifact but a physical constraint: the scheme must keep up with information propagating through the domain.
* Consistency plus stability equals convergence — the Lax equivalence theorem reduces "is my answer right?" to "is my scheme stable?"
* Spectral methods are exponentially accurate for smooth problems because they use a global basis (Fourier modes), while finite differences accumulate polynomial errors from local Taylor truncations.

## Classification

Second-order linear PDEs in two variables: $A u_{xx} + 2B u_{xy} + C u_{yy} + \text{lower order} = 0$.

The discriminant $B^2 - AC$ determines the type:

* **Elliptic** ($B^2 - AC < 0$): Laplace equation $\nabla^2 u = 0$. Things that have settled down.
* **Parabolic** ($B^2 - AC = 0$): Heat equation $u_t = \alpha \nabla^2 u$. Things that are settling down.
* **Hyperbolic** ($B^2 - AC > 0$): Wave equation $u_{tt} = c^2 \nabla^2 u$. Things that are sloshing around.

*Elliptic = rubber sheet draped over a frame. Parabolic = heat spreading until everything's the same temperature. Hyperbolic = waves in a pond.*

Each type needs different numerics and different boundary conditions.

## Finite Difference Methods

Replace continuous derivatives with discrete approximations on a grid:

$$
\frac{\partial u}{\partial x} \approx \frac{u_{i+1} - u_{i-1}}{2\Delta x} \quad (O(\Delta x^2))
$$

$$
\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2} \quad (O(\Delta x^2))
$$

*First derivative = slope between neighbors. Second derivative = curvature: how much a point deviates from the average of its neighbors.*

Halving $\Delta x$ quarters the error but quadruples the equations. Smooth solutions need fewer points; sharp gradients need more.

## The Heat Equation

1D heat equation $u_t = \alpha u_{xx}$. The **FTCS scheme** (Forward Time, Central Space):

$$
u_i^{n+1} = u_i^n + r(u_{i+1}^n - 2u_i^n + u_{i-1}^n), \qquad r = \alpha \Delta t / \Delta x^2.
$$

**Conditionally stable**: blows up unless $r \leq 1/2$. Make the time step too big and the solution explodes — a direct echo of the truncation-rounding tradeoff from [bounding errors](./bounding-errors).

**Backward Euler** (implicit): unconditionally stable, but requires solving a tridiagonal system each step.

**Crank-Nicolson**: averages explicit and implicit, giving $O(\Delta t^2, \Delta x^2)$ accuracy while remaining unconditionally stable.

## The Wave Equation

1D wave equation $u_{tt} = c^2 u_{xx}$, central differences in both time and space:

Stable when the **Courant number** $C = c\Delta t/\Delta x \leq 1$.

*Information travels at speed $c$. Your time step must be small enough that the scheme can keep up. If the wave moves faster than one grid cell per time step, the scheme misses information and explodes.*

## Elliptic Problems

Laplace's equation $\nabla^2 u = 0$ on a 2D grid gives the five-point stencil:

$$u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j} = 0$$

*At equilibrium, every point equals the average of its four neighbors. If it didn't, heat would flow.*

This produces a large sparse linear system. For a 2D grid with $N$ points per side, that's $N^2$ unknowns and a million-by-million matrix (for $N=1000$). Only tractable because it's sparse. Iterative methods (Jacobi, Gauss-Seidel, SOR) are preferred for large problems.

## Boundary Conditions

* **Dirichlet**: $u = g$ on boundary. Values known, directly substituted.
* **Neumann**: $\partial u/\partial n = g$. Use ghost points outside the domain.
* **Robin**: $\alpha u + \beta \partial u/\partial n = g$. Combines both.

## Spectral Methods

For smooth solutions with periodic boundaries, represent the solution in Fourier modes. Differentiation in physical space becomes multiplication in Fourier space:

$$\frac{\partial u}{\partial x} \longleftrightarrow ik\hat{u}_k$$

*Transform, multiply by $ik$, transform back. The [FFT](./fft) makes this $O(N\log N)$.*

Spectral methods achieve **exponential convergence** for smooth solutions, far outperforming finite differences. For non-periodic domains, Chebyshev spectral methods use Chebyshev polynomials with grid clustering near boundaries.

[[simulation reaction-diffusion]]

[[simulation heat-equation-2d]]

## Stability and Convergence

Here's a beautiful theorem: if your scheme is consistent and stable, you're guaranteed to get the right answer. If you follow the recipe (consistency) and don't burn the house down (stability), dinner will taste right (convergence).

The **Lax equivalence theorem**: for a consistent finite difference scheme on a well-posed **linear** PDE, stability is equivalent to convergence. You only need to verify stability.

**Important caveat:** Lax's theorem is a luxury of linearity. Nonlinear PDEs (Navier-Stokes, reaction-diffusion with nonlinear sources) can develop shocks and blow-up where additional conditions are needed.

**Von Neumann analysis**: substitute a Fourier mode $u_j^n = g^n e^{ij\theta}$ and require $|g| \leq 1$ for all wave numbers. The standard tool for CFL-type restrictions.

> **Challenge.** Here's a toy you can play with right now: implement FTCS for the 1D heat equation with $u(x,0) = \sin(\pi x)$ on $[0,1]$. Run with $r = 0.4$ (stable) and $r = 0.6$ (unstable). Watch the stable one smoothly decay while the unstable one explodes. The CFL condition $r \leq 0.5$ is the thin line between success and chaos.

## Summary

| PDE Type | Example | Stable Explicit Scheme | Implicit Alternative |
|----------|---------|----------------------|---------------------|
| Parabolic | Heat equation | FTCS ($r \leq 1/2$) | Crank-Nicolson |
| Hyperbolic | Wave equation | Leapfrog ($C \leq 1$) | Implicit leapfrog |
| Elliptic | Laplace equation | N/A (no time) | Direct solve or SOR |

---

## Wrapping Up

Every tool in this course was built for this moment. Error analysis taught you to distrust your computer just enough to use it wisely. LU decomposition and least squares gave you the machinery for the large linear systems that arise when you discretize a PDE. Eigenvalue analysis determines stability. The FFT powers spectral methods. ODE integrators time-march parabolic and hyperbolic PDEs.

PDEs are the language in which physics writes its deepest laws: heat, sound, light, quantum wavefunctions, fluid flow, electromagnetic fields. The numerical methods here are not approximations to that language — they are the tools that let you speak it on a finite machine. You now have everything you need to simulate the physical world at whatever resolution your computer can sustain. The rest is physics, imagination, and compute time.

## Check Your Understanding

1. The FTCS scheme for the heat equation is conditionally stable with $r \leq 1/2$. If you halve $\Delta x$, what must happen to $\Delta t$, and how does this affect the total number of time steps?
2. The five-point stencil says every interior point equals the average of its neighbors. Use this to argue that the solution's maximum must occur on the boundary.

## Challenge

Implement the 2D heat equation $u_t = \alpha(u_{xx} + u_{yy})$ on the unit square with $u = 0$ on the boundary and $u(x,y,0) = \sin(\pi x)\sin(\pi y)$. The exact solution is $u = e^{-2\pi^2\alpha t}\sin(\pi x)\sin(\pi y)$. Implement both FTCS and Crank-Nicolson. Compare: (a) stability limits, (b) error at $t = 0.1$ vs $\Delta t$.
