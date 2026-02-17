# Partial Differential Equations

## Classification

Partial differential equations (PDEs) are classified by the nature of their highest-order terms. A second-order linear PDE in two variables has the general form

$$
A u_{xx} + 2B u_{xy} + C u_{yy} + \text{lower order terms} = 0.
$$

The **discriminant** $B^2 - AC$ determines the type:

- **Elliptic** ($B^2 - AC < 0$): e.g., Laplace equation $\nabla^2 u = 0$. Describes equilibrium states.
- **Parabolic** ($B^2 - AC = 0$): e.g., heat equation $u_t = \alpha \nabla^2 u$. Describes diffusion processes.
- **Hyperbolic** ($B^2 - AC > 0$): e.g., wave equation $u_{tt} = c^2 \nabla^2 u$. Describes wave propagation.

Each type requires different numerical strategies and boundary conditions. Elliptic problems need boundary values on the entire domain boundary. Parabolic and hyperbolic problems need initial conditions plus boundary conditions.

## Finite Difference Methods

The fundamental idea is to replace continuous derivatives with **discrete approximations** on a grid. For a uniform grid with spacing $\Delta x$, the standard finite difference formulas are

$$
\frac{\partial u}{\partial x} \approx \frac{u_{i+1} - u_{i-1}}{2\Delta x} \quad \text{(central, } O(\Delta x^2)\text{)},
$$

$$
\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2} \quad \text{(} O(\Delta x^2)\text{)}.
$$

These replace the PDE at each interior grid point with an algebraic equation, producing a system of equations that can be solved by linear algebra techniques.

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

## Elliptic Problems

For Laplace's equation $\nabla^2 u = 0$ on a 2D grid, the five-point stencil gives

$$
u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j} = 0.
$$

This produces a large sparse linear system $A\mathbf{u} = \mathbf{b}$ where boundary values enter through the right-hand side. Direct solvers (LU decomposition) work for moderate grids, but **iterative methods** are preferred for large problems:

- **Jacobi iteration**: Update each point from its neighbors' previous values.
- **Gauss-Seidel**: Use updated values as soon as available (faster convergence).
- **Successive over-relaxation (SOR)**: Accelerates Gauss-Seidel with a relaxation parameter $\omega$.

For Poisson's equation $\nabla^2 u = f$, the same stencil applies with $f_{i,j}$ on the right-hand side.

## Boundary Conditions

The three standard types are:

- **Dirichlet**: $u = g$ on $\partial\Omega$. The boundary values are known and directly substituted.
- **Neumann**: $\partial u / \partial n = g$ on $\partial\Omega$. Discretized using one-sided or ghost-point differences.
- **Robin (mixed)**: $\alpha u + \beta \partial u / \partial n = g$. Combines the above.

Ghost points outside the domain are a clean way to implement Neumann conditions: introduce a fictitious point $u_{-1}$ and use the centered difference $u_1 - u_{-1} = 2\Delta x \cdot g$ to eliminate it.

## Spectral Methods

For problems with smooth solutions and periodic boundary conditions, **spectral methods** represent the solution in a basis of trigonometric functions. The spatial derivative of $u(x) = \sum_k \hat{u}_k e^{ikx}$ is computed exactly in Fourier space:

$$
\frac{\partial u}{\partial x} \longleftrightarrow ik\hat{u}_k.
$$

The **Fast Fourier Transform** (FFT) makes this approach computationally efficient with $O(N\log N)$ operations. Spectral methods achieve exponential convergence for smooth solutions, far outperforming finite differences.

For non-periodic domains, **Chebyshev spectral methods** use Chebyshev polynomials as the basis, with clustering of grid points near boundaries to handle boundary layers.

[[simulation reaction-diffusion]]

## Method of Lines

The **method of lines** (MOL) semi-discretizes the PDE by replacing spatial derivatives with finite differences while leaving time continuous. This converts the PDE into a system of ODEs:

$$
\frac{d\mathbf{u}}{dt} = \mathbf{F}(\mathbf{u}),
$$

which can then be integrated with any ODE solver (Euler, RK4, BDF). This approach cleanly separates spatial and temporal discretization, allowing established ODE solvers to handle time integration including adaptive step size control.

## Stability and Convergence

The **Lax equivalence theorem** states that for a consistent finite difference scheme applied to a well-posed linear PDE, **stability is equivalent to convergence**. This means we only need to verify stability (via von Neumann analysis or matrix methods) to guarantee that the numerical solution converges to the true solution as the grid is refined.

**Von Neumann stability analysis** substitutes a Fourier mode $u_j^n = g^n e^{ij\theta}$ into the difference scheme and requires the **amplification factor** $|g| \leq 1$ for all wave numbers $\theta$. This is the standard tool for determining CFL-type restrictions on the time step.

## Summary

| PDE Type | Example | Stable Explicit Scheme | Implicit Alternative |
|----------|---------|----------------------|---------------------|
| Parabolic | Heat equation | FTCS ($r \leq 1/2$) | Crank-Nicolson |
| Hyperbolic | Wave equation | Leapfrog ($C \leq 1$) | Implicit leapfrog |
| Elliptic | Laplace equation | N/A (no time) | Direct solve or SOR |
