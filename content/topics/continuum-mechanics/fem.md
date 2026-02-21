# Finite Element Method

## From Equations to Elements

Some differential equations don't have closed-form solutions — the geometry is irregular, the material varies, the boundary conditions resist analytical tricks. The **finite element method** (FEM) handles this by dividing the domain into small elements, assuming a simple solution on each, and stitching them together.

The key distinction from finite differences: FD approximates the *derivatives* in the equation. FEM approximates the *solution itself*.

## The Approximation

Write the unknown as a weighted sum of known basis functions:
$$
u^N(x) = \sum_{i=1}^N a_i \, \phi_i(x)
$$

Each $\phi_i$ is typically a polynomial that is nonzero only near node $i$. The coefficients $a_i$ are what we solve for. Substituting into the governing equation $\mathcal{L}(u) = f$ produces a **residual**:
$$
r^N(x) = \mathcal{L}(u^N) - f
$$

For the exact solution, $r = 0$ everywhere. For our approximation, we need a principled way to make it small.

## Galerkin's Method

The standard approach: require the residual to be **orthogonal** to every basis function:
$$
\int_\Omega r^N(x) \, \phi_i(x) \, dx = 0, \qquad i = 1, \ldots, N
$$

The intuition: the residual and the approximation error are linked. Making $r$ orthogonal to the approximation space produces the best approximation within that space. This yields $N$ equations for $N$ unknowns.

There are other approaches. *Least squares* minimizes $\int (r^N)^2 \, dx$. *Collocation* forces $r^N = 0$ at selected points. Galerkin dominates because it preserves symmetry of the underlying operator and connects directly to energy minimization.

## What Makes It "Finite Element"

Galerkin works with any basis. FEM makes three specific choices:

1. **Mesh the domain** into elements — intervals in 1D, triangles in 2D, tetrahedra in 3D.
2. **Use local basis functions** — each $\phi_i$ is a polynomial nonzero only on elements touching node $i$. This is the "finite" in finite element: the support is finite.
3. **Assemble** — global integrals become sums over elements, producing a **sparse** system $\mathbf{K}\mathbf{a} = \mathbf{f}$.

Sparsity is what makes FEM scale. Each node couples only to its neighbors, so systems with millions of unknowns remain tractable.

## The Variational Perspective

For elastic problems, the true displacement minimizes the **potential energy**:
$$
W(u) = \frac{1}{2}\int_\Omega \varepsilon : \sigma \, dV - \int_\Omega \mathbf{f} \cdot u \, dV
$$

The Galerkin equations are exactly the optimality conditions for this functional. Nature chooses the configuration that minimizes energy; FEM finds the best approximation within the element space.

## See It Work

The simulation below solves a 1D bar fixed at one end under a uniform distributed load. The exact displacement is a parabola — watch how piecewise-linear elements converge to it as you add more elements. Toggle the basis functions to see how local hat functions, each scaled by its nodal value, combine into the global solution.

[[simulation fem-1d-bar-sim]]

## Big Ideas

* FEM approximates the solution itself (as a sum of local basis functions), not the derivatives — this is what separates it from finite differences and gives it flexibility with irregular geometries.
* Galerkin's method turns the residual-minimization problem into a linear system: require the residual to be orthogonal to every basis function, and you get $N$ equations for $N$ unknowns.
* Sparsity is the killer feature: each basis function is nonzero only near its home node, so the global stiffness matrix $\mathbf{K}$ has almost all zero entries and systems with millions of unknowns stay tractable.
* The variational perspective reveals why FEM converges: nature minimizes potential energy, and FEM finds the best approximation within the element space — a projection onto a finite-dimensional subspace.

## What Comes Next

We have the theory. Next we get hands dirty with actual computation — setting up and solving FEM problems in Python with FEniCS.

## Check Your Understanding

1. In the Galerkin method, we require $\int r^N \phi_i \, dx = 0$ for every basis function $\phi_i$. Why is orthogonality the right condition? What would it mean if the residual were not orthogonal to the approximation space?
2. FEM produces a sparse stiffness matrix. Why? Which entries of $\mathbf{K}_{ij}$ are nonzero, and which are zero?
3. Adding more elements (refining the mesh) increases $N$. How does the error in the FEM solution typically scale with element size $h$? What does this tell you about the trade-off between accuracy and computational cost?

## Challenge

Consider the 1D Poisson equation $-d^2u/dx^2 = 1$ on $[0,1]$ with $u(0) = u(1) = 0$, whose exact solution is $u = x(1-x)/2$. Implement a 1D FEM solver by hand (no libraries): set up the weak form, build the stiffness matrix and load vector for $n = 2, 4, 8, 16$ linear elements, solve each system, and plot the error $\|u - u^N\|$ as a function of $n$. Verify that the error decreases as $h^2$ (where $h = 1/n$). Now repeat with quadratic elements ($n = 2, 4, 8$) and confirm the error scales as $h^3$. What is the computational cost (in terms of matrix size and solve time) of achieving the same error level with linear versus quadratic elements?
