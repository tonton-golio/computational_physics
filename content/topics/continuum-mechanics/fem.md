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

> **Other approaches.** *Least squares* minimizes $\int (r^N)^2 \, dx$. *Collocation* forces $r^N = 0$ at selected points. Galerkin dominates because it preserves symmetry of the underlying operator and connects directly to energy minimization.

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

## What's Next

We have the theory. Next we get hands dirty with actual computation — setting up and solving FEM problems in Python with FEniCS.
