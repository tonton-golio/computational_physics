# Finite Element Method

## From Equations to Elements

Imagine you need to figure out how a glacier deforms as it flows through a valley with irregular rock walls. No closed-form solution exists -- the geometry is too wild. So you do what any sensible person would do: chop the valley into thousands of tiny triangles, assume a simple solution on each one, and stitch them all together.

That's the **finite element method** (FEM). And unlike finite differences (which approximate *derivatives*), FEM approximates the *solution itself*.

## The Approximation

Write the unknown as a weighted sum of basis functions:
$$
u^N(x) = \sum_{i=1}^N a_i \, \phi_i(x)
$$

Each $\phi_i$ is typically a polynomial, nonzero only near node $i$. The coefficients $a_i$ are what we solve for. Substituting into the governing equation $\mathcal{L}(u) = f$ produces a **residual**:
$$
r^N(x) = \mathcal{L}(u^N) - f
$$

For the exact solution, $r = 0$ everywhere. We need a principled way to make it small.

## Galerkin's Method

Require the residual to be **orthogonal** to every basis function:
$$
\int_\Omega r^N(x) \, \phi_i(x) \, dx = 0, \qquad i = 1, \ldots, N
$$

The intuition: making the residual orthogonal to the approximation space produces the *best* approximation within that space. This gives $N$ equations for $N$ unknowns.

Other approaches exist -- least squares, collocation -- but Galerkin dominates because it preserves operator symmetry and connects to energy minimization.

## What Makes It "Finite Element"

Galerkin works with any basis. FEM makes three specific choices:

1. **Mesh the domain** into elements -- intervals in 1D, triangles in 2D, tetrahedra in 3D.
2. **Use local basis functions** -- each $\phi_i$ is a polynomial nonzero only on elements touching node $i$. This is the "finite" in finite element.
3. **Assemble** -- global integrals become sums over elements, producing a **sparse** system $\mathbf{K}\mathbf{a} = \mathbf{f}$.

Sparsity is what makes FEM scale. Each node couples only to its neighbors, so millions of unknowns remain tractable.

## The Variational Perspective

For elastic problems, the true displacement minimizes **potential energy**:
$$
W(u) = \frac{1}{2}\int_\Omega \varepsilon : \sigma \, dV - \int_\Omega \mathbf{f} \cdot u \, dV
$$

The Galerkin equations are exactly the optimality conditions. Nature chooses the energy-minimizing configuration; FEM finds the best approximation within the element space.

## See It Work

The simulation below solves a 1D bar fixed at one end under a uniform load. Watch piecewise-linear elements converge to the exact parabolic solution as you add more elements. Toggle the basis functions to see how local hat functions combine into the global solution.

[[simulation fem-1d-bar-sim]]

## Big Ideas

* FEM approximates the solution itself (as local basis functions), not the derivatives -- this gives it flexibility with irregular geometries that finite differences can't match.
* Galerkin's method: require the residual to be orthogonal to every basis function. $N$ equations for $N$ unknowns.
* Sparsity is the killer feature: each basis function is local, so the stiffness matrix is almost all zeros and huge systems stay tractable.
* Nature minimizes energy; FEM finds the best approximation within the element space -- a projection onto a finite-dimensional subspace.

## What Comes Next

We have the theory. Now we get our hands dirty with actual computation -- setting up and solving FEM problems in Python.

## Check Your Understanding

1. In Galerkin, we require $\int r^N \phi_i \, dx = 0$ for every basis function. Why is orthogonality the right condition?
2. FEM produces a sparse stiffness matrix. Why? Which entries of $\mathbf{K}_{ij}$ are nonzero?

## Challenge

Consider the 1D Poisson equation $-d^2u/dx^2 = 1$ on $[0,1]$ with $u(0) = u(1) = 0$, whose exact solution is $u = x(1-x)/2$. Implement a 1D FEM solver by hand (no libraries): set up the weak form, build the stiffness matrix and load vector for $n = 2, 4, 8, 16$ linear elements, solve each system, and plot the error $\|u - u^N\|$ as a function of $n$. Verify that the error decreases as $h^2$ (where $h = 1/n$). Now repeat with quadratic elements ($n = 2, 4, 8$) and confirm the error scales as $h^3$. What is the computational cost (in terms of matrix size and solve time) of achieving the same error level with linear versus quadratic elements?
