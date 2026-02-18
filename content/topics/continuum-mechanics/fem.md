# Finite Element Method

## The Rubber Sheet, Revisited — In Triangles

Remember the rubber sheet from the tensor section? Imagine you need to calculate exactly how it deforms under some complicated load. The shape is irregular, the forces are messy, and there's no analytical solution.

Here's the idea: instead of solving the whole sheet at once, **cut it into tiny triangles**. On each triangle, assume the solution is simple — maybe linear or quadratic. Then stitch all the triangles together, demanding that the solutions agree at the shared edges. That's the **finite element method** (FEM), and it's how we actually compute the deformation of real glaciers, airplane wings, bridges, and biological tissues.

## The Big Picture — Approximate the Solution, Not the Equation

FEM is fundamentally different from finite differences. In finite differences, you approximate the *derivatives* in the equation (replacing $\partial^2 u / \partial x^2$ with $(u_{i+1} - 2u_i + u_{i-1})/h^2$). In FEM, you approximate the *solution itself* as a combination of simple functions, then find the best combination.

Consider a differential equation $A(u) = f$. We approximate the solution as:
$$
u^N(x) = \sum_{i=1}^N a_i \, \phi_i(x)
$$

where the $\phi_i$ are **basis functions** (simple, known functions — usually polynomials that are nonzero only near a single mesh node) and the $a_i$ are unknown coefficients we need to find.

Substituting gives a **residual**:
$$
r^N(x) = A(u^N) - f
$$

If $u^N$ were the exact solution, the residual would be zero everywhere. It won't be, so our job is to choose the $a_i$ to make the residual as small as possible.

## How Small? — The Method of Weighted Residuals

Different choices for "as small as possible" give different methods:

### Least Squares Method
Minimize the total squared residual:
$$
\Pi = \int_0^L (r^N)^2 \, dx
$$

Setting $\partial \Pi / \partial a_i = 0$ gives $N$ equations for $N$ unknowns. This is clean and intuitive — you're literally minimizing the error in an $L^2$ sense.

### Collocation Method
Force the residual to be exactly zero at $N$ specific points:
$$
r^N(x_i) = 0, \qquad i = 1, \ldots, N
$$

This is useful when you care about accuracy in specific locations. Imagine working at a fastener company and getting complaints that screws keep breaking at the head. You don't need the stress field everywhere — you need it *right below the screwhead*. Collocation lets you concentrate accuracy where it matters.

### Galerkin's Method — The Winner
The most widely used approach: force the residual to be **orthogonal** to every basis function:
$$
\int_0^L r^N(x) \, \phi_i(x) \, dx = 0, \qquad i = 1, \ldots, N
$$

The intuition: the error $e = u - u^N$ and the residual $r$ are related (if one is zero, so is the other). Making $r$ orthogonal to the approximation space is the closest we can get to minimizing the error without knowing the true solution.

The Galerkin recipe:
1. **Compute the residual**: $r^N = A(u^N) - f$
2. **Force orthogonality**: $\int r^N \, \phi_i \, dx = 0$ for each $i$
3. **Solve** the resulting system of equations for the $a_i$

## The "Finite Element" Part — Choosing the Basis Functions

The general Galerkin framework doesn't tell you *how* to choose the $\phi_i$. This is where FEM adds its magic:

1. **Mesh the domain** — divide it into small elements (triangles in 2D, tetrahedra in 3D).
2. **Define local basis functions** — on each element, the basis functions are simple polynomials (linear, quadratic, etc.). Each $\phi_i$ is nonzero only in the elements surrounding node $i$, and zero everywhere else. This is the "finite" in "finite element" — the basis functions have finite support.
3. **Assemble** — the integrals over the whole domain become sums of integrals over individual elements. Each element contributes to a small block of the global matrix.

The result: a large but **sparse** linear system $\mathbf{K}\mathbf{a} = \mathbf{f}$, where $\mathbf{K}$ is the stiffness matrix and $\mathbf{f}$ is the load vector. Sparse means most entries are zero (because each basis function only overlaps with its neighbors), which makes the system efficient to solve even for millions of unknowns.

## Minimum Potential Energy — The Variational Perspective

For elastic problems, there's an elegant alternative viewpoint. The true displacement minimizes the **potential energy**:
$$
W(u) = \frac{1}{2}\int_\Omega \varepsilon(u) : \sigma(\varepsilon(u)) \, dV - \int_\Omega \mathbf{f} \cdot u \, dV
$$

Any perturbation from the true solution increases $W$. This is a **variational problem**: instead of solving a differential equation, we minimize a functional. The two formulations are equivalent (for linear problems, the Galerkin equations *are* the optimality conditions for the energy functional), but the variational perspective gives physical insight — nature chooses the configuration that minimizes energy.

The technique: instead of varying a scalar to minimize a function, we vary a *function* to minimize a *functional*. The calculus of variations gives us the tools, and the result is identical to the Galerkin weak form.

## Interactive FEM 1D Bar Demo

[[simulation fem-1d-bar-sim]]

## What We Just Learned

The finite element method approximates solutions to differential equations by dividing the domain into small elements with simple local basis functions. Galerkin's method makes the residual orthogonal to the approximation space, producing a sparse linear system. For elastic problems, this is equivalent to minimizing potential energy. FEM handles complex geometries, arbitrary boundary conditions, and heterogeneous materials — making it the workhorse of modern computational mechanics.

## What's Next

We have the theory. Now let's get our hands dirty with actual computation. The next section introduces the Python packages — especially FEniCS — that let you set up and solve FEM problems in remarkably few lines of code.
