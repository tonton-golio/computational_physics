# Python Packages -- Let's Play With the Universe

## Your Computational Toolkit

You've built up the theory: tensors, stress, strain, elasticity, fluid dynamics, FEM. Now it's time to *compute*. With modern Python packages, you can go from "here's my PDE" to "here's a beautiful simulation" in about 20 lines of code.

## FEniCS -- The Heavy Lifter

FEniCS solves PDEs by letting you write the weak form almost like math. Here's a complete script -- elastic bar under gravity:

```python
from fenics import *

mesh = UnitIntervalMesh(100)
V = FunctionSpace(mesh, 'Lagrange', 1)
bc = DirichletBC(V, Constant(0), 'near(x[0], 0)')

u = TrialFunction(V)
w = TestFunction(V)
f = Constant(-1.0)

a = inner(grad(u), grad(w)) * dx
L = f * w * dx

u_sol = Function(V)
solve(a == L, u_sol, bc)
plot(u_sol)
```

That's it. 15 lines. Key functions:

| Function | What it does |
|----------|-------------|
| `FunctionSpace(mesh, type, degree)` | Defines the approximation space |
| `TrialFunction(V)` | The unknown you solve for |
| `TestFunction(V)` | The weight function in the weak form |
| `DirichletBC(V, value, boundary)` | Boundary conditions |
| `solve(a == L, u, bcs)` | Solve the weak form |

For 2D:

```python
mesh = RectangleMesh(Point(0, 0), Point(width, height), nx, ny)
V = VectorFunctionSpace(mesh, 'Lagrange', 1)

def bottom(x, on_boundary):
    return on_boundary and near(x[1], 0)

bc = DirichletBC(V, Constant((0, 0)), bottom)
```

FEniCS handles meshing, assembly, and linear algebra behind the scenes. Your job is to express the physics in weak form.

## SymPy -- Algebra Without Tears

Symbolic computation inside Python. Invaluable for tensor calculations:

```python
from sympy import symbols, diff, simplify

x, y = symbols('x y')
vx = x**2 * y
vy = -x * y**2

eps_xy = (diff(vx, y) + diff(vy, x)) / 2
print(f"eps_xy = {eps_xy}")
```

Use `lambdify` to convert symbolic expressions into fast numerical functions.

## Plotly -- Interactive 3D Visualization

For 3D stress fields and deformation surfaces:

```python
import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface(z=stress_field, x=X, y=Y)])
fig.show()
```

## Shapely and Rasterio -- Real-World Geometry

Shapely for geometric operations (cross-sections, centroids). Rasterio for reading GeoTIFF terrain data into NumPy arrays.

## Gmsh -- Custom Meshes

For domains more complex than rectangles, Gmsh generates high-quality finite element meshes. Define geometries programmatically or through its GUI, export, and import into FEniCS.

## Big Ideas

* FEniCS lets you write PDEs in weak form almost exactly as you'd write them on paper -- the code mirrors the math.
* SymPy bridges pencil-and-paper algebra and numerical computation: derive symbolically, then `lambdify` for speed.
* The workflow is a closed loop: write the PDE, express it in weak form, encode in FEniCS, mesh the domain, visualize.

## What Comes Next

We've covered solids, fluids, and the computational tools to simulate them. Time for the grand finale: a review of every major idea, told through stories.

## Check Your Understanding

1. In the FEniCS code, `TrialFunction` and `TestFunction` look similar but play different roles. What is the difference, and why does FEM need both?
2. The weak form in FEniCS is `a = inner(grad(u), grad(w)) * dx`. How does this correspond to $\int_\Omega \nabla u \cdot \nabla w\,dx$?

## Challenge

Use FEniCS to solve the 2D elasticity problem of a rectangular plate ($1\,\text{m} \times 0.2\,\text{m}$) clamped at its left edge and loaded by a uniform downward body force of $10^4$ N/m^3 (representing gravity). Use $E = 10$ GPa and $\nu = 0.3$. Plot the vertical displacement field and find the maximum deflection at the free end. Now parameterize over plate thickness from 0.1 m to 0.5 m and plot maximum deflection versus thickness -- does it follow the $t^{-3}$ scaling predicted by Euler-Bernoulli beam theory? At what thickness does the beam approximation break down?
