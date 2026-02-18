# Python Packages — Let's Play With the Universe

## Your Computational Toolkit

You've spent the last several sections building up the theory: tensors, stress, strain, elasticity, fluid dynamics, the finite element method. Now it's time to *compute*. The good news: with modern Python packages, you can go from "here's my differential equation" to "here's a beautiful simulation of a bending beam" in about 20 lines of code.

This section introduces the packages you'll use throughout the rest of the course. Keep it open as a reference.

## FEniCS — The Heavy Lifter

FEniCS is a powerful finite element library that lets you solve PDEs by writing them in a form that looks almost like the math. Here's a taste — a complete script that computes the deformation of a 1D elastic bar under gravity:

```python
from fenics import *

# Create a mesh: 100 elements along a unit interval
mesh = UnitIntervalMesh(100)

# Define the function space: linear Lagrange elements
V = FunctionSpace(mesh, 'Lagrange', 1)

# Boundary condition: fixed at x=0
bc = DirichletBC(V, Constant(0), 'near(x[0], 0)')

# Define the problem: find u such that
# integral of (du/dx * dw/dx) dx = integral of f * w dx
u = TrialFunction(V)      # the unknown displacement
w = TestFunction(V)        # the weight/test function
f = Constant(-1.0)        # body force (gravity, pointing down)

# This is the weak form — compare to your notes!
a = inner(grad(u), grad(w)) * dx   # stiffness term
L = f * w * dx                      # load term

# Solve it
u_sol = Function(V)
solve(a == L, u_sol, bc)

# That's it. u_sol now contains the displacement field.
plot(u_sol)
```

That's 15 lines of real code, and you've just solved an elasticity problem. The key FEniCS functions:

| Function | What it does |
|----------|-------------|
| `UnitIntervalMesh(n)` | Creates a 1D mesh with $n$ elements |
| `RectangleMesh(...)` | Creates a 2D rectangular mesh |
| `FunctionSpace(mesh, type, degree)` | Defines the approximation space. `'Lagrange', 1` = linear polynomials |
| `VectorFunctionSpace(...)` | For vector-valued problems (2D/3D displacements) |
| `TrialFunction(V)` | The unknown function you're solving for |
| `TestFunction(V)` | The weight function in the weak form |
| `DirichletBC(V, value, boundary)` | Boundary conditions: "fix this value on this boundary" |
| `inner(a, b)` | The double-dot product ($a : b$) |
| `solve(a == L, u, bcs)` | Find the coefficients that satisfy the weak form |

For 2D problems, the syntax extends naturally:

```python
mesh = RectangleMesh(Point(0, 0), Point(width, height), nx, ny)
V = VectorFunctionSpace(mesh, 'Lagrange', 1)

# Boundary condition: fixed bottom edge
def bottom(x, on_boundary):
    return on_boundary and near(x[1], 0)

bc = DirichletBC(V, Constant((0, 0)), bottom)
```

FEniCS handles the meshing, assembly, and linear algebra behind the scenes. Your job is to express the physics in weak form — and you already know how to do that from the previous sections.

## SymPy — Algebra Without Tears

SymPy does symbolic computation inside Python. It's invaluable for tensor calculations where you need exact expressions rather than numbers:

```python
from sympy import symbols, diff, simplify, Matrix

x, y, z = symbols('x y z')

# Define a velocity field
vx = x**2 * y
vy = -x * y**2

# Compute the strain rate tensor
eps_xx = diff(vx, x)
eps_yy = diff(vy, y)
eps_xy = (diff(vx, y) + diff(vy, x)) / 2

print(f"Strain rate tensor:")
print(f"  eps_xx = {eps_xx}")
print(f"  eps_yy = {eps_yy}")
print(f"  eps_xy = {eps_xy}")
```

Use `lambdify` to convert symbolic expressions into fast numerical functions when you need to evaluate them on arrays.

## Plotly — Interactive 3D Visualization

Matplotlib is great for 2D plots, but for 3D stress fields and deformation surfaces, Plotly gives you interactive visualizations you can rotate, zoom, and explore:

```python
import plotly.graph_objects as go

# Plot a 3D surface of a stress field
fig = go.Figure(data=[go.Surface(z=stress_field, x=X, y=Y)])
fig.update_layout(title='Von Mises Stress')
fig.show()
```

## Shapely — Geometric Operations

For setting up domain geometries, computing cross-sections, and geometric preprocessing:

```python
from shapely.geometry import Polygon

# Define a glacier cross-section
glacier = Polygon([(0, 0), (10, 0), (8, 5), (2, 5)])
print(f"Area: {glacier.area}")
print(f"Centroid: {glacier.centroid}")
```

## Rasterio — Working With Real Terrain Data

For real-world applications (modeling glacier flow over actual terrain), Rasterio reads GeoTIFF and other raster formats used in geographic information systems:

```python
import rasterio
with rasterio.open('terrain.tif') as src:
    elevation = src.read(1)  # NumPy array of elevation data
```

## Gmsh — Custom Meshes

For domains more complex than rectangles, Gmsh generates high-quality finite element meshes. You can define geometries programmatically or through its GUI, export the mesh, and import it into FEniCS. See [gmsh.info](https://gmsh.info/).

## Gridap (Julia)

If you're curious about FEM in Julia, Gridap offers a similar high-level interface. The syntax and workflow are analogous to FEniCS, which is reassuring: the *concepts* transfer across languages. See [gridap.github.io](https://gridap.github.io).

## What We Just Learned

Modern computational tools let you express the physics in code that closely mirrors the math. FEniCS handles FEM problems, SymPy handles symbolic tensor calculations, and Plotly/Rasterio handle visualization and real-world data. These tools will be your companions for the rest of the course.

## What's Next

We've covered solids, fluids, and the computational tools to simulate them. Now it's time to bring everything together in the grand finale: a review of every major idea, told through stories and real-world examples.
