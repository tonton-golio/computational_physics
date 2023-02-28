
## __Introduction__
Beyond the usual Matplotlib, NumPy and SciPy packages, here are a few recommendations that may help in problem solving.

## __Shapely__
Shapely provides a way to conduct easy geometric calculations and representation.

[Shapely Documentation](https://shapely.readthedocs.io/en/stable/manual.html)

[GitHub](https://github.com/shapely/shapely/tree/main/docs/)

Highlighted functions: polygon, clip_by_rect,

## __SymPy__
SymPy allows maple/wolframalpha-like calculus analysis inside Python, it is very convenient for automating tensor calculations. Example functions for various tensors are given below.

[SymPy Documentation](https://docs.sympy.org/latest/index.html)

Highlighted functions: lambdify, symbols

## __Plotly__
Greatly extends the functionality of interactive 3D graphics compared to matplotlib.

[Plotly Documentation](https://plotly.com/graphing-libraries/)

Highlighted functions: graph_objects

## __Rasterio__

Geographic information systems use GeoTIFF and other formats to organize and store gridded raster datasets such as satellite imagery and terrain models. Rasterio reads and writes these formats and provides a Python API based on Numpy N-dimensional arrays and GeoJSON.

[Rasterio Documentation](https://rasterio.readthedocs.io/en/latest/)

Highlighted functions: 

## __Fenics__
For solving FEM systems. Brace yourself and memorize all the preculiarities of this package.

1D:
Highlighted functions:
UnitIntervalMesh: Defines the range and discretization used.
IntervalMesh: Defines the range and discretization used.
FunctionSpace: Defines the function space, *lagrange* means polynomial and deg is the order (1=linear).
Expression: Defines the basisfunction.
project: Projects the polynomial onto the functionspace. 
TrialFunction: The unknown function we wish to approximate. $u_z(z)$
TestFunction: The weight function space. $w_i$
solve: find the approximated constants.

2D: 
RectangleMesh(Point(,0,0), Point(width,height), width resolution, height resolution): Meshes a rectange, ideal for 2D modelling.
VectorFunctionSpace: 2D equivalent of the FunctionSpace function.
TrialFunction: The unknown function we wish to approximate. $u_z(z)$
TestFunction: The weight function space. $w_i$
inner(): The dobbelt dot product (:)

on_boundary and near(): on_boundary outputs 0 or 1 depending on whether the program iterates over all the points. near defines the tolerance, such that machine precision doesnt mess it up.
DirechletBC(on space V, set value Constant((0,0)), boundary): The actual boundary condition.
plot: plotting function from the Fenics package.

extracting to matplotlib looping over usol.

Implementing boundary conditions:
def bottom_boundary(x, on_boundary): return on_boundary and near(x\[0\], 0) 
bc1 = DirichletBC(V, Constant(0), bottom_boundary) # u_z = 0 (no displacement) at z=0 (bottom boundary)
bcs = \[bc1\]
*From Nicholas Notebook*