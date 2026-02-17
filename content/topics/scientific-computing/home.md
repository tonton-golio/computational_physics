# Scientific Computing

## Course overview

Scientific computing develops the **numerical methods** needed to solve problems in biology, physics, nanoscience, and chemistry that have no closed-form solution. The emphasis is on deriving algorithms, programming them, and understanding their error behavior.

- Accuracy: how close is the numerical answer to the true solution?
- Efficiency: how does the computational cost scale with problem size?
- Robustness: does the method work reliably across a range of inputs?
- Stability: do small perturbations in input produce small perturbations in output?

## Why this topic matters

- Most differential equations arising in science cannot be solved analytically.
- Linear systems with thousands of unknowns appear in finite-element modeling, data fitting, and network analysis.
- Optimization underlies machine learning, inverse problems, and experimental design.
- Understanding numerical error is essential for trusting computational results.

## Key mathematical ideas

- Matrix factorizations (LU, QR, SVD) and their role in solving linear systems.
- Iterative methods for nonlinear equations (Newton-Raphson, fixed-point iteration).
- Numerical integration of ODEs (Euler, Runge-Kutta) and stability theory.
- Finite-difference discretization of PDEs.
- The discrete Fourier transform and the FFT algorithm.
- Condition numbers and the propagation of rounding errors.

## Prerequisites

- Programming in Python with NumPy.
- Linear algebra: matrix operations, eigenvalues, vector spaces.
- Calculus: derivatives, integrals, Taylor series.

## Recommended reading

- Heath, *Scientific Computing: An Introductory Survey*.
- Trefethen and Bau, *Numerical Linear Algebra*.
- Press et al., *Numerical Recipes*.

## Learning trajectory

This module is organized from foundational linear algebra to advanced PDE methods:

- Error analysis and floating-point arithmetic.
- Linear equations: Gaussian elimination and LU factorization.
- Linear least squares and data fitting.
- Nonlinear equations: root finding, bisection, Newton's method.
- Nonlinear systems: Newton in higher dimensions, Broyden's method.
- Optimization: metaheuristics, simulated annealing, particle swarm.
- Eigenvalue problems: power method, inverse iteration, Rayleigh quotient.
- Eigenvalue algorithms: QR algorithm, Gershgorin circles, matrix functions.
- Initial value problems for ODEs: Euler and Runge-Kutta methods.
- Partial differential equations: finite differences, heat and wave equations.
- Fast Fourier Transform: DFT, FFT, and signal processing.
