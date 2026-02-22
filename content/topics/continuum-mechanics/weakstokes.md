# Weak Stokes Formulation

## Why "Weak"? -- From Perfect to Practical

Imagine trying to solve the Stokes equation $\nabla p = \eta \nabla^2 \mathbf{v}$ inside a glacier valley with jagged rock walls. The **strong form** demands the equation hold at every single point. Good luck with that geometry.

The **weak form** relaxes this. Instead of pointwise perfection, we require the equation to hold *on average* -- integrated against test functions. Sounds like we're giving something up. We're actually gaining something enormous: the ability to solve it numerically with the finite element method.

Think of it like grading a student. The strong form requires a perfect score on every question. The weak form requires the right overall average. For well-posed problems, these two demands are equivalent.

## The Weak Form

Start with steady Stokes flow:
$$
-\nabla \cdot \sigma + \nabla p = \mathbf{f}, \qquad \nabla \cdot \mathbf{v} = 0
$$

Multiply by a test function $\mathbf{w}$ (vanishes on Dirichlet boundaries), integrate over $\Omega$, and integrate by parts:
$$
\int_\Omega 2\eta\,\dot{\varepsilon}(\mathbf{v}) : \dot{\varepsilon}(\mathbf{w}) \, dV - \int_\Omega p \, (\nabla \cdot \mathbf{w}) \, dV = \int_\Omega \mathbf{f} \cdot \mathbf{w} \, dV + \int_{\Gamma_N} \mathbf{t} \cdot \mathbf{w} \, dS
$$

The incompressibility constraint tested against scalar $q$:
$$
\int_\Omega q \, (\nabla \cdot \mathbf{v}) \, dV = 0
$$

Notice what happened: second derivatives got distributed between $\mathbf{v}$ and $\mathbf{w}$ via integration by parts. We no longer need $\mathbf{v}$ to be twice differentiable -- once is enough. This is the technical payoff: we can use simple $C^0$ piecewise-polynomial elements (triangles with linear or quadratic polynomials) instead of the painful $C^1$ elements the strong form would require.

## From Weak Form to Linear System

In FEM, we approximate $\mathbf{v}$ and $p$ as combinations of basis functions. The weak form becomes:
$$
\begin{pmatrix} \mathbf{K} & \mathbf{G}^T \\ \mathbf{G} & \mathbf{0} \end{pmatrix} \begin{pmatrix} \mathbf{u} \\ \mathbf{p} \end{pmatrix} = \begin{pmatrix} \mathbf{f} \\ \mathbf{0} \end{pmatrix}
$$

This is a **saddle-point system**: $\mathbf{K}$ handles viscosity, $\mathbf{G}$ enforces incompressibility, and the zero block means pressure has no self-coupling -- it's determined entirely through its interaction with velocity.

And here's the warning you need to take seriously: pick the wrong finite element spaces and your pressure goes haywire -- spurious oscillations everywhere. This is the **inf-sup condition** (LBB condition). It's not bureaucratic pedantry. It's the difference between a working simulation and numerical garbage.

[[simulation fem-convergence]]

## Big Ideas

* The strong form demands pointwise perfection; the weak form demands the right average. For well-posed problems, they're equivalent, but the weak form is what computers can actually solve.
* Integration by parts is the magic trick: it halves the smoothness required of the solution, enabling simple piecewise-polynomial elements.
* The inf-sup condition is non-negotiable: wrong element pairs produce pressure oscillations that ruin your solution.

## What Comes Next

We have the equations in the right form. Now we need to learn how to actually *solve* them -- cutting the domain into pieces and solving each one. That's the finite element method.

## Check Your Understanding

1. What does it mean for a test function to "vanish on the boundary where velocities are prescribed"? What would go wrong if test functions were nonzero on Dirichlet boundaries?
2. The weak form has first derivatives of both $\mathbf{v}$ and $\mathbf{w}$, while the strong form has second derivatives of $\mathbf{v}$ alone. Why is first-differentiability easier to achieve in a finite element approximation?

## Challenge

**Challenge.** Apply what you have learned here to build a finite-element solver -- you will do exactly this in the next lesson on the Finite Element Method, where you derive the weak form of a 1D Poisson problem, assemble the stiffness matrix with hat functions, and compare numerical and exact solutions.
