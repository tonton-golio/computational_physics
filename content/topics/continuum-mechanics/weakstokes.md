# Weak Stokes Formulation

## Why "Weak"? — From Perfect to Practical

The Stokes equation $\nabla p = \eta \nabla^2 \mathbf{v}$ with $\nabla \cdot \mathbf{v} = 0$ is the **strong form**: it demands that the equation is satisfied at every single point in the domain. That's a tall order when your domain is a glacier valley with jagged rock walls and complex boundary conditions.

The **weak form** relaxes this demand. Instead of requiring the equation to hold pointwise, we require it to hold *on average* — specifically, when integrated against test functions. This might sound like we're giving something up, but we're actually gaining something enormous: the weak form can be solved numerically using the finite element method.

Think of it this way: the strong form is like requiring a student to know the answer at every point on the exam. The weak form is like requiring the student to get the right average score. The average-score requirement is easier to enforce, and for well-posed problems, it's equivalent to the pointwise requirement.

## The Cauchy Momentum Balance in Weak Form

Start with the Cauchy momentum balance for steady Stokes flow:
$$
-\nabla \cdot \sigma + \nabla p = \mathbf{f}, \qquad \nabla \cdot \mathbf{v} = 0
$$

where $\sigma = 2\eta\,\dot{\varepsilon}$ for a Newtonian fluid and $\mathbf{f}$ is the body force.

To get the weak form, we multiply the momentum equation by a **test function** $\mathbf{w}$ (a vector function that vanishes on the boundary where we've prescribed velocities), and integrate over the domain $\Omega$:
$$
\int_\Omega 2\eta\,\dot{\varepsilon}(\mathbf{v}) : \dot{\varepsilon}(\mathbf{w}) \, dV - \int_\Omega p \, (\nabla \cdot \mathbf{w}) \, dV = \int_\Omega \mathbf{f} \cdot \mathbf{w} \, dV + \int_{\Gamma_N} \mathbf{t} \cdot \mathbf{w} \, dS
$$

Similarly, the incompressibility constraint is tested against a scalar test function $q$:
$$
\int_\Omega q \, (\nabla \cdot \mathbf{v}) \, dV = 0
$$

Notice what happened: the second derivatives that were in the strong form got distributed (via integration by parts) between $\mathbf{v}$ and $\mathbf{w}$. We no longer need $\mathbf{v}$ to be twice differentiable — once differentiable is enough. This is the technical payoff of the weak form: weaker smoothness requirements on the solution.

## From Weak Form to Linear System

In the finite element method (next section), we'll approximate $\mathbf{v}$ and $p$ as combinations of basis functions. The weak form then becomes a system of linear equations:
$$
\begin{pmatrix} \mathbf{K} & \mathbf{G}^T \\ \mathbf{G} & \mathbf{0} \end{pmatrix} \begin{pmatrix} \mathbf{u} \\ \mathbf{p} \end{pmatrix} = \begin{pmatrix} \mathbf{f} \\ \mathbf{0} \end{pmatrix}
$$

where $\mathbf{K}$ is the viscosity matrix, $\mathbf{G}$ enforces incompressibility, $\mathbf{u}$ contains the unknown velocities at mesh nodes, and $\mathbf{p}$ contains the unknown pressures. This is a **saddle-point system** — it requires careful choice of finite element spaces to ensure stability (the inf-sup condition).

## Big Ideas

* The strong form demands the equation holds at every point. The weak form only demands it hold on average — when integrated against any test function. For well-posed problems, these two demands are equivalent, but the weak form is far easier to enforce numerically.
* Integration by parts is the magic trick: it redistributes one derivative from the solution onto the test function, halving the smoothness required of each. This is not a compromise — it's what allows FEM to use piecewise-polynomial approximations.
* The incompressibility constraint $\nabla \cdot \mathbf{v} = 0$ enters the weak form as a separate equation tested against pressure test functions, producing the saddle-point structure that couples velocity and pressure.
* The inf-sup condition (or LBB condition) is not bureaucratic pedantry: violate it by choosing the wrong finite element spaces, and your pressure solution will be polluted by spurious oscillations.

## What Comes Next

We have the equations in the right form. Now we need to learn how to actually *solve* them on a computer. That's the finite element method — where we cut the domain into tiny pieces and solve for each one.

## Check Your Understanding

1. What does it mean for a test function $\mathbf{w}$ to "vanish on the boundary where velocities are prescribed"? What would go wrong physically if test functions were allowed to be nonzero on Dirichlet boundaries?
2. The weak form has first derivatives of both $\mathbf{v}$ and $\mathbf{w}$, while the strong form has second derivatives of $\mathbf{v}$ alone. Integration by parts made this trade. Why is first-differentiability easier to achieve in a finite element approximation than second-differentiability?
3. The saddle-point matrix $\begin{pmatrix} \mathbf{K} & \mathbf{G}^T \\ \mathbf{G} & \mathbf{0} \end{pmatrix}$ has a zero block in the lower right. Why is there no pressure-pressure coupling, and what does this say about how pressure is determined in incompressible flow?

## Challenge

Derive the 1D weak form of the equation $-d^2u/dx^2 = f(x)$ on $[0,1]$ with $u(0) = u(1) = 0$. Multiply by a test function $w$ that vanishes at the endpoints, integrate by parts, and write down the resulting bilinear form $a(u,w)$ and linear form $\ell(w)$. Now discretize using piecewise-linear hat functions on a uniform mesh of $n$ elements. Write out the $n-1$ by $n-1$ stiffness matrix $\mathbf{K}$ explicitly and show that it's tridiagonal. For $f = 1$ and $n = 4$, solve the system and compare with the exact solution $u = x(1-x)/2$.
