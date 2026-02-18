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

## What We Just Learned

The weak form of the Stokes equations replaces pointwise satisfaction with integral satisfaction against test functions. Integration by parts lowers the smoothness requirements on the solution and produces a form ready for finite element discretization. The result is a saddle-point linear system coupling velocity and pressure unknowns.

## What's Next

We have the equations in the right form. Now we need to learn how to actually *solve* them on a computer. That's the finite element method — where we cut the domain into tiny pieces and solve for each one.
