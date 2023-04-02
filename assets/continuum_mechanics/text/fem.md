
## __Introduction__
MISSING: Weakform description, strongform description, 

The finite element modelling is a way to solve partial differential equations by approximating the *solution* to the differential equation rather than approximating the differential equation as done in the usual finite difference methods.
Its a method that is utilized in many engineering fields, such as structural analysis, heat transfer and fluid flow.

It works by discretizing the spatial dimensions into smaller subspaces called *finite elements*.
It is great for solving boundary value problems and it is easy to mesh complicated domains.

The *finite* part come from the basisfunctions usual finite structure (I.e. its 0 in all places but around a node).

## __Weighted Residuals__
Consider a 1D differential equation written as
$$
A(u)=f
$$
We then choose an approximate solution of the form
$$
u^N = \sum_{i=1}^N a_i \phi_i(x)
$$
where $\phi_i$ are approximation functions that we may choose and $a_i$ are unknown constants to be determined. 

Putting them together defines the *residual* $r$ of the approximation. The problem is then to choose $a_i$ such that the residual is minimized.
$$
r^N(x)=||A(u^N)-f||_{norm}
$$

The choice of norm defines the *moment* of the solution and will impact the accuracy and spatial resolution of the solution.

## __Least Squares Method__
If we choose the Euclidian norm and define it as
$$
\Pi (r^N) := ||r^N||^2 := \int_0^L (r^N(x))^2 dx
$$
The solution becomes a least squares solution. If we differentiate and constrain it to zero, we obtain the elements of the N by N matrix as
$$
\frac{\partial \Pi}{\partial a_i} = \int_0^L 2r \frac{\partial r}{\partial a_i} dx = 0
$$
which can be solved in the usual manner (See page on Scientific Computing or Applied Statistics).

By increasing the norm, we *punish* larger residuals harder. 

## __Collocation Method__
The least squares method (and for other moments) punishes spatially equally everywhere. This is not the only way, and a whole range of methods exists, generally called *Method of Weighted Residuals*.
We may not be interested in minimizing the *average* residual as done with the least squares method, but minimizing it around a general area. This can be done with the *Collocation method*, where we force the residuals to be zero at a number of locations
$$
r^N(x_i)=0
$$
or equivalently
$$
\int_0^L r^N(x) \delta (x-x_i)dx=0
$$

Imagine working at Würth and recieving complaints from customer service, that the produced screws are constantly breaking around the screwhead.
The job is to determine the minimum amount of steel to be added to reinforce the screw, just below the screwhead. This addition information regarding space of interest, motivates us to choose the collocation method. 
So that we may precisely determine the minimum amount of steel necessary to strengthen the screw exactly below the screwhead.

## __Galerkin's Method__
The most widely used method, is called *Galerkin\'s Method*. To illustrate, consider the relation shown in the above.
$$
u-u^N=e^N \rightarrow u=u^N + e^N
$$
Minimizing the error is now a question of achieving orthogonality, as the error will then be smallest. Unfortunately the error is unknown but we do know the residual. Ofcourse the error and the residual is not identical, but they share enough properties (For example, if one is zero, the other is zero), that we may proceed anyway.
Thus our problem is forcing $r^N \perp u^N$ by
$$
\int_0^L r^N(x)u^N(x)dx = \int_0^L r^N (x) \sum_{i=1}^N a_i \phi_i dx = 0
$$

The basic “recipe” for the Galerkin process is as follows:

Step 1: Compute the residual: $A(u^N)-f=r^N(x)$

Step 2: Force the residual to be orthogonal to each of the approximation functions: $ \int_0^L r^N (x) \sum_{i=1}^N a_i \phi_i dx = 0$

Step 3: Solve the set of coupled equations. The equations will be linear if the
differential equation is linear, and nonlinear if the differential equation is nonlinear.

The primary problem with such a general framework is that it provides no systematic
way of choosing the approximation functions, which is strongly dependent on issues
of possible nonsmoothness of the true solution. The basic Finite Element Method
has been designed to embellish and extend the fundamental Galerkin method by
constructing $\phi_i$ in order to deal with such issues. In particular:

It is based upon Galerkin’s method, It is computationally systematic and efficient and 
It is based on reformulations of the differential equation that remove the problems
of restrictive differentiability requirements.

## __Minimum potential energy principle
The weak problem is known as a variational problem, which is related to that of determning U such that the potential energy function W(u) is minimized

$$
W(u)=1/2 \int \epsilon (u) : \sigma(\epsilon(u))dV - \int f \cdot u dV
$$

If u mnimizes W(u) then any variation of u should lead to larger W(u)

Instead of varying a scalar, we vary a functino to minize a scalar.

Because of linearity, the change in some parameter of the function, will cause a linear change in the function W, ie. the original term plus additional terms. 

By doing the math, we see that the variation look like a derivative, but for a functional.
That means that we can employ similar methods as when minimizing ordinary function. ie. setting the functional derivative to zero and solve. That found function is the true minimizer of the energy.



**For the practical implementation of the finite element modelling, including choice of basisfunction $\phi_i$ and determining $a_i$ using the fenics python package, see the Finite Element Modelling Illustrator and Useful Python Packages topics.**