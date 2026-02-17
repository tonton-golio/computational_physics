# Viscous Flow

## Viscosity
Viscosity is a measure of a fluid's resistance to flow, or how easily it flows under stress. It is determined by the internal friction of a fluid and the cohesive forces between its molecules. High viscosity fluids are thick and resist deformation, while low viscosity fluids are thin and flow easily.

Write down all assumptions for the various descriptions.

*Newton's Law of Viscosity*, also known as shear or dynamics viscosity:
$$
\sigma_{xy}(y) = \eta \frac{dv_x(y)}{dy}
$$
Gives a linear relationship between the gradient and the stress. $\mu$ has units of  in units of Pa/s. Is used to model

*Kinematic viscosity*
For example used in gases
$$
v = \frac{\eta}{\rho}
$$

*Velocity-driven planar flow*:
$$
\frac{\partial v_x}{\partial t}= v\frac{\partial^2 v_x}{\partial y^2}
$$
Which is a typical diffusion equation, similar to a temperature diffusion equation.

*Isotrpoic viscous stress*
$$
\sigma = - p I + 2\eta \dot{\epsilon}
$$
where E is the strain rate tensor.
$$
\sigma_{ij}= - p \delta_{ij}+ \eta (\nabla_i v_j + \nabla_j v_i)
$$

*Incompressible, isotrpoic, homogenous fluids - Navier stokes description*.
$$
\frac{\partial \vec{v}}{\partial t} + (\vec{v}\cdot\nabla)\vec{v}= \vec{g}- \frac{1}{\rho_0}\nabla p + v\nabla^2 \vec{v}, \hspace{5mm}\nabla \cdot \vec{v}=0
$$
Assumptions: Divergence free $\nabla \cdot v = 0$, newtonian liquid $v$ is constant, isotropic ie. the same material properties everywhere.

Pseudo plastics and dilatant materials. Write description.

Reynolds number: The ratio of advective term and friction term in the navier stokes description.
$$
Re\approx-\frac{|(\vec{v}\cdot\nabla)\vec{v}|}{\vec{v}\nabla^2\vec{v}}\approx\frac{U^2/L}{vU/L^2}=\frac{UL}{v}
$$
Non-dimensional navier stokes description with reynolds number.
$$
(\vec{v}\cdot \nabla)\vec{v}= -  \nabla \vec{p}
$$
