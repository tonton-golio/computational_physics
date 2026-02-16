# Dynamics



## __Introduction__

Sample text


## __Definitions__
Mass definition
$$
M = \int_V \rho dV
$$

Momentum definition
$$
P = \int_V \rho \vec{v} dV
$$
where v is the specific momentum density, and combined with $\rho$ gives the momentum density


Angular momentum definition
$$
L = \int_V \vec{x} \times \rho \vec{v} dV
$$

Kinetic energy
$$
K = \int_V = \frac{1}{2}  \rho v^2 dV
$$

Conservation of mass
$$
\frac{d}{dt} \int_V \rho dV = - \int_S \rho v \cdot n dA 
$$

Using Gauss divergence theorem gives:

$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \vec{v}) dV = 0
$$
$$
= \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \vec{v})
$$



big D means derivative while travelling along some speed (With respect to a new reference). Named material derivative, comoving derivative. (Not yet defined)
$$
\frac{DM}{Dt} = 0
$$

$$
\frac{DP}{Dt} = - \vec{F}
$$

Q is some macroscopic quantity (Fx. temperature)
$$
Q = \int_V \rho q dV
$$
where q is the microscopic quantity

$$
\frac{DQ}{Dt}= int_V \rho \frac{Dq}{Dt}dV
$$

$$
= \frac{\partial Q}{\partial t} + \dot{Q}_{Boundary}
$$

Imagine a box, with various q's. Imagine the box moving, how much q is lost and how much is gained is given by the boundary.

$$
\frac{\partial}{\partial t} \int_V \rho q dV + \int_S \rho q(v\cdot n)dA
$$
$$
=\frac{\partial}{\partial t} \int_V \rho q dV+ \int_V \nabla \cdot (\rho q \vec{v}) dV
$$
$$
\frac{\partial (\rho q)}{\partial t} + \nabla \cdot (\rho q \vec{v})
$$

$$
= q (\frac{\partial \rho}{\partial t} + \nabla (\rho \vec{v})) + \rho (\frac{\partial q}{\partial t}+ (\vec{v}\cdot \nabla)q)
$$
The first term becomes zero, because of mass conservation leaving

$$
\frac{Dq}{Dt}=\rho (\frac{\partial q}{\partial t}+ (\vec{v}\cdot \nabla)q)
$$
The first term defines the local derivative and the second describes the change as the "box" is moving.
If we imagine flowspeed, the second describes the acceleration

The Lagrangian perspective is the moving perspective, the Eularian perspective is the static one.

$$
\frac{Dp}{Dt}=\vec{F}
$$

$$
\int_V \rho \frac{D\vec{v}}{Dt} dV
$$

$$
\frac{D\rho}{Dt}= \frac{\partial \rho}{\partial t} + (\vec{v}\cdot \vec{nabla})\rho
$$
$$
\frac{\partial \rho}{\partial t} +  \nabla \cdot (\rho v)- \rho(\nabla \cdot v)
$$

The first and second term becomes zero because of mass conservation
giving

$$
\frac{D\rho}{Dt}= - \rho (\nabla \cdot \vec{v})
$$

$$
\frac{Dx}{Dt}= (v \cdot \nabla) \vec{x}= v
$$

$$
\vec{v}=\frac{D\vec{u}}{Dt}=\frac{\partial \vec{U}}{\partial} + (\vec{v}\cdot \nabla)\vec{u}
$$

## __Cauchy Equation
$$
\rho \frac{D\vec{v}}{Dt}=f*
$$
Where f* is the body forces plus the divergence of the stress tensor
writing out the gives

$$
\rho \frac{\partial}
$$
