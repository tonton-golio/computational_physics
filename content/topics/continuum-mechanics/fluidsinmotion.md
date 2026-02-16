# Fluids in Motion



## __Introduction__

Newtonian and non-newtonianDilatant and pseudo-plastics (Power law coefficient, flow coefficient)
Deivatoric stress and strain rate plot. (n\<1, n=1 n\>1)

## __Ideal Flows__
Ideal flows are incompressible, has no viscocity (internal friction, ie. no resistance to shear stresses).

{{graph:wave-propagation}}

$$
\sigma = -p I
$$
$$
\nabla \cdot \sigma = - \nabla p
$$
where p is pressure.

Inputting into cauchy equation we get a simple description of flow.
$$
\nabla \cdot \vec{v}=0, \hspace{3mm} \frac{D\vec{v}}{Dt}=\vec{g}-\frac{\nabla p}{\rho}
$$
Which gives 4 equations.
From this a poisson equation can be made, which has non-local solutions, ie. if you change the pressure in one place, it must change in all places (consequence of incompressible).

Imagine a pipe that narrows down 

Bernouilli's Theorem.
$$
H= 1/2 * v^2 + \phi p p/ \rho
$$
where $\phi$ is the gravitational field ($zg$)
Must be constant \emph{along a flowline}. It seems similar to a energy conservation. diviing by g gives a total head/energy head. Ie,
$$
\frac{DH}{Dt}=0
$$

$$
\nabla_i H= 1/2  \nabla_i v^2 + \nabla_i \phi + 1/\rho \nabla_i p
$$

Vorticity $\omega$.
$$
\nabla H = \vec{v} \times (\nabla \times \vec{v}) - \frac{\partial\vec{v}}{\partial t}= \omega - \frac{\partial\vec{v}}{\partial t}
$$
How much is it circulating in a particular point

Taking the curl and rearranging gives

$$
\nabla \times \frac{\partial \vec{v}}{\partial t} = \nabla \times (\nabla \times \omega)
$$
Showing that if I have something that is vorticity free, it will remain vorticity free. 

In summation, add the following topics.
Streamfunctions, 2D flows, vorticity, circulation, stokes theorem. Reduce to qualitative conclusions, short chapter on ocean stuff.

## __Viscosity__
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

## __Channels and Pipes__
For steady flows, the navier stokes equation becomes;
$$
EQ 16.1
$$
From this we can derive assuming steady flow and imcompressible liquid, we get __pressure driven channel flow__ between two plates, the velocity field is given as
$$
v_x=\frac{G}{2 \eta}(a^2-y^2)
$$
So the velocity profile at one point between the plates is a parabella. 
For shear thickening material this shape will be more "sharp", for the opposite case it will be more square
__INCLUDE PICTURE__
Also works for other materials than water, ie. glaciers have $n>1$ and its profile is more square.

Reynolds number comment include.

__Gravity-driven planar flow__
Imagine two incline planes where gravity acts on the body of the water (Think Cauchys EQ.)
the solution then becomes
$$
0=g_y - \frac{\nabla_y p}{\rho}\hspace{4mm}0 = g_x + \mu \frac{\partial^2v_x}{\partial y^2}
$$
Observing the profile like above allows one to estimate the power flow law exponent $n$.
For $n=1$ the solution is a parabella 
$$
v_x=\frac{g_0 sin(\theta)}{2\mu}(a^2+y^2)
$$

__Laminar pipe flow__
for laminar pipe flow we have the velocity profile
$$
v_z = \frac{G}{4\eta}(a^2-r^2) \hspace{4mm} EQ 16.29
$$
$$
Q=\
$$


$$
EQ 16.29 + 16.39 + 16.32 + \text{SEE SLIDES FROM ABSA}
$$

## __Gravity Waves__
A ton of assumptions in this one, remember to review the notes from ABSA.

Dispersion law:
$$
\tau \approx \frac{\lambda}{\sqrt{g_0 d}}
$$

Shallow-water equations are:
$$
\frac{\partial v_x}{\partial t} + (v_x\nabla_x+v_y\nabla_y)v_x=-g_0 \nabla_x \eta + f v_y
$$

$$
\frac{\partial v_y}{\partial t} + (v_x\nabla_x+v_y\nabla_y)v_y=-g_0 \nabla_y \eta + f v_x
$$

$$
\frac{\partial v_y}{\partial t} + \nabla_x(hv_x)+ \nabla_y(hv_y)=0
$$
From this we can arrive at the 2D inhomogeneous wave equation in $\eta$ (Wave height).
$$
\left(\frac{\partial^2}{\partial t^2}+f^2\right)\eta - g D\nabla_H^2 \eta=0
$$

## __Creeping Flow - Newtonian Fluids__
For low Reynolds numbers and $\frac{Dv}{Dt}=0$. Examples are Heavy oils, honey and even tight crowds of people. Inertia is insignificant and internal friction dominates.
$$
Re = \frac{|(v\cdot\nabla)v|}{|v\cdot\nabla^2 v|} = \frac{UL}{v}
$$
For low reynolds numbers and steady flows, Navier Stokes EQ reduces 
$$
\nabla p = \eta \nabla^2 \vec{v}\hspace{1cm} \nabla \cdot v = 0
$$
An approximation usually called *Stokes flow*

*Drag and lift on a moving body*

Imagine a body in a moving flow, the reaction force is the only way the fluid can exert force on the body and it comes from the no-slip boundary condition on the body.
$$
R = \oint_S \sigma \cdot dS = D + L
$$
where D is the drag and act in the direction of the asymptotic flow and L is the lift acting perpendicular to the flow.
$$
D = D \hat{e}_U \hspace{1cm} L\cdot \hat{e}_U = 0
$$
Where $\hat{e}_U$ is the direction vector of the flow (normalized)
Additionally there can exist a torque on the body causing it to spin.

The drag term can be split into two terms, viscous forces and the suction, ie. shear and normal stresses. (In the book form=shear and skin)
See EQ 17.4-7.

Books gives a example of a spherical body. Difference between nearly ideal flow and creeping flow becomes that the shear stresses act additinally on the sphere and causes the isobars to stretch much further out into the flow (See fig. 13.4R and 17.1).
The drag on the sphere becomes
$$
D=6\pi \eta a U
$$
Which is a useful result for bodies that can be approximated as spheres in a moving newtonian fluids.

## __Creeping Flow - Non-Newtonian Fluids__
See slides - Start from Cauchys equations.


