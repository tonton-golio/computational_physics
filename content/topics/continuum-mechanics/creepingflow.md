# Creeping Flow

## Creeping Flow - Newtonian Fluids
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

## Creeping Flow - Non-Newtonian Fluids
See slides - Start from Cauchys equations.

