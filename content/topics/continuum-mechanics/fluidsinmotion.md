# Fluids in Motion



## Introduction

Newtonian and non-newtonianDilatant and pseudo-plastics (Power law coefficient, flow coefficient)
Deivatoric stress and strain rate plot. (n\<1, n=1 n\>1)

## Ideal Flows
Ideal flows are incompressible, has no viscocity (internal friction, ie. no resistance to shear stresses).

[[simulation elastic-wave]]

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
