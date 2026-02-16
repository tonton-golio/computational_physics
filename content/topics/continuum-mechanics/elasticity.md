# Elasticity



## __Introduction__
From mechanics, we are familiar with *Hooke's Law*. Describing the relationship between force and extention for an object attached to a spring as
$$
F = -kx
$$
where k is the spring constant. This is all well and dandy, but only describes elasticity in one dimension. 
In this topic we expand the law to 3 dimensions.

## __Young's Modulus, Poisson's Ratio and Lamé Coefficients__
Young's Modulus $E$ is a measure of the instretchability of a continuum. Its definition and its relation to the spring constant is
$$
E=\frac{\sigma_{xx}}{u_{xx}}=\frac{F/A}{x/L}=k\frac{L}{A}
$$
where A is the cross-section, L is the total length, F is the force and x is the absolute deformation which mean $x/L$ is the *relative* deformation.


Suppose a block of cheese is deformed by placing a weight on top of it. By intuition, we would assume that when we squeeze, 
Poisson's Ratio is a measure of the relative *shrinking* in the perpendicular direction to the 
## __Generalized Hooke's Law__
The general case, Hookes Law can be expressed as
$$
\sigma = E u
$$
Where E is a rank 4 tensor, relating all the components of the Cauchy strain tensor and the Cauchy stress tensor. 

The problem then becomes given in terms of the force density $f^*$.
$$
f^* = f + \nabla \cdot \sigma
$$
Where $f$ is the body forces like gravity. This is analytically unsolvable for most systems, we can however use finite difference approximations to solve it numerically.

## __Work and Energy__
Work can be defined for deformations in a continuum. 
Its defined by the ":" operator.
$$
W=\sigma:u
$$
where $\sigma$ is the Cauchy stress tensor and $u$ is the Cauchy strain tensor.
The ":" operator takes sum of the elementvise product, i.e.
$$
W=\sum_{ij} \sigma_{ij} u_{ij}
$$
The units will be in Pascal, which essentially is the same as Joules.

## __Linear elastostatics
-\nabla \cdot \sigma = f
\sigma = 2 \mu \epsilon. blah



## __Beam Profile__

## __Slender Rods__

## __Vibrations and Sound__
Sound, wirte EQ 24.5

Examples of vibration; platetectonics and earthquakes.
Bruel and Kjær uses microphones on objects and when vibrated using inverse problems can tell alot of the object.

Two types of waves, L-ongditudinal or P. Divergence and curl free, producing pressure and shear waves.

also known as primary and secondary waves
Pressure faster than shear.

For poisson coefficients of $1/3$, $c_L=2c_P$

Insert picture of plane wave propagation for comparision.
For pressure waves, the polarization vector is aligned with the wavevector $\vec{K}$.

For shear waves, there exists two polarization vectors perpendicular to the wavevector $\vec{K}$.


