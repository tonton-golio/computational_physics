# Tensor Fundamentals



## Introduction
The Cauchy stress tensor, Cauchy strain tensor, the stress deviator, gradient tensor and spin tensor are essential concepts in the field of continuum mechanics. The Cauchy stress tensor is a mathematical construct used to describe stress in a material. The Cauchy strain tensor describes the deformation of the material due to stress. The Cauchy strain tensor together with the spin tensor gives the combined gradient tensor. 
The stress deviator characterizes the state of stress in a material with respect to a reference pressure. Understanding these concepts is crucial for predicting the behavior of materials under different conditions. In this introduction, we will provide an overview of these concepts and their significance in the study of continuum mechnaics.

## Cauchy Stress Tensor
The Cauchy stress tensor is an mathematical object used to describe how forces propagate through a continuum. For 3 dimensions it is a rank 2 tensor, for 2 dimensions a rank 1 tensor and for 1 dimension a rank 0 tensor, these are usually denoted as
$$
\sigma_{3D} = \begin{pmatrix}
\sigma_{11} & \sigma_{12} & \sigma_{13}\\
\sigma_{21} & \sigma_{22} & \sigma_{23}\\
\sigma_{31} & \sigma_{32} & \sigma_{33}\\
\end{pmatrix},
\sigma_{2D} = \begin{pmatrix}
\sigma_{11} & \sigma_{12}\\
\sigma_{21} & \sigma_{22}\\
\end{pmatrix},
\sigma_{1D} = \sigma_{11}
$$

The diagonal elements represent *normal stresses* along the corresponding basis vectors. The off-diagonal elements represent the *shear stresses* that propagate through the continuum. 
The columns of $\sigma$ are the *traction vectors* of the Cauchy stress tensor. 
Each element of $\sigma$ has units of Pascal $Nm^{-2}$ and in order for it to be physical, symmetry is enforced as $\sigma=\sigma^T$.

For any solids that have forces propagation through them, there exist a basis for the Cauchy stress tensor such that only the diagonal element remain and no shear stresses are experienced. 
Normal and shear stresses are therefore a matter of perspective since they depend on the choosen basis, which is arbitrary. 
This "normal" basis is the eigenbasis of the Cauchy stress tensor.

## Stress Deviator and Invariants
It is convenient to introduce the *stress deviator* which is defined as the Cauchy stress tensor but removing the normal stresses or equivalently the pressure
$$
s=\sigma-p\mathbf{I}\hspace{0.3cm} \text{where}\hspace{0.3cm} p=\frac{1}{3}tr(\sigma)
$$
$$
\begin{pmatrix}
s_{11} & s_{12} & s_{13}\\
s_{21} & s_{22} & s_{23}\\
s_{31} & s_{32} & s_{33}\\
\end{pmatrix}
=
\begin{pmatrix}
\sigma_{11} - p & \sigma_{12} & \sigma_{13}\\
\sigma_{21} & \sigma_{22}- p & \sigma_{23}\\
\sigma_{31} & \sigma_{32} & \sigma_{33}- p\\
\end{pmatrix}
$$
By removing the pressure, the stress deviator allows us to define a new reference pressure. Imagine building a house, modeling how the forces propagate through the building and now having to find suitable materials to carry those forces. Those materials will come with a detailed stress test, but that test has most likely been performed in normal atmospheric conditions and the material was most likely forged in the same. It is also the information that is relevant for the building, as you are not interested in its structural strength in vacuum but at the pressure where it will be build.

Stresses relating to a similar system is therefore usually communicated in terms of the Stress Deviator with some reference pressure. 
The Cauchy stress tensor ($I$) and the stress deviator ($J$) have 3 invariants that are constant under rotation to any other basis.


$I_1=\sigma_1 +\sigma_2+\sigma_3$

$I_2=\sigma_1\sigma_2+\sigma_2\sigma_3+\sigma_3\sigma_1$

$I_3=\sigma_1\sigma_2\sigma_3$

$J_1=s_{kk}=0 $

$J_2=\frac{1}{2}tr(s^2)= \frac{1}{2}\left( tr(\sigma^2)-\frac{1}{3}tr(\sigma)^2\right)$

$J_3=\det(s_{ij}) = \frac{1}{3}\left( tr(\sigma^3)-tr(\sigma^2)tr(\sigma)+\frac{2}{9}tr(\sigma)^3\right)$

Especially the $J_2$ invariant is relevant, as is it linked to the *von Mises yield criterion*. For materials where the ratio of 
$$
\frac{\sigma_{shear,yielding}}{\sigma_{tensile,yielding}}=\frac{1}{\sqrt{3}}\approx 0.577
$$
The von Mises yield criterion can be used to model the failure point of a material when permanent deformation occurs, independent of choice of basis. This is given as
$$
P_{deformation}=\sqrt{3J_2}
$$

## Cauchy Strain Tensor
Similarily to the forces, displacement in position can also be modeled by a tensor, called the *Cauchy strain tensor*. 
Consider a vector field describing motion in a continuum
$$
\mathbf{v}(x,y,z)=\begin{pmatrix} v_x(x,y,z)\\ v_y(x,y,z)\\ v_z(x,y,z) \end{pmatrix}
$$
The Cauchy Strain Tensor is then given as
$$
\epsilon_{3D}=
\begin{pmatrix}
\frac{\partial v_x }{\partial x} & \frac{1}{2}\left( \frac{\partial v_x }{\partial y}+\frac{\partial v_y }{\partial x} \right) & \frac{1}{2}\left( \frac{\partial v_x }{\partial z}+\frac{\partial v_z }{\partial x} \right)\\
\frac{1}{2}\left( \frac{\partial v_x }{\partial y}+\frac{\partial v_y }{\partial x} \right) & \frac{\partial v_y }{\partial y} & \frac{1}{2}\left( \frac{\partial v_y }{\partial z}+\frac{\partial v_z }{\partial y} \right)\\
\frac{1}{2}\left( \frac{\partial v_x }{\partial z}+\frac{\partial v_z }{\partial x} \right) & \frac{1}{2}\left( \frac{\partial v_y }{\partial z}+\frac{\partial v_z }{\partial y} \right) & \frac{\partial v_z }{\partial z}\\
\end{pmatrix}
$$
and for 2 and 1 dimensions
$$
\epsilon_{2D}=
\begin{pmatrix}
\frac{\partial v_x }{\partial x} & \frac{1}{2}\left( \frac{\partial v_x }{\partial y}+\frac{\partial v_y }{\partial x} \right)\\
\frac{1}{2}\left( \frac{\partial v_x }{\partial y}+\frac{\partial v_y }{\partial x} \right) & \frac{\partial v_y }{\partial y}\\
\end{pmatrix},\hspace{3mm}
\epsilon_{1D}=
\frac{\partial v_x }{\partial x}
$$
The Cauchy strain tensor describes deformation in a continua. The eigenbasis of the Cauchy strain tensor has eigenvectors pointing in the direction of displacement and eigenvalues corresponding to the rate of displacement.
It is unitless, and describes relative deformation. To achieve length in units, multiply by or integrate along the corresponding dimension of the object it represents.

## Velocity Gradient and Spin Tensor
The velocity gradient tensor is given as
$$
\nabla\mathbf{v}_{3D}=
\begin{pmatrix}
\frac{\partial v_x }{\partial x} & \frac{\partial v_x }{\partial y} & \frac{\partial v_x }{\partial z} \\
\frac{\partial v_y }{\partial x} & \frac{\partial v_y }{\partial y} & \frac{\partial v_y }{\partial z} \\
\frac{\partial v_z }{\partial x} & \frac{\partial v_z }{\partial y} & \frac{\partial v_z }{\partial z} \\
\end{pmatrix}
$$
and for 2 and 1 dimensions
$$
\nabla\mathbf{v}_{2D}=
\begin{pmatrix}
\frac{\partial v_x }{\partial x} & \frac{\partial v_x }{\partial y}\\
\frac{\partial v_y }{\partial x} & \frac{\partial v_y }{\partial y}\\
\end{pmatrix},\hspace{3mm}
\nabla\mathbf{v}_{1D}=
\frac{\partial v_x }{\partial x}
$$
it gives the full description of the changes in deplacement of a continua and is equilevant to the *Jacobian* with respect to velocity and time. It can be decomposed into the Cauchy strain tensor and a *spin tensor*, corresponding to the symmetric and anti-symmetric component.
$$
\nabla\mathbf{v}=\frac{1}{2}\left(  \nabla\mathbf{v} + \nabla\mathbf{v}^T \ \right) + \frac{1}{2}\left(  \nabla\mathbf{v} - \nabla\mathbf{v}^T \ \right) = \epsilon + \mathbf{\omega}
$$