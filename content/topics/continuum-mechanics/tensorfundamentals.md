# Tensor Fundamentals

## The Ant on the Rubber Sheet

Imagine you're an ant walking on a rubber sheet while someone stretches it -- pulling right, squeezing from above. The ground under your feet stretches one way and compresses another. How it deforms depends on *which direction you're facing*. That direction-dependent description of stretching and squeezing? That's a tensor.

A scalar tells you one number at each point. A vector gives you magnitude and direction. A tensor tells you something richer: how a quantity *changes depending on which direction you look*. We need tensors because forces inside a material aren't just big or small -- they act differently in different directions. A beam might be compressed vertically but stretched horizontally, all at the same point.

Let's give these ideas precise names.

## The Cauchy Stress Tensor -- Six Little Hands

Picture cutting a tiny cube out of a stressed material. On each face, the surrounding material pushes and pulls. It's like having **six little hands** pressing and twisting every tiny cube inside the material.

The Cauchy stress tensor $\sigma$ captures all of this. In 3D, it's a $3 \times 3$ matrix:
$$
\sigma = \begin{pmatrix}
\sigma_{11} & \sigma_{12} & \sigma_{13}\\
\sigma_{21} & \sigma_{22} & \sigma_{23}\\
\sigma_{31} & \sigma_{32} & \sigma_{33}\\
\end{pmatrix}
$$

* **Diagonal elements** ($\sigma_{11}$, $\sigma_{22}$, $\sigma_{33}$): **normal stresses** -- pushing or pulling straight into each face. Positive = tension, negative = compression.
* **Off-diagonal elements** ($\sigma_{12}$, etc.): **shear stresses** -- sliding the faces sideways, like rubbing your hands together.

The columns are **traction vectors**: force per unit area on each face. Units: Pascal ($\text{N/m}^2$).

And here's the gorgeous part: if the torques didn't balance, the little cube would spin faster and faster forever. Angular momentum conservation forces the tensor to be symmetric: $\sigma = \sigma^T$. Only 6 independent components in 3D, not 9.

In 2D and 1D, the tensor shrinks:
$$
\sigma_{2D} = \begin{pmatrix}
\sigma_{11} & \sigma_{12}\\
\sigma_{21} & \sigma_{22}\\
\end{pmatrix}, \qquad
\sigma_{1D} = \sigma_{11}
$$

Here's a key insight: **normal and shear stresses are a matter of perspective**. They depend on your coordinate system. There always exists a special basis -- the eigenbasis -- where all shear stresses vanish. The stresses in that basis are the **principal stresses**, the purest description of the stress state at that point.

## The Stress Deviator -- Relative to What?

Sometimes you don't care about total stress -- you care about how it *deviates* from uniform pressure. Think about building a house. The materials were tested at atmospheric pressure. The house stands at atmospheric pressure. You want to know: how much *extra* stress does the structure experience?

The **stress deviator** strips away the uniform pressure:
$$
s = \sigma - p\,\mathbf{I} \qquad \text{where} \qquad p = \frac{1}{3}\text{tr}(\sigma)
$$

Written out:
$$
\begin{pmatrix}
s_{11} & s_{12} & s_{13}\\
s_{21} & s_{22} & s_{23}\\
s_{31} & s_{32} & s_{33}\\
\end{pmatrix}
=
\begin{pmatrix}
\sigma_{11} - p & \sigma_{12} & \sigma_{13}\\
\sigma_{21} & \sigma_{22} - p & \sigma_{23}\\
\sigma_{31} & \sigma_{32} & \sigma_{33} - p\\
\end{pmatrix}
$$

The deviator is what matters for predicting *shape change* and *failure*. Uniform pressure changes volume but doesn't change shape -- it's the deviator that warps, bends, and eventually breaks things.

## Invariants -- What Doesn't Change When You Rotate

The stress tensor looks different in different coordinate systems, but certain quantities stay the same no matter how you rotate. These **invariants** are the truly physical quantities.

For the Cauchy stress tensor:

$I_1 = \sigma_1 + \sigma_2 + \sigma_3$

$I_2 = \sigma_1\sigma_2 + \sigma_2\sigma_3 + \sigma_3\sigma_1$

$I_3 = \sigma_1\sigma_2\sigma_3$

For the stress deviator:

$J_1 = s_{kk} = 0$ (by construction -- we removed the pressure)

$J_2 = \frac{1}{2}\text{tr}(s^2) = \frac{1}{2}\left(\text{tr}(\sigma^2) - \frac{1}{3}\text{tr}(\sigma)^2\right)$

$J_3 = \det(s) = \frac{1}{3}\left(\text{tr}(\sigma^3) - \text{tr}(\sigma^2)\text{tr}(\sigma) + \frac{2}{9}\text{tr}(\sigma)^3\right)$

$J_2$ is the star. It connects to the **von Mises yield criterion**: a practical rule for predicting when a metal will permanently deform. The ratio of shear yield stress to tensile yield stress is:
$$
\frac{\sigma_{\text{shear,yield}}}{\sigma_{\text{tensile,yield}}} = \frac{1}{\sqrt{3}} \approx 0.577
$$

When this holds, permanent deformation begins when:
$$
\sigma_{\text{von Mises}} = \sqrt{3 J_2}
$$
exceeds the yield strength. Works regardless of your coordinates -- that's the power of invariants.

## The Cauchy Strain Tensor -- Measuring Deformation

We've described the *forces*. What about the *deformation*? Push on Rosie the Rubber Band -- how much does she actually stretch?

Consider a velocity field:
$$
\mathbf{v}(x,y,z) = \begin{pmatrix} v_x(x,y,z)\\ v_y(x,y,z)\\ v_z(x,y,z) \end{pmatrix}
$$

The **Cauchy strain tensor** captures how the material deforms:
$$
\epsilon = \begin{pmatrix}
\frac{\partial v_x}{\partial x} & \frac{1}{2}\left(\frac{\partial v_x}{\partial y}+\frac{\partial v_y}{\partial x}\right) & \frac{1}{2}\left(\frac{\partial v_x}{\partial z}+\frac{\partial v_z}{\partial x}\right)\\
\frac{1}{2}\left(\frac{\partial v_x}{\partial y}+\frac{\partial v_y}{\partial x}\right) & \frac{\partial v_y}{\partial y} & \frac{1}{2}\left(\frac{\partial v_y}{\partial z}+\frac{\partial v_z}{\partial y}\right)\\
\frac{1}{2}\left(\frac{\partial v_x}{\partial z}+\frac{\partial v_z}{\partial x}\right) & \frac{1}{2}\left(\frac{\partial v_y}{\partial z}+\frac{\partial v_z}{\partial y}\right) & \frac{\partial v_z}{\partial z}
\end{pmatrix}
$$

The pattern: diagonal entries measure *stretching* along each axis, off-diagonal entries measure *shearing*. The tensor is unitless. Its eigenvectors point in the directions of principal strain, and the eigenvalues tell you the rate of stretching in those directions.

## The Velocity Gradient and Spin Tensor

The strain tensor only captures *half* of what the velocity gradient is doing. The full **velocity gradient tensor** is:
$$
\nabla\mathbf{v} = \begin{pmatrix}
\frac{\partial v_x}{\partial x} & \frac{\partial v_x}{\partial y} & \frac{\partial v_x}{\partial z}\\
\frac{\partial v_y}{\partial x} & \frac{\partial v_y}{\partial y} & \frac{\partial v_y}{\partial z}\\
\frac{\partial v_z}{\partial x} & \frac{\partial v_z}{\partial y} & \frac{\partial v_z}{\partial z}
\end{pmatrix}
$$

This is just the Jacobian. It decomposes into symmetric (strain) and antisymmetric (spin) parts:
$$
\nabla\mathbf{v} = \underbrace{\frac{1}{2}\left(\nabla\mathbf{v} + \nabla\mathbf{v}^T\right)}_{\epsilon \text{ (strain)}} + \underbrace{\frac{1}{2}\left(\nabla\mathbf{v} - \nabla\mathbf{v}^T\right)}_{\omega \text{ (spin)}}
$$

Strain tells you how the material *deforms*. Spin tells you how it *rotates*. Together, they give the complete first-order picture of motion. And here's the punchline: only the strain part does mechanical work.

## Big Ideas

* A tensor is a machine that converts one vector (a surface normal) into another (the force on that surface) -- and it works correctly in any coordinate system.
* The stress tensor has only six independent components because angular momentum conservation forces symmetry.
* The stress deviator isolates what actually changes shape; invariants like $J_2$ let you predict failure regardless of your coordinates.
* Every velocity field splits cleanly into strain (deformation) and spin (rotation) -- only strain does work.

## What Comes Next

We have the language. Now let's use it. Next up: connecting stress and strain through Hooke's law, and seeing how real materials behave when you push on them.

## Check Your Understanding

1. Why must the stress tensor be symmetric? What physical catastrophe would occur if $\sigma_{12} \neq \sigma_{21}$?
2. You rotate your coordinate system by 45 degrees. The stress tensor entries all change. Yet the von Mises stress $\sqrt{3J_2}$ stays the same. Why?
3. The spin tensor $\omega$ is antisymmetric. What physical quantity does it represent, and why does it carry no energy?

## Challenge

For a 2D velocity field $v_x = Ay$, $v_y = 0$ (simple shear at rate $A$), compute the strain tensor $\varepsilon$ and the spin tensor $\omega$. Find the principal strains and the angle of the principal axes. Now imagine you double the shear rate. How do the principal strains change, and how does the principal axis orientation change? What does this tell you about the relationship between shear rate and the direction of maximum extension?
