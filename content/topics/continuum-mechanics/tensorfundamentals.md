# Tensor Fundamentals

## The Ant on the Rubber Sheet

Imagine you're an ant walking on a rubber sheet. Someone is stretching the sheet — pulling it to the right, squeezing it from above. As you walk, the ground under your feet stretches in one direction and compresses in another. The way the sheet deforms around you depends on *which direction you're facing*. That direction-dependent description of stretching and squeezing — that's a tensor.

A scalar (like temperature) tells you one number at each point. A vector (like velocity) tells you a magnitude and a direction. A tensor tells you something richer: it tells you how a quantity *changes depending on which direction you look*. In continuum mechanics, we need tensors because the forces inside a material aren't just big or small — they act differently in different directions. A beam might be compressed vertically but stretched horizontally, all at the same point.

Now let's give these ideas precise names so we don't have to wave our hands anymore.

## The Cauchy Stress Tensor — Six Little Hands

Imagine cutting a tiny cube out of the interior of a stressed material. On each face of that cube, the surrounding material is pushing and pulling. It's like having **six little hands** pressing and twisting on every tiny cube inside the material.

The Cauchy stress tensor $\sigma$ captures all of this. In 3D, it's a $3 \times 3$ matrix:
$$
\sigma = \begin{pmatrix}
\sigma_{11} & \sigma_{12} & \sigma_{13}\\
\sigma_{21} & \sigma_{22} & \sigma_{23}\\
\sigma_{31} & \sigma_{32} & \sigma_{33}\\
\end{pmatrix}
$$

What do the entries mean?

* **Diagonal elements** ($\sigma_{11}$, $\sigma_{22}$, $\sigma_{33}$) are the **normal stresses** — they push or pull straight into each face of the cube. Positive means tension (pulling apart), negative means compression (squeezing together).
* **Off-diagonal elements** ($\sigma_{12}$, $\sigma_{13}$, etc.) are the **shear stresses** — they slide the faces sideways, like rubbing your hands together.

The columns of $\sigma$ are the **traction vectors**: the force per unit area on each face. Every element has units of Pascal ($\text{N/m}^2$).

For the tensor to be physically consistent (no spontaneously spinning cubes!), angular momentum conservation requires it to be symmetric: $\sigma = \sigma^T$. That means $\sigma_{12} = \sigma_{21}$, and so on — only 6 independent components in 3D, not 9.

In 2D and 1D, the tensor shrinks accordingly:
$$
\sigma_{2D} = \begin{pmatrix}
\sigma_{11} & \sigma_{12}\\
\sigma_{21} & \sigma_{22}\\
\end{pmatrix}, \qquad
\sigma_{1D} = \sigma_{11}
$$

Here's a key insight: **normal and shear stresses are a matter of perspective**. They depend on which coordinate system you choose, which is arbitrary. There always exists a special basis — the eigenbasis — where the stress tensor is purely diagonal and all the shear stresses vanish. The stresses in that basis are called the **principal stresses**, and they represent the purest description of the stress state at that point.

## The Stress Deviator — Relative to What?

Sometimes you don't care about the total stress — you care about how the stress *deviates* from uniform pressure. Think about building a house. The materials were tested at atmospheric pressure. The house will stand at atmospheric pressure. So you want to know: how much *extra* stress does the structure experience beyond the background pressure?

The **stress deviator** strips away the uniform pressure part:
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

The deviator is what matters for predicting *shape change* and *failure*. Uniform pressure changes volume but doesn't change shape — it's the deviator that warps, bends, and eventually breaks things.

## Invariants — What Doesn't Change When You Rotate

The stress tensor looks different in different coordinate systems, but certain quantities remain the same no matter how you rotate your axes. These **invariants** are the truly physical quantities.

For the Cauchy stress tensor, the three invariants are:

$I_1 = \sigma_1 + \sigma_2 + \sigma_3$

$I_2 = \sigma_1\sigma_2 + \sigma_2\sigma_3 + \sigma_3\sigma_1$

$I_3 = \sigma_1\sigma_2\sigma_3$

For the stress deviator:

$J_1 = s_{kk} = 0$ (by construction — we removed the pressure)

$J_2 = \frac{1}{2}\text{tr}(s^2) = \frac{1}{2}\left(\text{tr}(\sigma^2) - \frac{1}{3}\text{tr}(\sigma)^2\right)$

$J_3 = \det(s) = \frac{1}{3}\left(\text{tr}(\sigma^3) - \text{tr}(\sigma^2)\text{tr}(\sigma) + \frac{2}{9}\text{tr}(\sigma)^3\right)$

The $J_2$ invariant is especially important because it connects to the **von Mises yield criterion**: a practical rule for predicting when a material will permanently deform. For many metals, the ratio of shear yield stress to tensile yield stress is:
$$
\frac{\sigma_{\text{shear,yield}}}{\sigma_{\text{tensile,yield}}} = \frac{1}{\sqrt{3}} \approx 0.577
$$

When this holds, the material will start to permanently deform when:
$$
\sigma_{\text{von Mises}} = \sqrt{3 J_2}
$$
exceeds the yield strength. This works regardless of your choice of coordinates — that's the power of invariants.

## The Cauchy Strain Tensor — Measuring Deformation

So far we've described the *forces* inside a material. But what about the *deformation* itself? If you push on Rosie the Rubber Band, how much does she actually stretch?

Consider a velocity field describing motion in a continuum:
$$
\mathbf{v}(x,y,z) = \begin{pmatrix} v_x(x,y,z)\\ v_y(x,y,z)\\ v_z(x,y,z) \end{pmatrix}
$$

The **Cauchy strain tensor** (or strain rate tensor) captures how the material deforms:
$$
\epsilon = \begin{pmatrix}
\frac{\partial v_x}{\partial x} & \frac{1}{2}\left(\frac{\partial v_x}{\partial y}+\frac{\partial v_y}{\partial x}\right) & \frac{1}{2}\left(\frac{\partial v_x}{\partial z}+\frac{\partial v_z}{\partial x}\right)\\
\frac{1}{2}\left(\frac{\partial v_x}{\partial y}+\frac{\partial v_y}{\partial x}\right) & \frac{\partial v_y}{\partial y} & \frac{1}{2}\left(\frac{\partial v_y}{\partial z}+\frac{\partial v_z}{\partial y}\right)\\
\frac{1}{2}\left(\frac{\partial v_x}{\partial z}+\frac{\partial v_z}{\partial x}\right) & \frac{1}{2}\left(\frac{\partial v_y}{\partial z}+\frac{\partial v_z}{\partial y}\right) & \frac{\partial v_z}{\partial z}
\end{pmatrix}
$$

Notice the pattern: the diagonal entries measure *stretching* along each axis, and the off-diagonal entries measure *shearing* — the tendency of the material to skew sideways.

The strain tensor is unitless (it describes *relative* deformation). Its eigenvectors point in the directions of principal strain, and the eigenvalues tell you the rate of stretching in those directions. To get actual lengths, multiply by the physical dimensions of the object.

## The Velocity Gradient and Spin Tensor — The Full Picture

The strain tensor only captures *half* of what the velocity gradient is doing. The full **velocity gradient tensor** is:
$$
\nabla\mathbf{v} = \begin{pmatrix}
\frac{\partial v_x}{\partial x} & \frac{\partial v_x}{\partial y} & \frac{\partial v_x}{\partial z}\\
\frac{\partial v_y}{\partial x} & \frac{\partial v_y}{\partial y} & \frac{\partial v_y}{\partial z}\\
\frac{\partial v_z}{\partial x} & \frac{\partial v_z}{\partial y} & \frac{\partial v_z}{\partial z}
\end{pmatrix}
$$

This is just the Jacobian of the velocity field. It can be decomposed into a symmetric part (the strain tensor) and an antisymmetric part (the **spin tensor**):
$$
\nabla\mathbf{v} = \underbrace{\frac{1}{2}\left(\nabla\mathbf{v} + \nabla\mathbf{v}^T\right)}_{\epsilon \text{ (strain)}} + \underbrace{\frac{1}{2}\left(\nabla\mathbf{v} - \nabla\mathbf{v}^T\right)}_{\omega \text{ (spin)}}
$$

The strain tensor tells you how the material *deforms*. The spin tensor tells you how it *rotates*. Together, they give the complete first-order picture of motion in a continuum.

## Big Ideas

* A tensor is a machine that converts one vector (like a surface normal) into another (like the force on that surface) in a way that works correctly in any coordinate system.
* The stress tensor has only six independent components — not nine — because angular momentum conservation forces it to be symmetric.
* The stress deviator strips away uniform pressure and isolates what actually changes shape; invariants like $J_2$ let you predict failure regardless of how you've drawn your coordinate axes.
* Every velocity field decomposes cleanly into two pieces: a strain part (deformation) and a spin part (rotation) — and only the strain part does mechanical work.

## What Comes Next

We have the language. Now let's use it. In the next section, we'll connect stress and strain through Hooke's law, explore Mohr's circle as a tool for visualizing stress states, and start to see how real materials behave when you push on them.

## Check Your Understanding

1. Why must the stress tensor be symmetric? What physical catastrophe would occur if $\sigma_{12} \neq \sigma_{21}$?
2. You rotate your coordinate system by 45°. The numerical entries of the stress tensor change. Yet the von Mises stress $\sqrt{3J_2}$ stays the same. Why?
3. The spin tensor $\omega$ is antisymmetric. What physical quantity does it represent, and why does it carry no energy?

## Challenge

For a 2D velocity field $v_x = Ay$, $v_y = 0$ (simple shear at rate $A$), compute the strain tensor $\varepsilon$ and the spin tensor $\omega$. Find the principal strains and the angle of the principal axes. Now imagine you double the shear rate. How do the principal strains change, and how does the principal axis orientation change? What does this tell you about the relationship between shear rate and the direction of maximum extension?
