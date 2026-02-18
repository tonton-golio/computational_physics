# Stress and Strain

## Getting Physical — The Rubber Band Experiment

Before we write a single equation, try this. Take a rubber band and hold it between your fingers. Now pull it straight — you feel it resist, and your fingers are being pulled toward each other. That's **normal stress**: force perpendicular to the surface, pulling the material apart.

Now try twisting the rubber band while holding it taut. Feel that sideways tug, the way the material wants to slide past itself? That's **shear stress**: force parallel to the surface.

Every point inside a stressed material experiences some combination of both. The stress tensor is just the bookkeeping system that tracks all of it.

## The Stress Tensor — Forces on Imaginary Surfaces

Imagine slicing through a stressed material with an imaginary plane. The orientation of that plane is described by its outward normal vector $\hat{n}$. The **traction vector** — the force per unit area that the material on one side of the cut exerts on the other — is:
$$
t_i = \sigma_{ij} \, n_j
$$

This is the stress tensor's job: you give it a direction (the normal to your imaginary cut), and it gives you back the force per unit area on that surface. It's a machine that converts orientations into forces.

The stress tensor is symmetric ($\sigma_{ij} = \sigma_{ji}$) because angular momentum must be conserved — otherwise every little cube of material would start spontaneously spinning. This symmetry means that in 3D, we have six independent stress components, not nine.

[[simulation stress-strain-sim]]

## Principal Stresses — Finding the Sweet Spot

Here's something remarkable: no matter how complicated the stress state, there's always a special orientation where all the shear stresses vanish. In this **principal coordinate system**, the stress tensor is diagonal:
$$
\sigma = \begin{pmatrix} \sigma_1 & 0 & 0 \\ 0 & \sigma_2 & 0 \\ 0 & 0 & \sigma_3 \end{pmatrix}
$$

The values $\sigma_1$, $\sigma_2$, $\sigma_3$ are the **principal stresses** — the eigenvalues of $\sigma_{ij}$. They tell you the maximum and minimum normal stresses at that point, and the eigenvectors tell you which directions they act in.

Finding principal stresses is just an eigenvalue problem. If you know the stress tensor in any coordinate system, you can always rotate to the principal system.

## The Strain Tensor — Measuring Deformation

When you push on a material, it deforms. The **strain tensor** quantifies that deformation. For small displacements $u_i$ from the original position:
$$
\varepsilon_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)
$$

The entries have clear physical meanings:

- **Normal strain** ($\varepsilon_{ii}$): fractional change in length along axis $i$. Positive means stretching, negative means compressing.
- **Shear strain** ($\varepsilon_{ij}$, $i \neq j$): change in angle between two originally perpendicular lines. It measures how much the material *skews*.

## Hooke's Law — The Simplest Possible Relationship

So stress causes strain, and strain implies stress. What's the relationship between them?

For a **linear elastic, isotropic** material (meaning it responds the same way in all directions, and the response is proportional to the load), the relationship is beautifully simple:
$$
\sigma_{ij} = \lambda \, \varepsilon_{kk} \, \delta_{ij} + 2\mu \, \varepsilon_{ij}
$$

Here $\lambda$ and $\mu$ are the **Lame parameters**, material constants that you can look up for steel, rubber, ice, or cheese. Equivalently, using Young's modulus $E$ and Poisson's ratio $\nu$:
$$
\varepsilon_{ij} = \frac{1+\nu}{E}\,\sigma_{ij} - \frac{\nu}{E}\,\sigma_{kk}\,\delta_{ij}
$$

What do these constants mean physically?

- **Young's modulus** $E$: how stiff is the material? Pull on it — $E$ tells you how much force per unit area you need to stretch it by a given fraction. Steel: $E \approx 200$ GPa. Rubber: $E \approx 0.01$ GPa.
- **Poisson's ratio** $\nu$: when you stretch something in one direction, how much does it thin in the other directions? For most materials, $\nu \approx 0.3$. For incompressible materials (like rubber), $\nu \approx 0.5$.

[[simulation stress-strain-curve]]

## Mohr's Circle — The World's Most Useful Clock Face for Stress

You have a stress state. You want to know: if I cut through the material at some angle, what normal stress and shear stress will I see on that surface? You *could* do the matrix rotation by hand every time. Or you could use **Mohr's circle**.

Here's how it works. Given principal stresses $\sigma_1 > \sigma_2 > \sigma_3$:

1. Draw a horizontal axis for normal stress $\sigma_n$ and a vertical axis for shear stress $\tau$.
2. Plot the points $(\sigma_1, 0)$, $(\sigma_2, 0)$, $(\sigma_3, 0)$ on the horizontal axis.
3. Draw circles: one connecting $\sigma_1$ and $\sigma_3$ (the big outer circle), one connecting $\sigma_1$ and $\sigma_2$, and one connecting $\sigma_2$ and $\sigma_3$.
4. Every possible stress state on any plane through that point lies *on or between* these three circles.

Think of it as a clock face for stress. The position on the circle tells you the normal and shear stress on a plane at a given angle. As you "rotate the clock hand" (change the orientation of your imaginary cut), you sweep around the circle.

The maximum shear stress jumps out immediately: it's the radius of the biggest circle:
$$
\tau_{\max} = \frac{\sigma_1 - \sigma_3}{2}
$$

No eigenvalue calculation needed. Just draw the circle, read off the answer. Engineers have been doing this on napkins for over a century.

[[simulation mohr-circle]]

## What We Just Learned

The stress tensor describes internal forces, the strain tensor describes deformation, and Hooke's law connects them linearly for elastic materials. Mohr's circle gives us a graphical shortcut to understand stress transformations without grinding through matrix algebra. These tools work for any material that obeys linear elasticity.

## What's Next

We've been treating materials as though they bounce back perfectly — pure elasticity. But real materials have limits. In the next section, we'll push further into elasticity: Young's modulus, Poisson's ratio, energy, vibrations, and what happens when you stretch a block of gouda cheese.
