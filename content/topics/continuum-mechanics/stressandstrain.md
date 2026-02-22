# Stress and Strain

## Getting Physical -- The Rubber Band Experiment

Imagine you've got a thick rubber band between your fingers. Pull it straight -- feel it resist, feel your fingers being tugged together. That's **normal stress**: force perpendicular to the surface, pulling the material apart.

Now twist the rubber band while holding it taut. Feel that sideways tug, the way the material wants to slide past itself? That's **shear stress**: force parallel to the surface.

Every point inside a stressed material experiences some combination of both. The stress tensor is the bookkeeping system that tracks all of it.

## The Stress Tensor -- Forces on Imaginary Surfaces

Imagine slicing through a stressed material with an imaginary plane. Its orientation is described by the outward normal $\hat{n}$. The **traction vector** -- force per unit area on that surface -- is:
$$
t_i = \sigma_{ij} \, n_j
$$

That's the stress tensor's job: give it a direction, get back a force per unit area. It's a machine that converts orientations into forces.

The stress tensor is symmetric ($\sigma_{ij} = \sigma_{ji}$) because if the torques didn't balance, every little cube would spin faster and faster forever -- and we'd all be in trouble. This means six independent components in 3D, not nine.

[[simulation stress-strain-sim]]

## Principal Stresses -- Finding the Sweet Spot

No matter how complicated the stress state, there's always a special orientation where all shear vanishes. In this **principal coordinate system**, the stress tensor is diagonal:
$$
\sigma = \begin{pmatrix} \sigma_1 & 0 & 0 \\ 0 & \sigma_2 & 0 \\ 0 & 0 & \sigma_3 \end{pmatrix}
$$

The values $\sigma_1$, $\sigma_2$, $\sigma_3$ are the **principal stresses** -- the eigenvalues of $\sigma_{ij}$. They tell you the maximum and minimum normal stresses at that point. Finding them is just an eigenvalue problem.

## The Strain Tensor -- Measuring Deformation

When you push on a material, it deforms. The **strain tensor** quantifies that:
$$
\varepsilon_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)
$$

* **Normal strain** ($\varepsilon_{ii}$): fractional change in length along axis $i$. Positive = stretching, negative = compressing.
* **Shear strain** ($\varepsilon_{ij}$, $i \neq j$): change in angle between two originally perpendicular lines. It measures skewing.

## Hooke's Law -- The Simplest Possible Relationship

Stress causes strain, and strain implies stress. What's the relationship?

For a **linear elastic, isotropic** material (same response in all directions, proportional to load), the answer is beautifully simple:
$$
\sigma_{ij} = \lambda \, \varepsilon_{kk} \, \delta_{ij} + 2\mu \, \varepsilon_{ij}
$$

Here $\lambda$ and $\mu$ are the **Lame parameters**. Equivalently, using Young's modulus $E$ and Poisson's ratio $\nu$:
$$
\varepsilon_{ij} = \frac{1+\nu}{E}\,\sigma_{ij} - \frac{\nu}{E}\,\sigma_{kk}\,\delta_{ij}
$$

What do the constants mean?

* **Young's modulus** $E$: how stiff? Steel: $\sim 200$ GPa. Rubber: $\sim 0.01$ GPa.
* **Poisson's ratio** $\nu$: when you stretch in one direction, how much does it thin in the others? Most materials: $\nu \approx 0.3$. Incompressible materials like rubber: $\nu \approx 0.5$.

[[simulation stress-strain-curve]]

## Mohr's Circle -- The World's Most Useful Clock Face

You have a stress state. You want to know: if I cut through the material at some angle, what normal and shear stress will I see? You *could* do the matrix rotation every time. Or you could use **Mohr's circle**.

Given principal stresses $\sigma_1 > \sigma_2 > \sigma_3$:

1. Draw a horizontal axis for normal stress $\sigma_n$ and a vertical axis for shear stress $\tau$.
2. Plot $(\sigma_1, 0)$, $(\sigma_2, 0)$, $(\sigma_3, 0)$ on the horizontal axis.
3. Draw circles connecting pairs: one big outer circle from $\sigma_1$ to $\sigma_3$, and two smaller ones.
4. Every possible stress state on any plane through that point lies *on or between* these circles.

Think of it as a clock face. The position tells you normal and shear stress at a given angle. As you rotate the clock hand, you sweep around the circle.

The maximum shear stress jumps right out:
$$
\tau_{\max} = \frac{\sigma_1 - \sigma_3}{2}
$$

No eigenvalue calculation needed. Draw the circle, read the answer. Engineers have been doing this on napkins for a century.

[[simulation mohr-circle]]

## Big Ideas

* The stress tensor is a machine: feed it a surface orientation, get back force per unit area. Its symmetry is angular momentum conservation in disguise.
* Principal stresses are the eigenvalues -- the stress state in the "sweet spot" basis where all shear vanishes.
* Hooke's law in 3D needs just two material constants ($E$ and $\nu$) because an isotropic solid has two independent ways to store elastic energy.
* Mohr's circle is a geometric picture of how stress rotates under change of basis -- maximum shear stress for free.

## What Comes Next

We've been treating materials as though they bounce back perfectly. In the next section, we push further into elasticity: energy, vibrations, wave speeds, and what happens when you stretch things too far.

## Check Your Understanding

1. Normal strain $\varepsilon_{ii}$ is dimensionless -- it's a ratio of lengths. What does it mean physically if $\varepsilon_{11} = -0.003$?
2. Hooke's law says $\sigma_{ij} = \lambda\varepsilon_{kk}\delta_{ij} + 2\mu\varepsilon_{ij}$. What is the role of the $\lambda\varepsilon_{kk}\delta_{ij}$ term? What would happen to the stress if the material were incompressible?

## Challenge

A thin steel plate is loaded with $\sigma_{11} = 200$ MPa, $\sigma_{22} = -100$ MPa, and $\sigma_{12} = 80$ MPa (plane stress, all other components zero). Draw Mohr's circle, find the principal stresses, find the maximum shear stress, and determine the orientation of the principal axes. Then apply the von Mises criterion: will this stress state cause yielding if the yield strength is 300 MPa?
