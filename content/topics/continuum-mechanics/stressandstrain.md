# Stress and Strain

## The stress tensor

The **stress tensor** $\sigma_{ij}$ describes the internal forces per unit area within a continuous material. For a surface element with outward normal $\hat{n}$, the traction vector (force per unit area) is:

$$
t_i = \sigma_{ij} n_j.
$$

The stress tensor is symmetric ($\sigma_{ij} = \sigma_{ji}$) as a consequence of angular momentum conservation. This means it has six independent components in 3D.

**Principal stresses** are the eigenvalues of $\sigma_{ij}$. In the principal coordinate system, the stress tensor is diagonal and the shear stresses vanish.

[[simulation stress-strain-sim]]

## The strain tensor

The **strain tensor** $\varepsilon_{ij}$ quantifies deformation. For small displacements $u_i$:

$$
\varepsilon_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right).
$$

- **Normal strain** ($\varepsilon_{ii}$): fractional change in length along axis $i$.
- **Shear strain** ($\varepsilon_{ij}$, $i \neq j$): change in angle between originally perpendicular directions.

## Hooke's law

For a **linear elastic** isotropic material, stress and strain are linearly related:

$$
\sigma_{ij} = \lambda \, \varepsilon_{kk} \, \delta_{ij} + 2\mu \, \varepsilon_{ij},
$$

where $\lambda$ and $\mu$ are the **Lame parameters**. Equivalently, using Young's modulus $E$ and Poisson's ratio $\nu$:

$$
\varepsilon_{ij} = \frac{1+\nu}{E}\sigma_{ij} - \frac{\nu}{E}\sigma_{kk}\delta_{ij}.
$$

- **Young's modulus** $E$: resistance to uniaxial stretching.
- **Poisson's ratio** $\nu$: ratio of transverse contraction to longitudinal extension.

[[simulation stress-strain-curve]]

## Mohr's circle

**Mohr's circle** provides a graphical method to determine the normal and shear stresses on any plane through a material point. Given principal stresses $\sigma_1 > \sigma_2 > \sigma_3$, the state of stress on a plane is represented by a point within three circles in the $(\sigma_n, \tau)$ plane.

The maximum shear stress is $\tau_{\max} = (\sigma_1 - \sigma_3)/2$.

[[simulation mohr-circle]]
