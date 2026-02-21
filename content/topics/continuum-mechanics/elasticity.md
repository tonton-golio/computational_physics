# Elasticity

## The Cheese Test

From basic mechanics, you know Hooke's Law: $F = -kx$. A spring, one dimension, done. But the world isn't one-dimensional. When you squeeze a block of gouda cheese, it doesn't just compress downward — it *bulges outward*. When you stretch a rubber band lengthwise, it gets thinner in the middle. Forces and deformations couple across dimensions, and the simple spring constant $k$ isn't enough anymore.

Why cheese? Because it's *perfect* for building intuition. It's solid enough to hold its shape (unlike honey), but soft enough that you can actually *see* the deformation with your eyes (unlike steel). A one cubic meter block of gouda is our mental laboratory for this section. Put a weight on top and watch what happens.

## Young's Modulus — How Stiff Is Your Cheese?

Young's modulus $E$ answers a simple question: *how hard do I have to pull (per unit area) to stretch the material by a certain fraction of its length?*

$$
E = \frac{\sigma_{xx}}{\varepsilon_{xx}} = \frac{F/A}{\Delta L / L} = k \frac{L}{A}
$$

It's the spring constant, but normalized by geometry — force per unit area divided by relative deformation. This lets you compare materials regardless of their shape or size:

| Material | Young's Modulus |
|----------|----------------|
| Diamond  | $\sim 1000$ GPa |
| Steel    | $\sim 200$ GPa  |
| Bone     | $\sim 15$ GPa   |
| Wood     | $\sim 10$ GPa   |
| Gouda    | $\sim 0.3$ GPa  |
| Rubber   | $\sim 0.01$ GPa |

## Poisson's Ratio — The Sideways Squeeze

Now put that weight on the cheese. It compresses vertically, but it also *bulges outward* horizontally. Poisson's ratio $\nu$ measures this coupling:

$$
\nu = -\frac{\varepsilon_{\text{transverse}}}{\varepsilon_{\text{longitudinal}}}
$$

For most solid materials, $\nu$ is between 0.2 and 0.5. A Poisson's ratio of 0.5 means the material is **incompressible** — it changes shape but not volume (rubber is close to this). A Poisson's ratio of 0 means the dimensions are decoupled — squeezing vertically has no effect on horizontal dimensions (cork is close to this, which is why it works so well as a bottle stopper).

## Generalized Hooke's Law — The Full 3D Picture

In the general case, Hooke's law becomes a tensor equation:
$$
\sigma_{ij} = C_{ijkl} \, \varepsilon_{kl}
$$

where $C_{ijkl}$ is a rank-4 **stiffness tensor** that relates all components of stress to all components of strain. For a general anisotropic material, this has 21 independent components. For an **isotropic** material (same properties in all directions), it collapses to just two parameters — the Lame coefficients $\lambda$ and $\mu$:
$$
\sigma_{ij} = \lambda \, \varepsilon_{kk} \, \delta_{ij} + 2\mu \, \varepsilon_{ij}
$$

The Lame coefficients relate to Young's modulus and Poisson's ratio by:
$$
\mu = \frac{E}{2(1+\nu)}, \qquad \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}
$$

## What Happens When You Stretch Too Far?

Hooke's law is a *linear* relationship — double the stress, double the strain. But you know from experience that this can't be the whole story. Stretch the rubber band far enough and it snaps. Squeeze the cheese hard enough and it crumbles.

Real materials have a **yield point** where the linear relationship breaks down. Beyond it, the material deforms *permanently* — it doesn't spring back. This is **plastic deformation**, and it's where materials science gets interesting. We won't dive deep into plasticity in this course, but it's important to know that linear elasticity has limits. The von Mises criterion from the previous section tells you *when* those limits are reached.

## Work and Energy in a Deformed Material

When you deform a material, you do work on it. That work gets stored as elastic potential energy (like compressing a spring). The work per unit volume is:
$$
W = \sigma : \varepsilon = \sum_{ij} \sigma_{ij} \, \varepsilon_{ij}
$$

The "$:$" operator is the **double contraction** — you multiply corresponding elements and sum them all up. The units work out to $\text{Pa} = \text{J/m}^3$: energy per unit volume, exactly what you'd expect for stored elastic energy.

## Linear Elastostatics — When Nothing Moves

If a material is in static equilibrium (no acceleration), the forces must balance everywhere:
$$
-\nabla \cdot \sigma = \mathbf{f}
$$

where $\mathbf{f}$ is the body force density (like gravity). Combined with Hooke's law ($\sigma = \lambda \, \text{tr}(\varepsilon) \, \mathbf{I} + 2\mu \, \varepsilon$) and the strain-displacement relation, this gives the **Navier-Cauchy equation** for elastostatics. It's analytically solvable for simple geometries, and it's what we'll solve numerically with FEM for everything else.

## Beam Profiles and Slender Rods

Many engineering structures — bridges, buildings, bones — can be modeled as slender beams. When a beam bends under load, the top is compressed and the bottom is stretched (or vice versa). There's a **neutral axis** in the middle where the strain is zero.

The classic Euler-Bernoulli beam theory gives the deflection $w(x)$ of a beam under load:
$$
EI \frac{d^4 w}{dx^4} = q(x)
$$

where $I$ is the second moment of area (a geometric property of the cross-section) and $q(x)$ is the distributed load. This single equation governs how bridges sag, diving boards flex, and tree branches bend in the wind.

## Vibrations and Sound — When Elasticity Meets Dynamics

Push a material and let go. If it's elastic, it springs back — and *overshoots*. Then it springs back again. This oscillation propagates through the material as a **wave**.

There are two kinds of elastic waves:

* **P-waves** (pressure/longitudinal): the material compresses and expands along the direction of propagation, like a slinky being pushed and pulled. The displacement is parallel to the wave vector $\vec{K}$.
* **S-waves** (shear/transverse): the material shears perpendicular to the direction of propagation, like a rope being wiggled. The displacement is perpendicular to $\vec{K}$, with two possible polarizations.

Their speeds are:
$$
c_P = \sqrt{\frac{\lambda + 2\mu}{\rho}}, \qquad c_S = \sqrt{\frac{\mu}{\rho}}
$$

P-waves are always faster than S-waves ($c_P > c_S$). For a Poisson's ratio of $1/3$, $c_P = 2\,c_S$.

This is how earthquakes work. When a fault ruptures, it sends out both P-waves and S-waves. The P-waves arrive first (that's why they're called "primary"), and the S-waves arrive later ("secondary"). The time delay between them tells seismologists how far away the earthquake was. This is also how Bruel & Kjaer uses microphones to characterize the vibrational properties of objects — they listen to the elastic waves and work backward to figure out the material properties (an inverse problem).

## Big Ideas

* Young's modulus and Poisson's ratio are the two numbers that completely characterize an isotropic elastic solid — everything else (Lame coefficients, bulk modulus, wave speeds) follows from them.
* A Poisson ratio of 0.5 means the material is perfectly incompressible: it changes shape without changing volume. A ratio of 0 means the transverse dimensions are completely indifferent to longitudinal loading.
* Elastic energy density is $\sigma : \varepsilon$ — a double contraction that sums up all the work done by every stress component on its corresponding strain.
* Two types of elastic waves always exist: P-waves (compressive, faster) and S-waves (shearing, slower). The time gap between them is your tape measure for earthquake distance.

## What Comes Next

We now have the toolkit to describe how solids deform and vibrate. But before we move to fluids, we need to bring in the computational tools you'll use for the rest of the course. Time to meet Python and its friends.

## Check Your Understanding

1. Steel has $E \approx 200$ GPa and $\nu \approx 0.3$. Cork has $\nu \approx 0$. What is special about cork's Poisson ratio, and why does it make cork ideal for wine-bottle stoppers?
2. For a material with $\nu = 1/3$, the P-wave speed is exactly twice the S-wave speed. Show this algebraically using the expressions for $c_P$ and $c_S$ in terms of $\lambda$, $\mu$, and $\rho$.
3. The Euler-Bernoulli beam equation $EI\,d^4w/dx^4 = q(x)$ is fourth-order. Why does a beam require four boundary conditions, and what are natural choices at a free end versus a clamped end?

## Challenge

A seismograph records two distinct arrivals from an earthquake. The P-wave arrives at 09:00:00 and the S-wave arrives at 09:01:30. The crust beneath the seismograph has $E = 70$ GPa, $\nu = 0.25$, and $\rho = 2700$ kg/m³. Compute the P- and S-wave speeds, estimate the distance to the earthquake, and determine the origin time of the quake. Now suppose a second seismograph 200 km away also records both arrivals — at what times does each wave arrive there?
