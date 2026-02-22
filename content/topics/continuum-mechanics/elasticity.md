# Elasticity

## The Cheese Test

Imagine squeezing a block of gouda. It doesn't just compress downward -- it *bulges outward*. Stretch a rubber band lengthwise and it gets thinner in the middle. Forces and deformations couple across dimensions, and a simple spring constant $k$ isn't enough anymore.

Why cheese? Because it's perfect for intuition. Solid enough to hold its shape (unlike honey), soft enough to *see* the deformation with your eyes (unlike steel). A one-cubic-meter block of gouda is our mental laboratory. Put a weight on top and watch.

## Young's Modulus -- How Stiff Is Your Cheese?

Young's modulus $E$ answers: *how hard do I have to pull (per unit area) to stretch it by a given fraction of its length?*

$$
E = \frac{\sigma_{xx}}{\varepsilon_{xx}} = \frac{F/A}{\Delta L / L} = k \frac{L}{A}
$$

It's the spring constant normalized by geometry. Compare any material regardless of shape:

| Material | Young's Modulus |
|----------|----------------|
| Diamond  | $\sim 1000$ GPa |
| Steel    | $\sim 200$ GPa  |
| Bone     | $\sim 15$ GPa   |
| Wood     | $\sim 10$ GPa   |
| Gouda    | $\sim 0.3$ GPa  |
| Rubber   | $\sim 0.01$ GPa |

## Poisson's Ratio -- The Sideways Squeeze

Put that weight on the cheese. It compresses vertically but *bulges outward* horizontally. Poisson's ratio $\nu$ measures this coupling:

$$
\nu = -\frac{\varepsilon_{\text{transverse}}}{\varepsilon_{\text{longitudinal}}}
$$

Most solids: $\nu$ between 0.2 and 0.5. A Poisson's ratio of 0.5 means **incompressible** -- changes shape but not volume (rubber). A ratio of 0 means dimensions are decoupled -- squeezing vertically has no effect on horizontal dimensions. Cork is close to this, which is why it works so brilliantly as a bottle stopper: you can squeeze it into the neck without it fighting back sideways.

## Generalized Hooke's Law -- The Full 3D Picture

In the general case:
$$
\sigma_{ij} = C_{ijkl} \, \varepsilon_{kl}
$$

where $C_{ijkl}$ is a rank-4 **stiffness tensor** with 21 independent components for a general anisotropic material. For an **isotropic** material, it collapses to just two parameters -- the Lame coefficients $\lambda$ and $\mu$:
$$
\sigma_{ij} = \lambda \, \varepsilon_{kk} \, \delta_{ij} + 2\mu \, \varepsilon_{ij}
$$

The connection:
$$
\mu = \frac{E}{2(1+\nu)}, \qquad \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}
$$

## What Happens When You Stretch Too Far?

Hooke's law is *linear* -- double the stress, double the strain. But stretch a rubber band far enough and it snaps. Squeeze cheese hard enough and it crumbles.

Real materials have a **yield point** where linearity breaks down. Beyond it, deformation is *permanent* -- the material doesn't spring back. This is **plastic deformation**. We won't dive deep into plasticity here, but it's important to know that linear elasticity has limits. The von Mises criterion tells you *when* those limits are reached.

## Work and Energy in a Deformed Material

Deform a material and you do work on it, stored as elastic potential energy:
$$
W = \sigma : \varepsilon = \sum_{ij} \sigma_{ij} \, \varepsilon_{ij}
$$

The "$:$" is the **double contraction** -- multiply corresponding elements and sum. Units: $\text{Pa} = \text{J/m}^3$. Energy per unit volume, exactly what you'd expect.

## Linear Elastostatics -- When Nothing Moves

Static equilibrium means forces balance everywhere:
$$
-\nabla \cdot \sigma = \mathbf{f}
$$

Combined with Hooke's law and the strain-displacement relation, this gives the **Navier-Cauchy equation** for elastostatics. Solvable analytically for simple geometries, solvable numerically with FEM for everything else.

Beams are a special case -- slender structures where the top is compressed and the bottom stretched (or vice versa), with a neutral axis in the middle. The key thing to know: beams need four boundary conditions (two at each end), which is why beam problems feel different from the rest of elasticity.

## Vibrations and Sound -- When Elasticity Meets Dynamics

Push a material and let go. If it's elastic, it springs back and *overshoots*. Then back again. This oscillation propagates as a **wave**.

Imagine you're standing on bedrock when an earthquake hits. You feel two distinct jolts. The first is a sharp compression -- the ground pushes and pulls along the direction the wave travels, like a slinky. That's a **P-wave** (primary). A minute later comes a rolling, sideways shudder -- the ground shears perpendicular to the wave direction. That's an **S-wave** (secondary). Their speeds:

$$
c_P = \sqrt{\frac{\lambda + 2\mu}{\rho}}, \qquad c_S = \sqrt{\frac{\mu}{\rho}}
$$

P-waves are always faster ($c_P > c_S$). For $\nu = 1/3$, $c_P = 2\,c_S$. The time delay between them tells seismologists how far away the earthquake was -- it's your tape measure for distance through rock.

## Big Ideas

* Young's modulus and Poisson's ratio completely characterize an isotropic elastic solid -- everything else follows from them.
* Cork's $\nu \approx 0$ makes it the perfect bottle stopper: squeeze it in and it doesn't fight back sideways.
* Two types of elastic waves always exist: P-waves (compressive, faster) and S-waves (shearing, slower). The time gap between them measures earthquake distance.

## What Comes Next

We now have the toolkit for how solids deform and vibrate. Next: the single equation that governs *all* continuous materials -- solids, fluids, and everything in between. The heartbeat of continuum mechanics.

## Check Your Understanding

1. Steel has $E \approx 200$ GPa and $\nu \approx 0.3$. Cork has $\nu \approx 0$. What is special about cork's Poisson ratio, and why does it make cork ideal for wine-bottle stoppers?
2. For a material with $\nu = 1/3$, the P-wave speed is exactly twice the S-wave speed. Show this algebraically using the expressions for $c_P$ and $c_S$ in terms of $\lambda$, $\mu$, and $\rho$.

## Challenge

A seismograph records two distinct arrivals from an earthquake. The P-wave arrives at 09:00:00 and the S-wave arrives at 09:01:30. The crust beneath the seismograph has $E = 70$ GPa, $\nu = 0.25$, and $\rho = 2700$ kg/m^3. Compute the P- and S-wave speeds, estimate the distance to the earthquake, and determine the origin time of the quake. Now suppose a second seismograph 200 km away also records both arrivals -- at what times does each wave arrive there?
