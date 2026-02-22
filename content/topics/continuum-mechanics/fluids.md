# Fluids at Rest

## The Cheese-vs-Water Test

Imagine setting a block of cheese on a table. It sits there. Now pour a glass of water on the same table. It goes everywhere.

That's the fundamental difference. A solid resists shear -- push sideways and it pushes back like a spring. A fluid *can't* resist shear. Apply even the tiniest shear stress and it flows. Might flow slowly (honey) or quickly (water), but it flows.

A **fluid** is a material with zero shear modulus. At rest, the only stress it sustains is pressure -- equal in all directions:
$$
\sigma_{ij} = -p \, \delta_{ij}
$$

Negative pressure (compression) equally on every face of every tiny cube.

## Pressure -- The Weight of Everything Above

The force on a small surface element $d\mathbf{S}$ due to pressure:
$$
d\mathbf{F} = -p \, d\mathbf{S}
$$

The minus sign: the fluid pushes *inward*, trying to expand. In a gravitational field, pressure increases with depth. You feel this in a swimming pool -- ears popping as you dive.

Static equilibrium:
$$
\nabla p = \rho \, \mathbf{g}
$$

This is the **hydrostatic equation**. For constant-density fluid with gravity pointing down:
$$
p(z) = p_0 + \rho g (z_0 - z)
$$

Every 10 meters of water adds about 1 atmosphere of pressure.

## Equation of State

For an ideal gas: $p = \rho R T / M_{\text{mol}}$. For liquids, the **bulk modulus** $K = -V \, dp/dV$ tells you how much pressure changes when you compress the fluid. Water: $K \approx 2.2$ GPa, which is why it's nearly incompressible under everyday conditions.

## Buoyancy -- Why Icebergs Float

A floating body displaces exactly its own weight of fluid -- full stop.

Here's why. Drop an object into a fluid. Gravity pulls it down. But pressure on the bottom surface exceeds pressure on the top (because pressure increases with depth). The net upward force is:
$$
\mathbf{F} = \int_V (\rho_{\text{body}} - \rho_{\text{fluid}}) \, \mathbf{g} \, dV
$$

Less dense than the fluid? It floats. Denser? It sinks.

A beautiful way to think about it: imagine replacing the object with an equal volume of fluid. That "fluid object" would be in perfect equilibrium. Now swap in the real object. If it's lighter, net upward force. If heavier, net downward force.

## Stability -- Will It Tip Over?

Floating isn't enough. A floating body must be *stable* -- nudge it, and it rocks back upright rather than capsizing.

Gravity acts through the **center of gravity** $\mathbf{x}_G$. Buoyancy acts through the **center of buoyancy** $\mathbf{x}_B$:

$$
\mathbf{x}_G = \frac{1}{M}\int_V \mathbf{x} \, \rho_{\text{body}} \, dV, \qquad \mathbf{x}_B = \frac{1}{M_{\text{displaced}}}\int_V \mathbf{x} \, \rho_{\text{fluid}} \, dV
$$

The total moment:
$$
\mathbf{M}_{\text{total}} = (\mathbf{x}_G - \mathbf{x}_B) \times M \, \mathbf{g}_0
$$

For stability, tilting the body must create a *restoring* moment. This is why ships have heavy keels (lower $\mathbf{x}_G$) and wide hulls (raise $\mathbf{x}_B$ when tilted).

## Big Ideas

* A fluid can't resist shear. At rest, the only allowed stress is isotropic pressure.
* The hydrostatic equation $\nabla p = \rho\mathbf{g}$ is Cauchy's equation with all velocities set to zero.
* Archimedes' principle: a floating body displaces its own weight of fluid. You don't need to know the shape -- just the mass and the fluid's density.
* Stability comes down to geometry: the center of buoyancy must shift outward faster than the center of gravity when the body tilts.

## What Comes Next

We've studied fluids sitting still. Now we set them in motion -- hoses, tea cups, and wind. The next section brings ideal flows, Euler's equations, and Bernoulli's theorem.

## Check Your Understanding

1. A submarine descends from the surface to 300 m depth in seawater ($\rho \approx 1025$ kg/m^3). By how much does the external pressure on the hull increase?
2. A hollow steel sphere has outer radius 10 cm and wall thickness 2 mm ($\rho_{\text{steel}} \approx 7800$ kg/m^3). Does it float in water? Show your calculation.

## Challenge

A rectangular barge (length $L$, width $W$, height $H$) is made of material with uniform density $\rho_b < \rho_w$. It floats with draft $d$ (depth below waterline). Derive an expression for $d$ in terms of the given quantities. Now tilt the barge by a small angle $\theta$ about its long axis. Find the new centers of gravity and buoyancy as functions of $\theta$, compute the restoring moment, and determine the condition on $H$, $W$, and $d$ for the barge to be stable. What happens to stability as you load the barge with cargo (increasing $\rho_b$ toward $\rho_w$)?
