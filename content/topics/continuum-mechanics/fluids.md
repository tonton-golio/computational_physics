# Fluids at Rest

## What Makes a Fluid a Fluid?

Pick up a block of cheese and set it on the table. It holds its shape. Now pour a glass of water on the table. It spreads everywhere.

That's the fundamental difference between solids and fluids. A solid resists being deformed — apply a shear stress and it pushes back with a restoring force, like a spring. A fluid *can't* resist shear. Apply even the tiniest shear stress and it flows. It might flow slowly (honey) or quickly (water), but it flows.

More precisely: a **fluid** is a material with zero shear modulus. It deforms continuously under any applied shear stress. Fluids resist compression (you can't easily squeeze water into a smaller volume), but they don't resist shearing.

This means fluids respond to stress in a very particular way:
* **Normal stress** (compression/tension) → yes, fluids push back through **pressure**.
* **Shear stress** → no resistance in equilibrium. Any shear causes flow.

At rest, the only stress a fluid can sustain is pressure — equal in all directions, like being squeezed uniformly from all sides. The stress tensor for a fluid at rest is:
$$
\sigma_{ij} = -p \, \delta_{ij}
$$

That's it: negative pressure (compression) equally on every face of every tiny cube.

## Pressure — The Weight of Everything Above

What *is* pressure? At a molecular level, it's the result of billions of molecules bombarding a surface. But in the continuum picture, we don't need molecules — we just need the fact that the fluid pushes back against compression.

The force on a small surface element $d\mathbf{S}$ due to pressure is:
$$
d\mathbf{F} = -p \, d\mathbf{S}
$$

The minus sign means the force points *inward* — the fluid pushes against the surface, trying to expand. The total pressure force on a body submerged in fluid is:
$$
\mathbf{F}_B = -\oint_S p \, d\mathbf{S}
$$

In a fluid at rest in a gravitational field, the pressure increases with depth. You've felt this in a swimming pool — your ears pop as you dive deeper. The condition for static equilibrium is:
$$
\nabla p = \rho \, \mathbf{g}
$$

This is the **hydrostatic equation**: pressure gradient equals body force density. For a constant-density fluid with gravity pointing down:
$$
p(z) = p_0 + \rho g (z_0 - z)
$$

Pressure increases linearly with depth. Every 10 meters of water adds about 1 atmosphere of pressure.

## Equation of State — How Pressure Relates to Density

For many applications, we need to know how pressure relates to other properties of the fluid, like density and temperature. This is the **equation of state**.

For an ideal gas: $p = \rho R T / M_{\text{mol}}$. For liquids, the relationship is more complex — the **bulk modulus** $K = -V \, dp/dV$ tells you how much the pressure changes when you compress the fluid. Water has a bulk modulus of about 2.2 GPa, which is why it's nearly incompressible under everyday conditions.

For advanced applications (like modeling phase transitions), the **van der Waals equation** accounts for molecular interactions and finite molecular size — but for most of this course, we'll treat fluids as either ideal gases or incompressible liquids.

## Buoyancy — Why Icebergs Float

Drop an object into a fluid. Gravity pulls it down. But the pressure of the fluid pushes up on its bottom surface more than it pushes down on its top surface (because pressure increases with depth). The result is an upward **buoyancy force**.

The gravitational force on the body:
$$
\mathbf{F}_G = \int_V \rho_{\text{body}} \, \mathbf{g} \, dV
$$

The buoyancy force from the surrounding fluid pressure:
$$
\mathbf{F}_B = -\oint_S p \, d\mathbf{S}
$$

Combining them gives **Archimedes' principle**:
$$
\mathbf{F} = \mathbf{F}_G + \mathbf{F}_B = \int_V (\rho_{\text{body}} - \rho_{\text{fluid}}) \, \mathbf{g} \, dV
$$

*The buoyant force equals the weight of the displaced fluid.* If the object is less dense than the fluid, it floats. If it's denser, it sinks.

For a uniform gravitational field:
$$
\mathbf{F} = (M_{\text{body}} - M_{\text{displaced fluid}}) \, \mathbf{g}_0
$$

Here's a beautiful way to think about it: imagine replacing the object with an equal volume of fluid. That "fluid object" would be in perfect equilibrium — the buoyancy exactly balances its weight. Now swap in the real object: if it's lighter, there's a net upward force; if heavier, a net downward force.

As the course quote goes: *"If the berg is full of water or if it's full of iceberg, it doesn't matter! Just imagine an iceberg made of water, a so-called waterberg."*

## Stability — Will It Tip Over?

Floating isn't enough. A floating body must also be *stable* — if you nudge it, it should rock back to its original position, not capsize.

Beyond the buoyant force balance, there must also be a balance of **moments**. The gravitational moment acts through the **center of gravity** $\mathbf{x}_G$, and the buoyancy moment acts through the **center of buoyancy** $\mathbf{x}_B$:

$$
\mathbf{x}_G = \frac{1}{M}\int_V \mathbf{x} \, \rho_{\text{body}} \, dV, \qquad \mathbf{x}_B = \frac{1}{M_{\text{displaced}}}\int_V \mathbf{x} \, \rho_{\text{fluid}} \, dV
$$

The total moment is:
$$
\mathbf{M}_{\text{total}} = (\mathbf{x}_G - \mathbf{x}_B) \times M \, \mathbf{g}_0
$$

For stability, this moment must be *restoring*: tilting the body should create a moment that pushes it back upright. This is why ships have heavy keels (to lower $\mathbf{x}_G$) and wide hulls (to raise $\mathbf{x}_B$ when tilted).

## Big Ideas

* A fluid is defined by what it cannot do: resist shear. Any shear stress, no matter how small, causes continuous flow. At rest, the only allowed stress is isotropic pressure.
* The hydrostatic equation $\nabla p = \rho\mathbf{g}$ is just Cauchy's equation with all velocities set to zero. Every 10 meters of water adds roughly one atmosphere of pressure.
* Archimedes' principle is surprisingly deep: a floating body displaces its own weight of fluid — full stop. You don't need to know the body's shape, just its mass and the fluid's density.
* Stability of a floating body comes down to geometry: the center of buoyancy must shift outward faster than the center of gravity when the body tilts.

## What Comes Next

We've studied fluids sitting still. Now we set them in motion. What happens when you turn on a hose, stir a cup of tea, or watch the wind blow? The next section brings us to ideal flows, Euler's equations, and Bernoulli's theorem.

## Check Your Understanding

1. A submarine descends from the surface to 300 m depth in seawater ($\rho \approx 1025$ kg/m³). By how much does the external pressure on the hull increase?
2. A hollow steel sphere has an outer radius of 10 cm and wall thickness of 2 mm ($\rho_{\text{steel}} \approx 7800$ kg/m³). Does it float in water? Show your calculation.
3. A tall ship has its center of gravity high up in the mast. Is this good or bad for stability? What would ship designers do to compensate?

## Challenge

A rectangular barge (length $L$, width $W$, height $H$) is made of material with uniform density $\rho_b < \rho_w$. It floats with draft $d$ (depth below waterline). Derive an expression for $d$ in terms of the given quantities. Now tilt the barge by a small angle $\theta$ about its long axis. Find the new centers of gravity and buoyancy as functions of $\theta$, compute the restoring moment, and determine the condition on $H$, $W$, and $d$ for the barge to be stable. What happens to stability as you load the barge with cargo (increasing $\rho_b$ toward $\rho_w$)?
