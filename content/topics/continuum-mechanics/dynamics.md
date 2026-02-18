# The Heartbeat Equation — Dynamics of Continua

## One Equation to Rule Them All

Here's something beautiful: whether you're modeling a glacier grinding through a valley, honey dripping off a spoon, or a steel beam vibrating after being struck — the fundamental equation is *the same*. It's the Cauchy momentum equation, and it's the **heartbeat** of continuum mechanics.

Everything we've done so far — stress tensors, strain tensors, Hooke's law — was building toward this. Now we put it all together.

## The Ingredients — Mass, Momentum, and Forces

Before we get to the big equation, let's make sure we know what we're tracking. For a chunk of material occupying volume $V$:

**Mass** — how much stuff is there:
$$
M = \int_V \rho \, dV
$$

**Momentum** — how much "oomph" does it carry:
$$
\mathbf{P} = \int_V \rho \, \mathbf{v} \, dV
$$

**Angular momentum** — how much is it spinning:
$$
\mathbf{L} = \int_V \mathbf{x} \times \rho \, \mathbf{v} \, dV
$$

**Kinetic energy** — how much energy is in the motion:
$$
K = \int_V \frac{1}{2} \rho \, v^2 \, dV
$$

## Conservation of Mass — Stuff Doesn't Disappear

The simplest conservation law: the mass in a volume $V$ can only change if stuff flows in or out through the surface $S$:
$$
\frac{d}{dt} \int_V \rho \, dV = -\oint_S \rho \, \mathbf{v} \cdot \hat{n} \, dA
$$

Using the divergence theorem to convert the surface integral into a volume integral:
$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \, \mathbf{v}) = 0
$$

This is the **continuity equation**. It says: the rate at which density changes at a point equals the rate at which mass flows away from that point. Nothing more, nothing less.

## The Material Derivative — Riding the Flow

Here's a subtlety that trips up everyone the first time. There are two ways to watch a flowing river:

- **Stand on the bank** (Eulerian view): you watch the water flow past you. At your fixed location, the velocity changes over time as different parcels of water arrive.
- **Jump in a boat** (Lagrangian view): you ride along with the water. The velocity you experience changes because you're moving to new locations.

The **material derivative** $D/Dt$ captures the boat-rider's perspective:
$$
\frac{Dq}{Dt} = \frac{\partial q}{\partial t} + (\mathbf{v} \cdot \nabla) q
$$

The first term is the *local* change (what happens at your fixed point). The second term is the *advective* change (what changes because you moved to a new location). Together, they give the total rate of change *experienced by a moving parcel of material*.

Think of it this way: imagine you're in a hot air balloon drifting east. The temperature at your location changes for two reasons: (1) the air around you might be heating up (local change), and (2) you're drifting into a region that was already warmer (advective change).

In the Lagrangian picture, conservation of mass becomes elegantly simple:
$$
\frac{DM}{Dt} = 0
$$

And the material derivative of density:
$$
\frac{D\rho}{Dt} = -\rho (\nabla \cdot \mathbf{v})
$$

For an incompressible material ($\nabla \cdot \mathbf{v} = 0$), the density of each parcel doesn't change as it moves — which makes sense, since incompressibility means volumes don't change.

## Transport of Any Quantity — The General Recipe

The material derivative works for *any* quantity, not just density. If $q$ is some specific (per-unit-mass) quantity like temperature or chemical concentration, then the total amount in a volume is $Q = \int_V \rho \, q \, dV$, and:
$$
\frac{DQ}{Dt} = \int_V \rho \frac{Dq}{Dt} \, dV
$$

This is the general transport equation. Imagine a box full of some quantity $q$, drifting with the flow. What enters and leaves through the boundaries changes $Q$:
$$
\frac{\partial (\rho q)}{\partial t} + \nabla \cdot (\rho q \, \mathbf{v}) = \rho \frac{Dq}{Dt}
$$

This works because the mass conservation terms cancel, leaving only the "ride along" derivative.

## Cauchy's Equation — The Heartbeat

Now we're ready. Newton's second law says momentum changes equal forces. For a continuum:
$$
\rho \frac{D\mathbf{v}}{Dt} = \mathbf{f} + \nabla \cdot \sigma
$$

That's it. That's the heartbeat equation. Let's unpack it:

- **Left side**: mass per unit volume times acceleration (in the material derivative sense — following the flow).
- **$\mathbf{f}$**: body forces, like gravity ($\rho \, \mathbf{g}$).
- **$\nabla \cdot \sigma$**: the divergence of the stress tensor — the net force per unit volume from all the internal stresses acting on the material.

This is **Cauchy's equation**, and it's universal. It doesn't care whether you're dealing with a solid, a liquid, or anything in between. The difference between solids and fluids comes in *later*, when you specify what $\sigma$ looks like:

- **For an elastic solid**: $\sigma = \lambda \, \text{tr}(\varepsilon) \, \mathbf{I} + 2\mu \, \varepsilon$ → you get the **Navier-Cauchy equation**.
- **For a viscous fluid**: $\sigma = -p\,\mathbf{I} + 2\eta \, \dot{\varepsilon}$ → you get the **Navier-Stokes equation**.

Same heartbeat. Different constitutive law. That's the deep unity of continuum mechanics.

## What We Just Learned

Conservation of mass gives the continuity equation. The material derivative lets us track quantities while riding along with the flow. Cauchy's equation is Newton's second law for continua — and it's the same equation for both solids and fluids. The only thing that changes is the constitutive relation between stress and strain.

## What's Next

Now we know the general equation of motion. Let's see what happens when we specialize to fluids — starting with fluids that aren't moving at all. Pressure, buoyancy, floating icebergs: fluids at rest.
