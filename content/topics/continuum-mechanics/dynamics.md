# The Heartbeat Equation -- Dynamics of Continua

## One Equation to Rule Them All

Imagine you could write a single equation that governs a glacier grinding through a valley, honey dripping off a spoon, *and* a steel beam ringing after being struck. You can. It's the Cauchy momentum equation -- the **heartbeat** of continuum mechanics.

Everything so far -- stress tensors, strain tensors, Hooke's law -- was building toward this moment.

## The Ingredients -- Mass, Momentum, and Forces

For a chunk of material occupying volume $V$:

**Mass**: $M = \int_V \rho \, dV$

**Momentum**: $\mathbf{P} = \int_V \rho \, \mathbf{v} \, dV$

**Angular momentum**: $\mathbf{L} = \int_V \mathbf{x} \times \rho \, \mathbf{v} \, dV$

**Kinetic energy**: $K = \int_V \frac{1}{2} \rho \, v^2 \, dV$

## Conservation of Mass -- Stuff Doesn't Disappear

Mass in a volume $V$ can only change if stuff flows in or out through the surface $S$:
$$
\frac{d}{dt} \int_V \rho \, dV = -\oint_S \rho \, \mathbf{v} \cdot \hat{n} \, dA
$$

Using the divergence theorem:
$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \, \mathbf{v}) = 0
$$

This is the **continuity equation**. Density changes at a point equal the rate mass flows away from it.

## The Material Derivative -- Riding the Flow

There are two ways to watch a river. **Stand on the bank** (Eulerian): the water flows past your fixed spot. **Jump in a boat** (Lagrangian): you ride along with the water.

The **material derivative** $D/Dt$ captures the boat-rider's view:
$$
\frac{Dq}{Dt} = \frac{\partial q}{\partial t} + (\mathbf{v} \cdot \nabla) q
$$

First term: local change at your fixed point. Second term: advective change because you moved somewhere new. Together: the total rate of change experienced by a moving parcel.

Think of it this way: you're in a hot air balloon drifting east. Temperature at your location changes because (1) the air around you heats up, and (2) you're drifting into a region that was already warmer.

For incompressible materials ($\nabla \cdot \mathbf{v} = 0$), the density of each parcel doesn't change as it moves.

## Transport of Any Quantity

The material derivative works for *any* per-unit-mass quantity $q$:
$$
\frac{\partial (\rho q)}{\partial t} + \nabla \cdot (\rho q \, \mathbf{v}) = \rho \frac{Dq}{Dt}
$$

Mass conservation terms cancel, leaving only the "ride along" derivative.

## Cauchy's Equation -- The Heartbeat

Newton's second law for a smear of matter:
$$
\rho \frac{D\mathbf{v}}{Dt} = \mathbf{f} + \nabla \cdot \sigma
$$

That's it. Left side: mass per volume times acceleration (following the flow). $\mathbf{f}$: body forces like gravity. $\nabla \cdot \sigma$: net internal stress force per unit volume.

This is **Cauchy's equation**, and it's universal. Solids and fluids differ only in what $\sigma$ looks like:

* **Elastic solid**: $\sigma = \lambda \, \text{tr}(\varepsilon) \, \mathbf{I} + 2\mu \, \varepsilon$ --> the **Navier-Cauchy equation**.
* **Viscous fluid**: $\sigma = -p\,\mathbf{I} + 2\eta \, \dot{\varepsilon}$ --> the **Navier-Stokes equation**.

Same heartbeat. Different constitutive law. That's the deep unity of continuum mechanics.

> **Reading it physically:** The left side is *how much this blob's velocity changes as it rides the flow*. The right side is *every force acting on it per unit volume* -- gravity, pressure, viscous stresses. Newton's second law, applied to a smear of matter.

[[simulation deformation-grid]]

[[simulation ps-wave-animation]]

## Big Ideas

* The continuity equation is "stuff doesn't disappear" written as a PDE.
* The material derivative $D/Dt$ is the derivative that rides with the flow -- what a boat observer experiences versus a bank observer.
* Cauchy's equation $\rho\,D\mathbf{v}/Dt = \mathbf{f} + \nabla\cdot\sigma$ is Newton's second law for continuous matter. Solids and fluids are just different choices for $\sigma$.

## What Comes Next

We know the general equation of motion. What's the simplest thing we can do with it? Drop the inertia. When viscosity dominates so completely that acceleration is negligible, Cauchy's equation collapses to the Stokes equation -- the world of honey, magma, and glaciers.

## Check Your Understanding

1. Give a physical example where the local term $\partial/\partial t$ is zero but the advective term $(\mathbf{v} \cdot \nabla)$ is nonzero. Then give an example of the reverse.
2. For an incompressible material, $D\rho/Dt = 0$. Show that this implies $\nabla \cdot \mathbf{v} = 0$ using the continuity equation.
3. Cauchy's equation for an elastic solid and a viscous fluid look structurally identical. Where exactly is the difference hiding, and why does it matter so enormously?

## Challenge

Consider a one-dimensional flow where density varies: $\rho(x,t)$ and $v(x,t)$ both depend on position and time. Start from the integral form of mass conservation (rate of change of mass in a control volume equals net flux through its boundaries), apply the divergence theorem, and derive the continuity equation in differential form. Then show that for the special case of steady flow ($\partial/\partial t = 0$), the continuity equation implies $\rho v = \text{const}$ along a streamline. Interpret this result for a converging pipe where the cross-section decreases.
