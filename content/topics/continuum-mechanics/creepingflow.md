# Creeping Flow

## The World Where Nothing Coasts

Imagine swimming through honey. You push forward, but the moment you stop pushing, you stop dead. No gliding, no coasting. The honey is so viscous it instantly kills any momentum. The fluid *forgets its own history*.

This is **creeping flow** (Stokes flow): the regime where viscosity completely dominates over inertia, Re $\ll 1$. It describes heavy oils, cellular-scale biology, magma oozing through rock, and glaciers.

## The Stokes Equations -- Navier-Stokes Without the Hard Part

When Re $\ll 1$, the advective term $(\mathbf{v} \cdot \nabla)\mathbf{v}$ becomes negligible. For steady flow:
$$
\nabla p = \eta \nabla^2 \mathbf{v}, \qquad \nabla \cdot \mathbf{v} = 0
$$

Pressure gradient balances viscous forces. And here's the gorgeous part: this equation is *linear*. Superposition works. Solutions are unique. The math becomes tractable.

That linearity has a wild consequence: creeping flow is **time-reversible**. Reverse all the forces and the flow runs backward through exactly the same states. This is why a scallop opening and closing its shell goes absolutely *nowhere* at low Re -- reciprocal motions produce zero net displacement. Microorganisms need non-reciprocal strokes (helical flagella, flexible cilia) to swim. Nature had to get creative.

## Stokes Drag -- The Formula That Explains Fog

Place a sphere of radius $a$ in a creeping flow at speed $U$. The math spits out:
$$
D = 6\pi \eta a U
$$

And the crazy part is it's *linear in velocity*. Not velocity squared like turbulent drag. Linear. This single result -- **Stokes' drag law** -- tells you:

* Drag $\propto$ velocity (not $v^2$)
* Drag $\propto$ radius (not $r^2$)
* Drag $\propto$ viscosity

It's how we measure fluid viscosity (drop a known sphere, time its fall), estimate sediment settling in water, and understand why fog droplets hang in the air so much longer than raindrops.

## Non-Newtonian Creeping Flow

For non-Newtonian fluids, we use a power-law constitutive relation. The effective viscosity depends on strain rate:
$$
\eta_{\text{eff}} = K \dot{\gamma}^{n-1}
$$

For glaciers, $n \approx 3$ (Glen's flow law), making ice strongly shear-thinning at glacier scales -- it flows more easily where shear rates are high, near the base and valley walls.

[[simulation stokes-flow-demo]]

## Big Ideas

* Creeping flow is the world where viscosity rules absolutely: stop pushing and the fluid stops instantly. No coasting.
* The Stokes equation is linear, which means superposition works and time runs backward just as well as forward. The scallop theorem forbids reciprocal swimming at low Re.
* Stokes' drag $D = 6\pi\eta a U$ is linear in velocity -- the formula behind sedimentation, aerosol dynamics, and viscometry.

## What Comes Next

We've seen what happens when viscosity dominates. Now let's meet the other limit: fluids sitting perfectly still. Pressure, buoyancy, and floating icebergs.

## Check Your Understanding

1. You reverse the pressure gradient driving a Stokes flow around a bump. What happens to the flow field? Contrast this with a high-Reynolds-number flow.
2. Microorganisms can't swim using a single hinged flap that opens and closes. What makes flagella and cilia effective at low Re, and why does a helix work when a flap doesn't?

## Challenge

A sphere of radius $a$ and density $\rho_s$ settles at terminal velocity through a fluid of density $\rho_f$ and viscosity $\eta$. Derive an expression for the terminal velocity using Stokes' drag law. Now consider a suspension of many identical spheres at volume fraction $\phi$ (fraction of space occupied by spheres). The effective viscosity increases as $\eta_{\text{eff}} = \eta(1 + 5\phi/2)$ for dilute suspensions (Einstein's formula). How does the terminal settling velocity change with $\phi$? At what $\phi$ does settling slow down by 10%? What happens physically at high $\phi$ that invalidates both Stokes' law and Einstein's formula?
