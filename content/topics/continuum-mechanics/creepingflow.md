# Creeping Flow

## When the Fluid Forgets Its Own History

Imagine swimming through honey. You push forward, but the moment you stop pushing, you stop moving. There's no gliding, no coasting, no inertia to speak of. The honey is so viscous that it instantly dampens any momentum. The fluid *forgets its own history* — it has no memory of what it was doing a moment ago.

This is **creeping flow** (also called **Stokes flow**): flow at very low Reynolds numbers, where viscosity completely dominates over inertia. It describes heavy oils, biological flows at the cellular scale, magma oozing through rock, and — importantly for us — glaciers.

## The Stokes Equations — Navier-Stokes Without the Hard Part

When Re $\ll 1$, the advective term $(\mathbf{v} \cdot \nabla)\mathbf{v}$ in the Navier-Stokes equation becomes negligible. For steady flow ($\partial \mathbf{v}/\partial t = 0$), the equations simplify dramatically:
$$
\nabla p = \eta \nabla^2 \mathbf{v}, \qquad \nabla \cdot \mathbf{v} = 0
$$

This is the **Stokes equation**: pressure gradient balances viscous forces. It's *linear* — which means superposition works, solutions are unique, and the mathematical machinery is much more tractable than for the full Navier-Stokes equation.

The linearity has a strange consequence: creeping flow is **time-reversible**. If you reverse all the forces, the flow runs backwards through exactly the same states. This is why microorganisms can't swim by reciprocal motions (the "scallop theorem") — a scallop opening and closing its shell would go nowhere at Re $\ll 1$.

## Drag and Lift — How the Fluid Pushes Back

Place a body in a creeping flow. The fluid exerts a force on the body through the no-slip boundary condition at its surface:
$$
\mathbf{R} = \oint_S \sigma \cdot d\mathbf{S} = \mathbf{D} + \mathbf{L}
$$

This reaction force splits into:
* **Drag** $\mathbf{D}$: the component in the direction of the flow — it resists the body's motion through the fluid.
* **Lift** $\mathbf{L}$: the component perpendicular to the flow.
* There can also be a **torque** that makes the body spin.

The drag itself has two contributions: **viscous drag** (shear stresses on the surface — the fluid "rubbing" against the body) and **pressure drag** (the pressure difference between the front and back of the body — the "suction" effect).

## The Sphere in Stokes Flow — A Classic Result

For a sphere of radius $a$ moving at speed $U$ through a creeping flow, the drag is:
$$
D = 6\pi \eta a U
$$

This is **Stokes' drag law**, and it's one of the most useful results in fluid mechanics. It tells you:
* Drag is proportional to velocity (not velocity squared, as in high-Re turbulent flow).
* Drag is proportional to radius (not radius squared).
* Drag is proportional to viscosity.

Stokes' law is how we measure the viscosity of fluids (drop a known sphere and time its fall), estimate the settling rate of sediment in water, and understand why fog droplets hang in the air so much longer than raindrops.

In creeping flow, the isobars (lines of constant pressure) stretch far out into the flow — much further than in nearly ideal (high-Re) flow. This is because viscous effects are felt over long distances when inertia is absent.

## Non-Newtonian Creeping Flow

For non-Newtonian fluids in creeping flow, we return to Cauchy's equation and use a power-law constitutive relation instead of the linear Newtonian one. The effective viscosity depends on the strain rate:
$$
\eta_{\text{eff}} = K \dot{\gamma}^{n-1}
$$

For glaciers, $n \approx 3$ (Glen's flow law), making ice a strongly shear-thinning material at glacier scales — it flows more easily where the shear rates are high, which is near the base and the valley walls.

## Big Ideas

* Creeping flow (Re $\ll 1$) is the world of viscosity as the absolute ruler: the moment you stop pushing, the fluid stops moving. There is no coasting.
* The Stokes equation is linear — which is remarkable. Superposition works, solutions are unique, and the math becomes tractable. Linearity is the direct consequence of dropping the nonlinear advection term.
* Time-reversibility is the strangest consequence of linearity: run the forces in reverse, and the flow retraces its path exactly. This is why the "scallop theorem" forbids reciprocal swimming strokes at low Re.
* Stokes' drag law $D = 6\pi\eta a U$ says drag is proportional to velocity (not $v^2$), to size (not size$^2$), and to viscosity. This is the formula behind sedimentation, aerosol dynamics, and viscometry.

## What Comes Next

To solve the Stokes equation for realistic geometries (like Gladys the Glacier flowing through an irregular valley), we need to reformulate it in **weak form** — a step that prepares the equation for numerical solution by the finite element method.

## Check Your Understanding

1. A raindrop of radius 1 mm falls at terminal velocity in air ($\eta \approx 1.8 \times 10^{-5}$ Pa·s, $\rho_{\text{air}} = 1.2$ kg/m³). At terminal velocity, Stokes drag equals the net gravitational force. Estimate the terminal velocity and check whether Re is small enough for Stokes' law to apply.
2. You reverse the direction of the pressure gradient driving a Stokes flow around a bump. What happens to the flow field? Contrast this with what would happen in a high-Reynolds-number flow.
3. Microorganisms can't swim using a single hinged "flap" that opens and closes — the scallop theorem forbids it. What makes flagella and cilia effective at low Re, and why does a helix work when a flap doesn't?

## Challenge

A sphere of radius $a$ and density $\rho_s$ settles at terminal velocity through a fluid of density $\rho_f$ and viscosity $\eta$. Derive an expression for the terminal velocity using Stokes' drag law. Now consider a suspension of many identical spheres at volume fraction $\phi$ (fraction of space occupied by spheres). The effective viscosity increases as $\eta_{\text{eff}} = \eta(1 + 5\phi/2)$ for dilute suspensions (Einstein's formula). How does the terminal settling velocity change with $\phi$? At what $\phi$ does settling slow down by 10%? What happens physically at high $\phi$ that invalidates both Stokes' law and Einstein's formula?
