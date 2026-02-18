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
- **Drag** $\mathbf{D}$: the component in the direction of the flow — it resists the body's motion through the fluid.
- **Lift** $\mathbf{L}$: the component perpendicular to the flow.
- There can also be a **torque** that makes the body spin.

The drag itself has two contributions: **viscous drag** (shear stresses on the surface — the fluid "rubbing" against the body) and **pressure drag** (the pressure difference between the front and back of the body — the "suction" effect).

## The Sphere in Stokes Flow — A Classic Result

For a sphere of radius $a$ moving at speed $U$ through a creeping flow, the drag is:
$$
D = 6\pi \eta a U
$$

This is **Stokes' drag law**, and it's one of the most useful results in fluid mechanics. It tells you:
- Drag is proportional to velocity (not velocity squared, as in high-Re turbulent flow).
- Drag is proportional to radius (not radius squared).
- Drag is proportional to viscosity.

Stokes' law is how we measure the viscosity of fluids (drop a known sphere and time its fall), estimate the settling rate of sediment in water, and understand why fog droplets hang in the air so much longer than raindrops.

In creeping flow, the isobars (lines of constant pressure) stretch far out into the flow — much further than in nearly ideal (high-Re) flow. This is because viscous effects are felt over long distances when inertia is absent.

## Non-Newtonian Creeping Flow

For non-Newtonian fluids in creeping flow, we return to Cauchy's equation and use a power-law constitutive relation instead of the linear Newtonian one. The effective viscosity depends on the strain rate:
$$
\eta_{\text{eff}} = K \dot{\gamma}^{n-1}
$$

For glaciers, $n \approx 3$ (Glen's flow law), making ice a strongly shear-thinning material at glacier scales — it flows more easily where the shear rates are high, which is near the base and the valley walls.

## What We Just Learned

Creeping flow occurs when viscosity overwhelms inertia (Re $\ll 1$). The Stokes equation is linear, time-reversible, and analytically tractable. Stokes' drag law gives the force on a sphere in creeping flow. Non-Newtonian creeping flows, governed by power-law rheology, describe glaciers and other geophysical systems.

## What's Next

To solve the Stokes equation for realistic geometries (like Gladys the Glacier flowing through an irregular valley), we need to reformulate it in **weak form** — a step that prepares the equation for numerical solution by the finite element method.
