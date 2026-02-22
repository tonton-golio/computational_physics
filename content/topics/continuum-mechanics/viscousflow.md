# Viscous Flow

## The Cafeteria Battle

Imagine a high-school cafeteria at lunchtime. A crowd streams toward the door. Their inertia wants to carry them straight -- they've got momentum. But there's friction: people bumping shoulders, friends grabbing friends, the crowd *smearing out* individual trajectories.

That's the battle at the heart of viscous flow: **inertia** versus **viscosity**. The Reynolds number tells you who's winning.

## Viscosity -- How Sticky Is Your Fluid?

Viscosity is a fluid's resistance to being sheared. Imagine two parallel plates with fluid between them. Slide the top plate while holding the bottom still. The fluid develops a velocity gradient. **Newton's law of viscosity**:
$$
\sigma_{xy} = \eta \frac{dv_x}{dy}
$$

Shear stress proportional to velocity gradient. The constant $\eta$ is **dynamic viscosity** (Pa$\cdot$s). Water: $\eta \approx 10^{-3}$ Pa$\cdot$s. Honey: $\eta \approx 2$--$10$ Pa$\cdot$s.

The **kinematic viscosity** $\nu = \eta/\rho$ is what shows up in the equations. Water: $\nu \approx 10^{-6}$ m$^2$/s. Air: $\nu \approx 1.5 \times 10^{-5}$ m$^2$/s.

## Viscosity as Diffusion

Here's a beautiful connection. Set a plate in motion suddenly. The velocity profile evolves as:
$$
\frac{\partial v_x}{\partial t} = \nu \frac{\partial^2 v_x}{\partial y^2}
$$

That's a **diffusion equation** -- the same form as heat diffusion. Viscosity *diffuses* momentum the way thermal conductivity diffuses heat. The viscous layer spreads at a rate $\sim \sqrt{\nu t}$. That's the boundary-layer thickness scaling, and it's beautiful: the layer grows like the square root of time, not linearly.

## The Viscous Stress Tensor

For a general incompressible Newtonian fluid:
$$
\sigma = -p\,\mathbf{I} + 2\eta\,\dot{\varepsilon}
$$

Compare to the elastic solid: $\sigma = -p\,\mathbf{I} + 2\mu\,\varepsilon$. Identical *structure*. The difference: fluids resist the *rate* of deformation, solids resist deformation itself.

## The Navier-Stokes Equation

Plug the viscous stress into Cauchy's equation for an incompressible, Newtonian fluid:
$$
\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} = \mathbf{g} - \frac{1}{\rho_0}\nabla p + \nu \nabla^2 \mathbf{v}, \qquad \nabla \cdot \mathbf{v} = 0
$$

Every term means something:

| Term | What it means |
|------|--------------|
| $\partial \mathbf{v}/\partial t$ | Local acceleration at a fixed point |
| $(\mathbf{v} \cdot \nabla)\mathbf{v}$ | Momentum carried by the flow itself |
| $\mathbf{g}$ | Gravity |
| $-\nabla p / \rho_0$ | Pressure gradient -- flow from high to low |
| $\nu \nabla^2 \mathbf{v}$ | Viscous diffusion -- friction smears velocity |

These equations can produce laminar flow, turbulence, boundary layers, vortex streets, and chaos -- all from four lines of math.

## The Reynolds Number -- Who Wins?

$$
\text{Re} = \frac{UL}{\nu}
$$

* **Re $\ll$ 1**: Viscosity dominates. Smooth, predictable creeping flow. Bacteria in mucus.
* **Re $\sim$ 1**: A fair fight.
* **Re $\gg$ 1**: Inertia dominates. Chaos and turbulence. Smoke, boat wakes.

| System | Re |
|--------|----|
| Bacterium | $\sim 10^{-4}$ |
| Blood in capillaries | $\sim 10^{-3}$ |
| Honey pouring | $\sim 10^{-1}$ |
| Water in a pipe | $\sim 10^3$ |
| Airplane wing | $\sim 10^7$ |
| Ocean currents | $\sim 10^9$ |

## Newtonian vs. Non-Newtonian

Everything above assumes **Newtonian**: stress linearly proportional to strain rate. Many fluids break this rule.

* **Shear-thinning** ($n < 1$): viscosity *drops* under shear. Ketchup, blood, paint. Shake the bottle -- the shearing makes it flow.
* **Shear-thickening** ($n > 1$): viscosity *rises* under shear. Cornstarch in water (oobleck), wet sand.

The power-law model: $\sigma = K \dot{\gamma}^n$. Navier-Stokes is the special case $n = 1$.

## Big Ideas

* Viscosity is momentum diffusion: it smooths velocity gradients the way thermal conductivity smooths temperature gradients.
* The Reynolds number $\text{Re} = UL/\nu$ is the single most important dimensionless number in fluid mechanics: inertia versus viscosity.
* The boundary-layer thickness grows as $\sqrt{\nu t}$ -- diffusive scaling at its finest.
* Non-Newtonian fluids (ketchup, blood, glacial ice) are far more common in nature than the Newtonian ideal.

## What Comes Next

Armed with the Navier-Stokes equation and the Reynolds number, we can now solve exact problems: flow through channels and pipes. Simple enough geometry for analytical solutions -- and the insight they give is deep.

## Check Your Understanding

1. The kinematic viscosity of air ($\nu \approx 1.5 \times 10^{-5}$ m^2/s) is actually *larger* than water's ($\nu \approx 10^{-6}$ m^2/s), even though air is far less "sticky." How is this possible, and what does it mean physically?
2. A bacterium ($L \sim 2\,\mu$m, $U \sim 30\,\mu$m/s) swims in water. Estimate its Reynolds number. What swimming strategy would be useless at this Re?

## Challenge

Consider a one-dimensional unsteady flow where a flat plate is suddenly set in motion at speed $U_0$ at time $t = 0$. The velocity $v_x(y,t)$ satisfies the diffusion equation $\partial v_x/\partial t = \nu\,\partial^2 v_x/\partial y^2$ with $v_x(0,t) = U_0$ and $v_x(\infty,t) = 0$. Use dimensional analysis to argue that the solution must take the form $v_x = U_0\,f(\eta)$ where $\eta = y/\sqrt{\nu t}$. Derive the ODE for $f(\eta)$, identify what boundary conditions it satisfies, and explain physically why the "thickness" of the viscous layer grows as $\sqrt{\nu t}$ rather than linearly in time.
