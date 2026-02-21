# Viscous Flow

## The Battle in a High-School Cafeteria

Imagine a high-school cafeteria at lunchtime. A crowd of students is streaming toward the door. Their inertia wants to carry them straight ahead — they've got momentum and they want to keep going. But there's also social friction: people bumping shoulders, friends grabbing friends, the crowd *smearing out* the individual trajectories.

That's the battle at the heart of viscous flow: **inertia** (the tendency to keep going straight) versus **viscosity** (the tendency to smear everything out). The Reynolds number tells you who's winning.

But we're getting ahead of ourselves. Let's start with what viscosity actually is.

## Viscosity — How Sticky Is Your Fluid?

Viscosity is a fluid's resistance to being sheared. Honey has high viscosity (hard to stir). Water has low viscosity (easy to stir). Air has very low viscosity (you barely notice it).

Imagine two parallel plates with fluid between them. You slide the top plate to the right while holding the bottom plate still. The fluid in between develops a velocity gradient — fast near the top plate, stationary near the bottom. **Newton's law of viscosity** says:
$$
\sigma_{xy} = \eta \frac{dv_x}{dy}
$$

The shear stress is proportional to the velocity gradient. The proportionality constant $\eta$ is the **dynamic viscosity** (units: Pa$\cdot$s). Water: $\eta \approx 10^{-3}$ Pa$\cdot$s. Honey: $\eta \approx 2$–$10$ Pa$\cdot$s. The fluid between the plates is being *sheared*, and viscosity determines how hard it pushes back.

A useful variant is the **kinematic viscosity**:
$$
\nu = \frac{\eta}{\rho}
$$

This is the viscosity "per unit density" — it's what shows up most often in the equations. For water: $\nu \approx 10^{-6}$ m$^2$/s. For air: $\nu \approx 1.5 \times 10^{-5}$ m$^2$/s.

## Velocity-Driven Planar Flow — Viscosity as Diffusion

The simplest viscous flow problem: one plate at rest, one plate suddenly set in motion. The velocity profile evolves as:
$$
\frac{\partial v_x}{\partial t} = \nu \frac{\partial^2 v_x}{\partial y^2}
$$

This is a **diffusion equation** — exactly the same form as heat diffusion! Viscosity *diffuses* momentum the same way thermal conductivity diffuses heat. The viscous layer spreads outward from the moving plate at a rate $\sim \sqrt{\nu t}$.

## The Viscous Stress Tensor

For a general incompressible Newtonian fluid, the stress tensor has two parts — pressure and viscous shear:
$$
\sigma_{ij} = -p\,\delta_{ij} + \eta\left(\frac{\partial v_i}{\partial x_j} + \frac{\partial v_j}{\partial x_i}\right)
$$

Or in compact notation:
$$
\sigma = -p\,\mathbf{I} + 2\eta\,\dot{\varepsilon}
$$

where $\dot{\varepsilon}$ is the strain rate tensor. Compare this to the elastic solid: $\sigma = -p\,\mathbf{I} + 2\mu\,\varepsilon$. The *structure* is identical — the difference is that fluids resist the *rate* of deformation while solids resist the deformation itself.

## The Navier-Stokes Equation — Nature's Accounting Book for Fluid Motion

So you want to know how the velocity of a viscous fluid changes over time? Here's the accounting book nature uses. Plug the viscous stress into Cauchy's equation, assume incompressible, isotropic, homogeneous Newtonian fluid:
$$
\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} = \mathbf{g} - \frac{1}{\rho_0}\nabla p + \nu \nabla^2 \mathbf{v}, \qquad \nabla \cdot \mathbf{v} = 0
$$

Every term has a physical meaning:

| Term | What it means |
|------|--------------|
| $\partial \mathbf{v}/\partial t$ | Local acceleration — how velocity changes at a fixed point |
| $(\mathbf{v} \cdot \nabla)\mathbf{v}$ | Advection — momentum carried by the flow itself |
| $\mathbf{g}$ | Gravity (or other body forces) |
| $-\nabla p / \rho_0$ | Pressure gradient — fluid flows from high to low pressure |
| $\nu \nabla^2 \mathbf{v}$ | Viscous diffusion — friction smears out velocity gradients |

The assumptions behind this equation: the fluid is **incompressible** ($\nabla \cdot \mathbf{v} = 0$), **Newtonian** ($\nu$ is constant), and **isotropic** (same material properties in all directions).

These equations are among the most studied in all of physics. They can produce laminar flow, turbulence, boundary layers, vortex streets, and chaos — all from four lines of math.

## The Reynolds Number — Who Wins the Cafeteria Battle?

Back to our cafeteria. The Reynolds number quantifies the competition between inertia and viscosity:
$$
\text{Re} = \frac{|\text{advective term}|}{|\text{viscous term}|} \approx \frac{U^2/L}{\nu U / L^2} = \frac{UL}{\nu}
$$

where $U$ is a characteristic velocity and $L$ is a characteristic length.

* **Re $\ll$ 1**: Viscosity dominates. The flow is smooth, predictable, and *creeping*. Think of bacteria swimming through mucus, or honey pouring off a spoon. Harry the Honey Drop lives here.
* **Re $\sim$ 1**: A fair fight. Inertia and viscosity are comparable.
* **Re $\gg$ 1**: Inertia dominates. The flow becomes chaotic, turbulent, unpredictable. Think of smoke rising from a campfire, or the wake behind a fast boat.

Some examples:

| System | Re |
|--------|----|
| Bacterium swimming | $\sim 10^{-4}$ |
| Blood flow in capillaries | $\sim 10^{-3}$ |
| Honey pouring | $\sim 10^{-1}$ |
| Water in a pipe | $\sim 10^3$ |
| Airplane wing | $\sim 10^7$ |
| Ocean currents | $\sim 10^9$ |

## Newtonian vs. Non-Newtonian — When the Rules Change

Everything above assumes the fluid is **Newtonian**: the stress is linearly proportional to the strain rate. But many fluids don't play by these rules.

* **Shear-thinning** (pseudoplastic, $n < 1$): viscosity *decreases* under shear. Examples: ketchup, blood, paint. This is why you shake the ketchup bottle — the shearing makes it flow.
* **Shear-thickening** (dilatant, $n > 1$): viscosity *increases* under shear. Examples: cornstarch in water (oobleck), wet sand at the beach.

The power-law model captures this: $\sigma = K \dot{\gamma}^n$, where $n$ is the flow behavior index. The Navier-Stokes equation as written above is the special case $n = 1$.

## Big Ideas

* Viscosity is momentum diffusion: just as heat diffuses from hot to cold, momentum diffuses from fast-moving fluid to slow-moving fluid, smoothing out velocity gradients.
* The Navier-Stokes equation adds a viscous diffusion term $\nu\nabla^2\mathbf{v}$ to Euler's equation. That single term is the difference between ideal and real fluids — and it contains multitudes.
* The Reynolds number $\text{Re} = UL/\nu$ is the single most important dimensionless number in fluid mechanics: it tells you the ratio of inertial to viscous forces, and whether the flow will be smooth or turbulent.
* Non-Newtonian fluids (ketchup, blood, glacial ice) break the proportionality between stress and strain rate — and they're far more common in nature than the Newtonian ideal.

## What Comes Next

Armed with the Navier-Stokes equation and the Reynolds number, we can now solve some beautiful exact problems: flow through channels and pipes. These are cases where the geometry is simple enough that we can find analytical solutions — and they give deep insight into how flows behave.

## Check Your Understanding

1. Viscosity has units of Pa·s. Show dimensionally that the viscous term $\nu\nabla^2\mathbf{v}$ in the Navier-Stokes equation has the same units as the acceleration term $\partial\mathbf{v}/\partial t$.
2. The kinematic viscosity of air ($\nu \approx 1.5 \times 10^{-5}$ m²/s) is actually *larger* than that of water ($\nu \approx 10^{-6}$ m²/s), even though air is far less "sticky." How is this possible, and what does it mean physically?
3. A bacterium ($L \sim 2\,\mu$m, $U \sim 30\,\mu$m/s) swims in water. Estimate its Reynolds number and describe qualitatively what its flow environment looks like. What swimming strategy would be useless at this Re?

## Challenge

Consider a one-dimensional unsteady flow where a flat plate is suddenly set in motion at speed $U_0$ at time $t = 0$. The velocity $v_x(y,t)$ satisfies the diffusion equation $\partial v_x/\partial t = \nu\,\partial^2 v_x/\partial y^2$ with $v_x(0,t) = U_0$ and $v_x(\infty,t) = 0$. Use dimensional analysis to argue that the solution must take the form $v_x = U_0\,f(\eta)$ where $\eta = y/\sqrt{\nu t}$. Derive the ODE for $f(\eta)$, identify what boundary conditions it satisfies, and explain physically why the "thickness" of the viscous layer grows as $\sqrt{\nu t}$ rather than linearly in time.
