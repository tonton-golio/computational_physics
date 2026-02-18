# Channels and Pipes

## The Beauty of Exact Solutions

Most of the time, the Navier-Stokes equation is too hard to solve analytically. But for a few special geometries — flows between parallel plates and through circular pipes — the equation simplifies enough that we can solve it exactly, by hand. These aren't just textbook exercises. They're the foundations for understanding everything from blood flow in arteries to glaciers sliding down valleys.

The key simplification: for **steady** ($\partial/\partial t = 0$), **fully developed** (the flow doesn't change along the channel), **unidirectional** flow, the nonlinear advection term $(\mathbf{v} \cdot \nabla)\mathbf{v}$ vanishes. The Navier-Stokes equation reduces to a balance between the pressure gradient (or gravity) and viscous friction.

## Pressure-Driven Channel Flow — The Parabolic Profile

Imagine two infinite parallel plates separated by a distance $2a$, with fluid pushed through by a pressure gradient $G = -dp/dx$. The no-slip boundary condition (fluid velocity is zero at the walls) and symmetry give:
$$
v_x(y) = \frac{G}{2\eta}(a^2 - y^2)
$$

The velocity profile is a **parabola**: fastest in the center, zero at the walls. This is **Poiseuille flow** between parallel plates.

The shape of this parabola tells you about the fluid. For a Newtonian fluid ($n = 1$), it's a perfect parabola. For a **shear-thinning** material ($n < 1$), the profile becomes more "blunted" — flatter in the middle, dropping sharply near the walls. For a **shear-thickening** material ($n > 1$), the profile becomes more "pointed" — sharper in the center.

Glaciers are a dramatic example: ice behaves as a shear-thickening fluid ($n > 1$ in Glen's flow law), so the velocity profile across a glacier valley is remarkably flat in the middle and drops off steeply near the valley walls. Gladys the Glacier knows her flow law.

## Gravity-Driven Planar Flow — The Waterfall Problem

Now tilt the channel. Instead of a pressure gradient pushing the fluid, gravity does the work. Think of rain flowing down a tilted roof, or a thin film of water on an inclined plane.

For a uniform film of thickness $a$ on a plane inclined at angle $\theta$:
$$
0 = g_x + \nu \frac{\partial^2 v_x}{\partial y^2}, \qquad 0 = g_y - \frac{1}{\rho}\frac{\partial p}{\partial y}
$$

where $g_x = g_0 \sin\theta$ drives the flow downhill. For a Newtonian fluid, the velocity profile is again parabolic:
$$
v_x(y) = \frac{g_0 \sin\theta}{2\nu}(2ay - y^2)
$$

Maximum velocity is at the free surface ($y = a$), zero at the wall ($y = 0$). By measuring the velocity profile — for instance, by tracking particles on the surface — you can estimate the power-law exponent $n$ and learn about the fluid's rheology.

## Laminar Pipe Flow — The Hagen-Poiseuille Result

The most practically important exact solution: steady flow through a circular pipe of radius $a$ driven by a pressure gradient $G$. The velocity profile in cylindrical coordinates is:
$$
v_z(r) = \frac{G}{4\eta}(a^2 - r^2)
$$

Again parabolic — fastest on the centerline, zero at the pipe wall. The **volume flow rate** (total discharge) is:
$$
Q = \int_0^a v_z(r) \, 2\pi r \, dr = \frac{\pi G a^4}{8\eta}
$$

This is the **Hagen-Poiseuille law**: flow rate scales with the *fourth power* of the pipe radius. Double the pipe diameter and you get 16 times the flow. This is why your arteries care so much about even a small amount of plaque buildup — a 10% reduction in radius cuts flow by over 34%.

The Reynolds number for pipe flow is Re $= UD/\nu$, where $U$ is the mean velocity and $D = 2a$ is the diameter. For Re $\lesssim 2300$, the flow is laminar and the Poiseuille solution holds. Above that, the flow transitions to turbulence and everything gets much more complicated.

## What We Just Learned

For simple geometries with steady, unidirectional flow, the Navier-Stokes equation yields exact analytical solutions. Channel flow and pipe flow both produce parabolic velocity profiles. The Hagen-Poiseuille law reveals the dramatic fourth-power dependence of flow rate on pipe radius. Non-Newtonian fluids modify the profile shape according to their power-law exponent.

## What's Next

We've seen flows driven by pressure and gravity in confined geometries. The next section takes us to the surface: gravity waves on water, shallow-water equations, and the physics of ocean waves.
