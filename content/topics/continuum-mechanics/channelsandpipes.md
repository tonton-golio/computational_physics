# Channels and Pipes

## When Navier-Stokes Plays Nice

Imagine a doctor staring at an angiogram. There's a tiny ring of plaque narrowing a coronary artery -- just a 10% reduction in radius. Doesn't sound like much. But that innocent-looking ring cuts blood flow by over a third. The physics behind this is one of the most beautiful exact solutions in fluid mechanics.

For **steady**, **fully developed**, **unidirectional** flow, the nonlinear advection term $(\mathbf{v} \cdot \nabla)\mathbf{v}$ vanishes. Navier-Stokes reduces to a balance between the pressure gradient and viscous friction -- solvable by hand.

## Pressure-Driven Channel Flow -- The Parabolic Profile

Two infinite parallel plates separated by distance $2a$, fluid pushed through by pressure gradient $G = -dp/dx$. No-slip at the walls, symmetry:
$$
v_x(y) = \frac{G}{2\eta}(a^2 - y^2)
$$

A **parabola**: fastest in the center, zero at the walls. This is **Poiseuille flow** between plates.

The parabola's shape tells you about the fluid. Newtonian ($n = 1$): perfect parabola. **Shear-thinning** ($n < 1$): flatter in the middle, drops sharply near walls. **Shear-thickening** ($n > 1$): sharper peak in the center.

## Gravity-Driven Planar Flow -- The Waterfall Problem

Tilt the channel. Instead of pressure, gravity drives the flow. Think rain streaming down a tilted roof.

For a film of thickness $a$ on a plane at angle $\theta$:
$$
v_x(y) = \frac{g_0 \sin\theta}{2\nu}(2ay - y^2)
$$

Maximum velocity at the free surface ($y = a$), zero at the wall ($y = 0$).

## Laminar Pipe Flow -- The Hagen-Poiseuille Result

Now the star. Steady flow through a circular pipe of radius $a$ driven by pressure gradient $G$:
$$
v_z(r) = \frac{G}{4\eta}(a^2 - r^2)
$$

Parabolic again. The **volume flow rate**:
$$
Q = \frac{\pi G a^4}{8\eta}
$$

This is the **Hagen-Poiseuille law**: flow rate scales with the *fourth power* of the pipe radius. Double the diameter, get 16 times the flow.

And here's why cardiologists lose sleep: a 10% reduction in artery radius cuts flow by over 34%. A 20% narrowing? Flow drops by more than half. That tiny ring of plaque is a death sentence for downstream tissue unless the heart compensates with higher pressure. The fourth-power law is unforgiving.

The Reynolds number for pipe flow: Re $= UD/\nu$. Below about 2300, the flow is laminar and Poiseuille holds. Above that, turbulence takes over.

[[simulation bernoulli-streamline]]

[[simulation poiseuille-vs-power-law]]

## Big Ideas

* For steady, fully developed flow, Navier-Stokes becomes a simple ODE -- solvable by hand.
* Both pressure-driven and gravity-driven flows produce parabolas for Newtonian fluids. The parabola sharpens ($n > 1$) or blunts ($n < 1$) for non-Newtonian materials.
* The Hagen-Poiseuille law $Q \propto a^4$ is one of the most consequential scaling laws in biology: a 10% reduction in artery radius cuts blood flow by more than a third.

## What Comes Next

We've seen flows driven by pressure and gravity in confined geometries. Next: gravity waves on water, shallow-water equations, and the physics of ocean waves and tsunamis.

## Check Your Understanding

1. A pipe of radius $a$ carries fluid at flow rate $Q$. Replace it with a pipe of radius $2a$ at the same pressure gradient. By what factor does flow rate increase? By what factor does mean velocity change?
2. In gravity-driven planar flow, maximum velocity is at the free surface, not the center. Why? Where would the maximum occur between two no-slip walls?

## Challenge

Blood ($\eta \approx 3 \times 10^{-3}$ Pa$\cdot$s, $\rho \approx 1060$ kg/m^3) flows through a capillary of radius 4 $\mu$m and length 1 mm, driven by a pressure difference of 25 Pa. Compute the volume flow rate using the Hagen-Poiseuille law. Then check the Reynolds number -- is laminar flow a reasonable assumption? Now suppose the capillary narrows to radius 3.5 $\mu$m due to plaque buildup. By what percentage does the flow rate drop at the same pressure difference? What additional pressure would be needed to restore the original flow rate?
