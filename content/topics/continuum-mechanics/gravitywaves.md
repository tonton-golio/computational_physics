# Gravity Waves

## The Stone in the Pond

Drop a stone in a pond. Ripples spread outward in concentric circles. Now imagine a fault slipping on the Pacific floor -- the resulting wave crosses the entire ocean in hours, arriving with nearly its original amplitude. Same physics, wildly different scales.

These are **gravity waves**: waves where gravity provides the restoring force. Water gets pushed above its equilibrium level, gravity pulls it back down, inertia carries it past, and the oscillation propagates outward.

## The Shallow-Water Equations

When the wavelength is much larger than the water depth -- tsunamis in the open ocean, tides in harbors, flood waves in rivers -- we can average the flow over depth and arrive at the **shallow-water equations**:
$$
\frac{\partial v_x}{\partial t} + (v_x \nabla_x + v_y \nabla_y)v_x = -g_0 \nabla_x \eta + f\,v_y
$$
$$
\frac{\partial v_y}{\partial t} + (v_x \nabla_x + v_y \nabla_y)v_y = -g_0 \nabla_y \eta - f\,v_x
$$
$$
\frac{\partial \eta}{\partial t} + \nabla_x(h\,v_x) + \nabla_y(h\,v_y) = 0
$$

Here $\eta$ is surface elevation, $h$ is water depth, $f$ is the Coriolis parameter, and $v_x$, $v_y$ are depth-averaged velocities. First two equations: momentum (Newton's law for each water column). Third: mass conservation (surface rises where more water flows in than out).

## The Shallow-Water Wave Speed

For small-amplitude waves over constant depth $D$, the equations linearize to a wave equation. The wave speed:
$$
c = \sqrt{g_0 D}
$$

And here's the punchline: it depends on depth, not wavelength. All frequencies travel at the same speed, so the wave shape is preserved as it propagates. This is why tsunamis are so dangerous -- a pulse of energy crosses entire ocean basins without spreading out.

In the deep ocean ($D \approx 4000$ m): $c \approx 200$ m/s $\approx 700$ km/h. A tsunami crossing 8000 km of open Pacific takes roughly 11 hours.

As it approaches shore and depth decreases, the wave slows down. But energy flux is conserved, so the wave *piles up* -- amplitude grows dramatically. A barely noticeable swell in open water becomes a devastating wall of water near shore.

With Earth's rotation ($f \neq 0$), waves slower than one cycle per half-day get deflected into rotating patterns rather than propagating freely.

[[simulation dispersion-relation]]

## Big Ideas

* Gravity waves exist because gravity and inertia compete: gravity pulls displaced water back, inertia carries it past, and the oscillation propagates.
* In shallow water, wave speed $c = \sqrt{g_0 D}$ depends only on depth -- all frequencies travel together and wave shapes are preserved.
* This dispersionless property makes tsunamis lethal: energy crosses oceans without spreading out.

## What Comes Next

We've covered the main exact solutions and wave phenomena. To solve the Stokes equation for realistic geometries (like Gladys flowing through an irregular valley), we need to reformulate it in **weak form** -- the step that prepares the equation for the finite element method.

## Check Your Understanding

1. Estimate the speed of a tsunami in the open Pacific (depth ~4000 m) in km/h. How long to cross 8000 km?
2. As a tsunami approaches shore and depth decreases from 4000 m to 10 m, by what factor does wave speed decrease? What must happen to amplitude to conserve energy flux?

## Challenge

A rectangular harbor of length $L = 500$ m and uniform depth $D = 5$ m is closed at one end and open to the ocean at the other. Model the harbor as a 1D resonator and find the natural resonant frequencies using the shallow-water wave equation with appropriate boundary conditions (zero velocity at the closed end, zero surface elevation perturbation at the open end). At which frequencies will the harbor experience resonance (a "harbor seiche")? If an incoming ocean swell has a period of 100 seconds, is the harbor at risk? What harbor geometry would bring a resonant mode to exactly that period?
