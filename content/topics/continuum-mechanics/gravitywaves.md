# Gravity Waves

## Ripples, Swells, and Tsunamis

Drop a stone in a pond. Ripples spread outward in concentric circles. Stand on a beach and watch the ocean: long, rolling swells march toward shore, steepen, and break. Somewhere on the other side of the Pacific, a submarine earthquake generates a wave that crosses the entire ocean in hours.

These are all **gravity waves** — waves where gravity provides the restoring force. When water is pushed up above its equilibrium level, gravity pulls it back down, and the resulting oscillation propagates outward.

This section develops the mathematical description of these waves, culminating in the shallow-water equations that govern everything from tidal bores to tsunamis.

## The Shallow-Water Equations

When the wavelength is much larger than the water depth (think tsunamis in the open ocean, tidal flows in harbors, or flood waves in rivers), we can average the flow over the depth and arrive at the **shallow-water equations**:
$$
\frac{\partial v_x}{\partial t} + (v_x \nabla_x + v_y \nabla_y)v_x = -g_0 \nabla_x \eta + f\,v_y
$$
$$
\frac{\partial v_y}{\partial t} + (v_x \nabla_x + v_y \nabla_y)v_y = -g_0 \nabla_y \eta - f\,v_x
$$
$$
\frac{\partial \eta}{\partial t} + \nabla_x(h\,v_x) + \nabla_y(h\,v_y) = 0
$$

Here $\eta$ is the surface elevation, $h$ is the water depth, $f$ is the Coriolis parameter (Earth's rotation matters for large-scale waves), and $v_x$, $v_y$ are the depth-averaged velocities.

The first two equations are momentum conservation (Newton's second law applied to each column of water). The third is mass conservation (the water surface rises where more water flows in than out).

## The Wave Equation and Dispersion

For small-amplitude waves over constant depth $D$, the shallow-water equations linearize to give a **2D wave equation** for the surface elevation:
$$
\left(\frac{\partial^2}{\partial t^2} + f^2\right)\eta - g_0 D \, \nabla_H^2 \eta = 0
$$

Without rotation ($f = 0$), this is a standard wave equation. Waves propagate at speed:
$$
c = \sqrt{g_0 D}
$$

This is the shallow-water wave speed: it depends on depth, not on wavelength. In the deep ocean ($D \approx 4000$ m), this gives $c \approx 200$ m/s $\approx 700$ km/h — which is why tsunamis cross oceans in hours.

The wave period for a wave of wavelength $\lambda$ is:
$$
\tau \approx \frac{\lambda}{\sqrt{g_0 D}}
$$

With rotation ($f \neq 0$), the minimum wave frequency is $f$ — waves slower than one cycle per half-day (at mid-latitudes) are deflected by the Coriolis force into rotating patterns rather than propagating freely.

## Big Ideas

* Gravity waves exist because two things compete: gravity pulls displaced water back to its resting level, and inertia carries it past. The oscillation between these two tendencies is the wave.
* In shallow water ($\text{wavelength} \gg \text{depth}$), the wave speed $c = \sqrt{g_0 D}$ depends only on depth — not on wavelength. All frequencies travel at the same speed, so the wave shape is preserved as it propagates.
* This dispersionless property explains why tsunamis are so dangerous: a pulse of energy crosses entire ocean basins without spreading out, arriving with nearly its original amplitude.
* Earth's rotation (the Coriolis force, parameterized by $f$) deflects large-scale gravity waves into rotating patterns and sets a minimum frequency below which free propagation is impossible.

## What Comes Next

We've now covered the main exact solutions and wave phenomena in fluid mechanics. But what about flows where viscosity absolutely dominates — where the fluid is so sticky or slow that inertia plays no role? That's creeping flow, and it describes everything from honey to glaciers.

## Check Your Understanding

1. The deep Pacific Ocean has an average depth of about 4000 m. Estimate the speed of a tsunami in the open ocean in km/h. How long does it take to cross 8000 km of open water?
2. As a tsunami approaches shore and the water depth decreases from 4000 m to 10 m, by what factor does the wave speed decrease? What must happen to the wave's amplitude to conserve energy flux?
3. What is the Coriolis parameter $f = 2\Omega\sin\phi$ at your latitude $\phi$? ($\Omega = 7.27 \times 10^{-5}$ rad/s.) What is the minimum wave period that can propagate freely without being deflected into a rotating pattern?

## Challenge

A rectangular harbor of length $L = 500$ m and uniform depth $D = 5$ m is closed at one end and open to the ocean at the other. Model the harbor as a 1D resonator and find the natural resonant frequencies using the shallow-water wave equation with appropriate boundary conditions (zero velocity at the closed end, zero surface elevation perturbation at the open end). At which frequencies will the harbor experience resonance (a "harbor seiche")? If an incoming ocean swell has a period of 100 seconds, is the harbor at risk? What harbor geometry would bring a resonant mode to exactly that period?
