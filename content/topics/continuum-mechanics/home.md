# Continuum Mechanics

## The Story So Far

Everything around you — the water in your glass, the ground beneath your feet, the glacier carving through a valley in Greenland — is made of atoms. Trillions upon trillions of them. But here's the thing: you don't need to track every single atom to understand how a bridge holds up, how honey pours, or why an iceberg floats. You just need to *pretend the world is smooth*.

That's the big trick of continuum mechanics. We ignore the atoms — the same way you ignore individual raindrops when you say "it's raining" — and instead treat matter as a continuous, smooth substance. From that single act of deliberate near-sightedness, an entire universe of physics unfolds.

This course is the story of that unfolding.

## The Journey

Here's where we're going, step by step:

**Act I — Setting the Stage**

We start by asking: *when can we get away with pretending matter is smooth?* Then we learn the mathematical language that lets us talk precisely about pushing, pulling, and deforming.

**Act II — Solids Fight Back**

We push on things and watch them push back. Stress, strain, elasticity — this is where you learn why cheese deforms, bridges hold, and earthquakes shake.

**Act III — Everything Moves**

The Cauchy equation — the heartbeat of continuum mechanics — shows up. It's the same equation whether you're squeezing a rubber ball or stirring honey. We see how solids and fluids are two sides of the same coin.

**Act IV — Fluids Flow**

We pour the honey. Pressure, buoyancy, ideal flows, then viscous flows. Beautiful cases we can solve by hand: channels, pipes, waves.

**Act V — When Things Get Sticky**

Creeping flow. Stokes equations. The world slows down and viscosity takes over. We develop the weak formulation and build the bridge to computation.

**Act VI — The Computer Takes Over**

For everything we can't solve by hand (which is most of the real world), we learn the finite element method — cutting the world into tiny triangles and stitching the answers together.

## Learning Trajectory

1. **Continuum Approximation** — When and why we can pretend matter is smooth.
2. **Tensor Fundamentals** — The language of pushing and pulling: stress tensors, strain tensors, and how they transform.
3. **Stress and Strain** — Traction vectors, principal stresses, Mohr's circle, and Hooke's law.
4. **Elasticity** — Young's modulus, Poisson's ratio, work, energy, vibrations, and the speed of sound.
5. **Python Packages** — Our computational toolkit. You'll need it from here on.
6. **The Heartbeat Equation** — Conservation laws, the material derivative, and Cauchy's equation — the one equation that rules them all.
7. **Fluids at Rest** — What is a fluid? Pressure, buoyancy, and why icebergs float the way they do.
8. **Fluids in Motion** — Ideal flows, Euler's equations, Bernoulli's theorem, and vorticity.
9. **Viscous Flow** — Viscosity, the Navier-Stokes equation, and the Reynolds number.
10. **Channels and Pipes** — Pressure-driven flow, gravity-driven flow, and laminar pipe flow. Beautiful exact solutions.
11. **Gravity Waves** — Shallow-water equations and dispersion.
12. **Creeping Flow** — Stokes flow: when viscosity dominates and inertia vanishes. Like swimming through honey.
13. **Weak Stokes Formulation** — Recasting the equations in weak form, preparing for computation.
14. **Finite Element Method** — Weighted residuals, Galerkin's method, and how we actually simulate glaciers and airplane wings.
15. **The Grand Finale** — Every big idea retold as one story. A glacier, an iceberg, and an earthquake walk into a bar...

## Meet the Cast

Throughout these notes, you'll run into three recurring characters:

- **Rosie the Rubber Band** — she stretches, she snaps back, she knows all about elasticity.
- **Harry the Honey Drop** — he flows, he creeps, he's impossibly viscous and proud of it.
- **Gladys the Glacier** — she's enormous, she's slow, and she's the real-world test case for everything we learn.

When the math gets abstract, these three will keep you grounded.

## Why This Matters

- Understanding stress and strain is essential for engineering, geophysics, and biomechanics.
- Fluid dynamics governs weather, ocean currents, blood flow, and industrial processes.
- The Navier-Stokes equations remain one of the most important unsolved problems in mathematics.
- Numerical simulation of continua underpins modern engineering and earth science.

## Prerequisites

- Vector calculus: gradient, divergence, curl.
- Linear algebra: matrices, eigenvalues, tensor notation.
- Ordinary differential equations.
- Basic thermodynamics and classical mechanics.

## Further Reading — For the Curious

If you want to feel the math in your bones, here are some recommendations:

- **Spencer**, *Continuum Mechanics* — clean and compact, a good desk companion.
- **Landau and Lifshitz**, *Theory of Elasticity* and *Fluid Mechanics* — deep and beautiful, like a masterclass from old-world physicists.
- **Batchelor**, *An Introduction to Fluid Dynamics* — the classic. Dense but rewarding.
- **Lautrup**, *Physics of Continuous Matter* — the textbook for this course. Practical, physical, and full of real examples.
- **Lamb**, *Hydrodynamics* — it's old but delicious, like a 100-year-old wine. Read it to feel the elegance.


## Visual and Simulation Gallery

[[figure continuum-stress-tensor]]

[[figure continuum-hookes-law]]

[[figure continuum-density-fluctuations]]
