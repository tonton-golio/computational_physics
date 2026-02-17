# Continuum Mechanics

## Course overview

Continuum mechanics describes the physics of **continuous matter**, both solids and fluids, from viscous liquids to elastic solids. The formalism connects microscopic forces to macroscopic observables through stress and strain tensors, conservation laws, and constitutive relations.

- Solids: deformation, elasticity, wave propagation.
- Fluids: viscous flow, potential flow, boundary layers.
- Numerics: finite-element and finite-difference methods for real-world problems.

## Why this topic matters

- Understanding stress and strain is essential for engineering design, geophysics, and biomechanics.
- Fluid dynamics governs weather, ocean currents, blood flow, and industrial processes.
- The Navier-Stokes equations remain one of the most important unsolved problems in mathematics.
- Numerical simulation of continua underpins modern engineering and earth science.

## Key mathematical ideas

- Tensor algebra: stress tensor, strain tensor, and their invariants.
- Conservation laws: mass, momentum, and energy in continuous media.
- Constitutive relations: Hooke's law for solids, Newtonian viscosity for fluids.
- The Navier-Cauchy equation for elastic solids and the Navier-Stokes equation for viscous fluids.
- Dimensionless analysis and the Reynolds number.
- Finite-element methods for boundary value problems.

## Prerequisites

- Vector calculus: gradient, divergence, curl.
- Linear algebra: matrices, eigenvalues, tensor notation.
- Ordinary differential equations.
- Basic thermodynamics and classical mechanics.

## Recommended reading

- Spencer, *Continuum Mechanics*.
- Landau and Lifshitz, *Theory of Elasticity* and *Fluid Mechanics*.
- Batchelor, *An Introduction to Fluid Dynamics*.

## Learning trajectory

This module progresses from the continuum approximation to numerical modeling:

1. **Continuum approximation** -- when and why matter can be treated as continuous.
2. **Tensor fundamentals** -- Cauchy stress and strain tensors, stress deviator, velocity gradient and spin tensor.
3. **Stress and strain** -- traction vectors, principal stresses, Hooke's law, Mohr's circle.
4. **Elasticity** -- Young's modulus, Poisson's ratio, generalized Hooke's law, work and energy, vibrations and sound.
5. **Dynamics** -- conservation laws, material derivative, Cauchy's equation.
6. **Fluids at rest** -- definition of fluids, pressure, buoyancy and stability.
7. **Fluids in motion** -- ideal flows, Euler equations, Bernoulli's theorem, vorticity.
8. **Viscous flow** -- viscosity, Navier-Stokes equation, Reynolds number.
9. **Channels and pipes** -- pressure-driven flow, gravity-driven flow, laminar pipe flow.
10. **Gravity waves** -- shallow-water equations, dispersion.
11. **Creeping flow** -- Stokes flow, drag and lift, non-Newtonian fluids.
12. **Weak Stokes formulation** -- weak form for Stokes equations.
13. **Finite element method** -- weighted residuals, Galerkin's method, minimum potential energy.
14. **Python packages** -- FEniCS, SymPy, Plotly, and other computational tools.
15. **Exam dispensation** -- review and exam preparation notes.

## Visual and Simulation Gallery

[[simulation stress-strain-curve]]

[[simulation mohr-circle]]

[[figure continuum-stress-tensor]]

[[figure continuum-hookes-law]]

[[figure continuum-density-fluctuations]]
