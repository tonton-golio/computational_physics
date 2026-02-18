# The Grand Finale — A Detective Story in Seven Acts

## How to Use This Section

Everything below is a compressed retelling of the entire course, organized around real problems. Think of each section as a detective story: there's a physical mystery, and the tools we've built are the detective's kit. If any section feels unfamiliar, go back to the corresponding section and re-read it with fresh eyes.

---

## Act 1: Hydrostatics and Buoyancy — Why the Titanic Sank (and Why the Iceberg Didn't)

**The Mystery:** An iceberg floats with about 90% of its volume below the waterline. Why exactly 90%? And what forces keep it stable?

**The Toolkit:**

*Continuum Approximation.* Before anything else, we check: can we treat ice and water as continua? The molecular separation length:
$$
L_{\text{mol}} = \left(\frac{M_{\text{mol}}}{\rho N_A}\right)^{1/3}
$$

For water: $L_{\text{mol}} \approx 3 \times 10^{-10}$ m. Our iceberg is $\sim 100$ m across. The Knudsen number is vanishingly small — the continuum approximation is spectacularly valid.

*Pressure and the hydrostatic equation:*
$$
\nabla p = \rho \, \mathbf{g}, \qquad \mathbf{F}_{\text{pressure}} = -\oint_S p \, d\mathbf{S}
$$

*Archimedes' principle:*
$$
\mathbf{F} = \mathbf{F}_G + \mathbf{F}_B = \int_V (\rho_{\text{ice}} - \rho_{\text{water}}) \, \mathbf{g} \, dV
$$

For equilibrium, the mass of displaced water equals the mass of the iceberg. Since $\rho_{\text{ice}} \approx 917$ kg/m$^3$ and $\rho_{\text{water}} \approx 1025$ kg/m$^3$, the fraction below the surface is $\rho_{\text{ice}}/\rho_{\text{water}} \approx 0.895$ — about 90%. As the course quote says: *"Just imagine an iceberg made of water, a so-called waterberg."*

*Stability:* The iceberg stays upright because its center of buoyancy $\mathbf{x}_B$ is above its center of gravity $\mathbf{x}_G$, creating a restoring moment when it tilts.

**Key equations:** Continuum approximation scales ($L_{\text{mol}}$, $L_{\text{micro}}$, $L_{\text{macro}}$), hydrostatic equation, Archimedes' principle, moment balance for floating body stability.

---

## Act 2: Stress and Strain — The Glacier's Crevasses

**The Mystery:** Glaciers develop deep cracks (crevasses) at their surfaces. Why at the surface and not deeper? And why do they stop at a certain depth?

**The Toolkit:**

*The Cauchy stress tensor* describes forces throughout the ice. Near the surface, the ice is being pulled apart by the glacier's flow — tensile stress. Deeper down, the weight of overlying ice creates compressive stress that overwhelms the tension.

*The strain tensor* captures how the ice deforms. The eigenvectors of strain point in the directions of maximum stretching — which is where crevasses open.

*The stress deviator* $s = \sigma - p\,\mathbf{I}$ strips away the confining pressure and reveals the "shape-changing" stress. Crevasses form where the deviatoric stress exceeds the tensile strength of ice.

*The von Mises criterion:* $\sigma_{\text{vM}} = \sqrt{3J_2}$ predicts the onset of failure independent of coordinate choice.

**Key equations:** Cauchy stress tensor, principal stresses, stress deviator, invariants, von Mises criterion.

---

## Act 3: Elasticity and Vibrations — The Earthquake's Double Punch

**The Mystery:** During an earthquake, you feel two distinct jolts. The first is a sharp, compressive "thump." A few seconds later comes a rolling, sideways shake. Why two?

**The Toolkit:**

*Hooke's law in 3D:* $\sigma_{ij} = \lambda\,\varepsilon_{kk}\,\delta_{ij} + 2\mu\,\varepsilon_{ij}$

*Elastic wave speeds:*
$$
c_P = \sqrt{\frac{\lambda + 2\mu}{\rho}}, \qquad c_S = \sqrt{\frac{\mu}{\rho}}
$$

P-waves (primary/pressure waves) are compressive and travel faster. S-waves (secondary/shear waves) are transverse and travel slower. For typical rock with $\nu \approx 1/3$: $c_P = 2\,c_S$.

The first jolt is the P-wave arriving; the second is the S-wave. The time delay between them is proportional to the distance to the earthquake. With three seismograph stations, you can triangulate the epicenter.

*The block of gouda:* Cheese has a Young's modulus of about 0.3 GPa — stiff enough to demonstrate elastic wave propagation but soft enough to deform visibly. It's the perfect teaching material.

**Key equations:** Generalized Hooke's law, Lame coefficients, P- and S-wave speeds, relationship between $E$, $\nu$, and wave speed.

---

## Act 4: The Cauchy Equation — One Heartbeat, Many Rhythms

**The Mystery:** Why do the equations for elastic solids and viscous fluids look so similar?

**The Toolkit:**

*Cauchy's equation:*
$$
\rho \frac{D\mathbf{v}}{Dt} = \mathbf{f} + \nabla \cdot \sigma
$$

This is the *same equation* for every continuous material. The constitutive law (the relationship between $\sigma$ and deformation) is what distinguishes them:

| Material | Constitutive law | Result |
|----------|-----------------|--------|
| Elastic solid | $\sigma = \lambda\,\text{tr}(\varepsilon)\,\mathbf{I} + 2\mu\,\varepsilon$ | Navier-Cauchy eq. |
| Viscous fluid | $\sigma = -p\,\mathbf{I} + 2\eta\,\dot{\varepsilon}$ | Navier-Stokes eq. |
| Power-law fluid | $\sigma = -p\,\mathbf{I} + 2K\dot{\gamma}^{n-1}\dot{\varepsilon}$ | Glen's flow law (glaciers) |

The material derivative $D/Dt = \partial/\partial t + (\mathbf{v} \cdot \nabla)$ connects the Eulerian and Lagrangian perspectives.

**Key equations:** Material derivative, Cauchy's equation, continuity equation, constitutive relations.

---

## Act 5: (Nearly) Ideal Flows — Bernoulli's Garden Hose

**The Mystery:** Pinch a garden hose and the water jets out faster. Why? And why does an airplane stay up?

**The Toolkit:**

*Euler's equations* for inviscid flow. *Bernoulli's theorem:*
$$
H = \frac{1}{2}v^2 + gz + \frac{p}{\rho} = \text{const along a streamline}
$$

Pinch the hose → area decreases → velocity increases (continuity) → pressure decreases (Bernoulli). This pressure difference is also what generates lift on an airplane wing.

*Vorticity* $\boldsymbol{\omega} = \nabla \times \mathbf{v}$ and *circulation* $\Gamma = \oint \mathbf{v} \cdot d\mathbf{l}$ describe the rotational structure of the flow. Kelvin's theorem: circulation is conserved in ideal flow.

*Gravity waves:* surface waves in shallow water propagate at $c = \sqrt{gD}$, independent of wavelength.

**Key equations:** Euler equations, Bernoulli's theorem, vorticity equation, shallow-water wave speed.

---

## Act 6: Viscous Flows — Honey, Ketchup, and Glaciers

**The Mystery:** Why does honey pour in a smooth stream while water from a faucet breaks into turbulent splashing?

**The Toolkit:**

*The Reynolds number:*
$$
\text{Re} = \frac{UL}{\nu}
$$

Honey has Re $\sim 10^{-1}$ (viscosity wins → smooth, laminar flow). A kitchen faucet has Re $\sim 10^3$–$10^4$ (inertia wins → turbulence).

*Navier-Stokes:*
$$
\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} = \mathbf{g} - \frac{\nabla p}{\rho} + \nu\nabla^2\mathbf{v}
$$

*Exact solutions:* Poiseuille flow in pipes ($v \propto a^2 - r^2$, flow rate $\propto a^4$), gravity-driven channel flow on inclined planes.

*Non-Newtonian behavior:* Custard (shear-thickening — run across it!), quicksand (shear-thickening — don't struggle!), ketchup (shear-thinning — shake the bottle!).

**Key equations:** Navier-Stokes, Reynolds number, Poiseuille flow, Hagen-Poiseuille law, power-law rheology.

---

## Act 7: Stokes Flow and FEM — Simulating Gladys the Glacier

**The Mystery:** How do you predict the flow of a glacier through a complex mountain valley?

**The Toolkit:**

*Stokes equations* (Re $\ll 1$):
$$
\nabla p = \eta\nabla^2\mathbf{v}, \qquad \nabla \cdot \mathbf{v} = 0
$$

*Weak formulation:* Multiply by test functions, integrate by parts, lower the smoothness requirements.

*Finite elements:* Mesh the glacier domain (irregular valley geometry), choose basis functions (Taylor-Hood elements for stable velocity-pressure coupling), assemble and solve the saddle-point system.

*Boundary conditions:* No-slip at the bedrock, free surface or stress-free at the top, inflow/outflow conditions at the glacier margins.

*FEniCS implementation:* Express the weak form in Python, let FEniCS handle meshing, assembly, and linear algebra. The glacier flows.

**Key equations:** Stokes equations, weak form, Galerkin FEM, saddle-point system.

---

## The Whole Story in One Breath

We started by pretending matter is smooth. We learned the language of tensors. We pushed on things and watched them push back (stress and strain). We discovered that the same Cauchy equation governs everything from steel beams to ocean currents. We solved beautiful exact problems for pipes and channels, then learned FEM for everything else. And along the way, Rosie stretched, Harry dripped, and Gladys kept grinding slowly toward the sea.

That's continuum mechanics. Now go pass the exam — and never look at a river, a rubber tire, or a cube of cheese the same way again.
