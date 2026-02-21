# The Grand Finale — A Detective Story in Seven Acts

## Seven Detective Stories — A Recap

Each act below is a mystery we solved during this course. If any feels unfamiliar, revisit the corresponding chapter with fresh eyes.

**Act 1 — Why the Iceberg Floats.** We opened with an audacious bet: pretend matter is smooth. The continuum approximation let us write down the hydrostatic equation and Archimedes' principle, explaining why exactly 90% of an iceberg hides below the waterline — and why it stays upright once it gets there.

**Act 2 — The Glacier's Crevasses.** The Cauchy stress tensor and its deviatoric part gave us the language to ask *where* inside a material things break. Crevasses open at the surface where tensile stress dominates and close at depth where compression wins.

**Act 3 — The Earthquake's Double Punch.** Hooke's law in three dimensions spawned two wave speeds: fast compressive P-waves and slower transverse S-waves. The time gap between the two jolts you feel in an earthquake is proportional to your distance from the epicenter.

**Act 4 — One Heartbeat, Many Rhythms.** Cauchy's equation turned out to be the single heartbeat behind every continuous material. Swap in one constitutive law and you get elastic solids; swap in another and you get viscous fluids, power-law glaciers, or anything in between.

**Act 5 — Bernoulli's Garden Hose.** Dropping viscosity gave us Euler's equations, Bernoulli's theorem, and the ideas of vorticity and circulation — enough to explain why pinching a hose speeds up the water and why airplanes stay aloft.

**Act 6 — Honey, Ketchup, and Turbulence.** Restoring viscosity brought the Navier-Stokes equations, the Reynolds number, and a zoo of exact solutions for pipes and channels. Non-Newtonian fluids like custard and ketchup showed that real materials can bend the rules in entertaining ways.

**Act 7 — Simulating Gladys the Glacier.** When geometry gets complicated, we turn differential equations into weak forms, project onto finite-element spaces, and let the computer solve the resulting sparse system. FEniCS turned the Stokes equations into a working glacier simulation.

---

[[simulation unified-map]]

[[simulation glacier-cross-section]]

## The Whole Story in One Breath

We started by pretending matter is smooth. We learned the language of tensors. We pushed on things and watched them push back (stress and strain). We discovered that the same Cauchy equation governs everything from steel beams to ocean currents. We solved beautiful exact problems for pipes and channels, then learned FEM for everything else. And along the way, Rosie stretched, Harry dripped, and Gladys kept grinding slowly toward the sea.

That's continuum mechanics. Now go pass the exam — and never look at a river, a rubber tire, or a cube of cheese the same way again.

## Big Ideas

* One equation — $\rho\,D\mathbf{v}/Dt = \mathbf{f} + \nabla\cdot\sigma$ — governs every continuous material. Glaciers, arteries, earthquake waves, and ocean tides all dance to the same heartbeat. What distinguishes them is the constitutive law: the material's personal recipe for turning deformation into stress.
* The route from a differential equation to a computer solution runs through weak form → Galerkin projection → sparse linear system → FEM solve. Every step has a physical justification, not just a mathematical one.

## What Comes Next

This is the end of the continuum mechanics arc — from the audacious claim that matter is smooth, through the language of tensors, through the heartbeat of Cauchy's equation, all the way to a working FEM simulation of a glacier. You've traveled from abstract principles to executable code.

Where do you go from here? The tools you now hold — weak formulations, FEM, dimensional analysis, the Reynolds number — transfer directly to heat transfer, electromagnetic field equations, geophysical modeling, and biomechanics. The constitutive law is what changes; the scaffolding stays the same. Pick a physical system you care about, write down what $\sigma$ looks like for it, and you already know how to solve the problem.

## Check Your Understanding

1. Cauchy's equation is the same for elastic solids and viscous fluids. Write down the two constitutive relations that turn it into the Navier-Cauchy equation (for solids) and the Navier-Stokes equation (for fluids). What is the fundamental physical difference between the two?
2. A tsunami travels across the open Pacific at $\approx 200$ m/s and slows to $\approx 10$ m/s as it approaches a shoreline where the depth has decreased to 10 m. By what factor does its amplitude increase, assuming energy flux is conserved?
3. The weak form of a PDE requires less smoothness from the solution than the strong form. Give a concrete example of a physical situation where the true solution has a kink or discontinuity — and explain why weak form is the right framework for handling it numerically.

## Challenge

Design a complete computational study of Gladys the Glacier. Her cross-section is a parabolic valley: $y = x^2/W$ for $-W \leq x \leq W$, with $W = 500$ m and maximum depth 60 m. Ice obeys Glen's flow law (power-law rheology with $n = 3$, $K \approx 2 \times 10^{-24}$ Pa$^{-3}$ s$^{-1}$) under the driving stress of a surface slope of $\theta = 3°$. Formulate the 2D Stokes problem in weak form, identify the appropriate boundary conditions (no-slip at the bedrock, stress-free at the surface), and outline the FEniCS implementation. Without computing, predict qualitatively how the velocity profile across the valley will differ from a Newtonian Poiseuille flow — then explain why that difference matters for predicting how fast icebergs calve from the glacier's terminus.
