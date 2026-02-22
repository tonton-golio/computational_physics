# The Grand Finale -- A Detective Story in Seven Acts

## Seven Detective Stories -- A Recap

A glacier calves an iceberg into the sea. The splash makes waves. The waves carry energy across the ocean. Somewhere a seismograph twitches. Every one of these moments connects back to an idea you've learned. Let's walk through the whole story.

**Act 1 -- Why the Iceberg Floats.** Picture a chunk of ice the size of a house sliding off a glacier into the Arctic Ocean. It bobs, tilts, and settles. We opened this course with an audacious bet: pretend matter is smooth. That single trick gave us the hydrostatic equation and Archimedes' principle -- explaining why exactly 90% of the berg hides below the waterline, and why it stays upright.

**Act 2 -- The Glacier's Crevasses.** A mountaineer peers into a deep blue crack in the ice. The Cauchy stress tensor and its deviatoric part gave us the language to ask *where* inside a material things break. Crevasses open at the surface where tensile stress dominates and close at depth where compression wins.

**Act 3 -- The Earthquake's Double Punch.** At 3 AM a jolt wakes you -- then a second, rolling shudder a minute later. Hooke's law in 3D spawned two wave speeds: fast compressive P-waves and slower transverse S-waves. The time gap between the two punches tells seismologists the distance to the epicenter.

**Act 4 -- One Heartbeat, Many Rhythms.** A rubber ball bounces, honey drips, a glacier creeps. Cauchy's equation turned out to be the single heartbeat behind every continuous material. Swap in one constitutive law and you get elastic solids; swap in another and you get viscous fluids, power-law glaciers, or anything in between.

**Act 5 -- Bernoulli's Garden Hose.** You're watering the garden and pinch the hose. The jet speeds up. Dropping viscosity gave us Euler's equations, Bernoulli's theorem, and the trade-off between speed and pressure that keeps airplanes aloft.

**Act 6 -- Honey, Ketchup, and Turbulence.** Harry the Honey Drop drips slowly off a spoon while the tap water in the sink swirls chaotically. Restoring viscosity brought the Navier-Stokes equations, the Reynolds number, and a zoo of exact solutions for pipes and channels.

**Act 7 -- Simulating Gladys the Glacier.** Gladys grinds through her irregular valley, and no closed-form solution will capture it. When geometry gets complicated, we turn PDEs into weak forms, project onto finite-element spaces, and let the computer solve the resulting sparse system.

---

[[simulation unified-map]]

[[simulation glacier-cross-section]]

## The Whole Story in One Breath

We started by pretending matter is smooth. We learned the language of tensors. We pushed on things and watched them push back. We discovered that the same Cauchy equation governs everything from steel beams to ocean currents. We solved beautiful exact problems for pipes and channels, then learned FEM for everything else. Rosie stretched, Harry dripped, and Gladys kept grinding slowly toward the sea.

That's continuum mechanics. Now go pass the exam -- and never look at a river, a rubber tire, or a cube of cheese the same way again.

## Big Ideas

* One equation -- $\rho\,D\mathbf{v}/Dt = \mathbf{f} + \nabla\cdot\sigma$ -- governs every continuous material. What distinguishes glaciers from arteries is the constitutive law: the material's personal recipe for turning deformation into stress.
* The route from PDE to computer solution: weak form --> Galerkin projection --> sparse linear system --> FEM solve. Every step has a physical justification, not just a mathematical one.

## What Comes Next

This is the end of the continuum mechanics arc. The tools you now hold -- weak formulations, FEM, dimensional analysis, the Reynolds number -- transfer directly to heat transfer, electromagnetism, geophysical modeling, and biomechanics. The constitutive law changes; the scaffolding stays the same. Pick a physical system you care about, write down what $\sigma$ looks like, and you already know how to solve the problem.

## Check Your Understanding

1. Cauchy's equation is the same for elastic solids and viscous fluids. Write down both constitutive relations. What is the fundamental physical difference?
2. A tsunami travels at ~200 m/s in the open Pacific and slows to ~10 m/s near shore (depth 10 m). By what factor does amplitude increase, assuming energy flux is conserved?
3. The weak form requires less smoothness from the solution than the strong form. Give a concrete physical situation where the true solution has a kink -- and explain why weak form is the right framework.

## Challenge

Design a complete computational study of Gladys the Glacier. Her cross-section is a parabolic valley: $y = x^2/W$ for $-W \leq x \leq W$, with $W = 500$ m and maximum depth 60 m. Ice obeys Glen's flow law (power-law rheology with $n = 3$, $K \approx 2 \times 10^{-24}$ Pa$^{-3}$ s$^{-1}$) under the driving stress of a surface slope of $\theta = 3$degrees. Formulate the 2D Stokes problem in weak form, identify the appropriate boundary conditions (no-slip at the bedrock, stress-free at the surface), and outline the FEniCS implementation. Without computing, predict qualitatively how the velocity profile across the valley will differ from a Newtonian Poiseuille flow -- then explain why that difference matters for predicting how fast icebergs calve from the glacier's terminus.
