# Inverse Problems

You drop a stone into a pond and ripples spread outward. That's a forward problem — given the cause, predict the effect. Now imagine you're standing at the edge and all you see are ripples. Can you figure out where the stone landed? That's an inverse problem: observe effects, work backwards to the cause.

This pattern is everywhere. In MRI, you measure electromagnetic responses at the body's surface and reconstruct what's happening inside. In seismology, earthquake waves travel through the Earth and you record wiggles at the surface, then try to image the interior. Remote sensing, radar, acoustics, materials science — always the same question: "what caused this?"

And here's the catch. The obvious approach — just invert the equations — almost always blows up. A tiny amount of noise in your measurements can send the answer flying off to absurd values. One hundredth of a percent perturbation at the boundary of a heat problem can produce interior temperatures of $10^{41}$. That single fact is why this whole subject exists.

The answer is almost never unique, but with the right tools — regularization, Bayesian inference, iterative solvers, Monte Carlo sampling — you can tame the instability and extract honest answers from incomplete data. Let's see how.
