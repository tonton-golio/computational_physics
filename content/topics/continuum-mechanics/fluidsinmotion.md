# Fluids in Motion

## The Garden Hose Experiment

Before a single equation, do this experiment in your head. You're holding a garden hose with water flowing steadily out the end. Now pinch the hose partway shut with your thumb. What happens?

The water speeds up. A narrow, fast jet shoots out.

*Why?* Because the same amount of water has to get through a smaller opening in the same amount of time. Water isn't appearing or disappearing — it's being conserved. That simple observation — **what goes in must come out** — is all the continuity equation is saying. The rest is just making that idea precise.

## Ideal Flows — The Simplest Starting Point

Let's start with the most idealized fluid imaginable: **incompressible** (you can't squeeze it), **inviscid** (no internal friction), and **irrotational** (no local spinning). This is an "ideal flow." Real fluids are never quite this perfect, but ideal flow gives us beautiful results that are surprisingly useful.

For an ideal fluid at rest, we had $\sigma = -p\,\mathbf{I}$. This remains true in motion — the only stress is pressure:
$$
\sigma = -p\,\mathbf{I} \qquad \Rightarrow \qquad \nabla \cdot \sigma = -\nabla p
$$

Plugging into Cauchy's equation, and using incompressibility ($\nabla \cdot \mathbf{v} = 0$), we get the **Euler equations**:
$$
\nabla \cdot \mathbf{v} = 0, \qquad \frac{D\mathbf{v}}{Dt} = \mathbf{g} - \frac{\nabla p}{\rho}
$$

Four equations (one scalar continuity equation, three components of the momentum equation) for four unknowns ($v_x$, $v_y$, $v_z$, $p$). The system is closed.

[[simulation elastic-wave]]

One striking consequence of incompressibility: taking the divergence of the momentum equation gives a **Poisson equation** for pressure. This means pressure is *non-local* — change the pressure somewhere, and it instantly adjusts everywhere. That's because we've assumed the fluid is perfectly incompressible: pressure waves travel at infinite speed (in reality they travel at the speed of sound, which is very fast but not infinite).

## Bernoulli's Theorem — Conservation of Energy Along a Streamline

Back to the garden hose. The water speeds up when you pinch the opening. But where does the extra kinetic energy come from? It comes from the pressure: the pressure drops where the velocity increases. This trade-off between pressure and velocity is **Bernoulli's theorem**.

Define the **Bernoulli function**:
$$
H = \frac{1}{2}v^2 + \phi + \frac{p}{\rho}
$$

where $\phi$ is the gravitational potential (e.g., $gz$). Along a streamline in steady flow:
$$
\frac{DH}{Dt} = 0
$$

$H$ is constant along a flowline. It's an energy conservation statement: kinetic energy + potential energy + pressure energy = constant. Divide everything by $g$ and you get what engineers call the **total head** — a quantity measured in meters that you can literally see as a height.

The gradient of $H$ connects to vorticity:
$$
\nabla H = \mathbf{v} \times (\nabla \times \mathbf{v}) - \frac{\partial \mathbf{v}}{\partial t}
$$

In steady, irrotational flow, $\nabla H = 0$ everywhere — meaning $H$ is constant not just along streamlines but throughout the entire flow. This is the strongest form of Bernoulli's theorem.

## Vorticity — The Local Spin

Imagine dropping a tiny ice skater into a flowing river. At some points, the current is uniform and she glides straight. At other points, the flow is faster on one side than the other, and she starts to **spin**. The rate at which she spins is the **vorticity**:
$$
\boldsymbol{\omega} = \nabla \times \mathbf{v}
$$

Vorticity is the curl of the velocity field — it measures the local rotation rate. If the vorticity is zero everywhere, the flow is **irrotational** and things simplify enormously (we can use a velocity potential). If vorticity is nonzero, the flow has internal structure — think of the swirling eddies behind a rock in a stream.

Taking the curl of Euler's equation gives the **vorticity equation**:
$$
\frac{\partial \boldsymbol{\omega}}{\partial t} = \nabla \times (\mathbf{v} \times \boldsymbol{\omega})
$$

A key consequence: **if a flow starts irrotational, it stays irrotational** (for an ideal fluid). Vorticity can't spontaneously appear in an inviscid flow — you need viscosity (friction) or boundaries to create it. This is **Kelvin's circulation theorem**, and it explains why potential flow is such a useful approximation far from surfaces.

## Circulation and Stokes' Theorem

The **circulation** $\Gamma$ around a closed curve is the line integral of velocity:
$$
\Gamma = \oint_C \mathbf{v} \cdot d\mathbf{l}
$$

By Stokes' theorem, this equals the flux of vorticity through any surface bounded by the curve:
$$
\Gamma = \int_S \boldsymbol{\omega} \cdot d\mathbf{S}
$$

Circulation is the "total spin" enclosed by a loop. Kelvin's theorem says that for an ideal fluid, the circulation around a material loop (one that moves with the fluid) is conserved. Vorticity isn't created or destroyed — it's just transported and stretched by the flow.

## What We Just Learned

Ideal flows obey the Euler equations. Bernoulli's theorem provides energy conservation along streamlines. Vorticity measures local rotation, and in ideal flows, an irrotational state is preserved. These are the building blocks for understanding everything from airplane wings to ocean currents.

## What's Next

Ideal flows are elegant but incomplete — they can't explain why the water in your cup eventually stops spinning after you stir it. That's because real fluids have *viscosity*: internal friction that dissipates energy. The next section introduces viscous flow and the Navier-Stokes equation.
