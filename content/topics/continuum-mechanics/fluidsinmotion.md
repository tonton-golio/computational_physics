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

## Big Ideas

* An ideal (inviscid, incompressible, irrotational) flow is governed by the Euler equations — four equations for four unknowns. Pressure becomes non-local: it instantly adjusts everywhere because incompressibility makes pressure waves travel infinitely fast.
* Bernoulli's theorem is energy conservation along a streamline: fast flow means low pressure, slow flow means high pressure. It explains garden hoses, airplane wings, and venturi meters.
* Vorticity $\omega = \nabla \times \mathbf{v}$ measures local spin; Kelvin's theorem says vorticity can't spontaneously appear in an ideal fluid — you need viscosity or boundaries to create it.
* Circulation $\Gamma = \oint \mathbf{v} \cdot d\mathbf{l}$ is the global version of vorticity: the total "spin" enclosed by a loop, conserved in ideal flow.

## What Comes Next

Ideal flows are elegant but incomplete — they can't explain why the water in your cup eventually stops spinning after you stir it. That's because real fluids have *viscosity*: internal friction that dissipates energy. The next section introduces viscous flow and the Navier-Stokes equation.

## Check Your Understanding

1. Water flows through a pipe that narrows from diameter 4 cm to diameter 2 cm. If the velocity in the wide section is 1 m/s, what is the velocity in the narrow section? What is the pressure difference between the two sections? (Use Bernoulli's theorem and $\rho = 1000$ kg/m³.)
2. An ideal flow starts with zero vorticity everywhere. Kelvin's theorem says vorticity can't be created. Yet we observe vorticity in real flows past obstacles. What is the resolution to this apparent paradox?
3. The Bernoulli function $H = \frac{1}{2}v^2 + gz + p/\rho$ is constant along a streamline in steady flow. Under what stronger condition is it constant throughout the entire flow field?

## Challenge

A wing profile creates faster flow over the top surface than the bottom. Model the wing as a 2D flat plate of chord length $c$ in a uniform flow of speed $U_\infty$. Using Bernoulli's theorem, if the top-surface velocity is $U_\infty + \Delta v$ and the bottom-surface velocity is $U_\infty - \Delta v$ (for small $\Delta v \ll U_\infty$), derive the lift force per unit span. Express the answer in terms of $\rho$, $U_\infty$, $c$, and $\Delta v/U_\infty$. Now consider: Kelvin's theorem says total circulation is conserved. If the wing generates positive circulation, where must an equal and opposite circulation appear?
