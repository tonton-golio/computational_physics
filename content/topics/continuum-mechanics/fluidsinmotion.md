# Fluids in Motion

## The Garden Hose Experiment

Imagine holding a garden hose with water flowing steadily. Now pinch the hose partway shut with your thumb. The water speeds up -- a narrow, fast jet shoots out.

*Why?* The same amount of water has to get through a smaller opening in the same time. That simple observation -- **what goes in must come out** -- is all the continuity equation is saying.

## Ideal Flows -- The Simplest Starting Point

Start with the most idealized fluid: **incompressible**, **inviscid**, and **irrotational**. Real fluids are never this perfect, but ideal flow gives surprisingly useful results.

For an ideal fluid, the only stress is pressure: $\sigma = -p\,\mathbf{I}$. Plug into Cauchy's equation with incompressibility ($\nabla \cdot \mathbf{v} = 0$) and you get the **Euler equations**:
$$
\nabla \cdot \mathbf{v} = 0, \qquad \frac{D\mathbf{v}}{Dt} = \mathbf{g} - \frac{\nabla p}{\rho}
$$

Four equations, four unknowns ($v_x$, $v_y$, $v_z$, $p$). The system is closed.

[[simulation elastic-wave]]

One striking consequence: pressure is *non-local*. Change it somewhere and it instantly adjusts everywhere. That's because incompressibility makes pressure waves travel at infinite speed (in reality, at the speed of sound -- very fast but not infinite).

## Bernoulli's Theorem -- Energy Conservation Along a Streamline

Back to the hose. The water speeds up when you pinch. Where does the kinetic energy come from? The pressure drops. This trade-off is **Bernoulli's theorem**.

The **Bernoulli function**:
$$
H = \frac{1}{2}v^2 + \phi + \frac{p}{\rho}
$$

where $\phi$ is the gravitational potential. Along a streamline in steady flow:
$$
\frac{DH}{Dt} = 0
$$

$H$ is constant along a flowline: kinetic energy + potential energy + pressure energy = constant. Divide by $g$ and you get what engineers call the **total head** -- a quantity in meters you can literally measure as a height.

The gradient of $H$ connects to vorticity:
$$
\nabla H = \mathbf{v} \times (\nabla \times \mathbf{v}) - \frac{\partial \mathbf{v}}{\partial t}
$$

In steady, irrotational flow, $\nabla H = 0$ everywhere -- $H$ is constant not just along streamlines but throughout the entire flow. That's the strongest Bernoulli.

## Vorticity -- The Local Spin

Imagine dropping a tiny ice skater into a flowing river. At some points the current is uniform and she glides straight. At other points the flow is faster on one side, and she starts to **spin**. That spin rate is the **vorticity**:
$$
\boldsymbol{\omega} = \nabla \times \mathbf{v}
$$

If vorticity is zero everywhere, the flow is **irrotational** and things simplify enormously. If nonzero, the flow has internal structure -- swirling eddies behind rocks.

Taking the curl of Euler's equation gives the **vorticity equation**:
$$
\frac{\partial \boldsymbol{\omega}}{\partial t} = \nabla \times (\mathbf{v} \times \boldsymbol{\omega})
$$

Here's the punchline: **if a flow starts irrotational, it stays irrotational** (for an ideal fluid). You need viscosity or boundaries to create vorticity. This is **Kelvin's circulation theorem**.

## Circulation and Stokes' Theorem

The **circulation** $\Gamma$ around a closed curve:
$$
\Gamma = \oint_C \mathbf{v} \cdot d\mathbf{l} = \int_S \boldsymbol{\omega} \cdot d\mathbf{S}
$$

Circulation is the "total spin" enclosed by a loop. Kelvin's theorem says that for an ideal fluid, circulation around a material loop is conserved. Vorticity isn't created or destroyed -- just transported and stretched.

## Big Ideas

* Bernoulli's theorem is energy conservation along a streamline: fast flow means low pressure, slow flow means high pressure. It explains garden hoses, airplane wings, and venturi meters.
* Vorticity $\omega = \nabla \times \mathbf{v}$ measures local spin; Kelvin's theorem says it can't spontaneously appear in an ideal fluid.
* Pressure in an incompressible fluid is non-local: it adjusts everywhere instantly.

## What Comes Next

Ideal flows can't explain why the water in your cup eventually stops spinning after you stir it. Real fluids have *viscosity* -- internal friction that dissipates energy. Next up: the Navier-Stokes equation.

## Check Your Understanding

1. Water flows through a pipe that narrows from diameter 4 cm to 2 cm. If the velocity in the wide section is 1 m/s, what is the velocity in the narrow section? What is the pressure difference? (Use Bernoulli and $\rho = 1000$ kg/m^3.)
2. Bernoulli's function $H = \frac{1}{2}v^2 + gz + p/\rho$ is constant along a streamline in steady flow. Under what stronger condition is it constant throughout the entire flow field?

## Challenge

A wing profile creates faster flow over the top surface than the bottom. Model the wing as a 2D flat plate of chord length $c$ in a uniform flow of speed $U_\infty$. Using Bernoulli's theorem, if the top-surface velocity is $U_\infty + \Delta v$ and the bottom-surface velocity is $U_\infty - \Delta v$ (for small $\Delta v \ll U_\infty$), derive the lift force per unit span. Express the answer in terms of $\rho$, $U_\infty$, $c$, and $\Delta v/U_\infty$. Now consider: Kelvin's theorem says total circulation is conserved. If the wing generates positive circulation, where must an equal and opposite circulation appear?
