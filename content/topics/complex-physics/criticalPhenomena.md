# Critical Phenomena

## The Miracle of Forgetting Details

Here's one of the most astonishing facts in all of physics. Take a magnet near its Curie temperature. Take a fluid near its liquid-gas critical point. Take a binary alloy near its unmixing transition. These systems are made of completely different stuff -- iron atoms, water molecules, copper and zinc.

And yet, near their critical points, they all behave *identically*.

The magnetization of iron follows the exact same power law as the density difference in water. The susceptibility diverges with the same exponent as the compressibility. How is this possible? The answer: near a critical point, the correlation length $\xi$ diverges to infinity, and the system "forgets" its microscopic details. All that matters is dimensionality and the symmetry of the order parameter. Everything else washes out.

This is **universality**, and it's the deepest result in the theory of phase transitions.

## Big Ideas

* Universality is the miracle that systems made of completely different stuff behave identically near their critical points, because the diverging correlation length erases all memory of microscopic details.
* The renormalization group explains universality: zoom out, average the small wiggles, and the big picture looks almost the same -- except a few numbers (the relevant ones) change. Keep zooming and those numbers flow to fixed points. That flow is universality itself.
* At the critical point, the correlation length diverges -- the system has structure at every scale simultaneously, which is why it looks the same whether you zoom in or out.
* Finite-size scaling and data collapse are the computational signatures of criticality: rescale simulation data from different system sizes and when they all fall on one curve, you've found the critical exponents.

## Universality

Systems in the same **universality class** share identical critical exponents. What determines the class? Just two things:

1. The **spatial dimensionality** $d$.
2. The **symmetry of the order parameter** (scalar, vector, tensor, etc.).

The liquid-gas critical point and the 3D Ising ferromagnet belong to the same class because both have a scalar order parameter in 3 dimensions -- even though one involves molecules bouncing around and the other involves quantum spins on a lattice. Nature doesn't care about the details. She cares about symmetry and dimension.

## Critical Exponents and Scaling Laws

Near $T_c$, thermodynamic quantities diverge or vanish as power laws. The **reduced temperature** $t = (T - T_c)/T_c$ measures distance from criticality.

* **Order parameter**: $m \sim |t|^\beta$ for $T < T_c$.
* **Susceptibility**: $\chi \sim |t|^{-\gamma}$.
* **Heat capacity**: $C \sim |t|^{-\alpha}$.
* **Correlation length**: $\xi \sim |t|^{-\nu}$.
* **Correlation function at $T_c$**: $G(r) \sim r^{-(d-2+\eta)}$.

These exponents aren't independent. **Scaling relations** connect them:

$$
\alpha + 2\beta + \gamma = 2 \qquad \text{(Rushbrooke)},
$$

$$
\gamma = \nu(2 - \eta) \qquad \text{(Fisher)},
$$

$$
\nu d = 2 - \alpha \qquad \text{(Josephson / hyperscaling)}.
$$

Know any two independent exponents and you can compute all the others.

## Mean-Field Theory and Its Limitations

Mean-field gives exponents: $\beta = 1/2$, $\gamma = 1$, $\alpha = 0$, $\nu = 1/2$. These are exact above $d_c = 4$. Below it, fluctuations bite:

| Exponent | Mean-field | 2D Ising | 3D Ising |
|----------|-----------|----------|----------|
| $\beta$  | 1/2       | 1/8      | 0.326    |
| $\gamma$ | 1         | 7/4      | 1.237    |
| $\nu$    | 1/2       | 1        | 0.630    |

In 2D, $\beta$ drops from $1/2$ to $1/8$ -- magnetization vanishes *much* more slowly than mean-field predicts. The susceptibility diverges almost twice as fast. Fluctuations matter enormously.

[[simulation phase-transition-ising-1d]]

Watch how the Ising model behaves near its critical point: enormous clusters form and dissolve, the correlation length grows, and the system fluctuates wildly between magnetized and unmagnetized states. This is what criticality looks like.

## The Renormalization Group

Zoom out, average the small wiggles, and the big picture looks almost the same -- except a few numbers (the relevant ones) change. Keep zooming and those numbers flow to fixed points. That flow is universality itself.

More precisely, the RG procedure goes:

1. **Block spins**: group neighbors into blocks, define a new effective spin for each.
2. **Rescale**: shrink the lattice back to its original size.
3. **Renormalize**: adjust coupling constants so the partition function is preserved.

This defines a flow in coupling-constant space. **Fixed points** correspond to scale-invariant systems -- the critical points. **Relevant** directions drive you away from criticality (temperature, field). **Irrelevant** directions shrink to zero (lattice structure, interaction range). Two systems with different microscopic details but the same relevant variables flow to the same fixed point. Same fixed point, same exponents. That's universality.

## Correlation Functions

The **two-point correlation function** measures how fluctuations at two points are statistically related:

$$
G(\mathbf{r}) = \langle s(\mathbf{0}) \, s(\mathbf{r}) \rangle - \langle s \rangle^2.
$$

Away from $T_c$, correlations decay exponentially: $G(r) \sim e^{-r/\xi}$. At $T_c$, $\xi \to \infty$ and correlations decay as a power law: $G(r) \sim r^{-(d-2+\eta)}$.

The divergence of $\xi$ means fluctuations at one point influence points arbitrarily far away. The system has no characteristic length scale, so it can't "know" about microscopic details. That's the physical origin of universality.

## Finite-Size Scaling

In simulations, $L$ is finite, so true divergences can't occur. **Finite-size scaling** connects finite-system data to infinite-system behavior:

$$
\chi(t, L) = L^{\gamma/\nu} \, \tilde{\chi}(t \, L^{1/\nu}),
$$

where $\tilde{\chi}$ is a universal scaling function. Plot $\chi / L^{\gamma/\nu}$ versus $t L^{1/\nu}$ for different system sizes. When the exponents are correct, all curves collapse onto a single master curve. This **data collapse** is deeply satisfying when it works.

[[simulation data-collapse]]

## What Comes Next

Critical phenomena are driven by thermal fluctuations tuned by temperature. But there's an entirely different kind of phase transition where temperature plays no role: the connectivity transition of [Percolation](percolation). Fill a lattice randomly at probability $p$, and at a critical $p_c$ a giant cluster suddenly spans the system. The same power-law scaling and universality classes appear -- in a completely different context.

## Check Your Understanding

1. The 3D Ising magnet and the liquid-gas critical point are in the same universality class. What is the shared symmetry?
2. The correlation function switches from exponential decay to power-law decay at $T_c$. What does this reveal about the structure of the system at criticality?
3. What happens to the data collapse when you use mean-field exponents instead of the correct ones?

## Challenge

Here is raw susceptibility data from 2D Ising simulations at system sizes $L = 16, 32, 64, 128$. Your task: find the critical exponents $\gamma/\nu$ and $1/\nu$ that make the data collapse. Plot $\chi / L^{\gamma/\nu}$ versus $(T - T_c) L^{1/\nu}$ for all four sizes. Start by guessing $\gamma/\nu = 1.75$ and $1/\nu = 1.0$ (the known 2D Ising values), then try deliberately wrong values to see the collapse fail. Use the interactive data-collapse simulation above to explore.
