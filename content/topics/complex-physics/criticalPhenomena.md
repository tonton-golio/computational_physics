# Critical Phenomena

## Universality

Near a continuous phase transition, systems that are microscopically very different exhibit the same macroscopic behavior. This remarkable property is called **universality**: the critical exponents depend only on a few features of the system, specifically the spatial dimensionality $d$ and the symmetry of the order parameter.

Systems in the same **universality class** share identical critical exponents. For example, the liquid-gas critical point and the 3D Ising ferromagnet belong to the same class despite being physically distinct.

## Critical exponents and scaling laws

Near the critical temperature $T_c$, thermodynamic quantities diverge or vanish as power laws. The **reduced temperature** $t = (T - T_c)/T_c$ measures distance from criticality.

Key critical exponents:

- **Order parameter**: $m \sim |t|^\beta$ for $T < T_c$.
- **Susceptibility**: $\chi \sim |t|^{-\gamma}$.
- **Heat capacity**: $C \sim |t|^{-\alpha}$.
- **Correlation length**: $\xi \sim |t|^{-\nu}$.
- **Correlation function at $T_c$**: $G(r) \sim r^{-(d-2+\eta)}$.

These exponents are not independent. **Scaling relations** connect them:

$$
\alpha + 2\beta + \gamma = 2 \qquad \text{(Rushbrooke)},
$$

$$
\gamma = \nu(2 - \eta) \qquad \text{(Fisher)},
$$

$$
\nu d = 2 - \alpha \qquad \text{(Josephson / hyperscaling)}.
$$

## Mean-field theory and its limitations

**Mean-field theory** replaces fluctuations with their average, yielding the Landau free energy:

$$
F(m) = a_0 + a_2 t \, m^2 + a_4 m^4 + \cdots
$$

Minimizing gives mean-field exponents: $\beta = 1/2$, $\gamma = 1$, $\alpha = 0$, $\nu = 1/2$.

Mean-field theory is exact above the **upper critical dimension** $d_c = 4$ for the Ising model. Below $d_c$, fluctuations are too strong to ignore, and the actual exponents differ from mean-field predictions.

| Exponent | Mean-field | 2D Ising | 3D Ising |
|----------|-----------|----------|----------|
| $\beta$  | 1/2       | 1/8      | 0.326    |
| $\gamma$ | 1         | 7/4      | 1.237    |
| $\nu$    | 1/2       | 1        | 0.630    |

[[simulation phase-transition-ising-1d]]

## The renormalization group

The **renormalization group** (RG) provides the theoretical framework for understanding universality and scaling. The idea is to systematically coarse-grain the system:

1. **Block spins**: group neighboring spins into blocks and define a new effective spin for each block.
2. **Rescale**: shrink the lattice back to its original size.
3. **Renormalize**: adjust the coupling constants so the partition function is preserved.

This defines a flow in the space of coupling constants. **Fixed points** of the RG flow correspond to scale-invariant systems (critical points). The critical exponents are determined by the eigenvalues of the linearized RG transformation at the fixed point.

Near a fixed point, the RG flow has **relevant** directions (which grow under iteration and drive the system away from criticality) and **irrelevant** directions (which shrink and do not affect universal behavior). This explains why microscopic details are irrelevant at the critical point.

## Correlation functions

The **two-point correlation function** measures how fluctuations at two points are statistically related:

$$
G(\mathbf{r}) = \langle s(\mathbf{0}) \, s(\mathbf{r}) \rangle - \langle s \rangle^2.
$$

Away from $T_c$, correlations decay exponentially:

$$
G(r) \sim e^{-r/\xi},
$$

where $\xi$ is the **correlation length**. At the critical point $\xi \to \infty$, and correlations decay as a power law:

$$
G(r) \sim \frac{1}{r^{d-2+\eta}},
$$

where $\eta$ is the anomalous dimension. The divergence of $\xi$ at $T_c$ means the system develops correlations on all length scales, which is the physical origin of scale invariance and universality.

## Finite-size scaling

In simulations, the system size $L$ is finite, so true divergences cannot occur. **Finite-size scaling** relates observables in a finite system to the infinite-system critical behavior:

$$
\chi(t, L) = L^{\gamma/\nu} \, \tilde{\chi}(t \, L^{1/\nu}),
$$

where $\tilde{\chi}$ is a universal scaling function. By plotting data for different system sizes as a function of $t L^{1/\nu}$, all curves collapse onto a single master curve when the correct exponents are used. This **data collapse** is a powerful method for extracting critical exponents from numerical simulations.

[[simulation ising-model]]
