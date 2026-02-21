# Critical Phenomena

## The Miracle of Forgetting Details

Here is one of the most astonishing facts in all of physics. Take a magnet near its Curie temperature. Take a fluid near its liquid-gas critical point. Take a binary alloy near its unmixing transition. These systems are made of completely different stuff — iron atoms with quantum spins, water molecules with hydrogen bonds, copper and zinc atoms with metallic interactions. Their microscopic physics could not be more different.

And yet, near their critical points, they all behave *identically*.

The magnetization of iron near $T_c$ follows the exact same power law as the density difference between liquid and gas water near its critical point. The susceptibility of the magnet diverges with the same exponent as the compressibility of the fluid. The correlation length grows the same way in both systems.

How is this possible? How can systems that are so different at the microscopic level produce the same macroscopic behavior at the critical point? The answer is that near a critical point, the system develops correlations on *all* length scales. The correlation length $\xi$ diverges to infinity, and the system "forgets" its microscopic details. All that matters is the spatial dimensionality $d$ and the symmetry of the order parameter. Everything else washes out.

This is **universality**, and it is the deepest result in the theory of phase transitions.

## Universality

Systems in the same **universality class** share identical critical exponents. What determines the class? Just two things:

1. The **spatial dimensionality** $d$.
2. The **symmetry of the order parameter** (scalar, vector, tensor, etc.).

The liquid-gas critical point and the 3D Ising ferromagnet belong to the same universality class because both have a scalar order parameter in 3 dimensions — even though one involves molecules bouncing around and the other involves quantum spins on a lattice. Nature does not care about the details. She cares about symmetry and dimension.

## Critical exponents and scaling laws

Near the critical temperature $T_c$, thermodynamic quantities diverge or vanish as power laws. The **reduced temperature** $t = (T - T_c)/T_c$ measures distance from criticality.

Key critical exponents:

* **Order parameter**: $m \sim |t|^\beta$ for $T < T_c$.
* **Susceptibility**: $\chi \sim |t|^{-\gamma}$.
* **Heat capacity**: $C \sim |t|^{-\alpha}$.
* **Correlation length**: $\xi \sim |t|^{-\nu}$.
* **Correlation function at $T_c$**: $G(r) \sim r^{-(d-2+\eta)}$.

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

These are not just empirical correlations. They follow from the mathematical structure of scale invariance. If you know any two independent exponents, you can compute all the others.

## Mean-field theory and its limitations

Recall from the mean-field section: mean-field theory replaces fluctuations with their average, yielding the Landau free energy:

$$
F(m) = a_0 + a_2 t \, m^2 + a_4 m^4 + \cdots
$$

Minimizing gives mean-field exponents: $\beta = 1/2$, $\gamma = 1$, $\alpha = 0$, $\nu = 1/2$.

Mean-field theory is exact above the **upper critical dimension** $d_c = 4$ for the Ising model. Below $d_c$, fluctuations are too strong to ignore, and the actual exponents differ from mean-field predictions — sometimes dramatically.

| Exponent | Mean-field | 2D Ising | 3D Ising |
|----------|-----------|----------|----------|
| $\beta$  | 1/2       | 1/8      | 0.326    |
| $\gamma$ | 1         | 7/4      | 1.237    |
| $\nu$    | 1/2       | 1        | 0.630    |

Look at those numbers. In 2D, $\beta$ drops from $1/2$ to $1/8$ — the magnetization vanishes *much* more slowly than mean-field predicts. The susceptibility diverges almost twice as fast ($\gamma = 7/4$ vs. $1$). Fluctuations matter enormously.

[[simulation phase-transition-ising-1d]]

In this simulation, you can watch how the Ising model behaves near its critical point: enormous clusters of aligned spins form and dissolve, the correlation length grows, and the system fluctuates wildly between magnetized and unmagnetized states. This is what criticality looks like.

## The renormalization group

The **renormalization group** (RG) provides the theoretical framework for understanding universality and scaling. The idea, due to Kenneth Wilson, is to systematically zoom out:

1. **Block spins**: group neighboring spins into blocks and define a new effective spin for each block.
2. **Rescale**: shrink the lattice back to its original size.
3. **Renormalize**: adjust the coupling constants so the partition function is preserved.

This defines a flow in the space of coupling constants. **Fixed points** of the RG flow correspond to scale-invariant systems — systems that look the same at every scale. These are exactly the critical points.

The critical exponents are determined by the eigenvalues of the linearized RG transformation at the fixed point. **Relevant** directions (which grow under iteration) drive the system away from criticality — they correspond to temperature and external field. **Irrelevant** directions (which shrink) do not affect the critical behavior — they correspond to microscopic details like lattice structure, interaction range, etc.

This is why universality works: the irrelevant directions all flow to zero, and only the relevant ones survive. Two systems with different microscopic details but the same relevant variables flow to the same fixed point and therefore share the same critical exponents.

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
G(r) \sim \frac{1}{r^{d-2+\eta}}.
$$

The divergence of $\xi$ at $T_c$ means the system develops correlations on all length scales. A fluctuation at one point influences points arbitrarily far away. This is the physical origin of scale invariance and universality — the system has no characteristic length scale, so it cannot "know" about microscopic details.

## Finite-size scaling

In simulations, the system size $L$ is finite, so true divergences cannot occur. **Finite-size scaling** relates observables in a finite system to the infinite-system critical behavior:

$$
\chi(t, L) = L^{\gamma/\nu} \, \tilde{\chi}(t \, L^{1/\nu}),
$$

where $\tilde{\chi}$ is a universal scaling function. By plotting data for different system sizes as a function of $t L^{1/\nu}$, all curves collapse onto a single master curve when the correct exponents are used. This **data collapse** is a powerful method for extracting critical exponents from numerical simulations — and it is deeply satisfying when it works.

## Big Ideas

* Universality is the miracle that systems made of completely different stuff — iron atoms, water molecules, copper alloys — behave identically near their critical points, because the diverging correlation length erases all memory of microscopic details.
* The renormalization group explains universality by showing that "irrelevant" directions in coupling-constant space flow to zero, leaving only symmetry and dimensionality to determine the critical exponents.
* At the critical point, the correlation length diverges to infinity — the system has structure at every scale simultaneously, which is why it looks the same whether you zoom in or out.
* Finite-size scaling and data collapse are the computational signatures of criticality: when you rescale simulation data from different system sizes and they all fall on one curve, you have found the critical exponents.

## What Comes Next

Critical phenomena are driven by thermal fluctuations tuned by temperature. But there is an entirely different kind of phase transition where temperature plays no role at all: the connectivity transition of [Percolation](percolation). Fill a lattice randomly with occupied sites at probability $p$, and at a critical $p_c$ a giant connected cluster suddenly spans the entire system. The same power-law scaling, universality classes, and critical exponents appear — in a completely different context. The comparison will sharpen your understanding of what is truly universal about critical phenomena.

## Check Your Understanding

1. Two systems belong to the same universality class if they have the same spatial dimensionality and the same symmetry of the order parameter. The 3D Ising magnet and the liquid-gas critical point are in the same class, even though one involves quantum spins and the other involves classical molecules. What is the shared symmetry?
2. The renormalization group says that "irrelevant" coupling constants flow to zero under coarse-graining. If irrelevant couplings (like next-nearest-neighbor interactions) do not affect the critical exponents, why do they matter at all for real experiments?
3. The correlation function decays exponentially away from $T_c$ ($G(r) \sim e^{-r/\xi}$) but as a power law at $T_c$ ($G(r) \sim r^{-(d-2+\eta)}$). What does the switch from exponential to power-law decay reveal about the structure of the system at criticality?

## Challenge

The finite-size scaling ansatz states that the susceptibility of a finite system of size $L$ obeys $\chi(t, L) = L^{\gamma/\nu} \tilde{\chi}(t L^{1/\nu})$. Design a numerical experiment to extract the critical exponents $\gamma$ and $\nu$ for the 2D Ising model from this relation. Specifically: what simulations would you run, how would you plot the data to achieve "collapse," and what systematic errors would you need to worry about? Estimate how large $L$ needs to be for the finite-size effects to be smaller than 10%.
