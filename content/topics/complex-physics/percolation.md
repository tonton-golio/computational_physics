# Percolation

Suppose you are painting a wall with random dots of ink. At first they are isolated islands — tiny splotches scattered here and there with gaps between them. Keep adding dots. Some start to merge into small clusters. Keep going. More clusters connect, forming larger and larger blobs.

Then, at a critical density, something extraordinary happens: a single cluster suddenly spans from one side of the wall all the way to the other. One more dot of ink, and the whole geometry changes. That sudden appearance of a giant connected cluster is a **phase transition** — but it is geometric, not energetic. No temperature is involved, no energy is minimized. It is purely about connectivity. Beautiful, isn't it?

This is **percolation**, and it turns out to have the same mathematical structure as the thermal phase transitions we studied earlier — critical exponents, scaling laws, universality, and all.

## Percolation theory

In **site percolation** on a lattice, each site is independently occupied with probability $p$ and empty with probability $1-p$. Occupied neighbors form clusters.

- For small $p$, only small isolated clusters exist.
- At the **percolation threshold** $p_c$, a giant cluster first spans the system.
- For $p > p_c$, the giant cluster contains a finite fraction of all sites.

The percolation threshold depends on the lattice geometry:

| Lattice | $p_c$ (site) |
|---------|-------------|
| Square | 0.5927 |
| Triangular | 0.5000 |
| Honeycomb | 0.6962 |

[[simulation percolation]]

Try the simulation: start with $p$ well below the threshold and slowly increase it. Watch the clusters grow, merge, and then suddenly — a single cluster connects the entire system. The transition is sharp and dramatic, just like the magnetic phase transition, even though no energy or temperature is involved.

## Critical exponents in percolation

Near the threshold, key quantities follow power laws in $|p - p_c|$, just like thermal phase transitions near $T_c$:

- **Order parameter** (fraction in spanning cluster): $P_\infty \sim (p - p_c)^\beta$ for $p > p_c$.
- **Mean cluster size** (excluding the spanning cluster): $\langle s \rangle \sim |p - p_c|^{-\gamma}$.
- **Correlation length** (typical cluster radius): $\xi \sim |p - p_c|^{-\nu}$.

In 2D: $\beta = 5/36$, $\gamma = 43/18$, $\nu = 4/3$. These exponents are universal within the percolation universality class — they do not depend on whether you use a square, triangular, or honeycomb lattice. Sound familiar? It is the same miracle of universality we saw for thermal phase transitions, just in a different universality class.

## The Bethe lattice

The **Bethe lattice** (Cayley tree) is an infinite tree where each node has exactly $z$ neighbors. Percolation on the Bethe lattice is exactly solvable:

$$
p_c = \frac{1}{z - 1}.
$$

For $z = 3$: $p_c = 1/2$. The Bethe lattice provides the **mean-field theory** for percolation and gives exact critical exponents $\beta = 1$, $\gamma = 1$, $\nu = 1/2$ — the same mean-field exponents we saw in the Ising model, because mean-field theory always gives the same exponents regardless of the specific system.

[[simulation bethe-lattice]]

In this simulation you can build a Bethe lattice and watch how occupied sites form clusters. Because every node looks the same (no loops, no boundaries), the math simplifies enormously — it is the percolation equivalent of mean-field theory.

> **Key Intuition.** Percolation is a geometric phase transition: at a critical occupation probability, a giant connected cluster suddenly spans the system. Near the threshold, the same power-law scaling and universality that govern thermal phase transitions appear here too — in a completely different universality class.

> **Challenge.** Take a piece of graph paper and randomly color squares with probability $p$. Start with $p = 0.3$, then $p = 0.5$, then $p = 0.7$. For each case, can you find a connected path of colored squares from the left edge to the right edge? You are doing a percolation experiment by hand. The critical threshold for a square lattice is about $p_c \approx 0.593$ — does your experiment roughly agree?

---

*At the percolation threshold, the spanning cluster is not a smooth blob — it is a tangled, wispy object full of holes at every scale. Zoom in, and you see the same kind of structure as when you zoom out. That is a fractal, and fractals are next.*
