# Percolation

Suppose you are painting a wall with random dots of ink. At first they are isolated islands — tiny splotches scattered here and there with gaps between them. Keep adding dots. Some start to merge into small clusters. Keep going. More clusters connect, forming larger and larger blobs.

Then, at a critical density, something extraordinary happens: a single cluster suddenly spans from one side of the wall all the way to the other. One more dot of ink, and the whole geometry changes. That sudden appearance of a giant connected cluster is a **phase transition** — but it is geometric, not energetic. No temperature is involved, no energy is minimized. It is purely about connectivity. Beautiful, isn't it?

This is **percolation**, and it turns out to have the same mathematical structure as the thermal phase transitions we studied earlier — critical exponents, scaling laws, universality, and all.

## Percolation theory

In **site percolation** on a lattice, each site is independently occupied with probability $p$ and empty with probability $1-p$. Occupied neighbors form clusters.

* For small $p$, only small isolated clusters exist.
* At the **percolation threshold** $p_c$, a giant cluster first spans the system.
* For $p > p_c$, the giant cluster contains a finite fraction of all sites.

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

* **Order parameter** (fraction in spanning cluster): $P_\infty \sim (p - p_c)^\beta$ for $p > p_c$.
* **Mean cluster size** (excluding the spanning cluster): $\langle s \rangle \sim |p - p_c|^{-\gamma}$.
* **Correlation length** (typical cluster radius): $\xi \sim |p - p_c|^{-\nu}$.

In 2D: $\beta = 5/36$, $\gamma = 43/18$, $\nu = 4/3$. These exponents are universal within the percolation universality class — they do not depend on whether you use a square, triangular, or honeycomb lattice. Sound familiar? It is the same miracle of universality we saw for thermal phase transitions, just in a different universality class.

## The Bethe lattice

The **Bethe lattice** (Cayley tree) is an infinite tree where each node has exactly $z$ neighbors. Percolation on the Bethe lattice is exactly solvable:

$$
p_c = \frac{1}{z - 1}.
$$

For $z = 3$: $p_c = 1/2$. The Bethe lattice provides the **mean-field theory** for percolation and gives exact critical exponents $\beta = 1$, $\gamma = 1$, $\nu = 1/2$ — the same mean-field exponents we saw in the Ising model, because mean-field theory always gives the same exponents regardless of the specific system.

[[simulation bethe-lattice]]

In this simulation you can build a Bethe lattice and watch how occupied sites form clusters. Because every node looks the same (no loops, no boundaries), the math simplifies enormously — it is the percolation equivalent of mean-field theory.

## Big Ideas

* Percolation is a phase transition without energy or temperature — the control parameter is simply the density of occupied sites, and the "order" is connectivity rather than magnetization.
* The percolation threshold $p_c$ depends on the lattice geometry, but the critical exponents near $p_c$ do not — universality again, in a different class from the Ising model.
* The Bethe lattice gives the mean-field theory of percolation: it is exactly solvable because it has no loops, and it gives the same mean-field exponents ($\beta = 1$, $\gamma = 1$) that appear in every mean-field theory.
* The spanning cluster at $p_c$ is not a compact blob — it is fractal, full of holes at every scale, a harbinger of the [Fractals](fractals) we encounter next.

## What Comes Next

The spanning cluster right at the percolation threshold is not a smooth, space-filling object — it has holes within holes within holes, self-similar structure all the way down. This is the geometry of a fractal. [Fractals](fractals) quantifies this self-similarity through the fractal dimension $d_f$, which is non-integer and tells you exactly how much space a critical cluster fills. The fractal geometry of percolation clusters is not a coincidence: it is the geometric signature of scale invariance at the critical point, the same scale invariance that produces power-law exponents in the thermodynamic quantities.

## Check Your Understanding

1. In site percolation, the "order parameter" is $P_\infty$, the fraction of sites in the spanning cluster. Above $p_c$ it is nonzero; below it is zero. Why is $P_\infty$ identically zero below $p_c$ in an infinite system, even though large finite clusters exist?
2. The percolation threshold for a triangular lattice ($p_c = 0.5$) is exactly solvable by a duality argument. Explain qualitatively why triangular and honeycomb lattices are "dual" to each other, and why this duality implies $p_c = 0.5$ for the triangular lattice.
3. The mean cluster size $\langle s \rangle$ diverges at $p_c$ from *both* sides — as $p \to p_c^-$ and as $p \to p_c^+$. Why does the average cluster size diverge even below the threshold, where the spanning cluster does not yet exist?

## Challenge

The Hoshen-Kopelman algorithm efficiently labels all clusters in a percolation configuration using a single pass through the lattice. Describe how the algorithm works (hint: it uses a union-find data structure). Implement it (or trace through a small example by hand) on a $5 \times 5$ lattice with occupation probability $p = 0.6$. Identify the spanning cluster, if one exists. Estimate the computational complexity of the algorithm as a function of the number of sites $N$.
