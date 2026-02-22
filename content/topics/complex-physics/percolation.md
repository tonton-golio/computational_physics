# Percolation

Suppose you're painting a wall with random dots of ink. At first they're isolated islands -- tiny splotches scattered here and there. Keep adding dots. Some merge into small clusters. Keep going.

Then, at a critical density, something extraordinary happens: a single cluster suddenly spans from one side all the way to the other. One more dot, and the whole geometry changes. That sudden appearance of a giant connected cluster is a **phase transition** -- but it's geometric, not energetic. No temperature, no energy minimized. Purely about connectivity.

This is **percolation**, and it has the same mathematical structure as the thermal phase transitions we studied -- critical exponents, scaling laws, universality, and all.

## Big Ideas

* Percolation is a phase transition without energy or temperature -- the control parameter is density, and the "order" is connectivity rather than magnetization.
* The percolation threshold $p_c$ depends on lattice geometry, but the critical exponents near $p_c$ do not -- universality again, in a different class from the Ising model.
* The Bethe lattice gives the mean-field theory of percolation: exactly solvable because it has no loops, same mean-field exponents that show up everywhere.
* The spanning cluster at $p_c$ isn't a compact blob -- it's fractal, full of holes at every scale, a harbinger of the [Fractals](fractals) we encounter next.

## Percolation Theory

In **site percolation**, each site is independently occupied with probability $p$, empty with $1-p$. Occupied neighbors form clusters.

* Small $p$: only small isolated clusters.
* At **$p_c$**: a giant cluster first spans the system.
* Above $p_c$: the giant cluster contains a finite fraction of all sites.

The threshold depends on lattice geometry:

| Lattice | $p_c$ (site) |
|---------|-------------|
| Square | 0.5927 |
| Triangular | 0.5000 |
| Honeycomb | 0.6962 |

[[simulation percolation]]

Try the simulation: start well below threshold and slowly increase $p$. Watch clusters grow, merge, and then suddenly -- one cluster connects everything. Sharp and dramatic, just like the magnetic transition, without any temperature in sight.

## Critical Exponents in Percolation

Near threshold, key quantities follow power laws in $|p - p_c|$:

* **Order parameter** (fraction in spanning cluster): $P_\infty \sim (p - p_c)^\beta$ for $p > p_c$.
* **Mean cluster size**: $\langle s \rangle \sim |p - p_c|^{-\gamma}$.
* **Correlation length**: $\xi \sim |p - p_c|^{-\nu}$.

In 2D: $\beta = 5/36$, $\gamma = 43/18$, $\nu = 4/3$. These are universal within the percolation class -- they don't depend on whether you use a square, triangular, or honeycomb lattice. Same miracle of universality, different universality class.

## The Bethe Lattice

The **Bethe lattice** (Cayley tree) is an infinite tree where each node has exactly $z$ neighbors. Percolation on it is exactly solvable:

$$
p_c = \frac{1}{z - 1}.
$$

For $z = 3$: $p_c = 1/2$. The Bethe lattice is the mean-field theory of percolation, giving exact exponents $\beta = 1$, $\gamma = 1$, $\nu = 1/2$.

[[simulation bethe-lattice]]

Because every node looks the same (no loops, no boundaries), the math simplifies enormously -- the percolation equivalent of mean-field theory.

## Duality and the Triangular Lattice

Why is $p_c = 1/2$ exact for the triangular lattice? Draw the dual honeycomb lattice and you'll see that occupied sites on one become empty bonds on the other. At $p = 1/2$ they are exactly the same problem -- so $p_c$ must be exactly $1/2$. Beautiful symmetry, isn't it?

## What Comes Next

The spanning cluster at the threshold isn't a smooth, space-filling object -- it has holes within holes within holes, self-similar structure all the way down. [Fractals](fractals) quantifies this through the fractal dimension $d_f$, which is non-integer and tells you exactly how much space a critical cluster fills. The fractal geometry of percolation clusters is the geometric signature of scale invariance at the critical point.

## Check Your Understanding

1. Why is $P_\infty$ identically zero below $p_c$ in an infinite system, even though large finite clusters exist?
2. The mean cluster size $\langle s \rangle$ diverges at $p_c$ from *both* sides. Why does it diverge even below threshold, where no spanning cluster exists?

## Challenge

The Hoshen-Kopelman algorithm efficiently labels all clusters using a single pass through the lattice (hint: union-find data structure). Implement it on a $5 \times 5$ lattice with $p = 0.6$. Identify the spanning cluster if one exists. Estimate the computational complexity as a function of $N$.
