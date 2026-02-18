# Percolation and Fractals

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

## Fractals and self-similarity

At the percolation threshold, the spanning cluster is not a smooth blob — it is a tangled, wispy, infinitely detailed object full of holes at every scale. Zoom in, and you see the same kind of structure as when you zoom out. This is a **fractal**: a geometric object that exhibits self-similarity across scales.

The **fractal dimension** $d_f$ characterizes how the mass of an object scales with its linear size:

$$
M(L) \sim L^{d_f}.
$$

For ordinary solid objects in $d$ dimensions, $d_f = d$ (a square has $d_f = 2$, a cube has $d_f = 3$). For fractals, $d_f$ is typically non-integer — the object is "too thin" to fill its embedding space, but "too thick" to be a lower-dimensional object.

Examples:

- Koch snowflake: $d_f = \log 4 / \log 3 \approx 1.26$.
- Sierpinski triangle: $d_f = \log 3 / \log 2 \approx 1.58$.
- Percolation cluster at $p_c$ in 2D: $d_f = 91/48 \approx 1.896$.

[[simulation fractal-dimension]]

Play with this simulation to see how fractal dimension is measured. The object has more "stuff" than a line ($d_f > 1$) but less than a plane ($d_f < 2$). It lives in between — and that fractional dimension is what makes it a fractal.

## The Mandelbrot set

The **Mandelbrot set** is defined in the complex plane as the set of values $c$ for which the iteration

$$
z_{n+1} = z_n^2 + c, \qquad z_0 = 0,
$$

remains bounded. The boundary of the Mandelbrot set is a fractal with infinite detail at every scale.

The **escape-time algorithm** colors each point $c$ by the number of iterations needed for $|z_n|$ to exceed a threshold (typically 2), producing the iconic visualizations of the set.

[[simulation mandelbrot-fractal]]

Zoom into the boundary of the Mandelbrot set. No matter how far you zoom, you keep finding new structure — spirals, tendrils, miniature copies of the whole set. This infinite self-similarity from a one-line formula ($z \to z^2 + c$) is one of the most stunning examples of complexity emerging from simplicity.

## Box-counting dimension

The **box-counting method** provides a practical way to estimate the fractal dimension of any set:

1. Cover the set with boxes of side length $\epsilon$.
2. Count the number $N(\epsilon)$ of boxes needed.
3. The fractal dimension is $d_f = -\lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log \epsilon}$.

In practice, $\log N(\epsilon)$ is plotted against $\log(1/\epsilon)$, and $d_f$ is estimated from the slope of the linear region.

> **Key Intuition.** Percolation is a geometric phase transition: at a critical occupation probability, a giant connected cluster suddenly spans the system. At the threshold, the critical cluster is a fractal — it has structure at every scale, just like a thermal system at its critical point has correlations at every scale. This is not a coincidence: percolation and thermal phase transitions share the same mathematical framework of scaling and universality.

> **Challenge.** Take a piece of graph paper and randomly color squares with probability $p$. Start with $p = 0.3$, then $p = 0.5$, then $p = 0.7$. For each case, can you find a connected path of colored squares from the left edge to the right edge? You are doing a percolation experiment by hand. The critical threshold for a square lattice is about $p_c \approx 0.593$ — does your experiment roughly agree?

---

*In percolation, we had to tune a parameter ($p$) to reach the critical point. But there are systems in nature that drive themselves to criticality without any external tuning. Drop sand on a pile, and it self-organizes to the exact slope where tiny inputs can cause huge avalanches. That is self-organized criticality, and it is next.*
