# Percolation and Fractals

## Percolation theory

**Percolation theory** studies the connectivity of random media. The fundamental question is: when does a connected pathway spanning the entire system first appear?

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

Near $p_c$, the system exhibits critical behavior analogous to thermal phase transitions.

[[simulation percolation]]

## Critical exponents in percolation

Near the threshold, key quantities follow power laws in $|p - p_c|$:

- **Order parameter** (fraction in spanning cluster): $P_\infty \sim (p - p_c)^\beta$ for $p > p_c$.
- **Mean cluster size** (excluding the spanning cluster): $\langle s \rangle \sim |p - p_c|^{-\gamma}$.
- **Correlation length** (typical cluster radius): $\xi \sim |p - p_c|^{-\nu}$.

In 2D: $\beta = 5/36$, $\gamma = 43/18$, $\nu = 4/3$. These exponents are universal within the percolation universality class.

## The Bethe lattice

The **Bethe lattice** (Cayley tree) is an infinite tree where each node has exactly $z$ neighbors. Percolation on the Bethe lattice is exactly solvable:

$$
p_c = \frac{1}{z - 1}.
$$

For $z = 3$: $p_c = 1/2$. The Bethe lattice provides the mean-field theory for percolation and gives exact critical exponents $\beta = 1$, $\gamma = 1$, $\nu = 1/2$.

[[simulation bethe-lattice]]

## Fractals and self-similarity

A **fractal** is a geometric object that exhibits self-similarity: it looks similar at different scales of magnification. Fractals arise naturally in percolation clusters at criticality.

The **fractal dimension** $d_f$ characterizes how the mass of an object scales with its linear size:

$$
M(L) \sim L^{d_f}.
$$

For ordinary objects in $d$ dimensions, $d_f = d$. For fractals, $d_f$ is typically non-integer.

Examples:

- Koch snowflake: $d_f = \log 4 / \log 3 \approx 1.26$.
- Sierpinski triangle: $d_f = \log 3 / \log 2 \approx 1.58$.
- Percolation cluster at $p_c$ in 2D: $d_f = 91/48 \approx 1.896$.

[[simulation fractal-dimension]]

## The Mandelbrot set

The **Mandelbrot set** is defined in the complex plane as the set of values $c$ for which the iteration

$$
z_{n+1} = z_n^2 + c, \qquad z_0 = 0,
$$

remains bounded. The boundary of the Mandelbrot set is a fractal with infinite detail at every scale.

The **escape-time algorithm** colors each point $c$ by the number of iterations needed for $|z_n|$ to exceed a threshold (typically 2), producing the iconic visualizations of the set.

[[simulation mandelbrot-fractal]]

## Box-counting dimension

The **box-counting method** provides a practical way to estimate the fractal dimension of any set:

1. Cover the set with boxes of side length $\epsilon$.
2. Count the number $N(\epsilon)$ of boxes needed.
3. The fractal dimension is $d_f = -\lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log \epsilon}$.

In practice, $\log N(\epsilon)$ is plotted against $\log(1/\epsilon)$, and $d_f$ is estimated from the slope of the linear region.
