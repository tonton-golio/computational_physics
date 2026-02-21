# Fractals

At the percolation threshold, the spanning cluster is not a smooth blob — it is a tangled, wispy, infinitely detailed object full of holes at every scale. Zoom in, and you see the same kind of structure as when you zoom out. This is a **fractal**: a geometric object that exhibits self-similarity across scales.

Fractals are not just curiosities — they appear at every critical point. The correlation structure of a magnet at $T_c$, the branching of a lightning bolt, the coastline of Norway — all are fractal. The mathematics of fractals provides the geometric language for understanding critical phenomena.

## Fractal dimension

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

> **Key Intuition.** Fractals are the geometry of critical phenomena. At a critical point — whether in percolation, magnetism, or any other phase transition — the system has structure at every scale. The fractal dimension quantifies exactly how much structure: it tells you whether the critical object is closer to a line, a surface, or something in between. This is not a coincidence: the same scaling laws that produce power-law divergences in thermodynamic quantities also produce fractal geometry in real space.

---

*In percolation, we had to tune a parameter ($p$) to reach the critical point. But there are systems in nature that drive themselves to criticality without any external tuning. Drop sand on a pile, and it self-organizes to the exact slope where tiny inputs can cause huge avalanches. That is self-organized criticality, and it is next.*
