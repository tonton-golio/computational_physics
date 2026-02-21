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

* Koch snowflake: $d_f = \log 4 / \log 3 \approx 1.26$.
* Sierpinski triangle: $d_f = \log 3 / \log 2 \approx 1.58$.
* Percolation cluster at $p_c$ in 2D: $d_f = 91/48 \approx 1.896$.

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

## Big Ideas

* Fractal dimension is non-integer because fractals are "too thin" to fill a plane but "too thick" to be a line — they occupy the twilight zone between dimensions.
* The Mandelbrot set is the most complex object you can define with a one-line formula: $z \to z^2 + c$ produces infinite self-similar detail from pure mathematics.
* Critical clusters are fractals because scale invariance at the critical point means the same kind of structure repeats at every length scale — the same reason power laws appear in thermodynamic quantities.
* The box-counting dimension is a practical tool: you can measure the fractal dimension of any real-world object (coastlines, trees, blood vessels) by counting boxes at different scales and fitting a slope.

## What Comes Next

Fractals appear at critical points because you must tune a control parameter — temperature to $T_c$, or occupation probability to $p_c$ — to arrive there. But some systems reach the critical point on their own, without anyone turning a knob. [Self-Organized Criticality](selfOrganizedCriticality) introduces the sandpile model, where adding sand grain by grain drives the pile to the critical slope automatically. Once there, the avalanche sizes follow a power law — and the avalanche shapes are fractal. You have seen all the ingredients; SOC brings them together into a single mechanism that may explain why power laws are everywhere in nature.

## Check Your Understanding

1. The Koch snowflake is built by repeatedly replacing each line segment with four segments of one-third the length. Use this self-similarity to derive the fractal dimension $d_f = \log 4 / \log 3$. Why does this formula work — what is the general principle for computing $d_f$ from a self-similar construction rule?
2. The Mandelbrot set boundary has fractal dimension 2 (it is a curve that fills area). Yet when you zoom in, you see thin tendrils and spirals, not a solid region. How can a curve have dimension 2?
3. The percolation cluster at $p_c$ in 2D has fractal dimension $d_f = 91/48 \approx 1.896$, which is close to but less than 2. What does it mean physically that the spanning cluster almost — but not quite — fills the 2D plane?

## Challenge

Measure the fractal dimension of a real-world object using box counting. Take a digital image of a coastline, a tree silhouette, or a lightning bolt. Convert it to black and white. Then, systematically cover it with grids of boxes of side length $\epsilon = 1, 2, 4, 8, \ldots$ pixels and count the number of boxes $N(\epsilon)$ that contain any black pixels. Plot $\log N$ against $\log(1/\epsilon)$ and extract the slope. Compare your measured fractal dimension to the theoretical values for similar objects (coastlines are typically $d_f \approx 1.2$–$1.5$, depending on how rugged they are). What are the main sources of error in your measurement?
