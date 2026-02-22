# Fractals

At the percolation threshold, the spanning cluster isn't a smooth blob -- it's a tangled, wispy, infinitely detailed object full of holes at every scale. Zoom in, and you see the same kind of structure as when you zoom out. This is a **fractal**: a geometric object that exhibits self-similarity across scales.

Fractals aren't curiosities -- they appear at every critical point. The correlation structure of a magnet at $T_c$, the branching of a lightning bolt, the coastline of Norway. The mathematics of fractals is the geometric language of critical phenomena.

## Big Ideas

* Fractal dimension is non-integer because fractals are "too thin" to fill a plane but "too thick" to be a line -- they live in the twilight zone between dimensions.
* Critical clusters are fractals because scale invariance at the critical point means the same structure repeats at every length scale.
* The Koch snowflake calculation is the clearest self-similarity example: replace each segment with four copies at one-third the size, and $d_f = \log 4 / \log 3$ falls right out.
* Box-counting is the practical tool: measure the fractal dimension of any real-world object by counting boxes at different scales and fitting a slope.

## Fractal Dimension

The **fractal dimension** $d_f$ characterizes how mass scales with size:

$$
M(L) \sim L^{d_f}.
$$

For ordinary objects, $d_f = d$ (a square has $d_f = 2$, a cube has $d_f = 3$). For fractals, $d_f$ is non-integer -- the object is "too thin" to fill its space, but "too thick" to be a lower-dimensional object.

Examples:

* Koch snowflake: $d_f = \log 4 / \log 3 \approx 1.26$.
* Sierpinski triangle: $d_f = \log 3 / \log 2 \approx 1.58$.
* Percolation cluster at $p_c$ in 2D: $d_f = 91/48 \approx 1.896$.

[[simulation fractal-dimension]]

Play with this simulation to see how fractal dimension is measured. The object has more "stuff" than a line ($d_f > 1$) but less than a plane ($d_f < 2$). It lives in between -- and that fractional dimension is what makes it a fractal.

## The Mandelbrot Set

Zoom into the boundary of the Mandelbrot set and new tendrils appear forever. A curve that somehow fills area -- dimension exactly 2. That's what scale invariance looks like when it goes wild. The set is defined by the iteration

$$
z_{n+1} = z_n^2 + c, \qquad z_0 = 0,
$$

and a point $c$ belongs to the set if the orbit stays bounded. The **escape-time algorithm** colors each point by how many iterations before $|z_n|$ exceeds 2, producing those iconic images.

[[simulation mandelbrot-fractal]]

No matter how far you zoom, you keep finding spirals, tendrils, miniature copies of the whole set. Infinite self-similarity from a one-line formula. That's complexity from simplicity at its purest.

## Box-Counting Dimension

The practical way to estimate $d_f$ for any set:

1. Cover the set with boxes of side $\epsilon$.
2. Count the number $N(\epsilon)$ needed.
3. $d_f = -\lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log \epsilon}$.

Plot $\log N(\epsilon)$ against $\log(1/\epsilon)$ and extract the slope.

## What Comes Next

Fractals appear at critical points because you have to tune a parameter -- temperature to $T_c$, density to $p_c$ -- to arrive there. But some systems reach the critical point on their own, without anyone turning a knob. [Self-Organized Criticality](selfOrganizedCriticality) introduces the sandpile model, where adding sand grain by grain drives the pile to the critical slope automatically. Once there, avalanche sizes follow a power law -- and the avalanche shapes are fractal.

## Check Your Understanding

1. The Koch snowflake replaces each segment with four segments of one-third the length. Derive $d_f = \log 4 / \log 3$. What's the general principle for computing $d_f$ from a self-similar construction?
2. The Mandelbrot set boundary has $d_f = 2$ -- a curve that fills area. Yet you see thin tendrils and spirals, not a solid region. How can a curve have dimension 2?

## Challenge

The percolation cluster at $p_c$ has $d_f = 91/48 \approx 1.896$. Generate a percolation cluster at threshold on a large lattice and measure its fractal dimension using box counting. Plot $\log N$ against $\log(1/\epsilon)$ and extract the slope. How close do you get to the theoretical value? What are your main sources of error?
