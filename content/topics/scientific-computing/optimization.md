# Optimization Methods

> *The quasi-Newton ideas from [Broyden's method](./nonlinear-systems) show up again here as BFGS — the same trick of building up curvature knowledge step by step, but now for finding minima instead of zeros.*

## Big Ideas

* Gradient descent follows the steepest local slope, which is almost never the most efficient direction — it zigzags in valleys and wastes effort.
* BFGS builds a running portrait of the landscape's curvature from gradient differences alone, converging superlinearly without ever computing a Hessian explicitly.
* Conjugate gradients trade convergence speed for memory: $O(n)$ storage instead of $O(n^2)$, making million-dimensional problems tractable.
* Metaheuristics (simulated annealing, particle swarms, genetic algorithms) are frameworks for escaping local minima at the cost of convergence guarantees.

## The Party in the Dark

Imagine you're at a party in a pitch-dark room trying to find the coldest spot (next to the AC). You can feel the temperature where you're standing and maybe which direction is cooler. That's gradient descent — just follow the chill.

But what if the room has multiple cold spots? You might get stuck in a corner that's cool but not the *coolest*. That's a local minimum. The big question is always: **How much do you know about the landscape, and how much can you afford to explore?**

## When to Use What

Three questions decide your method:
1. How slow are function evaluations (and gradients)?
2. How big is your space?
3. How ugly is the landscape? Convex, or riddled with local minima?

**Rules of thumb:**
* **Low dimensions, fast evaluations:** Grid search + Newton-Raphson. Works even for ugly landscapes.
* **Medium dimensions (~100):** BFGS if landscape is simple. BFGS + exploration if it's not.
* **High dimensions (up to millions):** Conjugate gradients. Slower convergence per step than BFGS, but you don't need to store a massive Hessian matrix.

Why can't you just use gradient descent for everything? Because it zigzags in narrow valleys and takes forever. BFGS and conjugate gradients learn the shape of the valley and take much better steps.

[[simulation rosenbrock-banana]]

## BFGS (Broyden-Fletcher-Goldfarb-Shanno)

Newton's method for optimization uses the Hessian $H$ to find the step: $H_k \Delta x_k = -\nabla f(x_k)$

*The Hessian tells you the curvature of the landscape, and Newton uses it to jump to the bottom of the local bowl. But computing the full Hessian is expensive.*

BFGS builds an approximation $B_k \approx H_k^{-1}$ using only gradient information:

$$B_{k+1} = \left(I - \frac{s_k y_k^T}{y_k^T s_k}\right) B_k \left(I - \frac{y_k s_k^T}{y_k^T s_k}\right) + \frac{s_k s_k^T}{y_k^T s_k}$$

where $s_k = x_{k+1} - x_k$ and $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$.

*Each step teaches you a bit more about the shape of the landscape.*

**The algorithm:**
1. **Start ignorant:** $B_0 = I$ (pretend the landscape is a simple bowl)
2. **Feel the slope:** $\Delta x_k = -B_k \nabla f(x_k)$
3. **Walk carefully:** Line search for step size (Wolfe conditions)
4. **Take the step:** $x_{k+1} = x_k + \alpha_k \Delta x_k$
5. **Learn from the step:** Update $B_{k+1}$
6. **Repeat** until $\|\nabla f\| < \text{tol}$

BFGS achieves **superlinear convergence** near a minimum. Beyond ~100 dimensions, storing the $n\times n$ matrix becomes prohibitive — that's where **L-BFGS** comes in, keeping only the last $m$ step pairs and reconstructing the matrix-vector product on the fly. Storage drops from $O(n^2)$ to $O(mn)$.

## Conjugate Gradient Method

For high-dimensional problems, even $O(n^2)$ storage is too much. CG uses only $O(n)$.

**Key idea:** Instead of steepest descent (which zigzags), choose directions that are _conjugate_ with respect to the Hessian: $d_i^T H d_j = 0$. This guarantees progress in one direction isn't undone by later steps.

*Steepest descent is a drunk stumbling downhill, zigzagging across a valley. Conjugate gradients is a sober hiker who never backtracks.*

**Nonlinear CG (Fletcher-Reeves):**
1. $d_0 = -\nabla f(x_0)$
2. Line search for $\alpha_k$
3. $x_{k+1} = x_k + \alpha_k d_k$
4. $\beta_{k+1} = \frac{g_{k+1}^T g_{k+1}}{g_k^T g_k}$
5. $d_{k+1} = -g_{k+1} + \beta_{k+1} d_k$

CG converges slower per iteration than BFGS, but each iteration is $O(n)$ instead of $O(n^2)$. Classic speed-memory tradeoff.

## Simulated Annealing

*Think of shaking a box of balls on a bumpy surface. At high temperature (lots of shaking), balls jump out of shallow dips and explore widely. As you cool (less shaking), they settle into the deepest valleys.*

1. Perturb $x$ by random $\Delta x$
2. Compute $\Delta E = f(x+\Delta x) - f(x)$
3. Accept with probability $P = e^{-\Delta E/k_BT}$
4. Cool: $T_{k+1} = \alpha T_k$

*The trick: sometimes accept a worse solution. This lets you escape local minima early on, then settle into the global minimum as temperature drops.*

## Particle Swarm Optimization

Maintain a swarm of particles $x_1, \dots, x_m \in \mathbb{R}^n$. Each step:
1. Pick a "free will" direction $\delta$ for each particle
2. Compute pairwise energy differences
3. Follow particles with better solutions
4. Update: $\Delta x_i = \sum_j \alpha_{ij}\phi_{ij}d_{ij} + \iota\delta_i$

Here's the trick for implementation — vectorize everything:

```python
fs = array([f(x) for x in xs])
F = fs[:, np.newaxis] - fs[np.newaxis, :]   # all pairwise energy differences at once
Phi = (1 - np.sign(F)) / 2                  # who's better than whom?
X += np.sum(alpha[:,:,np.newaxis] * phi[:,:,np.newaxis] * d, axis=1)  # swarm update
```

## Genetic Algorithms

Four ingredients for evolution:
1. **Representation:** A genetic code
2. **Mating:** Splitting and recombining genomes
3. **Selection Pressure:** Who survives to the next generation?
4. **Mutation:** Random perturbations to explore new territory

> **Challenge.** Here's a toy you can play with right now: implement simulated annealing in 20 lines of Python. Minimize the Rastrigin function $f(x) = 10n + \sum(x_i^2 - 10\cos(2\pi x_i))$ in 2D. Start at $T=100$, cool with $\alpha=0.999$. Plot the path — watch it jump around at first, then settle down.

---

## What Comes Next

Optimization brings us to the boundary between deterministic and stochastic computation. Behind every objective function lurks a matrix, and that matrix's eigenvalues govern everything: the shape of the bowl, the stiffness of the descent, the speed of convergence. Eigenvalue problems are next.

## Check Your Understanding

1. Why does steepest descent zigzag in an elongated valley, and how does BFGS fix this?
2. Simulated annealing sometimes accepts worse solutions. Why is this a feature, not a bug?
3. Conjugate gradient directions satisfy $d_i^T H d_j = 0$. Why does this prevent backtracking?

## Challenge

Minimize the 2D Rosenbrock function $f(x, y) = (1-x)^2 + 100(y - x^2)^2$ using (a) gradient descent with fixed learning rate, (b) BFGS via `scipy.optimize.minimize`, and (c) hand-rolled simulated annealing. Plot trajectories on a contour plot. Record function evaluations and final error for each.
