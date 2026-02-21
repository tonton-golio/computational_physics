# Optimization Methods

> *The quasi-Newton ideas from [Broyden's method](./nonlinear-systems) show up again here as BFGS — the same trick of building up curvature knowledge step by step, but now for finding minima instead of zeros.*

## Non-linear Optimization

### The party in the dark

Imagine you're at a party in a pitch-dark room and you're trying to find the coldest spot (next to the air conditioning). You can feel the temperature where you're standing, and maybe you can feel which direction is cooler. That's gradient descent — just follow the chill.

But what if the room has multiple cold spots? You might get stuck in a corner that's cool but not the *coolest*. That's a local minimum. Now imagine the room is full of people, all searching independently, and they shout to each other when they find somewhere cold. That's metaheuristic optimization — collective exploration beats individual greedy search.

The big question is always: **How much do you know about the landscape, and how much can you afford to explore?**

## When to use what:

### Questions that'll help decide on what to use
1. How slow are function evaluations (and gradients)?
2. How big is your space?
3. How ugly is your energy landscape? Convex, or many minima?

### Rules of thumb:
1. Fast function evaluations:
   1.  You can just take a linspace, evaluate the function at every point, and either use the minimum or feed that to a Newton Raphson method.  Both pretty and ugly energy landscapes works, it just changes the size of the linspace.
   2.  For medium dimensions (up to 100), you want to use BFGS if the energy landscape is simple. If it's complicated, you want to use BFGS + Exploration, as you need some way to escape local minima to find a global minima.
   3.  For high dimensions (up to 1M), use conjugated gradients. Takes longer to converge than BFGS, but you don't have to represent the high dimensional hessian matrix and so it works up to millions of dimensions. For a convoluted energy landscape, you want to use exploration as well.

2. Slow function evaluations
   1. Low to medium dimensions: If the energy landscape is simple, BFGS. Even if you have a complicated energy landscape, your search area is small enough that you can use BFGS with exploration.
   2. High dimensions: Simple energy landscape, use conjugate gradients. When it's expensive to evaluate the function, and you're in a high-dimensional complex landscape, the search space is too big for you to get anywhere. Here you need to think, and tailor your solution to fit your problem. Generally, you can try to use some sort of symmetry or structure of your problem and then use a metaheuristic to guide your solutions

Why can't you just use gradient descent for everything? Because gradient descent is like always walking downhill in the steepest direction — it zigzags in narrow valleys and takes forever to converge. BFGS and conjugate gradients are smarter: they learn the shape of the valley and take much better steps.

[[simulation rosenbrock-banana]]

## BFGS (Broyden-Fletcher-Goldfarb-Shanno)

BFGS is a quasi-Newton method for minimizing $f:\mathbb{R}^n\to\mathbb{R}$. Newton's method for optimization uses the Hessian $H$ to find the step:

$$H_k \Delta x_k = -\nabla f(x_k)$$

*This says: the Hessian tells you the curvature of the landscape, and Newton uses it to jump directly to the bottom of the local bowl. But computing the full Hessian is expensive.*

Computing the full Hessian is expensive ($O(n^2)$ storage, $O(n^3)$ to solve). BFGS builds an approximation $B_k \approx H_k^{-1}$ using only gradient information, similar to how Broyden's method approximates the Jacobian.

**BFGS update rule:** Given step $s_k = x_{k+1} - x_k$ and gradient change $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$:

$$B_{k+1} = \left(I - \frac{s_k y_k^T}{y_k^T s_k}\right) B_k \left(I - \frac{y_k s_k^T}{y_k^T s_k}\right) + \frac{s_k s_k^T}{y_k^T s_k}$$

*This says: update your curvature estimate using the step you just took and how much the gradient changed. Each step teaches you a bit more about the shape of the landscape.*

This is a rank-2 update (two outer products), so each step costs $O(n^2)$ instead of $O(n^3)$.

**Algorithm — a numbered story:**
1. **Start ignorant:** Set $B_0 = I$ (pretend the landscape is a simple bowl)
2. **Feel the slope:** Compute search direction $\Delta x_k = -B_k \nabla f(x_k)$
3. **Walk carefully:** Line search to find step size $\alpha_k$ along $\Delta x_k$ (Wolfe conditions)
4. **Take the step:** Update $x_{k+1} = x_k + \alpha_k \Delta x_k$
5. **Learn from the step:** Update $B_{k+1}$ using the formula above
6. **Repeat** until $\Vert \nabla f \Vert < \text{tol}$

BFGS achieves **superlinear convergence** near a minimum and works well up to medium dimensions (~100). Beyond that, storing the $n\times n$ matrix $B_k$ becomes prohibitive.

**L-BFGS** (limited-memory BFGS) avoids storing the full matrix by keeping only the last $m$ pairs $(s_k, y_k)$ and reconstructing the matrix-vector product implicitly. This reduces storage from $O(n^2)$ to $O(mn)$ and works up to millions of dimensions, though it converges somewhat slower than full BFGS.

## Conjugate Gradient Method

For high-dimensional problems where even $O(n^2)$ storage is too much, the conjugate gradient (CG) method uses only $O(n)$ storage by maintaining just a search direction vector.

**Key idea:** Instead of steepest descent (which zigzags), choose search directions that are _conjugate_ with respect to the Hessian: $d_i^T H d_j = 0$ for $i \neq j$. This guarantees that progress made in one direction is not undone by later steps.

*Think of it like this: if steepest descent is a drunk stumbling downhill (zigzagging back and forth across a valley), conjugate gradients is a sober hiker who remembers where they've already been and never backtracks.*

**Nonlinear CG (Fletcher-Reeves):**
1. **Start steep:** Set initial direction $d_0 = -\nabla f(x_0)$
2. **Slide downhill:** Line search to find $\alpha_k$ that minimizes $f(x_k + \alpha_k d_k)$
3. **Take the step:** $x_{k+1} = x_k + \alpha_k d_k$
4. **New gradient:** $g_{k+1} = \nabla f(x_{k+1})$
5. **Stay conjugate:** $\beta_{k+1} = \frac{g_{k+1}^T g_{k+1}}{g_k^T g_k}$
6. **New direction:** $d_{k+1} = -g_{k+1} + \beta_{k+1} d_k$

The Polak-Ribiere variant uses $\beta_{k+1} = \frac{g_{k+1}^T(g_{k+1}-g_k)}{g_k^T g_k}$ instead, which often performs better in practice.

CG converges slower than BFGS per iteration, but each iteration is $O(n)$ instead of $O(n^2)$, making it the method of choice for high-dimensional problems (up to millions of dimensions).

If conjugate gradients only uses $O(n)$ storage and works in millions of dimensions, why not always use it? Because BFGS converges faster per step — it has more information (the approximate Hessian). It's a classic speed-memory tradeoff.

## Metaheuristics

_Algorithm:_ A computational method with a guaranteed correct result after a finite number of steps.

_Heuristic:_ The same as an algorithm, but with no guarantees.

An algorithm has 2 guarantees: it gives you a "correct" result (as per your definition of correct; for optimization, this could mean finding a local minimum within some tolerance), and it happens after a finite number of steps. A heuristic doesn't guarantee either.

_Metaheuristic:_ A scheme for building heuristics. It's a framework where you have some overall structure (from your problem), then you take your metaheuristic scheme, tailor it to your problem, and then you produce a heuristic that you can run to get good answers for your problem

## Simulated Annealing:

Inspired by physical processes: annealing is when you let something cool slowly in order to get the correct hardened structure (in glass, metal, etc). 'Hardening' means that you find a position of atoms that minimizes the energy.

*Think of it as shaking a box of balls on a bumpy surface. At high temperature (lots of shaking), balls jump out of shallow dips and explore widely. As you cool down (less shaking), they settle into the deepest valleys.*

For minimizing $f:\mathbb{R}^n\to\mathbb{R}$ (maps from the high dimensional space to a scalar energy value):

* Maintain one state vector $x\in\mathbb{R}^n$
* Start with a high temperature $T_0$ ("high" is problem dependent).
* Gradually cool it down to 0 K

In each step:
1. Perturb $x$ by a random motion $\Delta x$
2. Let $\Delta E=(f(x+\Delta x)-f(x))\gamma$, where $\gamma$ is an optional energy unit.
3. Define a _transition probability_: $P(T, \Delta E) = e^{\Delta E/k_BT}$
4. Call a random number generator to get an $r\in[0,1]$ and accept new step if $P(T, \Delta E)\geq r$
5. Cool temperature: $T_{k+1} = \alpha T_k$

*The trick: sometimes accept a *worse* solution (uphill step) with probability that decreases as you cool. This lets you escape local minima early on, then settle into the global minimum as the temperature drops.*

## Particle Swarm Optimizations

* Maintain multiple state vectors (a whole swarm of particles), $x_1,x_2,\dots,x_m \in \mathbb{R^n}$

In each step:
1. Pick a "free will" direction $\delta\in\mathbb{R}^n$ for each particle
2. Calculate $M\times M$ matrices
   1. The displacements $d:d_{ij}=x_j-x_i$
   2. Energy differences $F:F_{ij}=F(x_j)-F(x_i)$
3. Follow everyone who has a better solution: $f(x_j) < f(x_i)$, then $i$ follows $j$. We can program this as $\phi_{ij}=\frac12 \left(1-\text{sign}(F_{ij})\right)$. If $F_{ij}>0$, $\phi_{ij}=1$; $F <0$ then $\phi=0; F=0$ then $\phi=\frac12$
4. The direction of flight combines "Free will direction" $\delta_i$ with swarm movement: The direction is it's own independent movement (scaled by _independence factor_ $\iota$) and an _attraction factor_ $\alpha_{i,j}$ which is how much you want to follow the swarm:
$$\Delta x_i=\sum_{j=1}^M\alpha_{i,j}\phi_{i,j}d_{i,j} + \iota\delta_i$$

The attraction term ($\alpha$) is something that we must decide for ourselves. For example, $\alpha_{i,j} = \beta e^{-\gamma ||\delta_{i,j}||^2}$ (gaussian decline with distance: exponential will drop too quickly). You can also use a Levy distribution: small immediately around you, then spikes up close to you and declines slowly. You don't get pulled right on top of yourself. Even something like $\alpha_{ij}=\beta(F_{i,j})e^{-\gamma ||\delta_{i,j}||^2}$, where the attraction depends on the force of attraction.

Note: when programming: **don't** make loops over $i$ and $j$: do 2.2. like
```python
fs = array([f(x) for x in xs])
F = fs[:, np.newaxis] - fs[np.newaxis, :]   # all pairwise energy differences at once
```
Similarly for 3.
```python
Phi = (1 - np.sign(F)) / 2                  # who's better than whom?
```
And for 4.
```python
X += np.sum(alpha[:,:,np.newaxis] * phi[:,:,np.newaxis] * d, axis=1)  # swarm update
```
We use newaxis in order to match $\alpha$ and $\phi$ with $d$, which is a rank 3 tensor.

Note that $\delta$, the free will step, can also be tweaked. The simplest thing to do is to use brownian motion: you get something that explores a local area really well, but doesn't go very far out. If you use the Levy distribution instead of the gaussian, you get a combination of local exploration with (sometimes) big jumps to a new place.

## Genetic Algorithms

### How to start an evolution:

1. **Representation:** A genetic code
2. **Mating:** Processes for splitting and recombining genomes
3. **Selection Pressure:** Who, and whose offspring, make up the next generation?
4. **Mutation:** Random perturbations (to get somewhere new)

Ideas $f:\mathbb{R}^{3N}\to\mathbb{R}$ representing $N$ particle position in space $\mathbb{R}^3$

> **Challenge.** Implement simulated annealing in 20 lines of Python. Minimize the Rastrigin function $f(x) = 10n + \sum(x_i^2 - 10\cos(2\pi x_i))$ in 2D. Start at temperature $T=100$, cool with $\alpha=0.999$. Plot the path — watch it jump around at first, then settle down.

---

## Big Ideas

* Gradient descent follows the steepest local slope, which is almost never the most efficient direction to travel — it zigzags in valleys and wastes effort.
* BFGS builds a running portrait of the landscape's curvature from gradient differences alone, converging superlinearly without ever computing a Hessian explicitly.
* Conjugate gradients trade convergence speed for memory: $O(n)$ storage instead of $O(n^2)$, making million-dimensional problems tractable.
* Metaheuristics (simulated annealing, particle swarms, genetic algorithms) are not algorithms — they are frameworks for inventing problem-specific heuristics that can escape local minima at the cost of convergence guarantees.

## What Comes Next

Optimization brings us to the boundary between deterministic and stochastic computation. All the methods here — gradient descent, BFGS, conjugate gradients — are designed to find minima of scalar functions. But behind many of those functions lurks a matrix, and the matrix's eigenvalues govern everything: the shape of the bowl, the stiffness of the descent, the speed of convergence.

Eigenvalue problems are the next stop. They answer a different question: not "what input minimizes this output?" but "what directions does this transformation preserve?" The answer controls natural frequencies in structures, energy levels in quantum systems, and the stability of every dynamical system we will encounter in the rest of the course.

## Check Your Understanding

1. Why does steepest descent zigzag in an elongated valley, and how does BFGS fix this? Answer in terms of what information each method uses at each step.
2. Simulated annealing occasionally accepts a step that *worsens* the objective. Why is this feature, not a bug, when searching a landscape with many local minima?
3. Conjugate gradient directions satisfy $d_i^T H d_j = 0$ for $i \neq j$. Why does this property guarantee that progress made in one direction is not destroyed by subsequent steps?

## Challenge

Minimize the 2D Rosenbrock function $f(x, y) = (1-x)^2 + 100(y - x^2)^2$ using (a) gradient descent with a fixed learning rate, (b) BFGS via `scipy.optimize.minimize`, and (c) simulated annealing with a hand-rolled implementation. For each method, plot the trajectory of iterates overlaid on a contour plot of $f$. Record the number of function evaluations and the final error. Then increase the dimension to 10 and repeat with gradient descent and BFGS only. At what dimension does the memory cost of BFGS become problematic, and what would you switch to?
