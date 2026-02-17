# Optimization Methods
## Non-linear Optimization

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
   2. High dimensions: Simple energy landscape, use conjugate gradients. When it's expensive to evaluate the function, and you're in a high-dimensional complex landscape, the search space is too big for you to get anywhere. Here you need to think, and tailor yourr solution to fit your problem. Generally, you can try to use some sort of symmetry or structrue of your problem and then use a metaheuristic to guide your solutions

## Metaheuristics

_Algorithm:_ Computatoinal metho with guaranteed correct result after finite steps.

_Heuristic:_ The same as an algorithm, but with no guarantees.

An algorithm has 2 guarantees: it gives you a "correct" result (as per your definition of correct, we were using tolerance to decide that), and it happens after a finite number of steps. A heuristic doesn't guarantee either.

_Metaheuristic:_ A scheme for building heuristics. It's a framework where you have some overall structure (from your problem), then you take your metaheuristic scheme, tailor it to your problem, and then you produce a heuristic that you can run to get good answers for your problem

## Simulated Annealing:

Inspired by physical processes: annealing is when you let something cool slowly in order to get the correct hardened structure (in glass, metal, etc). 'Hardening' means that you find a position of atoms that minimizes the energy.

For minimizing $f:\mathbb{R}^n\to\mathbb{R}$ (maps from the high dimensional space to a scalar energy value):

* Maintain one state vector $x\in\mathbb{R}^n$
* Start with a high temperature $T_0$ ("high" is problem dependent).
* Gradually cool it down to 0 K

In each step:
1. Perturn $x$ by a random motion $\Delta x$
2. Let $\Delta E=(f(x+\Delta x)-f(x))\gamma$, where $\gamma$ is an optional energy unit.
3. Define a _transition probability_: $P(T, \Delta E) = e^{\Delta E/k_BT}$
4. Call a random number generator to get an $r\in[0,1]$ and accept new step if $P(T, \Delta E)\geq r$
5. Cool temperature: $T_{k+1} = \alpha T_k$

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
F = fs[:, np.newaxis] - fs[np.newaxis, :]
```
Similarly for 3.
```python
Phi =(1-np.sign(F))/2
```
And for 4.
```python
X += np.sum(alpha[:,:,np.newaxis]*phi[:,:,np.newaxis]*d, axis=1)
```
We use newaxis in order to math $\alpha$ and $\phi$ with $d$, which is a rank 3 tensor.

Note that $\delta$, the free will step, can also be tweaked. The simplest thing to do is to use brownian motion: you get something that explores a local area really well, but doesn't go very far out. If you use the Levy distribution instead of the gaussian, you get a combination of local exploration with (sometimes) big jumps to a new place.

## Genetic Algorithms

### How to start an evolution:

1. **Representation:** A genetic code
2. **Mating:** Processes for splitting and recombining genomes
3. **Selection Pressure:** Who, and whose offspring, make up the next generation?
4. **Mutation:** Radom perturbations (to get somewhere new)

Ideas $f:\mathbb{R}^{3N}\to\mathbb{R}$ representing $N$ particle position in space $\mathbb{R}^3$
