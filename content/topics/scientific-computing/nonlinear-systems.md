# Nonlinear Systems

> *Everything we learned about Newton's method in 1D carries over — but now the derivative is a Jacobian matrix, and we solve a linear system at every step. Remember [linear equations](./linear-equations)? You'll need them here.*

## Nonlinear solvers

## Fixed point iteration (Recap)

Recall from the previous page: we can transform $f(x)=0$ to $g(x)=x$ and iterate $x_{k+1} = g(x_k)$. The key results were:

* If $|g'(x^*)| < 1$, $g$ is a contraction near $x^*$ and we get **linear convergence**: $|e_{k+1}| \leq c^k |e_0|$
* If $g'(x^*) = 0$, we get **quadratic convergence**: $|e_{k+1}| \leq \bar{c}^k |e_0|^{2^k}$

This all generalizes directly to higher dimensions, replacing $|g'|$ with $\Vert g' \Vert$ (the Jacobian norm).


## Newton's method

A quadratically convergent fixed point iteration solver. Take a point, take the tangent of the curve at that point, your new point is the tangent's x-intercept. Intuition: find zero for linear approximation, set as next $x$.

Taylor expand:
$$ f(x_{k+1}) = f(x_k) + f'(x_k) (x_{k+1}-x_k) + \mathcal{O}(\vert x_{k+1}-x_k\vert^2)$$
$$ \approx f(x_k) + f'(x_k) (x_{k+1}-x_k)$$
$$ 0 \approx f(x_k) + f'(x_k) \Delta x_{k+1}$$
$$ f'(x_k) \Delta x_{k+1} = f(x_k)$$

*This says: at each step, pretend the world is linear (use the tangent), solve the linear problem to find the step, then take that step.*

Remember the above line, cause we'll use it in higher dimensions as it's solving linear systems
$$ \Delta x_k = -\frac{f(x_k)}{f'(x_k)}  $$
This works in 1d, but is not general because we're dividing by a matrix (the jacobian) and not a number
$$ \implies g(x_k) = x_k + \Delta x_k = x_k - \frac{f(x_k)}{f'(x_k)}$$

if $f(x^*) = 0$,
$$g(x^*) = x^* - \frac{f(x)}{f'(x)} = x^*$$
Also,
$$ g'(x) = \frac{d}{dx}\left(x - \frac{f(x)}{f'(x)}\right)$$
$$ 1 - \frac{f'(x)f'(x) + f(x)f''(x)}{f'(x)^2} $$
$$  = - \frac{f(x)f''(x)}{f'(x)^2}$$
$$ \to 0 \quad \text{when}\quad f(x)=0$$

## Quasi-Newton/Secant Methods

These are methods that try to replicate the wonderful properties of Newton's method, but without having to evaluate the derivative. In higher dimensions, you don't want to be evaluating the derivative cause it's a massive matrix.

**Method**: Use secant (finite difference) instead of tangent (derivative).

$$f'(x) \approx \frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}$$

*This says: instead of knowing the exact slope, estimate it from the last two points. You lose a bit of convergence speed but save a huge amount of computation.*

In 1D, we have the following equation for $\Delta x$:
$$\Delta x_{k+1} = -\frac{f(x_k)}{\hat{f}'(x_k)}$$
where
$$\hat{f}'(x_k) = \frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}$$
For higher dimensions, we want to write it like
$$ \hat{f}'(x_k) \Delta x_{k+1} = - f(x_k)$$
Cause we'll solve a linear system instead

## Interactive Visualizations

### Newton's Method in 1D: Cobweb Diagram

Explore Newton's method interactively. Select functions with known roots, adjust initial guess `x₀`, and step through iterations. The cobweb diagram visualizes the fixed-point iteration `x = g(x)` where `g(x) = x - f(x)/f'(x)`.

```tsx
import Newton1D from '@/components/visualization/nonlinear-equations/Newton1D';
&lt;Newton1D /&gt;
```

<Newton1D />

Observe quadratic convergence near the root and potential overshoots or divergences.

### 2D Nonlinear Optimization: Contour Descent and Basins of Attraction

The Himmelblau function has four minima:
* ≈ (3.0, 2.0)
* ≈ (-2.8, 3.1)
* ≈ (-3.8, -3.3)
* ≈ (3.6, -1.8)

Compare Gradient Descent (GD) and Newton's method. Toggle basins to see attraction regions from initial points grid. Newton's method has smaller, more precise basins due to curvature info.

```tsx
import Himmelblau2D from '@/components/visualization/nonlinear-equations/Himmelblau2D';
&lt;Himmelblau2D /&gt;
```

<Himmelblau2D />

**Research Insights & Gaps Filled:**
* **Basin Attractors**: Visualizes fractal-like boundaries in practice, sensitive to method.
* **Contour Descent**: Paths show GD zigzagging (Rosenbrock-like valley), Newton direct.
* **Interactivity**: Sliders for params/init, animation reveals dynamics.
* **Gaps**: Few web interactives compare Newton/secant(quasi) in 2D basins; this adds Newton vs GD proxy for secant ideas.

For secant methods like Broyden/BFGS, paths approximate Newton without exact derivs/Hessians.

## Going to higher dimensions

Imagine you're a blind mountain climber feeling the slope under your boots. In one dimension, you only feel "uphill" or "downhill." But in $n$ dimensions, the slope has a direction — it's a vector (the gradient), and the rate of change along any particular direction is the directional derivative. The Jacobian matrix collects all these directional sensitivities into one object.

Notation: $f: \mathbb{R}\to\mathbb{R}$, $F: \mathbb{R}^n\to\mathbb{R}^n$

1D taylor expansion:
$$f(x_{k+1}) = f(x_k + \Delta x_{k+1})$$
$$f(x_{k+1}) = f(x_k) + f'(x_k)\Delta x_{k+1} + \mathcal{O}(|\Delta x_{k+1}|^2)$$

n-dimensional case:
$$F(x_{k+1}) = F(x_k + \Delta x_{k+1})$$
$$F(x_{k+1}) = F(x_k) + F'(x_k)\Delta x_{k+1} + \mathcal{O}(|\Delta x_{k+1}|^2)$$

*Here $F'(x_k)$ is the Jacobian — an $n \times n$ matrix of all partial derivatives. Each row tells you how one output component changes with all the inputs.*

To go to the next step, we don't divide by $f'(x)$ like we do in the 1D case, but instead we solve the linear system
$$F'(x_k) \Delta x_{k+1} = - F(x)$$
where we know the first and last terms and want to find $\Delta x_{k+1}$

*This is the key connection: each Newton step in n dimensions requires solving a [linear system](./linear-equations). The Jacobian plays the role of the coefficient matrix, and $-F(x_k)$ is the right-hand side.*

In the secant method, we just use $\hat{F}'(x)$ instead. This raises an extra problem: We need to find a good $\hat{F}'$

How expensive is computing the Jacobian? For $n$ unknowns, the Jacobian has $n^2$ entries. If each partial derivative requires a function evaluation, that's $n$ extra evaluations per step (using finite differences). For $n = 1000$, that's 1000 extra function evaluations — which is why quasi-Newton methods that avoid this cost are so valuable.

## Broyden's Secant Method

Secant equation:
$$\hat{F}'_{k+1} \Delta x_k = \Delta F_k$$

We know $\Delta x_k$ and $\Delta F_k$ from step $k$.

The problem: we have $n^2$ unknowns (entries of $\hat{F}'_{k+1}$) but only $n$ equations (the secant equation). What do we do?

We need to determine some $B_{k+1}$ that satisfies the secant equation $B_{k+1} \Delta x_k = \Delta y_k$, where $\Delta y_k = \Delta F_k$.
1. "Absorb" $\Delta x_k$, and "Produce" $\Delta y_k$.
   $$\left( \Delta y_k \frac{\Delta x_k^T}{\Vert\Delta x_k \Vert^2}\right)\Delta x = \Delta y_k $$
   The left part of the big bracket produces $\Delta y_k$, the right part eliminates $\Delta x_k$

2. Decide what $B_{k+1}$ does on the rest of $\mathbb{R}^n$. Anything that satisfies the secant equation will do 1., now we're into the arbitrary stuff. Broyden decides to let it act on the rest of the space in the same way as before (i.e, same as $B_k$). In math,
   $$ B_{k+1} = \Delta y_k \frac{\Delta x_k^T}{\Vert \Delta x_k\Vert^2} + B_k\left( I-P_{\Delta x_k}\right)$$

   *This says: update the Jacobian approximation along the direction we just stepped in (because we have new info there), and leave it unchanged in every other direction (because we don't know anything new there). Minimal change, maximum information.*

   $\left( I-P_{\Delta x_k}\right)$ is the rest of the space

### How to compute action on orthogonal component to $\Delta x$

$$P_{\Delta x_k} = \frac{\Delta x_k \Delta x_k^T}{\Vert \Delta x_k \Vert^2} $$
$$ \implies I = P_{\Delta x_k} = I - \frac{\Delta x_k \Delta x_k^T}{\Vert \Delta x_k \Vert^2} $$
In conclusion:
   $$ B_{k+1} = \Delta y_k \frac{\Delta x_k^T}{\Delta x_k^T \Delta x_k} + B_k - B_k\left( \frac{\Delta x_k \Delta x_k^T}{\Vert \Delta x_k \Vert^2} \right)$$
Simplifying, this becomes:

$$ B_k + \frac{\Delta y_k \Delta x_k^T}{\Vert \Delta x_k\Vert^2} - B_k \frac{\Delta x_k \Delta x_k^T}{\Vert \Delta x_k\Vert^2} $$

Let's do the inverse instead

$$ B_{k+1}\Delta x_k = \Delta y_k $$
$$ \Delta x_k = B^{-1}_k\Delta y_k$$

There's nothing wrong with calculating the inverse here, we're using the same amount of information.

Let's build $\tilde{B}_{k+1}$ (which does the effect of the inverse, I think?) by swapping the roles of $\Delta x$ and $\Delta y$.

What do we start out with? We could start out with an approximation to the Jacobian, but there's no reason to: if we start out with the identity, we'll get a reasonable step that'll converge slowly, and as we gain information it'll get better and better.

Why start with the identity instead of a better initial guess? Because it works! The identity gives you a gradient-descent-like first step. After a few iterations, Broyden's update will have learned enough about the curvature that it starts to look like the real Jacobian. It's like learning to drive — you start cautiously and get bolder as you learn the road.

## n-D to 1-D: Constrained equations on a curve

If you have a function $F: \mathbb{R}^m\to\mathbb{R}^n$ and $\gamma: \mathbb{R}\to\mathbb{R}^m$. ($\gamma$ is a parameterized curve in the higher dimensional space).

Compose $F \circ \gamma: \mathbb{R}\to\mathbb{R}^n, \qquad F(\gamma(t))\in\mathbb{R}^n$.

Directional value (component along $\gamma(t)$):
$$ F(\gamma(t))^T \gamma'(t) $$
i.e, dot product

**Special case**: straight line:
$$ \gamma(t) = x_0 + td$$
where $d$ is a vector with the direction of the line.
$$\gamma(t)' =d$$
$$\implies F_\gamma (t) = F(\gamma(t))^T d$$

> **Challenge.** Implement 2D Newton's method: solve $F(x,y) = (x^2 + y^2 - 1, \; x - y) = 0$ (intersection of circle and line). Start from $(2, 2)$ and watch it converge in ~5 steps. Print the Jacobian at each step to see how it guides you.

---

## Big Ideas

* Multidimensional Newton's method is just one-dimensional Newton's method with division replaced by solving a linear system — the structure is identical, only the cost changes.
* Computing the full Jacobian costs $O(n^2)$ function evaluations via finite differences; for large $n$, this alone is prohibitive, which is why quasi-Newton methods matter.
* Broyden's update is the principle of minimum change: update the Jacobian approximation only along the direction where you just learned something, and leave everything else alone.
* The secant condition is not enough to determine the new Jacobian uniquely — you have $n$ equations and $n^2$ unknowns — so every quasi-Newton method encodes an assumption about what to do with the remaining freedom.

## What Comes Next

Root-finding and optimization look like different problems, but they share the same engine: at every step, linearize around the current point, solve the linear problem, and take the step. The difference is whether you are chasing a zero of $F(x)$ or a zero of $\nabla f(x)$.

Optimization adds one more ingredient: a *scalar* objective function, which means you can use line searches to guarantee that each step actually improves the solution. The BFGS algorithm is exactly Broyden's method adapted for this setting, and the quasi-Newton approximation of the Hessian plays the same role as Broyden's approximate Jacobian — building curvature knowledge one step at a time.

## Check Your Understanding

1. Newton's method in $n$ dimensions requires solving a linear system $F'(x_k)\Delta x = -F(x_k)$ at every step. Why is it better to *solve* this system rather than explicitly compute $[F'(x_k)]^{-1}$ and multiply?
2. Broyden's update satisfies the secant equation $B_{k+1}\Delta x_k = \Delta F_k$ but changes $B_k$ minimally. In what sense is "minimally"? What is being minimized?
3. If you start Broyden's method with $B_0 = I$ (the identity), the first step is identical to gradient descent. Why, and what does this say about how much information the identity encodes about the problem?

## Challenge

Solve the system of nonlinear equations $F(x, y) = (x^2 - y - 1, \; x - y^3 + 1) = 0$ using both full Newton's method (computing the exact Jacobian at each step) and Broyden's method (starting from $B_0 = I$). Start from three different initial points. For each run, record the error $\|F(x_k)\|$ at each iteration and plot convergence curves on a log scale. Compare the number of *function evaluations* used by each method, not just the number of iterations, and explain which method is more efficient and under what circumstances.


