# Nonlinear Systems

> *Everything we learned about Newton's method in 1D carries over — but now the derivative is a Jacobian matrix, and we solve a linear system at every step. Remember [linear equations](./linear-equations)? You'll need them here.*

## Big Ideas

* Multidimensional Newton's method is just 1D Newton with division replaced by solving a linear system — the structure is identical, only the cost changes.
* Computing the full Jacobian costs $O(n^2)$ function evaluations via finite differences; for large $n$, this alone is prohibitive, which is why quasi-Newton methods matter.
* Broyden's update is the principle of minimum change: update the Jacobian approximation only along the direction where you just learned something, and leave everything else alone.
* The secant condition gives $n$ equations for $n^2$ unknowns — every quasi-Newton method encodes an assumption about what to do with the remaining freedom.

## Fixed Point Iteration (Recap)

From the previous lesson: transform $f(x)=0$ to $g(x)=x$ and iterate $x_{k+1} = g(x_k)$. If $|g'(x^*)| < 1$, linear convergence. If $g'(x^*) = 0$, quadratic. This generalizes directly to higher dimensions, replacing $|g'|$ with $\|g'\|$ (the Jacobian norm).

## Newton's Method

Take a point, take the tangent, your new point is where the tangent hits zero. In 1D:

$$f'(x_k) \Delta x_{k+1} = -f(x_k)$$

*At each step, pretend the world is linear, solve the linear problem, take the step.*

$$ g(x) = x - \frac{f(x)}{f'(x)} \implies g'(x^*) = 0 \text{ when } f(x^*)=0$$

## Quasi-Newton/Secant Methods

These replicate Newton's properties without evaluating the derivative. In higher dimensions, you don't want to compute the Jacobian — it's a massive matrix.

$$f'(x) \approx \frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}$$

*Instead of knowing the exact slope, estimate it from the last two points. You lose a bit of convergence speed but save a huge amount of computation.*

## Interactive Visualizations

### Newton's Method in 1D: Cobweb Diagram

Explore Newton's method interactively. Select functions, adjust `x_0`, and step through iterations:

[[simulation newton-1d]]

### 2D Nonlinear: Contour Descent and Basins of Attraction

Compare Gradient Descent and Newton's method on the Himmelblau function (four minima). Toggle basins to see which starting points lead where:

[[simulation himmelblau-2d]]

## Going to Higher Dimensions

Imagine you're a blind mountain climber feeling the slope under your boots. In one dimension, you only feel "uphill" or "downhill." In $n$ dimensions, the slope is a vector (the gradient), and the Jacobian matrix collects all the directional sensitivities into one object.

n-dimensional Newton:
$$F'(x_k) \Delta x_{k+1} = - F(x_k)$$

*Each Newton step in n dimensions requires solving a [linear system](./linear-equations). The Jacobian is the coefficient matrix, $-F(x_k)$ is the right-hand side.*

How expensive is computing the Jacobian? For $n$ unknowns, the Jacobian has $n^2$ entries. Using finite differences, that's $n$ extra function evaluations per step. For $n = 1000$, that's 1000 extra evaluations — which is why quasi-Newton methods are so valuable.

## Broyden's Secant Method

The secant equation: $\hat{F}'_{k+1} \Delta x_k = \Delta F_k$

We know $\Delta x_k$ and $\Delta F_k$ from step $k$. Problem: $n^2$ unknowns but only $n$ equations. What do we do?

Here's Broyden's beautiful idea — put the update picture right next to the secant equation so you can see the "why":

Build $B_{k+1}$ that:
1. **Absorbs** $\Delta x_k$ and **produces** $\Delta y_k$:
   $$\left( \Delta y_k \frac{\Delta x_k^T}{\Vert\Delta x_k \Vert^2}\right)\Delta x = \Delta y_k$$

2. **Acts on everything else the same as before** (Broyden's choice — minimal change):
   $$ B_{k+1} = B_k + \frac{(\Delta y_k - B_k \Delta x_k)\Delta x_k^T}{\Delta x_k^T \Delta x_k}$$

*Update the Jacobian approximation along the direction you just stepped in (because you have new info there), and leave it unchanged in every other direction (because you don't know anything new there). Minimal change, maximum information.*

### Working with the inverse instead

$$ B_{k+1}\Delta x_k = \Delta y_k \implies \Delta x_k = B^{-1}_{k+1}\Delta y_k$$

Build $\tilde{B}_{k+1}$ (the inverse effect) by swapping the roles of $\Delta x$ and $\Delta y$.

Why start with $B_0 = I$? Because it works! The identity gives you a gradient-descent-like first step. After a few iterations, Broyden's update learns enough about the curvature that it starts looking like the real Jacobian. It's like learning to drive — you start cautiously and get bolder as you learn the road.

## n-D to 1-D: Constrained Equations on a Curve

If you have $F: \mathbb{R}^m\to\mathbb{R}^n$ and $\gamma: \mathbb{R}\to\mathbb{R}^m$ (a parameterized curve), compose them: $F \circ \gamma: \mathbb{R}\to\mathbb{R}^n$.

For a straight line $\gamma(t) = x_0 + td$:
$$F_\gamma(t) = F(\gamma(t))^T d$$

> **Challenge.** Implement 2D Newton's method: solve $F(x,y) = (x^2 + y^2 - 1, \; x - y) = 0$ (intersection of circle and line). Start from $(2, 2)$ and watch it converge in ~5 steps. Print the Jacobian at each step to see how it guides you.

---

## What Comes Next

Root-finding and optimization share the same engine: linearize, solve the linear problem, take the step. The difference is whether you're chasing a zero of $F(x)$ or a zero of $\nabla f(x)$. Optimization adds line searches to guarantee each step actually improves things — and BFGS is exactly Broyden's method adapted for that setting.

## Check Your Understanding

1. Newton's method in $n$ dimensions solves $F'(x_k)\Delta x = -F(x_k)$ at every step. Why solve instead of computing $[F'(x_k)]^{-1}$?
2. Broyden's update satisfies the secant equation but changes $B_k$ minimally. In what sense is "minimally"?
3. Starting with $B_0 = I$, the first Broyden step looks like gradient descent. Why?

## Challenge

Solve $F(x, y) = (x^2 - y - 1, \; x - y^3 + 1) = 0$ using both full Newton (exact Jacobian each step) and Broyden's method ($B_0 = I$). Record the error $\|F(x_k)\|$ at each iteration and plot convergence on a log scale. Compare the number of *function evaluations* (not just iterations) and explain which method is more efficient.
