# Nonlinear Systems
## Nonlinear solvers


## Fixed point iteration equation solvers (Recap)

We can transform $f(x)=0$ to $g(x)=x$ in multiple ways, but we need to pick one that doesn't blow up in our face. We can converge to our solution with $x_{k+1} = g(x_k)$. Solutions are $x^* = g(x^*)$

This doesn't require anything to be one dimensional, it holds in higher dimensional space as well.

First, let's suppose $|g'(x)| \leq c \lt 1$ on a ball $k_\delta$ of radius $\delta$.

$$ \Vert e_{x_k + 1}\Vert = \Vert x_{k+1}-x^* \Vert$$
$$ = \Vert x_{k+1}-g(x^*) \Vert$$
$$ = \Vert g(x_k)-g(x^*) \Vert$$

Because of the _mean value theorem_, there exists an $x' \in [x_k, x^*]$

$$ = \Vert g'(x')(x_k - x^*) \Vert$$
$$ \leq \Vert g'(x')\Vert\,\Vert(x_k - x^*) \Vert $$

If $x_k \in k_\delta:$

$$ \leq c \,\Vert(x_k - x^*) \Vert $$
$$ \leq c^k \,\Vert(x_0 - x^*) \Vert $$

We can build the ball $k_\delta$ from continuity of $g'$:

**Continuity**: A function $f: X\to Y$ is continuous if for any $\epsilon>0 \quad \exists \quad \delta >0$ such that
$$\Vert x - x' \Vert \lt \delta \implies \Vert f(x) - f(x') \Vert < \epsilon$$

Assume $|g'(x^*)| <1$, then
* Pick $0 < \epsilon<1-|g'(x^*)|$ and set $c = |g'(x^*)| + \epsilon$
* Then by continuity of $g'$ at $x^*$, there is a $\delta>0$ such that $\Vert x - x' \Vert \lt \delta \implies \Vert f(x) - f(x') \Vert < \epsilon$
* $\Vert g'(x) \Vert - \Vert g'(x^*)\Vert\lt\Vert g'(x) - g'(x^*)\Vert$ so $\Vert g'(x) \Vert \leq \Vert g'(x^*)\Vert + c$, where $c<1 \forall x\in k_\delta$

## Constructing quadratic convergence:

If $g'(x^*) = 0$ then the second term in the taylor expanion is 0

$$ g(x) = g(x^*)+g'(x^*)(x-x^*)+\frac{g''(x^*)(x-x^*)^2}{2} $$

By continuity, we can find a ball $k_\delta$ for which $\Vert g''(x)\Vert <\bar{c}$, where $\bar{c}=\Vert g''(x^*)\Vert + \epsilon$.

$$ \Vert e_{k+1} \Vert = \Vert g(x_k)-g(x^*)\Vert $$
$$ \leq \bar{c}^2\Vert x_k - x^* \Vert^2  $$
$$ \leq \bar{c}^k\Vert x_0 - x^* \Vert^{2^k}  $$

**Requirements for g:**

* $|g'(x^*)|<1$ at all solutiond $x^*$
* if $0<|g'(x^*)|<1$, we get linear convergence: $\vert e_{k+1} \vert\leq c^k | e_0|$
* if $0=|g'(x^*)|$, we get quadratic convergence: $\vert e_{k+1} \vert\leq \bar{c}^k | e_0|^{2^k}$


## Newton's method

A quadratically convergent fixed point iteration solver. Take a point, take the tangent of the curve at that point, your new point is the tangent's x-intercept. Intuition: find zero for linear approximation, set as next $x$.

Talor expand:
$$ f(x_{k+1}) = f(x_k) + f'(x_k) (x_{k+1}-x_k) + \mathcal{O}(\vert x_{k+1}-x_k\vert^2)$$
$$ \approx f(x_k) + f'(x_k) (x_{k+1}-x_k)$$
$$ 0 \approx f(x_k) + f'(x_k) \Delta x_{k+1}$$
$$ f'(x_k) \Delta x_{k+1} = f(x_k)$$
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

$$f'(x) \\approx \\frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}$$
In 1D, we have the following equation for $\\Delta x$
$$\\Delta x_{k+1} = -\\frac{f(x_k)}{\\hat{f}'(x_k)}$$
where
$$\\hat{f}'(x_k) = \\frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}$$
For higher dimensions, we want to write it like
$$ \\hat{f}'(x_k) \\Delta x_{k+1} = - f(x_k)$$
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
- ≈ (3.0, 2.0)
- ≈ (-2.8, 3.1)
- ≈ (-3.8, -3.3)
- ≈ (3.6, -1.8)

Compare Gradient Descent (GD) and Newton's method. Toggle basins to see attraction regions from initial points grid. Newton's method has smaller, more precise basins due to curvature info.

```tsx
import Himmelblau2D from '@/components/visualization/nonlinear-equations/Himmelblau2D';
&lt;Himmelblau2D /&gt;
```

<Himmelblau2D />

**Research Insights & Gaps Filled:**
- **Basin Attractors**: Visualizes fractal-like boundaries in practice, sensitive to method.
- **Contour Descent**: Paths show GD zigzagging (Rosenbrock-like valley), Newton direct.
- **Interactivity**: Sliders for params/init, animation reveals dynamics.
- **Gaps**: Few web interactives compare Newton/secant(quasi) in 2D basins; this adds Newton vs GD proxy for secant ideas.

For secant methods like Broyden/BFGS, paths approximate Newton without exact derivs/Hessians.


These are methods that try to replicate the wonderful properties of Newton's method, but without having to evaluate the derivative. In higher dimensions, you don't want to be evaluating the derivative cause it's a massive matrix.

**Method**: Use secant (finite difference) instead of tangent (derivative).

$$f'(x) \approx \frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}$$
In 1D, we have the following equation for $\Delta x$
$$\Delta x_{k+1} = -\frac{f(x_k)}{\hat{f}'(x_k)}$$
where
$$\hat{f}'(x_k) = \frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}$$
For higher dimensions, we want to write it like
$$ \hat{f}'(x_k) \Delta x_{k+1} = - f(x_k)$$
Cause we'll solve a linear system instead

## Going to higher dimensions:

Notation: $f: \mathbb{R}\to\mathbb{R}$, $F: \mathbb{R}^n\to\mathbb{R}^n$

1D taylor expansion:
$$f(x_{k+1}) = f(x_k + \Delta x_{k+1})$$
$$f(x_{k+1}) = f(x_k) + f'(x_k)\Delta x_{k+1} + \mathcal{O}(|\Delta x_{k+1}|^2)$$

n-dimensional case:
$$F(x_{k+1}) = F(x_k + \Delta x_{k+1})$$
$$F(x_{k+1}) = F(x_k) + F'(x_k)\Delta x_{k+1} + \mathcal{O}(|\Delta x_{k+1}|^2)$$

To go to the next step, we don't divide by $f'(x)$ like we do in the 1D case, but instead we solve the linear system
$$F'(x_k) \Delta x_{k+1} = - F(x)$$
where we know the first and last terms and want to find $\Delta x_{k+1}$

In the secant method, we just use $\hat{F}'(x)$ instead. This raises an extra problem: We need to find a good $\hat{F}'$

## Broyden's Secant Method

Secant equation:
$$\hat{F}'_{k+1} \Delta x_k = \Delta F_k$$

We know $\Delta x_k$ and $\Delta F'_k$ from step k.

The problem: we have $n^2$ unknowns and 1 equation. What do we do?

We need to determine some $B_{k+1}$ that satisfies the secant equation.
1. "Absorb" $\Delta x_k$, and "Produce" $\Delta y_k$.
   $$\left( \Delta y_k \frac{\Delta x_k^T}{\Vert\Delta x_k \Vert^2}\right)\Delta x = \Delta y_k $$
   The left part of the big bracket produces $\Delta y_k$, the right part eliminates $\Delta x_k$

2. Decide what $B_{k+1}$ does on the rest of $\mathbb{R}^n$. Anything that satisfies the secant equation will do 1., now we're into the arbitrary stuff. Broyden decides to let it act on the rest of the space in the same way as before (i.e, same as $B_k$). In math,
   $$ B_{k+1} = \Delta y_k \frac{\Delta x_k^T}{\Vert \Delta x_k\Vert^2} + B_k\left( I-P_{\Delta x_k}\right)$$
   $\left( I-P_{\Delta x_k}\right)$ is the rest of the space

### How to compute action on orthogonal component to $\Delta x$

$$P_{\Delta x_k} = \frac{\Delta x_k \Delta x_k^T}{\Vert \Delta x_k \Vert^2} $$
$$ \implies I = P_{\Delta x_k} = I - \frac{\Delta x_k \Delta x_k^T}{\Vert \Delta x_k \Vert^2} $$
In conclusion:
   $$ B_{k+1} = \Delta y_k \frac{\Delta x_k^T}{\Delta x_k^T \Delta x_k} + B_k - B_k\left( \frac{\Delta x_k \Delta x_k^T}{\Vert \Delta x_k \Vert^2} \right)$$
Note: verify the derivation above carefully before using.

It really looks like

$$ B_k + \frac{\Delta y_k \Delta x_k^T}{\Vert \Delta x_k\Vert^2} - B_k \frac{\Delta x_k \Delta x_k^T}{\Vert \Delta x_k\Vert^2} $$

Let's do the inverse instead

$$ B_{k+1}\Delta x_k = \Delta y_k $$
$$ \Delta x_k = B^{-1}_k\Delta y_k$$

There's nothing wrong with calculating the inverse here, we're using the same amount of information.

Let's build $\tilde{B}_{k+1}$ (which does the effect of the inverse, I think?) by swapping the roles of $\Delta x$ and $\Delta y$.

What do we start out with? We could start out with an approximation to the Jacobian, but there's no reason to: if we start out with the identity, we'll get a reasonable step that'll converge slowly, and as we gain information it'll get better and better.

## n-D to 1-D: Constrained equations on a curve

If you have a function $F: \mathbb{R}^m\to\mathbb{R}^n$ and $\gamma: \mathbb{R}\to\mathbb{R}^m$. ($\gamma$ is a parameterized curve in the highere dimensional space).

Compose $F \circ \gamma: \mathbb{R}\to\mathbb{R}^n, \qquad F(\gamma(t))\in\mathbb{R}^n$.

Directional value (component along $\gamma(t)$):
$$ F(\gamma(t))^T \gamma'(t) $$
i.e, dot product

**Special case**: straight line:
$$ \gamma(t) = x_0 + td$$
where $d$ is a vector with the direction of the line.
$$\gamma(t)' =d$$
$$\implies F_\gamma (t) = F(\gamma(t))^T d$$


