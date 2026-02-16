# Nonlinear Equations and Optimization



# Header 1
## Notation:

* $f(x) = 0$: 1 dimensional stuff. $f:\mathbb{R} \to \mathbb{R}$
* $F(x) = 0$: Higher dimensional stuff, $F$ is a matrix an $x$ a vector. $F:\mathbb{R}^n\to\mathbb{R}^m$

Complex numbers are complicated, so we'll only work with reals.


## Introduction

Emergent macroscopic behaviour comes out of high dimensionality of linear systems. For example, you don't figure out the aerodynamics of a plane by using the schrodinger equation for every atom. You can make a simpler theory based on the emergent behaviour, which could be non-linear despite the underlying rules being linear. You could also just make up a non-linear problem, like in economics.

**Linear Systems:** 😊
* We know exactly how many solutions exist (by looking at the matrix's rank)
* We have methods to find exact solutions (if they exist) or approximate solutions (if they don't exist).
* We can find the full solution space of the problem by adding the kernel
* We can routinely solve for billions of dimensions: it's very efficient

**Non-linear systems:** 💀
* No idea how many (if any) solutions. All we can hope for is rules of thumb, heuristics, and we can look for something that works as much as possible and fails rarely. We won't get great results in a finite number of steps, sometimes it gets closer and sometimes it doesn't.
* No fail-proof solvers
* No way of knowing if we've found all the solutions
* Even 1 dimensional solutions can take ages

## How many solutions?

Globally, *anything is possible*. $e^{-x}$ has no solutions, but $(e^{-x}-\delta)$ (where $\delta$ is a small number) does. $\sin(x)$ has countably infinite solutions, while $\text{erf}(x) -\frac12$ has uncountably infinite solutions.

Locally, we can sometimes work it out. In 1D, we can look at where $f(x)$ changes sign, and we can assume that there's a root in between (Intermediate Value Theorem, assuming the function is continuous). In particular, there are $>1$ roots in the region, and there are an odd number of roots.

## General algorithm construction scheme

In general, you want to find an algorithm which does the following:
1. Find invariant that guarantees existence of a solution in the search space
2. Design operation that preserves 1. and shrinks search space

It's a tried and true scheme for coming up with algorithms, Euclid did it thousands of years ago, and we'll do it now. Let's use this to build an equation solver on a bracket. (A *bracket* is an interval in which $f(x)$ changes sign).

## Bisection method
1. Our invariant: $a<b$ and $\text{sign}(f(a)) \neq \text{sign}(f(b))$
2. Set $m = \frac{a+b}{2}$ and evaluate $S_m = \text{sign}(f(m))$
3. If $S_m == S_a$: $a=m$
4. If $S_m == S_b$: $b=m$
5. If $S_m == 0$: We've found the root

Note: you should not use $m = \frac{a+b}{2}$ because of floating point error: use  $m = a + \frac{b-a}{2}$

```python
def bisection(f, a, b, tolerance):
    n_steps = np.log2((b - a) / tolerance)
    S_a, S_b = sign(f(a)), sign(f(b))
    for i in range(nsteps):
        m = a + (b - a) / 2
        s_m = sign(m)
        if S_m == S_a:
            a = m
        else:
            b = m
    return m
```

## Conditionings

The conditioning for evaluating $f(x)$ is approximately $\vert \frac{x f'(x)}{f(x)} \vert$ (taylor expansion) and the absolute error is $\vert f'(x)\vert$ When evaluating $f^{-1}$, the conditional number is $\approx \vert  \frac{f(x)}{x f'(x)} \vert$ aand the absolute error is $\vert\frac{1}{f'(x)} \vert$

As you get a high sensitivity in the inverse, you get a low sensitivity in the inversion and vice-versa. The function doesn't need to have an inverse in order to find the inverse in a local region.

We're trying to look for $f(x)=0$: when we're close to zero, we never use the relative accuracy but always the absolute one.

## Convergence

$e_{k}$ (the error at the $k^{\text{th}}$ step) $= x_k - x^*$

$$E^k_{rel} = \frac{e_k}{x^*}  = \frac{x_k - x^*}{x^*}$$

We need to look at the number of significant bits, because it's exact unlike significant decimal digits.

	""")
	st.latex(r"""
\begin{align*}
\text{bits/step} &= -\log_2(E^{k+1}_{rel}) - \left(-\log_2(E^k_{rel}) \right)   \\
& = \log_2\left(\frac{\frac{x_k - x^*}{x^*}}{\frac{x_{k+1} - x^*}{x^*}} \right) \\
& = \log_2\left(\frac{\vert x_k - x^*\vert}{\vert x_{k+1} - x^*\vert} \right)   \\
& = \log_2\left(\frac{\vert e_k\vert}{\vert e_{k+1}\vert}\right) 				\\
& = -\log_2\left(\frac{\vert e_{k+1}\vert}{\vert e_k\vert}\right) 
\end{align*}
""")
	st.markdown(r"""
If $\lim_{k\to 0} \frac{\vert e_{k+1}\vert}{\vert e_k\vert^r} = c$, and $0 \leq c \lt 1$, method converges with rate r=1 $\implies$ linear, r=2 $\implies$ quadratic, etc

## Fixed point solvers

A systeamtic approach that works (also in n-dimensions). It uses the fixed point theorem: here we'll use _Banache's theorem_. It doesn't just hold in a vector space, but even in a metric space.

Let $S$ be  closed set $S \subseteq \mathbb{R}^n$
and $g:\mathbb{R}^n \to \mathbb{R}^n$, if there exists $0\leq c \lt 1$ such that
$$\Vert g(x) - g(x') \Vert \leq c\Vert x - x'\Vert$$
for $x, x' \in S$, the we call $g$ a _contraction_ on $S$ and we are guaranteed a solution to $g(x)=x$ on $S$, which is
$$x^k = \lim_{k \to \infty} g^k(x_0)$$
for any $x_0 \in S$

Question: Can we transform "$f(x)=0$" to "$g(x)=x$"? The answer is yes, it's easy, but most choices are terrible.

For example, if you pick $g(x) = x - f(x)$, you usually repel solutions. Look at example 5-8 in the book, it gives 4 different ways of rewriting, some are repulsors and some attracters.

How can we make it attractive? Let's analyze the error (in 1D, because it's easier):

$$\vert\, e_{k+1} \,\vert = \vert\, x_{k+1} - x^* \,\vert $$
$$ = \vert\, x_{k+1} - g(x^*) \,\vert $$
$$ = \vert\, g(x_k) - g(x^*) \,\vert $$

Now we bring in the _Mean value theorem_: $\exists \theta \in [x_{k+1}, x^*]$ for which
$$ = \vert\,g'(\theta) (x_k - x^*)  \,\vert$$
$$ = \vert\,g(\theta)\,\vert \vert\, (x_k - x^*)  \,\vert$$
If we can bound $Sup_\theta \vert g'(\theta) \vert \leq c$

$$ = c  \vert (x_k - x^*)  \vert$$
$$ = c\, e_k $$

By continuity, if $\vert g'(x) = x \lt 1$, then for every $0 \lt \epsilon$, there exists $\delta$ such that $\vert g'(x) \vert \leq c + \epsilon$ when $\vert x-x^*\vert \lt \delta$

Thus, if $\vert g'(x^*) \vert \lt 1$, then $g$ is a contraction around $x^* \to x^*$

If $g'(x^*) =0$, then by taylor expanding, we can see that
$$g(x) = g(x^*) + 0 + \underbrace{g''(\theta)} (x - x^*)^2$$
by MVT again, $\exists\, \theta \in [x, x^*]$
$$ \implies \vert e_{k+1} \vert = \vert g(x_k) - g(x^*) \vert $$
$$ = \vert g''(\theta) \vert \vert x_k-x^*\vert^2 $$
$$ \lt \vert g''(x) \vert  \vert e_k \vert^2 $$

If we set $g$ in this way, we have quadratic convergence: the number of bits you gained is squared every timme. The closer you get, the quicker you converge.

## Newton's Method

Set $0 = f(x_k) + \Delta x_k f(x_k) + \mathcal{O}(\Delta x ^2)$
$$ \Delta x = \frac{-f(x_k)}{f'(x_k)} $$
$$ \implies x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)} $$
$\left(g(x) = x -\frac{-f(x_k)}{f'(x_k)}\right)$
$$g'(x) = 1 - \frac{f'(x)f'(x)}{f'(x)^2} + \frac{f(x) f''(x)}{f'(x)^2} $$
$$ = \frac{f(x) f''(x)}{f'(x)^2} $$

Which is Newton's method: you have to start close enough, but once you do it converges rapidly

# Header 2
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
Be sceptical because it's fucking wrong, this guy's just btec Sneppen

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
	


# Header 3
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