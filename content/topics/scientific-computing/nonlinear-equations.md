# Nonlinear Equations

> *We just mastered linear systems where everything was guaranteed. Now those guarantees vanish. Welcome to the real world.*

# Root Finding in One Dimension
## Notation:

* $f(x) = 0$: 1 dimensional stuff. $f:\mathbb{R} \to \mathbb{R}$
* $F(x) = 0$: Higher dimensional stuff, $F$ is a matrix an $x$ a vector. $F:\mathbb{R}^n\to\mathbb{R}^m$

Complex numbers are complicated, so we'll only work with reals.


## Introduction

Emergent macroscopic behaviour comes out of high dimensionality of linear systems. For example, you don't figure out the aerodynamics of a plane by using the schrodinger equation for every atom. You can make a simpler theory based on the emergent behaviour, which could be non-linear despite the underlying rules being linear. You could also just make up a non-linear problem, like in economics.

**Linear Systems:**
* We know exactly how many solutions exist (by looking at the matrix's rank)
* We have methods to find exact solutions (if they exist) or approximate solutions (if they don't exist)
* We can find the full solution space of the problem by adding the kernel
* We can routinely solve for billions of dimensions: it's very efficient

**Non-linear systems:**
* No idea how many (if any) solutions. All we can hope for is rules of thumb, heuristics, and we can look for something that works as much as possible and fails rarely. We won't get great results in a finite number of steps; sometimes it gets closer and sometimes it doesn't.
* No fail-proof solvers
* No way of knowing if we've found all the solutions
* Even 1 dimensional solutions can take ages

## How many solutions?

Globally, *anything is possible*. $e^{-x} = 0$ has no solutions, but $(e^{-x}-\delta) = 0$ (where $\delta$ is a small positive number) does. $\sin(x) = 0$ has countably infinite solutions. Even a simple-looking polynomial like $x^5 - x = 0$ has multiple roots that are hard to predict without analysis.

Locally, we can sometimes work it out. In 1D, we can look at where $f(x)$ changes sign, and we can assume that there's a root in between (Intermediate Value Theorem, assuming the function is continuous). In particular, there are $>1$ roots in the region, and there are an odd number of roots.

## General algorithm construction scheme

Here's a powerful recipe for inventing algorithms:
1. Find invariant that guarantees existence of a solution in the search space
2. Design operation that preserves 1. and shrinks search space

*This says: first make sure you're looking in the right place, then systematically make that place smaller. It's a tried and true scheme — Euclid did it thousands of years ago, and we'll do it now.*

Let's use this to build an equation solver on a bracket. (A *bracket* is an interval in which $f(x)$ changes sign).

## Bisection method

**Step 1:** Establish the invariant: $a<b$ and $\text{sign}(f(a)) \neq \text{sign}(f(b))$

*The function crosses zero somewhere between a and b. We're sure of it.*

**Step 2:** Cut the interval in half: set $m = \frac{a+b}{2}$ and evaluate $S_m = \text{sign}(f(m))$

**Step 3:** Keep the half that still brackets the root:
* If $S_m == S_a$: the root is in the right half, so $a=m$
* If $S_m == S_b$: the root is in the left half, so $b=m$
* If $S_m == 0$: We've found the root

Note: you should not use $m = \frac{a+b}{2}$ because of floating point error: use  $m = a + \frac{b-a}{2}$

```python
def bisection(f, a, b, tolerance):
    n_steps = int(np.ceil(np.log2((b - a) / tolerance)))
    S_a = np.sign(f(a))
    for i in range(n_steps):
        m = a + (b - a) / 2      # midpoint without overflow
        S_m = np.sign(f(m))
        if S_m == S_a:
            a = m                 # root is in right half
        else:
            b = m                 # root is in left half
    return m
```

*Bisection gains exactly one bit of accuracy per step. Slow but sure — the tortoise of root-finding.*

## Conditionings

The conditioning for evaluating $f(x)$ is approximately $\left| \frac{x f'(x)}{f(x)} \right|$ (from Taylor expansion), and the absolute error is $|f'(x)|$. When evaluating $f^{-1}$, the condition number is $\approx \left|  \frac{f(x)}{x f'(x)} \right|$ and the absolute error is $\left|\frac{1}{f'(x)} \right|$.

*This says: if a function has a steep slope near the root, the root is easy to find (well-conditioned). If the function barely grazes zero (shallow crossing), the root is hard to pin down.*

As you get a high sensitivity in the inverse, you get a low sensitivity in the inversion and vice-versa. The function doesn't need to have an inverse in order to find the inverse in a local region.

We're trying to look for $f(x)=0$: when we're close to zero, we never use the relative accuracy but always the absolute one.

## Convergence

$e_{k}$ (the error at the $k^{\text{th}}$ step) $= x_k - x^*$

$$E^k_{rel} = \frac{e_k}{x^*}  = \frac{x_k - x^*}{x^*}$$

We need to look at the number of significant bits, because it's exact unlike significant decimal digits.

$$
\begin{align*}
\text{bits/step} &= -\log_2(E^{k+1}_{rel}) - \left(-\log_2(E^k_{rel}) \right)   \\
& = \log_2\left(\frac{\frac{x_k - x^*}{x^*}}{\frac{x_{k+1} - x^*}{x^*}} \right) \\
& = \log_2\left(\frac{\vert x_k - x^*\vert}{\vert x_{k+1} - x^*\vert} \right)   \\
& = \log_2\left(\frac{\vert e_k\vert}{\vert e_{k+1}\vert}\right) 				\\
& = -\log_2\left(\frac{\vert e_{k+1}\vert}{\vert e_k\vert}\right)
\end{align*}
$$

If $\lim_{k\to 0} \frac{\vert e_{k+1}\vert}{\vert e_k\vert^r} = c$, and $0 \leq c \lt 1$, method converges with rate r=1 $\implies$ linear, r=2 $\implies$ quadratic, etc

**How fast is fast?** Think of convergence rates like this:
* **Linear (r=1):** Like earning simple interest — you gain a fixed number of correct digits each step. Bisection does this: one bit per step, steady and reliable.
* **Quadratic (r=2):** Like compound interest on steroids — the number of correct digits *doubles* each step. Start with 3 good digits, then 6, then 12, then 24. Newton's method does this.
* **Cubic (r=3):** Even wilder — correct digits *triple* each step. Rayleigh quotient iteration achieves this for eigenvalue problems.

## Fixed point solvers

Picture a drunkard trying to walk home. He starts somewhere, takes a step according to some rule, and ends up somewhere new. If the rule is a contraction — meaning each step brings him closer to home regardless of where he currently is — then he's guaranteed to eventually stumble through his front door. That's fixed-point iteration.

A systematic approach that works (also in n-dimensions). It uses the fixed point theorem: here we'll use _Banach's fixed point theorem_. It doesn't just hold in a vector space, but even in a metric space.

Let $S$ be  closed set $S \subseteq \mathbb{R}^n$
and $g:\mathbb{R}^n \to \mathbb{R}^n$, if there exists $0\leq c \lt 1$ such that
$$\Vert g(x) - g(x') \Vert \leq c\Vert x - x'\Vert$$
for $x, x' \in S$, the we call $g$ a _contraction_ on $S$ and we are guaranteed a solution to $g(x)=x$ on $S$, which is
$$x^k = \lim_{k \to \infty} g^k(x_0)$$
for any $x_0 \in S$

*This says: if every step of the iteration brings points closer together (by at least a factor c < 1), then all roads lead to Rome. The fixed point exists and is unique.*

Question: Can we transform "$f(x)=0$" to "$g(x)=x$"? The answer is yes, it's easy, but most choices are terrible.

For example, if you pick $g(x) = x - f(x)$, you usually repel solutions. Look at example 5-8 in the book, it gives 4 different ways of rewriting, some are repulsors and some attracters.

How do you know which rewriting will converge? Check the derivative at the fixed point! If $|g'(x^*)| < 1$, it converges. If $|g'(x^*)| > 1$, it diverges. If $|g'(x^*)| = 0$, you've hit the jackpot — quadratic convergence.

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

By continuity, if $|g'(x^*)| < 1$, then for every $0 < \epsilon$, there exists $\delta$ such that $|g'(x)| \leq |g'(x^*)| + \epsilon$ when $|x-x^*| < \delta$

Thus, if $\vert g'(x^*) \vert \lt 1$, then $g$ is a contraction around $x^* \to x^*$

If $g'(x^*) =0$, then by taylor expanding, we can see that
$$g(x) = g(x^*) + 0 + \underbrace{g''(\theta)} (x - x^*)^2$$
by MVT again, $\exists\, \theta \in [x, x^*]$
$$ \implies \vert e_{k+1} \vert = \vert g(x_k) - g(x^*) \vert $$
$$ = \vert g''(\theta) \vert \vert x_k-x^*\vert^2 $$
$$ \lt \vert g''(x) \vert  \vert e_k \vert^2 $$

If we set $g$ in this way, we have quadratic convergence: the number of correct bits roughly doubles every step. The closer you get, the quicker you converge.

## Newton's Method

Here's the trick that makes quadratic convergence happen automatically:

Set $0 = f(x_k) + \Delta x_k f(x_k) + \mathcal{O}(\Delta x ^2)$
$$ \Delta x = \frac{-f(x_k)}{f'(x_k)} $$
$$ \implies x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)} $$

*This says: at each step, replace the curve with its tangent line, and go where the tangent line crosses zero. Repeat.*

$\left(g(x) = x -\frac{-f(x_k)}{f'(x_k)}\right)$
$$g'(x) = 1 - \frac{f'(x)f'(x)}{f'(x)^2} + \frac{f(x) f''(x)}{f'(x)^2} $$
$$ = \frac{f(x) f''(x)}{f'(x)^2} $$

Watch the beautiful part: at the root where $f(x^*) = 0$, we get $g'(x^*) = 0$, which is exactly the condition for quadratic convergence!

Which is Newton's method: you have to start close enough, but once you do it converges rapidly.

> **Challenge.** Implement Newton's method in 3 lines of Python and find $\sqrt{2}$ by solving $f(x) = x^2 - 2 = 0$. Start from $x_0 = 1$. Print the error at each step and verify it roughly squares each time.

---

## Big Ideas

* Nonlinear equations offer none of the guarantees of linear ones — you may have zero solutions, one, many, or infinitely many, and there is no formula that tells you which.
* Bisection is the tortoise: it always converges if a bracket exists, gains exactly one bit per step, and cannot be fooled.
* Newton's method is the hare: it doubles your correct digits every step near a root, but it can diverge catastrophically if you start too far away.
* The condition number of root-finding is the reciprocal of the slope — a root where the function barely grazes zero is genuinely hard to locate precisely.

## What Comes Next

One-dimensional root-finding already reveals the full drama of nonlinear computation: the tension between global reliability and local speed, and the role of the derivative as the key to fast convergence. In $n$ dimensions, the derivative becomes a Jacobian matrix — an $n \times n$ object — and each Newton step requires solving a linear system rather than performing a simple division.

This is where linear algebra pays its debts. Every Newton step in higher dimensions is exactly the kind of problem studied in the linear equations lesson: a known matrix, a known right-hand side, and an unknown step to compute. The condition number of the Jacobian will determine how trustworthy each step is.

## Check Your Understanding

1. Bisection on $[0, 10]$ has already run for 20 steps. What is the maximum possible distance between the current midpoint and the true root?
2. Newton's method achieves quadratic convergence because $g'(x^*) = 0$ at the root. Explain in your own words why this derivative being zero leads to the error squaring each step.
3. A function has a double root: $f(x) = (x - r)^2$. What happens to Newton's method near $r$, and why does convergence degrade?

[[simulation basins-of-attraction]]

## Challenge

Implement both bisection and Newton's method to find all roots of $f(x) = x^5 - 5x^3 + 4x$ on $[-3, 3]$. First plot the function to identify brackets for bisection. Then experiment with different starting points for Newton's method, recording whether each starting point leads to convergence and which root it finds. Map out the basins of attraction: which starting points lead to which roots? Are there starting points where Newton's method diverges entirely?
