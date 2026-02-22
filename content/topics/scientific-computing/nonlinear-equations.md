# Nonlinear Equations

> *We just mastered linear systems where everything was guaranteed. Now those guarantees vanish. Welcome to the real world.*

## Big Ideas

* Nonlinear equations offer none of the guarantees of linear ones — you may have zero solutions, one, many, or infinitely many, and there's no formula that tells you which.
* Bisection is the tortoise: it always converges if a bracket exists, gains exactly one bit per step, and cannot be fooled.
* Newton's method is the hare: it doubles your correct digits every step near a root, but it can diverge catastrophically if you start too far away.
* The condition number of root-finding is the reciprocal of the slope — a root where the function barely grazes zero is genuinely hard to locate precisely.

# Root Finding in One Dimension

## Notation:

* $f(x) = 0$: 1D stuff. $f:\mathbb{R} \to \mathbb{R}$
* $F(x) = 0$: Higher dimensions. $F:\mathbb{R}^n\to\mathbb{R}^m$

## The Nonlinear Wilderness

Emergent macroscopic behaviour comes out of high dimensionality. You don't figure out the aerodynamics of a plane by using Schrodinger's equation for every atom. Simpler theories based on emergent behaviour can be nonlinear even when the underlying rules are linear.

**Linear Systems** gave us: exact solution counts, fail-proof solvers, full solution spaces, and routines that handle billions of dimensions.

**Nonlinear systems** give us: no idea how many solutions exist, no fail-proof solvers, no way to know if we've found them all, and even 1D can take ages.

## How many solutions?

Globally, *anything is possible*. $e^{-x} = 0$ has none. $\sin(x) = 0$ has infinitely many.

Locally, we can sometimes work it out. In 1D, if $f$ changes sign on an interval, there's a root in between (Intermediate Value Theorem). There's an odd number of roots in that interval.

## General Algorithm Construction

Here's a powerful recipe for inventing algorithms:
1. Find an invariant that guarantees a solution exists in your search space
2. Design an operation that preserves that invariant and shrinks the search space

*First make sure you're looking in the right place, then systematically make that place smaller. Euclid did this thousands of years ago.*

## Bisection Method

**Step 1:** Establish the invariant: $a<b$ and $\text{sign}(f(a)) \neq \text{sign}(f(b))$. The function crosses zero somewhere in there. We're sure of it.

**Step 2:** Cut in half: $m = a + \frac{b-a}{2}$ and evaluate $S_m = \text{sign}(f(m))$

**Step 3:** Keep the half that still brackets the root.

(Note: use $m = a + \frac{b-a}{2}$, not $\frac{a+b}{2}$, to avoid floating-point overflow.)

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

## Conditioning

If a function has a steep slope near the root, the root is easy to find (well-conditioned). If the function barely grazes zero, the root is hard to pin down. The absolute conditioning of root-finding is $\left|\frac{1}{f'(x)}\right|$.

## Convergence

$e_{k} = x_k - x^*$. We measure convergence by how many significant bits we gain per step.

If $\lim_{k\to 0} \frac{\vert e_{k+1}\vert}{\vert e_k\vert^r} = c$ with $0 \leq c < 1$:
* **Linear (r=1):** Like simple interest — a fixed number of correct digits per step. Bisection does this.
* **Quadratic (r=2):** Like compound interest on steroids — correct digits *double* each step. 3, then 6, then 12, then 24. Newton's method does this.

## Fixed Point Solvers

Picture a drunkard trying to walk home. He starts somewhere, takes a step according to some rule, and ends up somewhere new. If the rule is a contraction — each step brings him closer regardless of where he is — he's guaranteed to stumble through his front door. That's fixed-point iteration.

We use _Banach's fixed point theorem_: if $g$ is a contraction on a closed set $S$ (meaning $\|g(x) - g(x')\| \leq c\|x - x'\|$ for some $c < 1$), then $g(x) = x$ has a unique solution, and iterating $x_{k+1} = g(x_k)$ from any starting point in $S$ will find it.

*If every step brings points closer together, all roads lead to Rome.*

How do you know if your choice of $g$ will converge? Check the derivative at the fixed point. If $|g'(x^*)| < 1$, it converges. If $|g'(x^*)| = 0$, you've hit the jackpot — quadratic convergence.

The error analysis (1D): by the Mean Value Theorem, $|e_{k+1}| = |g'(\theta)||e_k|$ for some $\theta$ between $x_k$ and $x^*$. If $|g'|$ is bounded by $c < 1$ near the root, each step shrinks the error.

If $g'(x^*) = 0$, Taylor expanding gives $|e_{k+1}| \leq |g''(\theta)||e_k|^2$ — quadratic convergence.

## Newton's Method

Here's the trick that makes quadratic convergence happen automatically:

$$ x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)} $$

*At each step, replace the curve with its tangent line, and go where the tangent crosses zero. Repeat.*

Watch the beautiful part: set $g(x) = x - f(x)/f'(x)$ and compute $g'(x)$. At the root where $f(x^*) = 0$, you get $g'(x^*) = 0$, which is exactly the condition for quadratic convergence. It happens automatically.

You have to start close enough, but once you do it converges like wildfire.

> **Challenge.** Implement Newton's method in 3 lines of Python and find $\sqrt{2}$ by solving $f(x) = x^2 - 2 = 0$. Start from $x_0 = 1$. Print the error at each step and verify it roughly squares each time.

---

[[simulation basins-of-attraction]]

## What Comes Next

One-dimensional root-finding already reveals the full drama: the tension between global reliability and local speed, and the role of the derivative as the key to fast convergence. In $n$ dimensions, the derivative becomes a Jacobian matrix, and each Newton step requires solving a linear system — exactly what we studied in the linear equations lesson.

## Check Your Understanding

1. Bisection on $[0, 10]$ has run for 20 steps. What's the maximum possible distance between the current midpoint and the true root?
2. Newton's method achieves quadratic convergence because $g'(x^*) = 0$. Explain in your own words why this makes the error square each step.
3. Try Newton's method on $f(x) = x^2$ starting from $x_0 = 1$. What happens, and why is convergence slower than for a simple root?

## Challenge

Implement both bisection and Newton's method to find all roots of $f(x) = x^5 - 5x^3 + 4x$ on $[-3, 3]$. First plot the function to identify brackets. Then try three different starting points for Newton and see which root each one finds. Are there starting points where Newton diverges entirely?
