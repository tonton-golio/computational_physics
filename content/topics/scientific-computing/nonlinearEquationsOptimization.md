# Nonlinear Equations and Optimization



# Root Finding in One Dimension
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
