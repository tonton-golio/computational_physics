# Bounding Errors
*Why your computer is a liar and how to catch it in the act*

## The first surprise

Suppose you try to compute $\sqrt{2}$ on a calculator. You punch in 2, hit the square root button, and get 1.41421356... That looks pretty good. But here's the thing: your calculator is lying to you. Not maliciously — it's doing its best with only a finite number of digits. The true $\sqrt{2}$ goes on forever, and the calculator has to stop somewhere. Every number it gives you is a tiny lie.

The whole game of scientific computing starts right here: **understanding how much your computer is lying, and making sure the lies stay small enough that your answers are still useful.**

## Sources of Error and Error Definitions

**Sources of approximation** include:
- Modelling (simplifications of the physical system)
- Empirical measurements
- Previous computations
- Truncation/discretization
- Rounding

**Absolute error** and **relative error** are different in the obvious manner, i.e., abs. error
= approx. value - true value, and rel. error = abs. error / true value.

**Data error and computational error**: the hats indicate an approximation;

$$
\begin{align*}
\text{total error} &= \hat{f}(\hat{x})-f(x)\\
&= \left(\hat{f}(\hat{x})-f(\hat{x})\right) &+&\left(f(\hat{x})-f(x)\right)\\
&= \text{computational error} &+& \text{propagated data error}\\
E_\text{tot} &= E_\text{comp} &+& E_\text{data}
\end{align*}
$$

*In plain English: the total error is the sum of (a) mistakes the algorithm makes and (b) mistakes already baked into the input data.*

Here $\hat{x}$ is the approximate input, $\hat{f}$ is the approximate function, $f$ is the true function, and $x$ is the true input.

> **You might be wondering...** "Can these two errors cancel each other out?" Yes, sometimes they do! But you can't count on it. That's like hoping two wrongs make a right — occasionally true, but a terrible strategy.

## Truncation Error vs Rounding Error
Computational error can be split into truncation error $E_\text{trunc}$ and rounding error $E_\text{round}$.

**Truncation error** stems from:
- Simplifications of the physical model (frictionless, etc.)
- Finite basis sets
- Truncations of infinite series (replacing derivatives with finite differences)

**Rounding error** contains everything that comes from working on a finite computer:
- Accumulated rounding error (from finite arithmetic)

**Forward vs. backward error**

Forward error is the error in the output, backward error is the error in the input.

*Think of it this way: forward error asks "How wrong is my answer?" Backward error asks "What slightly different question did I actually solve perfectly?"*


## Sensitivity, Conditioning, and Floating Point
**Sensitivity and conditioning**

Condition number: $\text{COND}(f) \equiv \frac{|\Delta y / y|}{|\Delta x / x|} = \frac{|x \Delta y|}{|y \Delta x|}$

In the limit $\Delta x \to 0$, this becomes $\text{COND}(f) = \left|\frac{x f'(x)}{f(x)}\right|$, which measures how sensitive the relative output is to relative changes in input.

*This says: the condition number is how much the problem itself amplifies relative errors. A condition number of 100 means a 1% input error could become a 100% output error. Yikes.*

> **You might be wondering...** "Is a bad condition number the computer's fault?" No! Conditioning is a property of the *problem*, not the algorithm. Even a perfect computer with infinite precision would struggle with an ill-conditioned problem. It's like trying to balance a pencil on its tip — the physics makes it hard, not your fingers.

**Stability and accuracy**

- Fixed points have each bit correspond to a specific scale.
- Floating point (32 bit) has: 1 sign bit (0=positive, 1=negative), 8 exponent bits, and 23 mantissa bits. Machine epsilon is $\epsilon \approx 2^{-23} \approx 1.2 \times 10^{-7}$ for single precision.

Overflow and underflow refer to the largest and smallest numbers that can be contained in a floating point representation.


## Example: Finite Difference Error Trade-off

Here's the beautiful part — watch what happens when two types of error fight each other.

Computational error of first order finite difference: $$ f'(x) \approx \frac{f(x+h) - f(x)}{h} \equiv \hat{f'}(x) $$

*This says: estimate the slope by looking at two nearby points and computing rise over run.*

Taylor expand:
$$ f(x+h) = f(x) + h f'(x) + \frac{h^2}{2} f''(\theta), \qquad \lvert \theta - x \rvert \leq h $$
$$ \frac{f(x+h) - f(x)}{h} = f'(x) + \frac{h}{2} f''(\theta) $$
$$ \hat{f'}(x) - f'(x) = \frac{h}{2} f''(\theta) $$
$$ M \equiv \sup_{|\theta - x| \leq h} \lvert f''(\theta) \rvert $$
$$ E_\text{trunc} = |\hat{f'}(x) - f'(x)| \leq \frac{M}{2} h \quad \sim O(h) $$

*Truncation error shrinks as h gets smaller — make the step tiny, get a better derivative.*

What about rounding error? When we compute $f(x+h)$ and $f(x)$ in floating point, each has a relative error bounded by machine epsilon $\epsilon$. The subtraction $f(x+h) - f(x)$ can lose significant digits (cancellation), and dividing by $h$ amplifies this. The result is:
$$ E_\text{round} \leq \frac{2\epsilon}{h} \quad \sim O\left(\frac{1}{h}\right) $$

*Rounding error grows as h gets smaller — the very thing that helps truncation error makes rounding error worse!*

The factor of 2 arises because we subtract two quantities, each with rounding error up to $\epsilon |f|$.

If you decrease $h$, you decrease truncation error but increase rounding error:

$$ E_\text{comp} = \frac{M}{2} h + \frac{2\epsilon}{h} $$

*Two forces pulling in opposite directions. There's a sweet spot in the middle.*

What value of $h$ minimizes it? Differentiate and set to zero:

$$ 0 = \frac{dE_\text{comp}}{dh} = \frac{M}{2} - \frac{2\epsilon}{h^2} $$
$$ h^2 = \frac{4\epsilon}{M} $$
Since $h$ is a step size (positive by definition):
$$ h_\text{optimal} = 2 \sqrt{\frac{\epsilon}{M}} $$
(Note that $\epsilon$ here is a bound on the relative rounding error.)

> **Challenge:** Try this in Python in 30 seconds. Compute the derivative of $\sin(x)$ at $x=1$ using $h = 10^{-1}, 10^{-2}, \dots, 10^{-16}$. Plot the error. You'll see it drop, hit a sweet spot, then climb back up. That's the truncation-rounding tug of war, right there on your screen.

## Propagated Data Error

The problem can either expand or contract the error from your data, and it's important to understand what it does.

Absolute forward data error = $f(\hat{x}) - f(x) \equiv \Delta y$

Relative forward data error = $ \frac{\Delta y}{y} = \frac{f(\hat{x}) - f(x)}{f(x)}$

We also use a _Condition Number_: How much a change in the data affects a change in the result.
$$ \text{COND}(f) = \frac{\lvert \Delta y/y \rvert}{ \lvert \Delta x/x \rvert} = \frac{\lvert x \Delta y \rvert}{ \lvert y \Delta x \rvert} $$

It may be intuitive that if you start with 4 digits of input, your output will be correct to 4 digits at max. This isn't the case: Consider a function $f(x) = x^{\frac{1}{10}}$, and let's analyze the errors.

$$ E_\text{data} = f(\hat{x}) - f(x) $$
$$ (x+\Delta x)^\frac{1}{10} - x^\frac{1}{10} $$
The relative error will be
$$ E^\text{rel}_\text{data} = \frac{(x+\Delta x)^\frac{1}{10} - x^\frac{1}{10}}{x^\frac{1}{10}} $$
Now Taylor expand: $(x + \Delta x)^{1/10} = x^{1/10}(1 + \Delta x/x)^{1/10} \approx x^{1/10}\left[1 + \frac{1}{10}\frac{\Delta x}{x} + \frac{(1/10)(-9/10)}{2}\left(\frac{\Delta x}{x}\right)^2 + \cdots\right]$
$$ \Delta y / y = \frac{1}{10} \Delta{x} /x + O\left((\Delta x / x)^2\right)$$
The leading factor of $1/10$ means the relative error shrinks: you can have an additional significant digit in the output. Start with 3 correct digits, end up with 4, etc.

> **You might be wondering...** "Wait, are we creating information out of nowhere?" It sure looks like it! But here's the rubber-band analogy: imagine stretching a rubber band between two marks on a ruler. The marks have some uncertainty. Now compress the rubber band to one-tenth its length. The marks get closer together — the *absolute* uncertainty shrinks. But you haven't created information; you've just compressed the range. Go the other way (stretch the band to 10x), and the uncertainty blows up. The function $x^{1/10}$ compresses; $x^{10}$ stretches. No free lunch, just redistribution.

**In general**:
- $\sqrt{x}$ has 1 more significant bit as compared to $x$
- $x^{1/10^n}$ has $n$ more decimal significant digits
- $x^2$ is 1 fewer bit significant
- $x^{10^n}$ has $n$ fewer decimal significant digits

But information theory tells us that information cannot be gained out of nowhere: what's going on?

The resolution is that taking a root does not create information. The condition number $\text{COND}(x^{1/10}) = 1/10 < 1$, so the function _compresses_ errors, but the total information content is preserved. The output has more correct digits because the function maps a wide input range into a narrow output range. Going the other direction ($x^{10}$) _amplifies_ errors, and you lose digits. The two are consistent: no free lunch, just redistribution.

---

**What we just learned in one sentence:** Your computer rounds every number it touches, and the condition number tells you how much the problem amplifies those tiny lies into big ones.

**What's next and why it matters:** Now that we know the machine is lying to us, let's learn to solve linear systems of equations — the one type of problem where we have exact, reliable tools. That's the foundation everything else is built on.
