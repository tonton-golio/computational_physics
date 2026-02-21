# Bounding Errors

## The first surprise

Suppose you try to compute $\sqrt{2}$ on a calculator. You punch in 2, hit the square root button, and get 1.41421356... That looks pretty good. But here's the thing: your calculator is lying to you. Not maliciously — it's doing its best with only a finite number of digits. The true $\sqrt{2}$ goes on forever, and the calculator has to stop somewhere. Every number it gives you is a tiny lie.

The whole game of scientific computing starts right here: **understanding how much your computer is lying, and making sure the lies stay small enough that your answers are still useful.**

## Sources of Error and Error Definitions

**Sources of approximation** include:
* Modelling (simplifications of the physical system)
* Empirical measurements
* Previous computations
* Truncation/discretization
* Rounding

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

Can these two errors cancel each other out? Yes, sometimes they do! But you can't count on it. That's like hoping two wrongs make a right — occasionally true, but a terrible strategy.

## Truncation Error vs Rounding Error
Computational error can be split into truncation error $E_\text{trunc}$ and rounding error $E_\text{round}$.

**Truncation error** stems from:
* Simplifications of the physical model (frictionless, etc.)
* Finite basis sets
* Truncations of infinite series (replacing derivatives with finite differences)

**Rounding error** contains everything that comes from working on a finite computer:
* Accumulated rounding error (from finite arithmetic)

**Forward vs. backward error**

Forward error is the error in the output, backward error is the error in the input.

*Think of it this way: forward error asks "How wrong is my answer?" Backward error asks "What slightly different question did I actually solve perfectly?"*


## Sensitivity, Conditioning, and Floating Point
**Sensitivity and conditioning**

Think of the condition number as an **error amplifier** built into the problem itself. Before you choose an algorithm, before you write a single line of code, the mathematics already has an opinion about how many of your input digits will survive to the output. A condition number of 1 means every input digit comes through intact. A condition number of $10^6$ means you lose six decimal digits on the spot — not because the algorithm is bad, but because the problem is inherently sensitive. The whole point of learning about conditioning first is that no clever algorithm can rescue digits the problem has already destroyed.

Condition number: $\text{COND}(f) \equiv \frac{|\Delta y / y|}{|\Delta x / x|} = \frac{|x \Delta y|}{|y \Delta x|}$

In the limit $\Delta x \to 0$, this becomes $\text{COND}(f) = \left|\frac{x f'(x)}{f(x)}\right|$, which measures how sensitive the relative output is to relative changes in input.

*This says: the condition number is how much the problem itself amplifies relative errors. A condition number of 100 means a 1% input error could become a 100% output error. Yikes.*

Is a bad condition number the computer's fault? No! Conditioning is a property of the *problem*, not the algorithm. Even a perfect computer with infinite precision would struggle with an ill-conditioned problem. It's like trying to balance a pencil on its tip — the physics makes it hard, not your fingers.

**Stability and accuracy**

* Fixed points have each bit correspond to a specific scale.
* Floating point (32 bit) has: 1 sign bit (0=positive, 1=negative), 8 exponent bits, and 23 mantissa bits. Machine epsilon is $\epsilon \approx 2^{-23} \approx 1.2 \times 10^{-7}$ for single precision.

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

> **Challenge.** Try this in Python in 30 seconds. Compute the derivative of $\sin(x)$ at $x=1$ using $h = 10^{-1}, 10^{-2}, \dots, 10^{-16}$. Plot the error. You'll see it drop, hit a sweet spot, then climb back up. That's the truncation-rounding tug of war, right there on your screen.

[[simulation error-vs-h]]

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

Are we creating information out of nowhere? It sure looks like it! But here's the rubber-band analogy: imagine stretching a rubber band between two marks on a ruler. The marks have some uncertainty. Now compress the rubber band to one-tenth its length. The marks get closer together — the *absolute* uncertainty shrinks. But you haven't created information; you've just compressed the range. Go the other way (stretch the band to 10x), and the uncertainty blows up. The function $x^{1/10}$ compresses; $x^{10}$ stretches. No free lunch, just redistribution.

**In general**:
* $\sqrt{x}$ has 1 more significant bit as compared to $x$
* $x^{1/10^n}$ has $n$ more decimal significant digits
* $x^2$ is 1 fewer bit significant
* $x^{10^n}$ has $n$ fewer decimal significant digits

But information theory tells us that information cannot be gained out of nowhere: what's going on?

The resolution is that taking a root does not create information. The condition number $\text{COND}(x^{1/10}) = 1/10 < 1$, so the function _compresses_ errors, but the total information content is preserved. The output has more correct digits because the function maps a wide input range into a narrow output range. Going the other direction ($x^{10}$) _amplifies_ errors, and you lose digits. The two are consistent: no free lunch, just redistribution.

---

## Big Ideas

* Every number in a computer is a rounded approximation — the machine is always lying to you a tiny bit, and understanding that lie is the first step to controlling it.
* Total error splits cleanly into computational error (the algorithm's fault) and propagated data error (the input's fault); you need to track both.
* The condition number belongs to the *problem*, not the algorithm — an ill-conditioned problem will defeat even a perfect computer.
* Truncation error and rounding error pull in opposite directions as you shrink a step size, so there is always a sweet spot, and blindly making $h$ smaller eventually makes things *worse*.

## What Comes Next

The bad news is that floating-point arithmetic corrupts every number it touches. The good news is that for linear systems of equations, we have algorithms that are both exact (in exact arithmetic) and well-understood in the presence of rounding. The condition number you just learned about will reappear there as the precise measure of how much you should trust your answer.

Understanding error bounds is not just bookkeeping — it is the lens through which every subsequent method should be judged. When a linear solver, an eigenvalue algorithm, or a PDE integrator gives you a number, the condition number tells you how many of those digits to believe.

## Check Your Understanding

1. A function $f(x) = x^{10}$ is applied to an input with a 0.1% relative error. Roughly what relative error do you expect in the output, and why?
2. You decrease the step size $h$ in a finite-difference derivative estimate and the error starts *increasing*. What is happening, and at what value of $h$ is the error minimized?
3. Two errors cancel in a particular computation, giving a surprisingly accurate answer. Should you be relieved or suspicious, and why?

## Challenge

Write a program that computes the condition number of $f(x) = x^n$ for $n = -10, -5, -2, -1, 0, 1, 2, 5, 10$ and plots $\text{COND}(f)$ versus $n$. Then pick a single input $x$ with a known relative error of $10^{-4}$ and, for each $n$, predict the output relative error from the condition number alone. Compare your prediction against the actual error computed numerically. Where does the bound tighten, and where does it remain loose?
