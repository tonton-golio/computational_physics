# Bounding Errors

## Big Ideas

* Every number in a computer is a rounded approximation — the machine is always lying to you a tiny bit, and understanding that lie is the first step to controlling it.
* Total error splits cleanly into computational error (the algorithm's fault) and propagated data error (the input's fault); you need to track both.
* The condition number belongs to the *problem*, not the algorithm — an ill-conditioned problem will defeat even a perfect computer.
* Truncation error and rounding error pull in opposite directions as you shrink a step size, so there's always a sweet spot, and blindly making $h$ smaller eventually makes things *worse*.

## The first surprise

Suppose you try to compute $\sqrt{2}$ on a calculator. You punch in 2, hit the square root button, and get 1.41421356... Looks pretty good. But here's the thing: your calculator is lying to you. Not maliciously — it's doing its best with a finite number of digits. The true $\sqrt{2}$ goes on forever, and the calculator has to stop somewhere. Every number it gives you is a tiny lie.

The whole game of scientific computing starts right here: **understanding how much your computer is lying, and making sure the lies stay small enough that your answers are still useful.**

## Sources of Error

**Sources of approximation** include:
* Modelling (simplifications of the physical system)
* Empirical measurements
* Previous computations
* Truncation/discretization
* Rounding

**Absolute error** = approx. value - true value. **Relative error** = abs. error / true value. Simple enough.

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

Can these two cancel each other out? Sure, sometimes. But counting on that is like hoping two wrongs make a right — occasionally true, terrible strategy.

## Truncation Error vs Rounding Error

**Truncation error** comes from simplifying the math — frictionless models, finite basis sets, replacing derivatives with finite differences.

**Rounding error** comes from working on a finite machine — accumulated rounding from finite arithmetic.

**Forward vs. backward error**: Forward error asks "How wrong is my answer?" Backward error asks "What slightly different question did I actually solve perfectly?" Both are useful lenses.

## Sensitivity, Conditioning, and Floating Point

Think of the condition number as an **error amplifier** built into the problem itself. Before you choose an algorithm, before you write a single line of code, the mathematics already has an opinion about how many of your input digits will survive to the output. A condition number of 1 means every digit comes through intact. A condition number of $10^6$ means you lose six decimal digits on the spot — not because the algorithm is bad, but because the problem is inherently touchy. No clever algorithm can rescue digits the problem has already destroyed.

Condition number: $\text{COND}(f) \equiv \frac{|\Delta y / y|}{|\Delta x / x|} = \frac{|x \Delta y|}{|y \Delta x|}$

In the limit $\Delta x \to 0$: $\text{COND}(f) = \left|\frac{x f'(x)}{f(x)}\right|$

*A condition number of 100 means a 1% input error could become a 100% output error. Yikes.*

Is a bad condition number the computer's fault? No! Conditioning is a property of the *problem*. It's like trying to balance a pencil on its tip — the physics makes it hard, not your fingers.

**Floating point**: 32-bit floats use 1 sign bit, 8 exponent bits, and 23 mantissa bits. Machine epsilon: $\epsilon \approx 2^{-23} \approx 1.2 \times 10^{-7}$ for single precision.

## Example: Finite Difference Error Trade-off

Here's the gorgeous part — watch what happens when two types of error fight each other.

$$ f'(x) \approx \frac{f(x+h) - f(x)}{h} \equiv \hat{f'}(x) $$

*Estimate the slope by looking at two nearby points. Rise over run.*

Taylor expand:
$$ f(x+h) = f(x) + h f'(x) + \frac{h^2}{2} f''(\theta), \qquad \lvert \theta - x \rvert \leq h $$

So the truncation error is:
$$ E_\text{trunc} \leq \frac{M}{2} h \quad \sim O(h) $$
where $M \equiv \sup_{|\theta - x| \leq h} \lvert f''(\theta) \rvert$.

*Truncation error shrinks as h gets smaller. Good.*

But rounding error grows! When we subtract $f(x+h) - f(x)$ in floating point, cancellation eats our digits, and dividing by tiny $h$ amplifies the damage:
$$ E_\text{round} \leq \frac{2\epsilon}{h} \quad \sim O\left(\frac{1}{h}\right) $$

*The very thing that helps truncation error makes rounding error worse!*

Two forces pulling in opposite directions. There's a sweet spot:

$$ E_\text{comp} = \frac{M}{2} h + \frac{2\epsilon}{h} $$

Differentiate and set to zero:
$$ h_\text{optimal} = 2 \sqrt{\frac{\epsilon}{M}} $$

> **Challenge.** Try this in Python in 30 seconds. Compute the derivative of $\sin(x)$ at $x=1$ using $h = 10^{-1}, 10^{-2}, \dots, 10^{-16}$. Plot the error. You'll see it drop, hit a sweet spot, then climb back up. That's the truncation-rounding tug of war, right there on your screen.

[[simulation error-vs-h]]

## Propagated Data Error

The problem can either expand or contract the error from your data. Understanding which is critical.

Absolute forward data error = $f(\hat{x}) - f(x) \equiv \Delta y$

Relative forward data error = $ \frac{\Delta y}{y} = \frac{f(\hat{x}) - f(x)}{f(x)}$

Here's an example that'll mess with your intuition. Consider $f(x) = x^{\frac{1}{10}}$. Taylor expand $(x + \Delta x)^{1/10}$ and you find:
$$ \Delta y / y = \frac{1}{10} \Delta{x} /x + O\left((\Delta x / x)^2\right)$$

The leading factor of $1/10$ means the relative error *shrinks*. Start with 3 correct digits, end up with 4. Are we creating information out of nowhere?

Here's the rubber-band analogy: imagine stretching a rubber band between two marks on a ruler. Now compress it to one-tenth its length. The marks get closer together — the uncertainty shrinks. But you haven't created information; you've just compressed the range. Go the other way (stretch to 10x), and the uncertainty blows up. No free lunch, just redistribution.

**In general**:
* $\sqrt{x}$ has 1 more significant bit compared to $x$
* $x^{1/10^n}$ has $n$ more decimal significant digits
* $x^2$ has 1 fewer bit significant
* $x^{10^n}$ has $n$ fewer decimal significant digits

---

## What Comes Next

Floating-point arithmetic corrupts every number it touches — that's the bad news. The good news is that for linear systems of equations, we have algorithms that are both exact (in exact arithmetic) and well-understood in the presence of rounding. The condition number you just met will reappear there as the precise measure of how much you should trust your answer.

## Check Your Understanding

1. A function $f(x) = x^{10}$ is applied to an input with a 0.1% relative error. Roughly what relative error do you expect in the output, and why?
2. You decrease the step size $h$ in a finite-difference derivative estimate and the error starts *increasing*. What's happening?
3. Two errors cancel in a particular computation, giving a surprisingly accurate answer. Should you be relieved or suspicious?

## Challenge

Write a program that computes the condition number of $f(x) = x^n$ for $n = -10, -5, -2, -1, 0, 1, 2, 5, 10$. Then pick a single input $x$ with a known relative error of $10^{-4}$ and, for each $n$, predict the output relative error from the condition number alone. Compare your prediction against the actual error computed numerically. Where does the bound tighten, and where does it remain loose?
