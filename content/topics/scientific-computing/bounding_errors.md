# Bounding Errors



## Header 1
**Sources of approximation** include modelling, 
empirical measurements, previous computations, truncation/discretization, rounding.
 
**Absolute error** and **relative error** are different in the obvious manner, i.e., abs. error 
= approx. value- true value, and rel. error = abs. error / true value.
**Data error and computational errror**, the hats indicate an approximation;

$$
\begin{align*}
\text{total error} &= \hat{f}(\hat{x})-f(x)\\
&= \left(\hat{f}(\hat{x})-f(\hat{x})\right) &+&\left(f(\hat{x})-f(x)\right)\\
&= \text{computational error} &+& \text{propagated data error}\\
E_\text{tot} &= E_\text{comp} &+& E_\text{data}
\end{align*}
$$

## Header 2
Computational error can be split int truncation error $E_{trunc}$ and rounding error E_{round}.

**Truncation error** contains:
* Simplifications of the physciial model (frictionless, etc)
* Finite basis sets
* Truncations of infinite series

**Rounding error** contains everything that comes from working on a finite computer
* Accumulated rounding error (from finite arithmetic)

**Forward vs. backward error**

foward error is the error in the output, backward error is the error in the input.


## Header 3
**Sensitivity and conditioning**

Condition number: $COND(f) \equiv \frac{|\frac{\Delta y}{y}|}{|\frac{\Delta x}{x}|} = \frac{|x\Delta y|}{|y\Delta x|}$

**Stability and accuracy**

* Fixed points have each bit correspond to a specific scale.
* floating point (32 bit) has: 1 sign bit (0=postive, 1=negative), 8 exponent bits, and 23 mantissa bits. 

overflow and underflow; refers to the largest and smallest numbers that can be 
contained in a floating point.


## Example

Computational error of first order finite difference: $$ f'(x) = \frac{f(x+h) - f(x)}{h} \def \hat{f'}(x) $$
Taylor expand:
$$ f(x+h) = f(x) + h*f'(x) + \frac{h^2}{2} f''(\theta), \qquad \lvert \theta - x \rvert \leq h $$
$$ \frac{f(x+h) - f(x)}{h} = f'(x) + \frac{h}2 f''(\theta) $$
$$ \hat{f'}(x) - f'(x) = \frac{h}2 f''(\theta) $$
$$ M \def \sup{\theta - x = h} \lvert f''(\theta) \rvert $$
$$ E_{trunc} = \hat{f'}(x) - f'(x) \leq \frac{M}2 h \quad \sim O(h) $$

What about rounding error? Assume that for $f$, it's bounded by $\epsilon$
$$ E_{round} \leq \frac{2\epsilon}h \quad \sim O\left(\frac1h\right) $$ 
(comes from floating point somehow... use significant digits?)

If you decrease $h$, you decrease truncation error but increase rounding error

$$ E_{comp} = \frac{M}2 h + \frac{2\epsilon}h $$
What value of $h$ minimizes it? Differentiate

$$ 0 = \frac{M}2 - \frac{2\epsilon}{h^2} $$
$$ h^2 = \frac{4\epsilon}M $$
$h$ can't be negative, so
$$ h = \frac{2 \sqrt{\epsilon}}{\sqrt{M}} $$
(Note that $\epsilon$ is a bound)

**Propagated Data Error**: The problem can either expand or contract the error from your data, and it's importat to understand what it does

Absolute forward data error = $f(\hat{x}) - f(x) \equiv \Delta y$

Relative forward data error = $ \frac{\Delta y}{y} = \frac{f(\hat{x}) - f(x)}{f(x)}$

We also use a _Condition Number_: How much a change in the data affects a change in the result.
$$ COND(f) = \frac{\lvert \Delta y/y \rvert}{ \lvert \Delta x/x \rvert} = \frac{\lvert x \Delta y \rvert}{ \lvert y \Delta x \rvert} $$

It may be intutive that if you start with 4 digits of input, your ouput will be correct to 4 digits at max. This isn't the case: Consider a function $f(x) = x^{\frac{1}{10}}$, and let's analyze the errors.

$$ E_{data} = f(\hat{x}) - f(x) $$
$$ (x+\Delta x)^\frac{1}{10} - x^\frac{1}{10} $$
The relative error will be
$$ E^{rel}_{data} = \frac{(x+\Delta x)^\frac{1}{10} - x^\frac{1}{10}}{x^\frac{1}{10}} $$
Now Taylor expand
$$ = \frac{x^\frac1{10} + \Delta x x^\frac{-9}{10} - x^\frac{1}{10}}{x^\frac{1}{10}} + O(\frac{\Delta x^2}{x^\frac1{10}})$$
$$ \Delta y / y = \frac1{10} \Delta{x} /x + O(quadratic)$$
You can have an additional significant digit in the output: start with 3, end up with 4, etc

**In general**:
* $\sqrt{x}$ has 1 more significant bit as compared to $x$
* $x^\frac1{10^n}$ has n more decimal significant digits
* x^2 is 1 fewer bit significant
* x^10^n has n fewer decimal sig digits 

But information theory tells us that information cannot be gained out of nowhere: what's going on?

**Truncaiton error and rounding error** are the two parts of computational error. 
Truncation error stems from truncating infinite series, or replacing derivatives 
with finite differences. Rounding error is like the error from like floating point accuracy.

Truncation error, $E_\text{trunc}$ can stem from; 
* simplification of physical model
* finite basis sets
* truncations of infinite series
* ...

*Example:* computational error of $1^\text{st}$ order finite difference

$$
\begin{align*}
f'(x) \approx \frac{f(x+h)-f(x)}{h}\equiv\hat{f'(x)}\\
f(x+h) = f(x)+hf'(x)+ \frac{h^2}{2}f''(\theta), |\theta-x|\leq h\\
\frac{f(x+h)-f(x)}{h} = f'(x) + \frac{h}{2}f''(\theta)\\
\hat{f'(x)} - f'(x) = \frac{h}{2}f''(\theta), \text{let} \equiv \text{Sup}_{|\theta-x|\leq h} (f''(\theta))\\
E_\text{trunc} = \hat{f'(x)}-f'(x)\leq \frac{M}{2}h\sim O(h)
\end{align*}
$$

But what about the rounding error? 
(assume R.E. for $f$ is $\epsilon \Rightarrow E_\text{ronud} \leq \frac{2\epsilon}{h}\sim 0(\frac{1}{h})$

$$
\begin{align*}
E_\text{comp} = \frac{M}{2}h + \frac{2\epsilon}{h}\\
0 = \frac{d}{dh}E_\text{comp} = \frac{M}{2}-\frac{2\epsilon}{h^2}\\
\frac{M}{2} = \frac{2\epsilon}{h^2} 
\Leftrightarrow h^2 = \frac{4\epsilon}{M}\Leftrightarrow h_\text{optimal} = 2\sqrt{\frac{\epsilon}{M}}
\end{align*}
$$