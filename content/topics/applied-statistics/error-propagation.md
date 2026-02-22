# Central Limit Theorem and Error Propagation

## The Drunkard's Walk to Gaussian

Picture a drunkard stumbling home. Each step is random -- left, right, long, short. After one step, the position could be anywhere. After a thousand? Here's the crazy thing: if you repeated the walk many times, the final positions would form a bell curve. Every single time. No matter how wild each step is.

That's the **central limit theorem**. Take $N$ independent samples from *any* distribution (with finite variance), average them, and that average is approximately Gaussian. The original distribution can be uniform, exponential, bimodal, whatever -- the average converges to a bell curve.

[[simulation applied-stats-sim-3]]

This is why Gaussian error analysis works almost everywhere. Your measurement is typically many small independent influences added together. Each might follow some weird distribution, but their sum -- your measurement -- is Gaussian. The CLT is your license to use Gaussian statistics.

**Caveats**: the contributing distributions should have similar standard deviations. If one source dwarfs the others, the sum looks like that dominant source. And distributions with no finite variance (like the Cauchy) never converge -- no amount of averaging tames them.

> **Challenge.** Roll a single die -- perfectly uniform, not a bell curve at all. Now roll five dice and average them. Do this twenty times. Plot your averages. They'll cluster around 3.5 in a bell shape. That's the CLT at work, turning uniform randomness into Gaussian.

## Error Propagation

You've measured some inputs $x_i$ with uncertainties $\sigma(x_i)$ and you compute a derived quantity $y(x_i)$. What's the uncertainty on $y$?

Here's the intuition: imagine walking on a hill. A small wobble in your position causes a wobble in your altitude. On a **steep** hill, a tiny wobble produces a big altitude change. On a **flat** hill, the same wobble barely matters. Error propagation is the same idea -- the steeper $y$ changes with $x$, the more the input error gets amplified.

Measure the radius of a circle with some uncertainty, and you want the area $A = \pi r^2$. The derivative $dA/dr = 2\pi r$ says a 1% error in $r$ becomes roughly a 2% error in $A$. For a sphere's volume ($V = \frac{4}{3}\pi r^3$), that same 1% becomes 3%. The steeper the function, the bigger the amplification.

Formally, for one variable:

$$
\sigma(y) = \frac{\partial y}{\partial x}\sigma(x_i)
$$

For multiple variables with correlations, using the covariance matrix $V_{ij}$ (from [introduction and concepts](./introduction-concepts)):

$$
\sigma_y^2 = \sum_{i,j}^n \frac{\partial y}{\partial x_i} \frac{\partial y}{\partial x_j} V_{ij}
$$

If inputs are uncorrelated, only diagonal terms survive. This tells you which measurement dominates the error budget -- improve *that* one.

### Addition

$$
y = x_1 + x_2 \implies \sigma_y^2 = \sigma_{x_1}^2 + \sigma_{x_2}^2 + 2V_{x_1, x_2}
$$

Errors add in quadrature when uncorrelated. Combining independent measurements always improves precision.

### Multiplication

$$
y = x_1 x_2 \implies \sigma_y^2 = (x_2\sigma_{x_1})^2 + (x_1\sigma_{x_2})^2 + 2x_1 x_2 V_{x_1, x_2}
$$

And here's a beautiful consequence: by engineering *negative* error correlations, you can make errors partially cancel. Harrison's gridiron pendulum is a gorgeous example. He arranged brass and steel rods so that thermal expansion in one partially cancels the other -- the two metals expand in the same direction, but their contributions to the pendulum length have opposite signs. He exploited that covariance term two centuries before anyone wrote the formula.

### When Analytical Propagation Fails

The formulas assume $y(x)$ is smooth and roughly linear over the uncertainty range. When that breaks down -- thresholds, discontinuities, wild nonlinearity -- you simulate instead. Draw random inputs from their error distributions, compute $y$ for each draw, and look at the spread. This Monte Carlo approach (next section) handles anything.

[[simulation applied-stats-sim-8]]

> **Challenge.** Explain error propagation using only the example of measuring a room's area from length and width. No equations. One minute.

## Big Ideas

* The CLT is why Gaussian error analysis works: averages of random quantities converge to bell curves, no matter what the individuals look like.
* Error propagation asks "how steeply does my answer change when I wiggle each input?" Steep functions amplify errors; shallow ones suppress them.
* Negative covariance is an opportunity, not a problem. Harrison exploited it to cancel thermal drift two centuries early.
* When the function isn't smooth, the derivative formula fails -- simulation is the honest alternative.

## What Comes Next

The CLT and error propagation give you pen-and-paper tools for tracking uncertainty. But they assume smooth functions, Gaussian errors, and tractable derivatives. When those assumptions break, you replace the calculation with brute-force simulation: generate random inputs, push them through a million times, and look at the spread. That's Monte Carlo -- and it connects back to the CLT in a satisfying way.

## Check Your Understanding

1. You measure $L = 10.0 \pm 0.2$ cm for the side of a square. What's the uncertainty on the area? Show how the derivative formula gives a factor of 2, and explain in words why squaring amplifies relative error.
2. Two correlated quantities with $V_{xy} < 0$. How does this affect the uncertainty on $z = x + y$? Can the total uncertainty be smaller than either individual one?
3. "If I average enough measurements, the CLT guarantees Gaussianity, no matter what." When is this wrong?

## Challenge

Design an experiment to measure the volume of a small irregular rock using a ruler and a graduated cylinder. Write down the formula, identify all inputs and uncertainties, and propagate them. One of your measurements enters as a *difference* (water level before and after). What does this do to the relative uncertainty compared to measuring a rectangular block?
