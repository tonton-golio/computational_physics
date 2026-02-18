# Central Limit Theorem and Error Propagation

## The Drunkard's Walk to Gaussian

Imagine a drunkard stumbling home from a bar. Each step is random — left, right, long, short, forward, backward. After one step, the drunkard could be anywhere. After ten steps, the position is hard to predict. But after a *thousand* steps? Something remarkable happens: if you repeated this walk many times, the final positions would form a bell curve. Every time. No matter how erratic each individual step is.

That is the **central limit theorem** (CLT) in action. Take $N$ independent samples from *any* distribution (as long as it has finite variance), compute their mean, and that mean will be approximately Gaussian. The more samples you average, the better the approximation. The original distribution can be uniform, exponential, bimodal, or anything else — the average still converges to a bell curve.

[[simulation applied-stats-sim-3]]

This is why Gaussian error analysis works almost everywhere. Your measurement is typically the result of many small, independent influences added together. Each influence might follow some complicated distribution, but their sum — your measurement — will be Gaussian. The CLT is the license that lets you use Gaussian statistics.

But there are caveats. For the CLT to work well, the contributing distributions should have similar standard deviations. If one source of variation dwarfs all others, the sum looks like that dominant source, not a Gaussian. And distributions with very heavy tails (like the Cauchy distribution, which has no finite variance) never converge — no amount of averaging tames them.

**The practical rule**: if each contribution has finite variance and no single contribution dominates, the sum is approximately Gaussian. This is the foundation on which the rest of the course builds.

> **Try this at home.** Roll a single die and note the result. Now roll five dice and compute the average. Do this twenty times. Plot your twenty averages on a number line. You'll see them clustering around 3.5 in a roughly bell-shaped pattern — even though a single die roll is perfectly uniform. That's the CLT at work.

## Error Propagation

Now suppose you've measured some input quantities $x_i$, each with uncertainty $\sigma(x_i)$, and you compute a derived quantity $y(x_i)$. What is the uncertainty on $y$?

The intuition is simple: a small wiggle in the input causes a wiggle in the output. The size of that output wiggle depends on how steeply $y$ changes with respect to $x$. If $y$ varies gently, input errors stay small. If $y$ varies steeply, input errors get amplified.

Think of it this way. You measure the radius of a circle with some uncertainty, and you want the area $A = \pi r^2$. The derivative $dA/dr = 2\pi r$ tells you that a 1% error in $r$ becomes roughly a 2% error in $A$ — the squaring amplifies it. For the volume of a sphere ($V = \frac{4}{3}\pi r^3$), the same 1% error in $r$ becomes a 3% error in $V$. The steeper the function, the bigger the amplification.

Formally, for a function of one variable:

$$
\sigma(y) = \frac{\partial y}{\partial x}\sigma(x_i)
$$

This works when $y$ is smooth around $x_i$ — when the slope is roughly constant over the uncertainty range.

For multiple variables with correlations, the general formula uses the covariance matrix $V_{ij}$ (which we met in lesson 1):

$$
\sigma_y^2 = \sum_{i,j}^n \frac{\partial y}{\partial x_i} \frac{\partial y}{\partial x_j} V_{ij}
$$

If there are no correlations between the inputs, only the diagonal terms survive — each input contributes independently to the total uncertainty. This lets you identify which measurement dominates the error budget and focus your effort on improving *that* one.

### Addition

$$
y = x_1 + x_2 \implies \sigma_y^2 = \sigma_{x_1}^2 + \sigma_{x_2}^2 + 2V_{x_1, x_2}
$$

Errors add in quadrature (when uncorrelated). This is why combining independent measurements always improves precision.

### Multiplication

$$
y = x_1 x_2 \implies \sigma_y^2 = (x_2\sigma_{x_1})^2 + (x_1\sigma_{x_2})^2 + 2x_1 x_2 V_{x_1, x_2}
$$

Dividing by $y^2$ gives the relative uncertainties. Here's a beautiful consequence: by engineering *negative* error correlations, you can make errors partially cancel.

Alex encounters this trick in the lab. Harrison's gridiron pendulum uses two metals with different thermal expansion coefficients — brass and steel — arranged so that when temperature rises, the brass rods push the pendulum bob *down* while the steel rods push it *up*. The expansions partially cancel, keeping the pendulum length stable. The key insight: the two metals expand in the same direction physically, but their *contributions to the total length* have opposite signs. Harrison exploited the covariance term $V_{x_1, x_2}$ to engineer a pendulum that barely changes length with temperature — a mechanical implementation of error cancellation, two centuries before anyone wrote down the formula.

### When Analytical Propagation Fails

The formulas above assume $y(x)$ is smooth and approximately linear over the uncertainty range. When that breaks down — sharp thresholds, discontinuities, highly nonlinear functions — you need a different approach: **simulation**.

Choose random inputs $x_i$ from their error distributions, compute $y$ for each draw, and look at the resulting spread in $y$. This Monte Carlo approach (developed fully in the next section) handles arbitrary functions, non-Gaussian inputs, and complex correlations. If $y(x)$ is not smooth, the output distribution may not be Gaussian even when the inputs are — the simulation reveals this automatically.

[[simulation applied-stats-sim-8]]

---

**What we just learned, and why it matters.** The CLT explains why Gaussian statistics appear everywhere: averages of random quantities converge to a bell curve. Error propagation tells you how uncertainties flow through calculations — steeply varying functions amplify errors, gently varying ones suppress them. Together, these two ideas let you attach meaningful error bars to almost any derived quantity. But when the math gets too complicated for pen and paper, you need to let the computer do the experiment for you. That's simulation — and it's where we go next.

> **Challenge.** Explain error propagation to a friend using only the example of measuring a room's area from its length and width. No equations. One minute.
