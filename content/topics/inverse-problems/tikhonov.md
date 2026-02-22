# Big Problems Need Clever Tricks

We now know what [regularization](./regularization) does and *why* it works — it's a Gaussian prior ([Bayesian inversion](./bayesian-inversion)). The Tikhonov formula gives a closed-form MAP estimate. One matrix inversion, done.

But here's the catch. That formula requires building $\mathbf{G}^T\mathbf{G} + \epsilon^2\mathbf{I}$ and inverting it. If your model has 100 parameters, no problem. If it has a million — a 3D seismic tomography model, a full-waveform inversion grid, a climate reanalysis — that matrix has $10^{12}$ entries. You can't even store it, let alone invert it.

So we need a different strategy: instead of solving the problem in one shot, we *walk* toward the answer, one step at a time.

---

## Steepest Descent

The simplest idea: at each step, move in the direction that decreases the objective fastest. For the Tikhonov objective

$$
J(\mathbf{m}) = \|\mathbf{d} - \mathbf{Gm}\|^2 + \epsilon^2\|\mathbf{m}\|^2,
$$

the gradient is

$$
\nabla J = -2\mathbf{G}^T(\mathbf{d} - \mathbf{Gm}) + 2\epsilon^2\mathbf{m}.
$$

The update rule:

$$
\mathbf{m}_{k+1} = \mathbf{m}_k - \alpha_k \nabla J(\mathbf{m}_k),
$$

where $\alpha_k$ is the step size (learning rate).

Sounds simple. And it is — but the devil is in the details.

[[simulation steepest-descent]]

Watch what happens: step size too large and the iterates overshoot and bounce off the walls of a narrow valley. Too small, and you're tiptoeing toward the answer for an eternity. Just right — smooth convergence to the same answer the closed-form gives, but obtained without ever building the full matrix.

The beautiful thing is that each iteration only requires *matrix-vector products* $\mathbf{G}\mathbf{v}$ and $\mathbf{G}^T\mathbf{v}$ — not the full matrix $\mathbf{G}^T\mathbf{G}$. For large sparse systems, this is the difference between feasible and impossible.

---

## Beyond Steepest Descent

Steepest descent works but it can be agonizingly slow when the problem is ill-conditioned — eigenvalues spanning many orders of magnitude, which is exactly the situation in inverse problems. Picture walking down a narrow zigzag valley: steepest descent bounces from wall to wall, making progress only at the narrow angle between bounces.

Conjugate gradients fix this. Instead of bouncing, CG uses information from previous steps to carve a direct path through the valley. It actively avoids redundant search directions, which is why it converges so much faster for the deep ill-conditioning typical of inverse problems.

Other workhorses:

* **L-BFGS:** approximates curvature using a limited memory of past gradients.
* **Truncated Newton methods:** solve the Newton system approximately using a few CG iterations. Excellent for nonlinear problems.

The common thread: none of these need the full Hessian matrix. They all work with matrix-vector products, which means they scale to the problems that matter.

[[simulation l-curve-construction]]

[[simulation conjugate-gradient-race]]

**Early stopping is itself regularization.** Here's the punchline of this whole lesson. Iterative methods typically fit the large-scale features first and the noise last. If you stop before full convergence, you get a model that captures real structure and leaves the noise behind. That's regularization by another name — no penalty term needed. Stopping iteration $k$ plays the same role as $\epsilon$ in Tikhonov.

---

## Why the Penalty Term Is Physics, Not Math

It's tempting to think of the regularization term $\epsilon^2\|\mathbf{m}\|^2$ as a mathematical trick — something we added to make the numerics work. But that misses the point entirely.

The penalty term encodes **real physical beliefs** about the world:

* **$\|\mathbf{m}\|^2$ (Tikhonov):** the model should have bounded energy. Don't put structure where you don't need it.
* **$\|\nabla \mathbf{m}\|^2$ (smoothness):** neighboring parameters should be similar. The Earth doesn't change density by a factor of ten from one meter to the next.
* **$\|\mathbf{m}\|_1$ (sparsity):** most parameters should be zero. The model is simple, with a few localized features.

Each choice tells the inversion something different about what "reasonable" looks like. Notice what we're really doing: encoding our physical intuition as mathematics. The penalty term is where domain knowledge enters the computation. Get it right, and you extract signal. Get it wrong, and you hallucinate structure or miss it entirely.

---

## When Iterative Methods Shine

Use the closed-form solution when you can. Use iterative methods when you must — and you must when:

* The model has more than ~$10^4$ parameters
* $\mathbf{G}$ is only available as a function (you can compute $\mathbf{G}\mathbf{v}$ but never write down $\mathbf{G}$ explicitly)
* The forward model is nonlinear (you linearize at each step and solve iteratively)
* You want to monitor convergence and stop early as an implicit regularization strategy

---

So here's the thread: the bottleneck in large-scale inversion is never the physics — it's the cost of building and inverting a dense matrix. Iterative methods dissolve that bottleneck by working only with matrix-vector products. The choice of penalty is a scientific statement, not a numerical decision. And early stopping is regularization by another name — fit the big features first, stop before the noise creeps in, and walk away with a better model than you'd get by running to convergence.

## What Comes Next

With iterative methods in hand, you can solve the Tikhonov problem at any scale. But both the closed-form formula and the iterative solver share the same limitation: they recover a single model — the MAP estimate. If the posterior has multiple modes, or if parameter trade-offs create curved ridges in the probability landscape, a point estimate misses the story entirely.

Linear tomography shows what a complete inversion workflow looks like in practice, from constructing the sensitivity matrix to choosing regularization to diagnosing resolution with checkerboard tests. It is the bridge between the abstract machinery developed so far and the concrete geometry of a real imaging problem.

## Let's Make Sure You Really Got It
1. Why does the steepest descent algorithm tend to zigzag in narrow, elongated valleys, and how do conjugate gradients avoid this behavior?
2. Explain why early stopping in iterative optimization acts as a form of regularization. What property of iterative methods makes this work — and what does it imply about the order in which the algorithm recovers information from the data?
3. For a problem where $\mathbf{G}$ is only available as a function (you can compute $\mathbf{G}\mathbf{v}$ but never write $\mathbf{G}$ explicitly), which of the methods discussed here still apply, and why?
