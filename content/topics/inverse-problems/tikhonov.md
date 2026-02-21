# Iterative Methods and Large-Scale Tricks

We now know what [regularization](./regularization) does and *why* it works — it's a Gaussian prior ([Bayesian inversion](./bayesian-inversion)). The Tikhonov formula gives a closed-form MAP estimate. One matrix inversion, done.

But here's the catch. That formula requires building $\mathbf{G}^T\mathbf{G} + \epsilon^2\mathbf{I}$ and inverting it. If your model has 100 parameters, no problem. If it has a million parameters — a 3D seismic tomography model, a full-waveform inversion grid, a climate reanalysis — that matrix has $10^{12}$ entries. You can't even store it, let alone invert it.

So we need a different strategy: instead of solving the problem in one shot, we *walk* toward the answer, one step at a time. That's iterative optimization.

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

Things to look for in the simulation:

* Try extreme learning rates — watch the iterates overshoot and oscillate (too large) or crawl (too small)
* Find the Goldilocks step size where the path curves smoothly toward the minimum
* Notice how the path zigzags in narrow valleys — that's why conjugate gradients and L-BFGS exist

Watch what happens in the simulation above:

* **Step size too large:** the iterates overshoot, oscillate wildly, or diverge entirely. The algorithm is trying to sprint down a narrow valley and keeps bouncing off the walls.
* **Step size too small:** convergence is glacially slow. You're tiptoeing toward the answer and might not get there in your lifetime.
* **Just right:** smooth convergence to the regularized solution — the same answer the closed-form gives, but obtained without ever building the full matrix.

The beautiful thing is that each iteration only requires *matrix-vector products* $\mathbf{G}\mathbf{v}$ and $\mathbf{G}^T\mathbf{v}$ — not the full matrix $\mathbf{G}^T\mathbf{G}$. For large sparse systems, this is the difference between feasible and impossible.

---

## Beyond Steepest Descent

Steepest descent works but it can be slow, especially when the problem is ill-conditioned (eigenvalues spanning many orders of magnitude — exactly the situation in inverse problems). Better options:

* **Conjugate gradients:** uses information from previous steps to avoid redundant search directions. Converges much faster for quadratic objectives.
* **L-BFGS:** approximates the curvature of the objective using a limited memory of past gradients. The workhorse of large-scale optimization.
* **Truncated Newton methods:** solve the Newton system approximately using a few CG iterations. Excellent for nonlinear problems.

The common thread: none of these need the full Hessian matrix. They all work with matrix-vector products, which means they scale to the problems that matter.

[[simulation l-curve-construction]]

[[simulation conjugate-gradient-race]]

Things to look for in the simulation:

* CG converges in exactly 2 steps for any 2D quadratic — regardless of condition number
* Crank the condition number up and watch SD zigzag wildly while CG cuts straight through
* Check the cost-vs-iteration chart: SD shows linear convergence (straight line on log scale), CG shows superlinear (drops to machine zero in 2 steps)
* Try different start positions — CG's 2-step convergence is universal for 2D quadratics

---

## Why the Penalty Term Is Physics, Not Math

It's tempting to think of the regularization term $\epsilon^2\|\mathbf{m}\|^2$ as a mathematical trick — something we added to make the numerics work. But that misses the point entirely.

The penalty term encodes **real physical beliefs** about the world:

* **$\|\mathbf{m}\|^2$ (Tikhonov):** the model should have bounded energy. Don't put structure where you don't need it.
* **$\|\nabla \mathbf{m}\|^2$ (smoothness):** neighboring parameters should be similar. The Earth doesn't change density by a factor of ten from one meter to the next.
* **$\|\mathbf{m}\|_1$ (sparsity):** most parameters should be zero. The model is simple, with a few localized features.

Each choice tells the inversion something different about what "reasonable" looks like. This mirrors how coarse-grid parameterizations work in climate and Earth-system models — the grid resolution itself imposes a smoothness assumption.

Notice what we're really doing: encoding our physical intuition as mathematics. The penalty term is where domain knowledge enters the computation. Get it right, and you extract signal. Get it wrong, and you hallucinate structure or miss it entirely.

---

## When Iterative Methods Shine

Use the closed-form solution when you can. Use iterative methods when you must — and you must when:

* The model has more than ~$10^4$ parameters
* $\mathbf{G}$ is only available as a function (you can compute $\mathbf{G}\mathbf{v}$ but never write down $\mathbf{G}$ explicitly)
* The forward model is nonlinear (you linearize at each step and solve iteratively)
* You want to monitor convergence and stop early as an implicit regularization strategy

Early stopping is itself a form of regularization: iterative methods typically fit the large-scale features first and the noise last. Stopping before full convergence can give you a better model than running to completion.

---

## Big Ideas
* The bottleneck in large-scale inversion is never the physics — it is the cost of building and inverting a dense matrix. Iterative methods dissolve this bottleneck by working only with matrix-vector products.
* The choice of penalty — $\|\mathbf{m}\|^2$ versus $\|\nabla\mathbf{m}\|^2$ versus $\|\mathbf{m}\|_1$ — is not a numerical decision. It is a scientific statement about what physically plausible models look like.
* Early stopping is regularization by another name: iterative methods fit large-scale features first and noise last, so stopping before convergence gives you smoother, more stable models.
* Conjugate gradients don't just converge faster than steepest descent — they actively avoid redundant search directions, which is the key to handling the deep ill-conditioning typical of inverse problems.

## What Comes Next

With iterative methods in hand, you can solve the Tikhonov problem at any scale. But both the closed-form formula and the iterative solver share the same limitation: they recover a single model — the MAP estimate. If the posterior has multiple modes, or if parameter trade-offs create curved ridges in the probability landscape, a point estimate misses the story entirely.

Linear tomography shows what a complete inversion workflow looks like in practice, from constructing the sensitivity matrix to choosing regularization to diagnosing resolution with checkerboard tests. It is the bridge between the abstract machinery developed so far and the concrete geometry of a real imaging problem.

## Check Your Understanding
1. Why does the steepest descent algorithm tend to zigzag in narrow, elongated valleys, and how do conjugate gradients avoid this behavior?
2. Explain why early stopping in iterative optimization acts as a form of regularization. What property of iterative methods makes this work — and what does it imply about the order in which the algorithm recovers information from the data?
3. For a problem where $\mathbf{G}$ is only available as a function (you can compute $\mathbf{G}\mathbf{v}$ but never write $\mathbf{G}$ explicitly), which of the methods discussed here still apply, and why?

## Challenge

Implement the conjugate gradient method for the Tikhonov system $(\mathbf{G}^T\mathbf{G} + \epsilon^2\mathbf{I})\mathbf{m} = \mathbf{G}^T\mathbf{d}$ without ever forming the matrix $\mathbf{G}^T\mathbf{G}$ explicitly. Use only matrix-vector products with $\mathbf{G}$ and $\mathbf{G}^T$. Then design an experiment: for a fixed ill-conditioned $\mathbf{G}$, compare the recovered model at iteration $k$ (early stopping) with the Tikhonov solution at various $\epsilon$ values. Can you find a mapping between stopping iteration and effective regularization strength? Does the relationship depend on the condition number of $\mathbf{G}$?

