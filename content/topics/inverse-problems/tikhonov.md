# Iterative Methods and Large-Scale Tricks

The [previous lesson](./regularization) gave us the Tikhonov formula — a closed-form solution that stabilizes inversion beautifully. One matrix inversion, done.

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

Watch what happens in the simulation above:

- **Step size too large:** the iterates overshoot, oscillate wildly, or diverge entirely. The algorithm is trying to sprint down a narrow valley and keeps bouncing off the walls.
- **Step size too small:** convergence is glacially slow. You're tiptoeing toward the answer and might not get there in your lifetime.
- **Just right:** smooth convergence to the regularized solution — the same answer the closed-form gives, but obtained without ever building the full matrix.

The beautiful thing is that each iteration only requires *matrix-vector products* $\mathbf{G}\mathbf{v}$ and $\mathbf{G}^T\mathbf{v}$ — not the full matrix $\mathbf{G}^T\mathbf{G}$. For large sparse systems, this is the difference between feasible and impossible.

---

## Beyond Steepest Descent

Steepest descent works but it can be slow, especially when the problem is ill-conditioned (eigenvalues spanning many orders of magnitude — exactly the situation in inverse problems). Better options:

- **Conjugate gradients:** uses information from previous steps to avoid redundant search directions. Converges much faster for quadratic objectives.
- **L-BFGS:** approximates the curvature of the objective using a limited memory of past gradients. The workhorse of large-scale optimization.
- **Truncated Newton methods:** solve the Newton system approximately using a few CG iterations. Excellent for nonlinear problems.

The common thread: none of these need the full Hessian matrix. They all work with matrix-vector products, which means they scale to the problems that matter.

---

## Why the Penalty Term Is Physics, Not Math

It's tempting to think of the regularization term $\epsilon^2\|\mathbf{m}\|^2$ as a mathematical trick — something we added to make the numerics work. But that misses the point entirely.

The penalty term encodes **real physical beliefs** about the world:

- **$\|\mathbf{m}\|^2$ (Tikhonov):** the model should have bounded energy. Don't put structure where you don't need it.
- **$\|\nabla \mathbf{m}\|^2$ (smoothness):** neighboring parameters should be similar. The Earth doesn't change density by a factor of ten from one meter to the next.
- **$\|\mathbf{m}\|_1$ (sparsity):** most parameters should be zero. The model is simple, with a few localized features.

Each choice tells the inversion something different about what "reasonable" looks like. This mirrors how coarse-grid parameterizations work in climate and Earth-system models — the grid resolution itself imposes a smoothness assumption.

[[figure climate-grid]]

Notice what we're really doing: encoding our physical intuition as mathematics. The penalty term is where domain knowledge enters the computation. Get it right, and you extract signal. Get it wrong, and you hallucinate structure or miss it entirely.

---

## When Iterative Methods Shine

Use the closed-form solution when you can. Use iterative methods when you must — and you must when:

- The model has more than ~$10^4$ parameters
- $\mathbf{G}$ is only available as a function (you can compute $\mathbf{G}\mathbf{v}$ but never write down $\mathbf{G}$ explicitly)
- The forward model is nonlinear (you linearize at each step and solve iteratively)
- You want to monitor convergence and stop early as an implicit regularization strategy

Early stopping is itself a form of regularization: iterative methods typically fit the large-scale features first and the noise last. Stopping before full convergence can give you a better model than running to completion.

---

## Takeaway

When the Tikhonov formula is too expensive to compute directly, iterative optimization gives you the same answer — or a better one — using only matrix-vector products. The regularization term isn't just numerical medicine; it's where your physical knowledge of the problem lives.

For the probabilistic interpretation of these choices — why the penalty is really a prior — see [Bayesian Inversion](./bayesian-inversion).

---

## Further Reading

Nocedal & Wright's *Numerical Optimization* is the standard reference for iterative methods. For the inverse-problems angle, see Vogel's *Computational Methods for Inverse Problems*. But the best way to understand this is to play with the steepest descent simulation above and watch how the step size and regularization interact.
