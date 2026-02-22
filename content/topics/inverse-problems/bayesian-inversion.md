# Bayesian Inversion — Why Regularization Works

Here's a question that might have been nagging you. In the [regularization lesson](./regularization), we added a penalty term $\epsilon^2\|\mathbf{m}\|^2$ and said "this keeps the model from going crazy." But *why that particular penalty*? Why not something else? And how do we know if $\epsilon$ is too big or too small?

The beautiful thing is that there's an answer — and it comes from probability. Every regularization choice has a precise probabilistic meaning. The penalty is a **prior belief** about the model. The data misfit is a **likelihood**. And the regularized solution is the **most probable model** given both your beliefs and your data.

Once you see this, regularization stops feeling like a hack and starts feeling like honest science.

---

## Regularization as a Gaussian Prior

The Tikhonov objective

$$
E_\epsilon(\mathbf{m}) = \|\mathbf{d} - \mathbf{Gm}\|^2 + \epsilon^2\|\mathbf{m}\|^2
$$

is equivalent to maximizing the posterior probability

$$
\sigma(\mathbf{m}) = \rho_m(\mathbf{m})\,L(\mathbf{m}), \qquad L(\mathbf{m}) = \rho_d(g(\mathbf{m}))
$$

where the prior $\rho_m$ is a zero-mean Gaussian with covariance proportional to $\epsilon^{-2}\mathbf{I}$, and the likelihood $L$ assumes Gaussian noise on the data. (Important caveat: this equivalence holds specifically under Gaussian noise and a quadratic prior. Non-Gaussian noise or non-quadratic penalties lead to different posterior shapes — but the core insight that regularization encodes prior belief remains universal.)

> Minimizing the Tikhonov objective is mathematically identical to finding the mode of a Gaussian posterior. **Regularization is a prior in disguise.**

What does that mean in plain language? The prior says: "I expect the model parameters to be small — close to zero — and I'm about $1/\epsilon$ uncertain about each one." Large $\epsilon$ means a tight prior (you're very confident the model is small). Small $\epsilon$ means a loose prior (you're letting the data do the talking).

Each colored line in the prior distribution represents a possible model *before* seeing any data. The prior is telling the model: "I expect you to be smooth, like a gentle hillside, not like the Alps on a bad day." The spread of the lines shows your uncertainty. When you combine this prior with data, the posterior (shaded region) shrinks — the data has taught you something.

[[simulation prior-likelihood-posterior]]

[[simulation posterior-walker-arena]]

---

## Weighted Formulation: Data and Model Covariance

The basic Tikhonov formulation assumes all data points have equal uncertainty and the model is isotropically smooth. In the real world? Never.

Some seismometers are more precise than others. Some model parameters are better constrained by geology than others. To handle this, introduce **data covariance** $\mathbf{C}_D$ and **model covariance** $\mathbf{C}_M$:

$$
\mathbf{V}^T\mathbf{V} = \mathbf{C}_D^{-1}, \qquad \mathbf{W}^T\mathbf{W} = \mathbf{C}_M^{-1}.
$$

Transform to "whitened" variables:

$$
\bar{\mathbf{d}} = \mathbf{Vd}, \qquad \bar{\mathbf{m}} = \mathbf{Wm}, \qquad \bar{\mathbf{G}} = \mathbf{VGW}^{-1}.
$$

Now solve the standard Tikhonov problem in these new variables. The effect: noisy data points get downweighted automatically. Model parameters with strong prior constraints are penalized more heavily.

Notice what we just did — we turned our beliefs about noise and about geology into two simple matrices. That's the whole game of science: turn intuition into numbers, then let the numbers argue with the data.

---

## The Linear-Gaussian Closed Form

When both the prior and the noise are Gaussian and the forward model is linear, the posterior is also Gaussian — and we can write it down exactly. No sampling, no optimization, just algebra.

When everything is Gaussian, the MAP estimate and the posterior mean are the same number, and the covariance has a beautiful closed form — but the real point is this: regularization is just a prior in disguise. Set $\mathbf{C}_M = \epsilon^{-2}\mathbf{I}$ and $\mathbf{C}_D = \mathbf{I}$, and you recover the Tikhonov formula from the [regularization lesson](./regularization). The "regularization parameter" $\epsilon$ was the ratio of noise standard deviation to prior standard deviation all along.

And here's the gorgeous part: the posterior covariance depends on $\mathbf{G}$, $\mathbf{C}_M$, and $\mathbf{C}_D$, but **not on the data** $\mathbf{d}$. The data shifts the posterior mean but doesn't change its shape. This means you can compute how uncertain your answer will be *before you collect a single measurement* — which is the foundation of experimental design. If two survey configurations give different posterior covariances, you can pick the one with smaller uncertainty and know you're making an optimal investment.

---

## From Point Estimates to Posterior Exploration

So far, we've been finding the *peak* of the posterior — the single most probable model. That's useful, but it's not the whole story.

Think about it. The Earth doesn't care which of ten different fault geometries you pick — they all wiggle the surface the same way. So the answer isn't a single line on a map; it's a whole cloud of possible lines. That cloud *is* the real answer.

When the posterior is complex — multimodal (several distinct peaks), asymmetric (skewed to one side), or high-dimensional (hundreds of parameters) — the point estimate misses crucial structure. You need to explore the full distribution.

How much did the data actually narrow down the possibilities? That's something we can quantify with information theory — the tools in the [information and entropy lesson](./information-entropy). But to actually *explore* the posterior, we need sampling methods.

This is exactly what [Monte Carlo methods](./monte-carlo-methods) give us: instead of finding one best answer, we generate thousands of plausible models, weighted by how well they explain the data. The spread of those models tells you what you know and what you don't.

But before we sample — we need to scale up. When the model has millions of parameters, the Tikhonov formula is too expensive. [Iterative Methods](./tikhonov) shows how to find the MAP estimate without ever building the full matrix.

---

So here's the takeaway: every regularization choice is secretly a prior — own that belief rather than hiding it in notation. Data covariance and model covariance encode everything you know before the inversion starts. And the MAP estimate is just one number extracted from a distribution. For linear-Gaussian problems, it's sufficient. For everything else, it's a starting point, not the answer.

## What Comes Next

Finding the peak of the posterior is tractable for Gaussian problems, but inverse problems in the real world are rarely that clean. The model might have a million parameters, making the closed-form Tikhonov matrix too large to even store in memory. The answer to this is iterative optimization: rather than solving the system in one shot, you walk toward the solution step by step, using only matrix-vector products with $\mathbf{G}$ and $\mathbf{G}^T$.

These iterative strategies carry forward the same prior encoded by the regularization term, now expressed as gradient updates rather than a matrix inverse. Understanding how to apply them efficiently — and when to stop early as a form of implicit regularization — opens the door to the large-scale inversions that appear in seismology, climate science, and medical imaging.

## Let's Make Sure You Really Got It
1. Show explicitly why minimizing the Tikhonov objective $\|\mathbf{d} - \mathbf{Gm}\|^2 + \epsilon^2\|\mathbf{m}\|^2$ is equivalent to maximizing a posterior probability with a zero-mean Gaussian prior on $\mathbf{m}$ and Gaussian noise on $\mathbf{d}$. What assumptions are required?
2. In the weighted formulation, why is it important to whiten the data before solving the standard Tikhonov problem? What goes wrong if you ignore the off-diagonal structure in $\mathbf{C}_D$?
3. The posterior distribution over a model with a million parameters cannot be visualized directly. How would you extract useful scientific conclusions from it?
