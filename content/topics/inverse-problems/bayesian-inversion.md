# Bayesian Inversion

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

where the prior $\rho_m$ is a zero-mean Gaussian with covariance proportional to $\epsilon^{-2}\mathbf{I}$, and the likelihood $L$ assumes Gaussian noise on the data.

What does that mean in plain language? The prior says: "I expect the model parameters to be small — close to zero — and I'm about $1/\epsilon$ uncertain about each one." Large $\epsilon$ means a tight prior (you're very confident the model is small). Small $\epsilon$ means a loose prior (you're letting the data do the talking).

[[figure gaussian-process]]

Look at this picture carefully. Each colored line is a sample from the prior distribution — a possible model *before* seeing any data. The prior is telling the model: "I expect you to be smooth, like a gentle hillside, not like the Alps on a bad day." The spread of the lines shows your uncertainty. When you combine this prior with data, the posterior (shaded region) shrinks — the data has taught you something.

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

## From Point Estimates to Posterior Exploration

So far, we've been finding the *peak* of the posterior — the single most probable model. That's useful, but it's not the whole story.

Think about it. The Earth doesn't care which of ten different fault geometries you pick — they all wiggle the surface the same way. So the answer isn't a single line on a map; it's a whole cloud of possible lines. That cloud *is* the real answer.

When the posterior is complex — multimodal (several distinct peaks), asymmetric (skewed to one side), or high-dimensional (hundreds of parameters) — the point estimate misses crucial structure. You need to explore the full distribution.

How much did the data actually narrow down the possibilities? That's something we can quantify with information theory — the tools in the [information and entropy lesson](./information-entropy). But to actually *explore* the posterior, we need sampling methods.

This is exactly what [Monte Carlo methods](./monte-carlo-methods) give us: instead of finding one best answer, we generate thousands of plausible models, weighted by how well they explain the data. The spread of those models tells you what you know and what you don't.

---

## Takeaway

The Bayesian viewpoint unifies regularization, uncertainty quantification, and model comparison into a single framework. Regularization is not a trick — it is the deterministic shadow of a probabilistic prior. And the posterior distribution, not the point estimate, is the honest answer to an inverse problem.

---

## Further Reading

Tarantola's *Inverse Problem Theory* develops the probabilistic viewpoint with real physical intuition. Kaipio & Somersalo's *Statistical and Computational Inverse Problems* is more mathematical but very thorough. Stuart's *Inverse Problems: A Bayesian Perspective* (Acta Numerica, 2010) is an excellent survey. But first, make sure the connection between $\epsilon$ and the prior variance has really sunk in — that insight will carry you through everything that follows.
