# Information, Entropy, and Uncertainty

In 1948, Claude Shannon asked a deceptively simple question: **how much does one message reduce my uncertainty?** He was thinking about telegraph wires and telephone cables, but his answer turned out to be universal. It applies to everything from compression algorithms to DNA to — you guessed it — inverse problems.

Here's the connection. You start an inverse problem with uncertainty about the model (the prior). You collect data. After inversion, you have less uncertainty (the posterior). Shannon's framework lets you *measure* exactly how much the data taught you — in bits, in nats, in precise mathematical units.

That's what this lesson is about: quantifying information. And it turns out to be the deepest reason we regularize.

---

## Shannon Entropy: Measuring Uncertainty

If $X$ is a random variable taking values with probabilities $p_i$, the **Shannon entropy** is:

$$
H(X) = -\sum_i p_i \log p_i.
$$

What does this mean? Think of it as the average surprise. If one outcome is nearly certain ($p_1 \approx 1$), there's almost no surprise when it happens — entropy is low. If all outcomes are equally likely, every observation is maximally surprising — entropy is high.

[[figure claude-shannon]]

A fair coin: $H = \log 2 \approx 0.693$ nats. A loaded coin (99% heads): $H \approx 0.056$ nats. The loaded coin is boring — you almost always know what's coming. The fair coin keeps you guessing.

[[simulation entropy-demo]]

Things to look for in the simulation:

* Drag the distribution toward uniform — entropy climbs to its maximum (maximum ignorance)
* Concentrate probability on a single outcome — entropy drops toward zero (you know the answer)
* Every inverse problem starts somewhere on this curve, and data moves you toward the minimum

Drag the probability slider and watch entropy change. The maximum is at uniform distribution (maximum ignorance). The minimum is at a spike (you know the answer). Every inverse problem starts somewhere on this curve — and data moves you toward the minimum.

---

## KL Divergence: The Extra Surprise

Now suppose you have two distributions: $P$ (the truth) and $Q$ (your model). The **Kullback-Leibler divergence** measures how much *extra* surprise you experience by using the wrong model:

$$
D_{\mathrm{KL}}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}.
$$

Think of $D_{\mathrm{KL}}(P \| Q)$ as the "extra surprise tax" you pay every time you use model $Q$ when reality follows $P$. Using the wrong model always costs you — never the other way around.

If your model $Q$ matches reality $P$ perfectly, there's no extra surprise: $D_{\mathrm{KL}} = 0$. If your model assigns low probability to events that actually happen often, you're constantly caught off guard — $D_{\mathrm{KL}}$ is large.

Three things to remember:

* $D_{\mathrm{KL}} = 0$ *only* when the distributions match exactly
* It's always non-negative (you can't be *less* surprised by using the wrong model)
* It's **asymmetric**: $D_{\mathrm{KL}}(P \| Q) \neq D_{\mathrm{KL}}(Q \| P)$. Being wrong about likely events costs more than being wrong about rare ones.

[[simulation kl-divergence]]

Things to look for in the simulation:

* Make the two distributions identical and verify $D_{\mathrm{KL}} = 0$
* Shift one distribution's mean and watch the divergence grow — the "surprise tax" increases
* Swap $P$ and $Q$ and notice the asymmetry: $D_{\mathrm{KL}}(P \| Q) \neq D_{\mathrm{KL}}(Q \| P)$

---

## Why This Matters for Inverse Problems

These aren't abstract mathematical toys. They connect directly to every concept in this course.

### Measuring what the data taught you

Remember the [Bayesian framework](./bayesian-inversion)? You start with a prior, collect data, and get a posterior. The KL divergence between posterior and prior —

$$
D_{\mathrm{KL}}(\text{posterior} \| \text{prior})
$$

— measures exactly how much the data changed your beliefs. Large value: the data was highly informative, you learned a lot. Small value: the data told you nothing you didn't already know. This is the quantitative answer to "was the experiment worth running?"

### Why we regularize (the deep reason)

Here's something most textbooks don't tell you. Regularization is about controlling how much information you extract from the data.

With no regularization ($\epsilon \approx 0$), you're extracting *everything* from the data — including the noise. The inferred model has low entropy relative to the prior (you've committed to very specific parameter values), but much of that "information" is actually noise. You've fooled yourself into thinking you know more than you do.

With too much regularization ($\epsilon$ huge), you're ignoring the data entirely. The posterior is basically the prior. Entropy stays high — you haven't learned anything.

The right regularization extracts the signal and leaves the noise behind. Information theory gives you a principled way to find that balance: maximize the genuine information gain while penalizing overfitting.

### Model comparison and selection

When two different models both fit the data, which is better? Residuals alone can't answer this — a more complex model will always fit better. KL divergence provides a principled comparison: which model's predicted distribution is closest to the observed data distribution? This is the information-theoretic foundation of model selection criteria like AIC and BIC.

### Monte Carlo diagnostics

When running [MCMC](./monte-carlo-methods), the entropy of the sampled posterior tells you whether the chain has explored the full range of plausible models. If the sampled entropy is much lower than expected, your chain might be stuck in one region. If it matches the theoretical posterior entropy, you can be more confident your samples represent the true posterior.

---

## The Big Picture

Step back and look at the entire course through this lens:

1. **Foundations**: inverse problems are ill-posed — entropy of the naive solution is all wrong
2. **[Regularization](./regularization)**: stabilize by controlling model complexity — limiting how much "information" (real + noise) you extract
3. **[Bayesian framework](./bayesian-inversion)**: the prior sets the entropy floor, the data reduces it
4. **[Iterative methods](./tikhonov)**: find the MAP efficiently when the problem is huge
5. **[Tomography](./linear-tomography)**: the complete linear workflow
6. **[Monte Carlo](./monte-carlo-methods)**: explores the posterior, whose entropy tells you what you actually know
7. **[Geophysical examples](./geophysical-inversion)**: the answer is always a distribution
8. **Information theory** (here): quantifies all of this precisely

Entropy is not a footnote. It's the thread that runs through the entire course.

---

## Big Ideas
* Entropy measures ignorance — not mystery, not complexity, but precisely how spread-out your probability distribution is. More spread means more ignorance, and the uniform distribution is total ignorance.
* KL divergence is the surprise tax you pay for using the wrong model. It is always non-negative, it is zero only when your model matches reality, and its asymmetry is not a defect — it reflects the real asymmetry between truth and approximation.
* Regularization is information control: too little regularization extracts noise as if it were signal; too much leaves the prior unchanged. The right amount extracts exactly what the data knows.
* The answer to an inverse problem is always a distribution, not a number — and entropy is the one quantity that tells you how wide that distribution really is.

## What Comes Next

This is the end of the arc. Start from a broken, ill-posed problem — one where direct inversion amplifies noise into nonsense. Regularize to stabilize it, and you have made an implicit prior. Make the prior explicit through the Bayesian framework, and you have a posterior distribution. When the posterior is too complex or too large for closed-form solutions, you walk toward it iteratively or sample from it with Markov chains. Apply all of this to linear imaging problems and nonlinear geophysical field cases, and you find that uncertainty is not an embarrassment to be swept under the rug — it is a precise, quantifiable object. Information theory closes the circle: entropy and KL divergence give you the vocabulary to ask, and answer, how much any experiment can possibly teach you.

The tools here — regularization, Bayesian inference, iterative optimization, tomography, Monte Carlo sampling, information theory — are not a menu of separate techniques. They are facets of a single idea: science is the process of updating beliefs in proportion to evidence. Everything in this course is a rigorous way to do that.

## Check Your Understanding
1. A fair six-sided die has Shannon entropy $H = \log 6$. A loaded die that always lands on 4 has $H = 0$. What is the entropy of a die that lands on 4 with probability 0.9 and on each other face with probability 0.02? Is this closer to the fair or the loaded die, and does that match your intuition?
2. Explain in words why $D_{\mathrm{KL}}(P \| Q) \neq D_{\mathrm{KL}}(Q \| P)$. Give a concrete example of two distributions where the asymmetry is large and explain what makes it so.
3. If the KL divergence from posterior to prior is nearly zero after running an inversion, what does this tell you about the experiment? List two different causes that could produce this result and explain how you would distinguish between them.

## Challenge

Design an "experiment value" calculator for a simple linear inverse problem $\mathbf{d} = \mathbf{Gm} + \boldsymbol{\eta}$. Before collecting data, use the prior $p(\mathbf{m})$ and the noise model to compute the expected KL divergence from the posterior to the prior — this is the expected information gain. Implement this for a 1D deconvolution problem as a function of the number of observations $N$ and the noise level $\sigma$. Plot expected information gain vs. $N$ and vs. $\sigma$. At what point does adding more data yield diminishing returns — and how does this threshold shift as you change the regularization strength? Does the optimal regularization (in the sense of maximizing genuine information gain while penalizing overfitting) correspond to the L-curve corner from the very first lesson?

