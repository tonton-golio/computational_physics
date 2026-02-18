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

Drag the probability slider and watch entropy change. The maximum is at uniform distribution (maximum ignorance). The minimum is at a spike (you know the answer). Every inverse problem starts somewhere on this curve — and data moves you toward the minimum.

---

## KL Divergence: The Extra Surprise

Now suppose you have two distributions: $P$ (the truth) and $Q$ (your model). The **Kullback-Leibler divergence** measures how much *extra* surprise you experience by using the wrong model:

$$
D_{\mathrm{KL}}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}.
$$

Here's the intuition. If your model $Q$ matches reality $P$ perfectly, there's no extra surprise: $D_{\mathrm{KL}} = 0$. If your model assigns low probability to events that actually happen often, you're constantly caught off guard — $D_{\mathrm{KL}}$ is large.

Three things to remember:

- $D_{\mathrm{KL}} = 0$ *only* when the distributions match exactly
- It's always non-negative (you can't be *less* surprised by using the wrong model)
- It's **asymmetric**: $D_{\mathrm{KL}}(P \| Q) \neq D_{\mathrm{KL}}(Q \| P)$. Being wrong about likely events costs more than being wrong about rare ones.

[[simulation kl-divergence]]

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
2. **Regularization**: stabilize by controlling model complexity — equivalently, by limiting how much "information" (real + noise) you extract
3. **Bayesian framework**: the prior sets the entropy floor, the data reduces it
4. **Monte Carlo**: explores the posterior, whose entropy tells you what you actually know
5. **Information theory** (here): quantifies all of this precisely

Entropy is not a footnote. It's the thread that runs through the entire course.

---

## Takeaway

Information-theoretic tools — entropy, KL divergence, mutual information — quantify what the data taught you and what remains unknown. They explain *why* regularization works (it controls information extraction), *how much* the data helped (KL divergence between posterior and prior), and *whether* your sampling explored the full answer (posterior entropy). They provide the deepest justification for everything we've done.

---

## Further Reading

Shannon's original 1948 paper is surprisingly readable and still worth your time. Cover & Thomas's *Elements of Information Theory* is the standard textbook. For the inverse-problems connection, see Tarantola's chapter on information and resolution, or Mackay's *Information Theory, Inference, and Learning Algorithms* which beautifully bridges the gap. But honestly, play with the entropy demo first — watch how moving a single probability changes the whole landscape of uncertainty.
