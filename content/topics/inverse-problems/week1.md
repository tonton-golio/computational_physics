# Week 1 - Information, Entropy, and Uncertainty

Inverse problems are not only about fitting curves.
They are about deciding which model parameters are actually supported by the data.

---

## Information as Uncertainty Reduction

If $X$ is a random variable with probabilities $p_i$, the **Shannon entropy** is

$$
H(X)=-\sum_i p_i\log p_i.
$$

High entropy means high uncertainty.
Low entropy means observations are informative and concentrated.

[[figure claude-shannon]]

[[simulation entropy-demo]]

---

## Comparing Candidate Distributions

When we compare two probability models $P$ and $Q$, a standard choice is the **Kullback-Leibler divergence**:

$$
D_{\mathrm{KL}}(P\|Q)=\sum_i P(i)\log\frac{P(i)}{Q(i)}.
$$

Interpretation:

- $D_{\mathrm{KL}}=0$ only when the distributions match
- Larger values indicate larger mismatch
- It is asymmetric, so direction matters

[[simulation kl-divergence]]

---

## Why This Matters for Inversion

In inversion, we often compare:

- observed data distribution vs model-predicted distribution
- prior parameter belief vs posterior parameter belief

These comparisons guide model selection and regularization strength.

A practical takeaway is simple: **fit quality is not enough**.
You also want a model that is informative, stable, and physically plausible.

---

## Week 1 Takeaway

Information-theoretic tools quantify whether your model explains data efficiently.
In later weeks, we combine this idea with regularized optimization and uncertainty-aware inference.
