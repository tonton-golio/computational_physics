# Follow the Leader and Hedge

## Follow the Leader (FTL)

FTL picks the action minimizing cumulative past loss:

$$
a_t \in \arg\min_{a \in \mathcal{A}} \sum_{s=1}^{t-1}\ell_s(a).
$$

Strengths:

- Simple and parameter-free.
- Works well for strongly convex or stable losses.

Weakness:

- Can oscillate badly in adversarial sequences, causing linear regret.

[[simulation ftl-instability]]

## Hedge / Exponential Weights

Setting: $N$ experts, full-information losses $\ell_t(i)\in[0,1]$.

Initialize weights $w_1(i)=1$ and choose

$$
p_t(i)=\frac{w_t(i)}{\sum_j w_t(j)}.
$$

Update after observing losses:

$$
w_{t+1}(i)=w_t(i)\exp(-\eta \ell_t(i)).
$$

With $\eta \asymp \sqrt{\log N / T}$, regret scales as

$$
R_T = O(\sqrt{T\log N}).
$$

Proof idea: potential $\Phi_t=\sum_i w_t(i)$, multiplicative update bounds, and second-order control.

[[simulation hedge-weights-regret]]
