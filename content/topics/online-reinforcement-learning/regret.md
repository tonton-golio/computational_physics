# The Notion of Regret

Regret is the primary performance metric in online learning.
It compares the learner's cumulative loss to a benchmark policy, often the best fixed action in hindsight.

For losses, external regret over horizon $T$ is

$$
R_T = \sum_{t=1}^{T} \ell_t(a_t) - \min_{a \in \mathcal{A}} \sum_{t=1}^{T} \ell_t(a).
$$

Sublinear regret, $R_T = o(T)$, implies average regret $R_T/T \to 0$.

## Why regret replaces generalization error

- No i.i.d. requirement.
- Works under adversarial sequences.
- Naturally handles sequential adaptation.
- Provides minimax comparisons and lower bounds.

## Variants

- **Expected regret**: $\mathbb{E}[R_T]$.
- **High-probability regret**: bounds that hold with probability at least $1-\delta$.
- **Pseudo-regret** (stochastic settings): compares against the best expected arm/policy.
- **External regret** vs **internal/swap regret**.

## Adversarial vs stochastic interpretation

- Stochastic bandits: losses/rewards sampled from fixed distributions.
- Adversarial online learning: losses can be arbitrary or adaptive.
- Minimax lower bounds differ by feedback model:
  - Full information: typically $\Omega(\sqrt{T \log N})$ scale.
  - Bandit feedback: typically $\Omega(\sqrt{KT})$ scale.

## Practical examples

- Portfolio learning compared with the best stock in hindsight.
- Expert aggregation compared with the best expert sequence baseline.

[[simulation regret-growth-comparison]]
