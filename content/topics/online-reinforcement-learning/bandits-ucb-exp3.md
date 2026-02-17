# Stochastic and Adversarial Bandits: UCB1 and EXP3

Assume $K$ arms and horizon $T$.

## UCB1 (stochastic bandits)

For each arm $a$, let $\widehat{\mu}_a(t)$ be empirical mean and $n_a(t)$ pull count.
UCB1 chooses

$$
a_t \in \arg\max_a \left[\widehat{\mu}_a(t) + \sqrt{\frac{2\log t}{n_a(t)}}\right].
$$

Interpretation: optimism under uncertainty.

- Gap-dependent regret: $O\!\left(\sum_{a:\Delta_a>0}\frac{\log T}{\Delta_a}\right)$.
- Gap-free regret: $O(\sqrt{KT\log T})$.

## EXP3 (adversarial bandits)

EXP3 adapts exponential weights to partial feedback.

- Maintain weights $w_t(i)$.
- Build exploration-smoothed sampling distribution $p_t$.
- Observe only chosen loss/reward.
- Use importance-weighted estimate to update all weights.

Typical regret rate:

$$
R_T = O(\sqrt{KT\log K}).
$$

Lower bounds are $\Omega(\sqrt{KT})$ in adversarial bandits, so this is near-optimal up to log factors.

[[simulation multi-armed-bandit]]
[[simulation bandit-regret-comparison]]
[[simulation bernoulli-trials]]
