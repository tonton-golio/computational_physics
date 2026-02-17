# Online RL in Average-Reward and Discounted Settings

Discounted RL optimizes

$$
\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right].
$$

Average-reward RL targets long-run gain

$$
g^\pi(s)=\lim_{T\to\infty}\frac{1}{T}\mathbb{E}\left[\sum_{t=1}^{T}r_t \mid s_1=s\right].
$$

## Gain and bias

- **Gain** captures steady-state average reward.
- **Bias** captures transient advantage before steady state.

This decomposition is central in continuing tasks without episodic reset.

## Representative algorithmic ideas

- Relative value iteration.
- Differential TD / differential Q-learning.
- R-learning style methods.
- Optimism under uncertainty in model-based online RL.

## Regret in RL

For online RL, regret compares cumulative reward to the optimal policy in the same environment class.
Modern methods include optimistic and posterior-sampling approaches (for example, UCRL- and PSRL-style methods).

[[simulation average-reward-vs-discounted]]
