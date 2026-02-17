# Temporal-Difference Learning, SARSA, and Q-Learning

TD methods bootstrap from current estimates instead of waiting for episode termination.

## TD(0) state-value update

$$
V(s_t)\leftarrow V(s_t)+\alpha\left[r_{t+1}+\gamma V(s_{t+1})-V(s_t)\right].
$$

The bracketed term is the TD error.

## SARSA (on-policy control)

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\right].
$$

SARSA learns the value of the behavior policy (often epsilon-greedy).

## Q-Learning (off-policy control)

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_{t+1}+\gamma \max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\right].
$$

With sufficient exploration and suitable step sizes in tabular settings, Q-learning converges to $Q^*$.

## Exploration

- Epsilon-greedy.
- Softmax/Boltzmann exploration.
- Optimism and UCB-like bonuses.

[[simulation sarsa-vs-qlearning]]
[[simulation concentration-bounds]]
[[simulation stochastic-approximation]]
