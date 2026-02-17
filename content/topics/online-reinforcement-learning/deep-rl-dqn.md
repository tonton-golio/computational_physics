# Function Approximation and Deep Q-Learning

Tabular methods do not scale to high-dimensional or continuous state spaces.
Function approximation replaces lookup tables with parameterized models.

## From linear approximation to neural networks

- Linear value approximation is simple but limited.
- Deep neural networks enable large-scale representation learning.

## DQN essentials

- Replay buffer to decorrelate samples.
- Target network for stable bootstrapping targets.
- Gradient clipping and robust losses.
- Double-Q style targets to reduce overestimation.

## Stability challenge: the deadly triad

Instability appears when combining:

1. Function approximation.
2. Bootstrapping.
3. Off-policy training.

DQN mitigates, but does not eliminate, these risks.

## Beyond DQN (brief)

- Policy gradients.
- Actor-critic methods.
- Continuous-control variants.

[[simulation dqn-stability]]
[[simulation activation-functions]]
[[simulation cartpole-learning-curves]]
