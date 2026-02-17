# Online and Reinforcement Learning

## Course overview

Online learning and reinforcement learning study **sequential decision making**.
Unlike classical offline machine learning, data are not i.i.d. and the learner can change future data through its actions.

- Offline ML: fixed dataset, i.i.d. assumptions, generalization error.
- Online/RL: repeated interaction, potentially non-stationary or adversarial feedback, continual adaptation.

Core interaction loop:

1. Observe state or context.
2. Choose an action.
3. Receive feedback (reward/loss, full or partial).
4. Update the decision rule.

[[figure mdp-agent-environment-loop]]

[[figure orl-cartpole-learning-image]]

## Why this topic matters

- Adversarial games and planning (for example, chess-like settings).
- Repeated investment and portfolio decisions.
- Spam filtering and security screening under adaptive opponents.
- Online advertising and recommendation systems.
- Routing and control in networks and robotics.
- Sequential medical decision support.

## Key mathematical ideas

- Regret minimization and no-regret learning.
- Online convex optimization and mirror-style updates.
- Bandit feedback and exploration-exploitation trade-offs.
- Markov decision processes and Bellman operators.

## Prerequisites

- Probability and random variables.
- Linear algebra.
- Basic optimization and machine learning.
- Differential and integral calculus.

## Recommended reading

- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd edition).
- Cesa-Bianchi and Lugosi, *Prediction, Learning, and Games*.
- Lattimore and Szepesvari, *Bandit Algorithms*.
- Hazan, *Introduction to Online Convex Optimization*.

## Learning trajectory

This module is organized from foundational online learning concepts through bandits to full reinforcement learning:

1. **The Notion of Regret** -- the central performance metric for online learning.
2. **Forms of Feedback and Problem Settings** -- full-information, bandit, and contextual feedback models.
3. **Follow the Leader and Hedge** -- expert advice algorithms and exponential weights.
4. **Stochastic and Adversarial Bandits: UCB1 and EXP3** -- core bandit algorithms.
5. **Contextual Bandits and EXP4** -- bandits with side information.
6. **MDPs and Dynamic Programming** -- Markov decision processes and planning.
7. **Monte Carlo Methods for RL** -- model-free value estimation from episodes.
8. **Temporal-Difference Learning, SARSA, and Q-Learning** -- bootstrapping methods for control.
9. **Function Approximation and Deep Q-Learning** -- scaling RL with neural networks.
10. **Online RL in Average-Reward and Discounted Settings** -- continuing tasks and regret in RL.
11. **Assignments and Project Ideas** -- empirical and theoretical exercises.
