# MDPs and Dynamic Programming

## Markov Decision Processes

An MDP is a tuple

$$
\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle
$$

with states, actions, transition kernel, reward function, and discount factor.

Policies can be deterministic or stochastic. Value functions:

$$
V^\pi(s)=\mathbb{E}^\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s\right],\quad
Q^\pi(s,a)=\mathbb{E}^\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s,a_0=a\right].
$$

## Bellman equations

- Policy evaluation:
  $V^\pi = r^\pi + \gamma P^\pi V^\pi$.
- Optimality:
  $V^*(s)=\max_a\left[r(s,a)+\gamma\sum_{s'}P(s'|s,a)V^*(s')\right]$.

For discounted finite MDPs, Bellman operators are contractions, giving unique fixed points.

## Dynamic programming algorithms

- Policy evaluation: iterative Bellman backup.
- Policy iteration: evaluate policy then greedy improve.
- Value iteration: repeatedly apply Bellman optimality operator.

These are planning methods (model known), not model-free learning.

[[simulation gridworld-mdp]]
[[simulation dp-convergence]]
[[simulation mdp-simulation]]
