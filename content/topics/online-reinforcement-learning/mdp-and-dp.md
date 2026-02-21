# MDPs and Dynamic Programming

## The world remembers

Up to now, the world had no memory. You pulled a slot-machine lever, you got a payout, and the world reset. Your action did not change what happened next. Now the world *remembers* what you did yesterday. That changes everything.

If you are a robot navigating a warehouse, and you turn left at the first aisle, you end up in a different part of the building than if you turned right. The state of the world — your position — depends on your previous action. And the reward you get (finding the package vs hitting a wall) depends on that state. This is a **Markov Decision Process**: a world where your actions have consequences that ripple forward in time.

## The MDP tuple

An MDP is described by five things:

$$
\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle
$$

$\mathcal{S}$ is the set of states — all the places the world can be. $\mathcal{A}$ is the set of actions — all the moves you can make. $P$ is the transition kernel — for every state and action, it tells you the probability of ending up in each next state. $R$ is the reward function — what you earn for taking an action in a state. And $\gamma$ is the discount factor, a number between 0 and 1 that says how much you care about future rewards compared to immediate ones.

A **policy** $\pi$ tells you what to do in each state. It can be deterministic ("always go left in state 5") or stochastic ("go left with probability 0.7, right with probability 0.3"). The goal is to find the policy that maximizes cumulative discounted reward.

## Value functions: scoring a policy

How good is a policy $\pi$? We measure it with value functions. The state-value function $V^\pi(s)$ tells you the expected total discounted reward if you start in state $s$ and follow $\pi$ forever:

$$
V^\pi(s)=\mathbb{E}^\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s\right].
$$

The action-value function $Q^\pi(s,a)$ tells you the same thing, but conditioned on taking action $a$ first and then following $\pi$:

$$
Q^\pi(s,a)=\mathbb{E}^\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s,a_0=a\right].
$$

Think of $V^\pi(s)$ as the "worth" of being in state $s$ under policy $\pi$. A state near the goal with a good policy has high value. A state far from the goal or served by a bad policy has low value.

## The Bellman equation: the fixed point where ice meets water

The Bellman equation is the most important equation in reinforcement learning. It says: the value of a state is the immediate reward plus the discounted value of wherever you end up next.

$$
V^*(s)=\max_a\left[r(s,a)+\gamma\sum_{s'}P(s'|s,a)V^*(s')\right].
$$

Read it aloud: the optimal value of state $s$ is the best you can do by choosing the action that maximizes your immediate reward plus $\gamma$ times the expected value of the next state.

This equation has a **fixed point** — a unique solution $V^*$ that does not change when you apply the Bellman operator. Think of it like the temperature where ice and water are happy together: 0°C. If you heat the ice, it melts and comes back to 0°. If you cool the water, it freezes and comes back to 0°. The fixed point is the equilibrium. The Bellman equation works the same way — if you start with any guess for $V$ and keep applying the operator, you converge to $V^*$. The operator is a **contraction** in the discounted case: every application brings you closer to the fixed point. That guarantees convergence and uniqueness.

For policy evaluation (not optimizing, just scoring a fixed policy), the Bellman equation becomes linear: $V^\pi = r^\pi + \gamma P^\pi V^\pi$. This you can solve by matrix inversion in small problems, or by iterating the backup operator until convergence.

## Dynamic programming: planning when you know the model

When you know the transition probabilities and rewards — when you have a perfect map of the world — you can solve the MDP with dynamic programming. These are **planning** methods, not learning methods. You are doing math on the model, not interacting with the world.

**Policy evaluation** takes a fixed policy $\pi$ and computes $V^\pi$ by iteratively applying the Bellman backup. You start with an arbitrary guess and keep updating each state's value based on its successors' values. Each sweep brings you closer to the true value.

**Policy iteration** alternates between two steps: evaluate the current policy (compute $V^\pi$), then improve it by acting greedily with respect to $V^\pi$ (pick the action that looks best given the current values). This creates a new, better policy, and you repeat. Policy iteration converges in a finite number of steps because there are only finitely many deterministic policies.

**Value iteration** is more direct — you repeatedly apply the Bellman *optimality* operator, which combines evaluation and improvement into one step. It converges to $V^*$ as the number of sweeps grows.

These methods are elegant and exact, but they require knowing $P$ and $R$. In the real world, you usually do not have that luxury. The next lessons are about what happens when you have to *learn* the values by interacting with the environment.

[[simulation gridworld-mdp]]
[[simulation dp-convergence]]
[[simulation mdp-simulation]]

## Big Ideas
* The Markov property is a compression miracle: all the history of how you got here is irrelevant; the current state contains everything you need to act optimally. That is not always true of the real world, but it is a useful lie.
* The Bellman equation is a self-consistency condition. The value of a state must equal what you get now plus the discounted value of where you end up. Any other assignment is unstable and will correct itself under iteration.
* Dynamic programming requires a perfect map. The moment you do not know the transition probabilities, the whole edifice becomes learning rather than planning. The next lessons are about learning without a map.
* The discount factor $\gamma$ is not just a mathematical convenience — it shapes behavior. A small $\gamma$ produces a myopic agent; $\gamma$ close to 1 produces a far-sighted one, but also a harder optimization problem.

## What Comes Next

Dynamic programming solves MDPs perfectly, but only when you already know the model — the transition probabilities and rewards. In the real world, you almost never have that. You have to learn by interacting.

The next lesson introduces Monte Carlo methods: run complete episodes, observe the total reward, and use those returns to estimate value functions. No model needed, no Bellman backup — just the law of large numbers applied to full trajectories. It is the simplest possible way to learn from experience in a sequential decision problem, and understanding its trade-offs — low bias, high variance, requires episode termination — sets up the more sophisticated TD methods that follow.

## Check Your Understanding
1. The Bellman optimality operator is a contraction with factor $\gamma$ under the max-norm. What does this mean geometrically, and why does it guarantee that value iteration converges to $V^*$?
2. Policy iteration alternates evaluation and greedy improvement. Why is the improved policy guaranteed to be at least as good as the old one after each improvement step?
3. Suppose you have a 100-state MDP and you want to compute $V^\pi$ for a fixed policy $\pi$. When would you prefer matrix inversion over iterative policy evaluation, and when would you prefer iteration?

## Challenge
Design a small MDP (5 states, 2 actions) where policy iteration converges in exactly 2 improvement steps but value iteration requires many sweeps to achieve the same accuracy. Compute both the exact optimal policy and the number of value-iteration sweeps needed to get within $\epsilon = 0.01$ of $V^*$ in the $\ell^\infty$ norm. What does this reveal about the relationship between policy and value convergence rates?
