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

---

*We just learned how to plan in a world with memory — MDPs and dynamic programming give us the tools to compute optimal policies when we know the model. But what if we do not know the model? Next lesson, we learn Monte Carlo methods: play the game until it ends, write down the score, and learn from experience. No model required.*
