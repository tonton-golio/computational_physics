# MDPs and Dynamic Programming

## The world remembers

Up to now, the world had no memory. You pulled a lever, got a payout, and everything reset. Your action didn't change what happened next. Now the world *remembers*. And that changes everything.

Picture a robot navigating a warehouse. Turn left at the first aisle, and you're in a different part of the building than if you turned right. Your position depends on your previous action. And the reward -- finding the package vs hitting a wall -- depends on that position. This is a **Markov Decision Process**: a world where your actions have consequences that ripple forward in time.

## The MDP tuple

An MDP is five things:

$$
\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle
$$

$\mathcal{S}$ is all the places the world can be. $\mathcal{A}$ is all the moves you can make. $P$ is the transition kernel -- for every state and action, it tells you the probability of landing in each next state. $R$ is the reward function. And $\gamma$ is the discount factor, a number between 0 and 1 saying how much you care about future rewards versus immediate ones.

A **policy** $\pi$ tells you what to do in each state. The goal: find the policy that maximizes cumulative discounted reward.

## Value functions: scoring a policy

How good is a policy? We measure it with value functions. The state-value function $V^\pi(s)$ tells you the expected total discounted reward starting in state $s$ and following $\pi$ forever:

$$
V^\pi(s)=\mathbb{E}^\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s\right].
$$

The action-value function $Q^\pi(s,a)$ tells you the same thing, but conditioned on taking action $a$ first:

$$
Q^\pi(s,a)=\mathbb{E}^\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s,a_0=a\right].
$$

Think of $V^\pi(s)$ as the "worth" of being somewhere. A state near the goal with a good policy? High value. A state far from the goal with a bad policy? Low value.

## The Bellman equation

Here's the most important equation in reinforcement learning. It says: the value of a state is the immediate reward plus the discounted value of wherever you end up next.

$$
V^*(s)=\max_a\left[r(s,a)+\gamma\sum_{s'}P(s'|s,a)V^*(s')\right].
$$

Read it aloud: the optimal value of state $s$ is the best you can do by choosing the action that maximizes your immediate reward plus $\gamma$ times the expected value of the next state. That's the whole idea.

This equation has a **fixed point** -- a unique solution $V^*$ that doesn't budge when you apply the Bellman operator. If you start with any guess for $V$ and keep applying the operator, you converge to $V^*$. The operator is a **contraction**: every application brings you closer. That guarantees convergence and uniqueness.

## Why the discount factor is weird

The discount factor $\gamma$ deserves a closer look, because it carries a philosophical assumption that isn't always justified.

Why $\gamma < 1$? Mathematically, without discounting, the infinite sum $\sum_{t=0}^{\infty} r_t$ can diverge. Discounting forces convergence and makes the Bellman operator a contraction. Good.

But $\gamma$ also says tomorrow's reward is worth less than today's -- not because of anything in the environment, but by convention. Small $\gamma$ makes a myopic agent that grabs whatever's nearby. $\gamma$ close to 1 makes a far-sighted agent, but the optimization gets harder (contraction weakens, convergence slows).

For **episodic tasks** -- games that end, mazes with exits -- discounting is fine. The episode ends, so the sum is finite anyway. $\gamma$ just softly prefers sooner rewards.

For **continuing tasks** -- a server running 24/7, a robot that never shuts down -- the discount factor gets awkward. There's no natural reason the ten-thousandth step should matter less than the first. Here, $\gamma$ is a mathematical convenience pretending to be a modeling choice. The [final lesson](./average-reward-online-rl) resolves this with the average-reward formulation, which removes $\gamma$ entirely.

For now, keep using $\gamma$. But remember it's a choice, not a law.

## Dynamic programming: planning with a perfect map

When you know the transition probabilities and rewards -- when you have a perfect map of the world -- you can solve the MDP with dynamic programming. Dynamic programming is just "guess and check" done in the smartest possible order.

**Value iteration** repeatedly applies the Bellman optimality operator, combining evaluation and improvement into one step. Start with any guess, keep sweeping, and converge to $V^*$.

**Policy iteration** alternates two steps: evaluate the current policy (compute $V^\pi$), then improve it greedily (pick the action that looks best given current values). Repeat. Converges in finitely many steps because there are only finitely many deterministic policies.

**Policy evaluation** computes $V^\pi$ for a fixed policy by iterating the Bellman backup, or by solving the linear system $V^\pi = r^\pi + \gamma P^\pi V^\pi$ directly for small problems.

These methods are elegant and exact, but they need the model. No model, no dynamic programming. The next lessons are about learning without a map.

[[simulation gridworld-mdp]]
[[simulation dp-convergence]]
[[simulation mdp-simulation]]

## What Comes Next

Dynamic programming solves MDPs perfectly, but only when you know the model. In the real world, you have to learn by interacting. The next lesson introduces Monte Carlo methods: run complete episodes, observe returns, and estimate value functions -- no model needed. Understanding its trade-offs (low bias, high variance, requires episodes to end) sets up the TD methods that follow.

Later, in the [final lesson](./average-reward-online-rl), we'll see how optimism-based planning (UCRL) brings together model-based DP with the regret analysis from the opening lessons.

## Challenge (Optional)

Design a small MDP (5 states, 2 actions) where policy iteration converges in exactly 2 improvement steps but value iteration requires many sweeps to reach the same accuracy. Compute the exact optimal policy and the number of value-iteration sweeps needed to get within $\epsilon = 0.01$ of $V^*$ in the $\ell^\infty$ norm. What does this reveal about the relationship between policy and value convergence rates?
