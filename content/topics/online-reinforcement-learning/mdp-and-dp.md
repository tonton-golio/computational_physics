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

*Try it yourself: in a 3-state chain (start → middle → goal), what does the value function look like when $\gamma = 0$ versus $\gamma = 0.99$? Pause and sketch both before reading on.*

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

### A note on the discount factor

The discount factor $\gamma$ appears everywhere in this lesson — in the value functions, the Bellman equations, the contraction argument. It is easy to treat it as just another number and move on. But it deserves a closer look, because it carries a philosophical assumption that is not always justified.

Why $\gamma < 1$? The mathematical reason is clear: without discounting, the infinite sum $\sum_{t=0}^{\infty} r_t$ can diverge. The discount factor forces convergence by making far-future rewards exponentially small. It also makes the Bellman operator a contraction, which guarantees a unique fixed point and the convergence of value iteration and policy iteration. Without $\gamma < 1$, the fixed-point machinery breaks down.

But there is a deeper issue. The discount factor says that tomorrow's reward is worth less than today's — not because of any feature of the environment, but by modeling convention. A small $\gamma$ produces a myopic agent that grabs whatever is nearby. A $\gamma$ close to 1 produces a far-sighted agent, but also a harder optimization problem (the contraction factor weakens, convergence slows, and numerical conditioning degrades).

For **episodic tasks** — games that end, mazes with exits, problems with a clear terminal state — discounting is a reasonable modeling choice. The episode ends, so the sum is finite even without discounting. Here, $\gamma$ acts as a soft preference for sooner rewards, and nothing goes badly wrong.

For **continuing tasks** — a server that runs 24/7, a robot that never shuts down, a factory line that keeps going — the discount factor becomes philosophically awkward. There is no natural reason why the ten-thousandth time step should matter less than the first. The agent will run just as long tomorrow as it did today. In these settings, $\gamma$ is a mathematical convenience masquerading as a modeling choice, and the resulting policies can depend sensitively on its exact value in ways that have no physical meaning.

This tension motivates the **average-reward formulation** introduced in the [final lesson](./average-reward-online-rl), which removes $\gamma$ entirely and instead asks: what is my average reward per time step in steady state? That formulation sidesteps the awkward question of how much to discount the future by refusing to discount at all.

For now, keep using $\gamma$ — it works well for the episodic and finite-horizon problems in the next several lessons. But remember that it is a choice, not a law.

## Big Ideas
* The Markov property is a compression miracle: all the history of how you got here is irrelevant; the current state contains everything you need to act optimally. That is not always true of the real world, but it is a useful lie.
* The Bellman equation is a self-consistency condition. The value of a state must equal what you get now plus the discounted value of where you end up. Any other assignment is unstable and will correct itself under iteration.
* Dynamic programming requires a perfect map. The moment you do not know the transition probabilities, the whole edifice becomes learning rather than planning. The next lessons are about learning without a map.
* The discount factor $\gamma$ is not just a mathematical convenience — it shapes behavior. As discussed above, it is a modeling choice that works well for episodic tasks but becomes philosophically awkward for continuing tasks. The [final lesson](./average-reward-online-rl) resolves this with the average-reward formulation.

## What Comes Next

Dynamic programming solves MDPs perfectly, but only when you know the model. In the real world, you have to learn by interacting. The next lesson introduces Monte Carlo methods: run complete episodes, observe returns, and estimate value functions — no model needed. Understanding its trade-offs (low bias, high variance, requires episode termination) sets up the TD methods that follow.

Later, in the [final lesson](./average-reward-online-rl), we will see how optimism-based planning (UCRL) brings together the model-based perspective from dynamic programming with the regret analysis from the opening lessons.

## Check Your Understanding
1. The Bellman optimality operator is a contraction with factor $\gamma$ under the max-norm. What does this mean geometrically, and why does it guarantee that value iteration converges to $V^*$?
2. Policy iteration alternates evaluation and greedy improvement. Why is the improved policy guaranteed to be at least as good as the old one after each improvement step?
3. Suppose you have a 100-state MDP and you want to compute $V^\pi$ for a fixed policy $\pi$. When would you prefer matrix inversion over iterative policy evaluation, and when would you prefer iteration?

## Challenge (Advanced)

**Advanced challenge (optional).** This exercise is aimed at students who want to go beyond the core material.

Design a small MDP (5 states, 2 actions) where policy iteration converges in exactly 2 improvement steps but value iteration requires many sweeps to achieve the same accuracy. Compute both the exact optimal policy and the number of value-iteration sweeps needed to get within $\epsilon = 0.01$ of $V^*$ in the $\ell^\infty$ norm. What does this reveal about the relationship between policy and value convergence rates?
