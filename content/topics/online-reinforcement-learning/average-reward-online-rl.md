# Online RL in Average-Reward and Discounted Settings

## What if the game never ends?

Every algorithm we have built so far assumed the game eventually stops. An episode starts, rewards accumulate, the episode ends, and you get a clean total. But what about a server that runs 24/7? A robot that never shuts down? A factory production line that just keeps going? There is no "end of episode." There is no final score to look back on.

When the game never ends, the discounted return $\sum_{t=0}^{\infty}\gamma^t r_t$ still makes mathematical sense — the geometric discount makes far-future rewards negligible. But as we discussed in the [MDP lesson](./mdp-and-dp), the discount factor becomes philosophically awkward for continuing tasks: there is no natural reason why tomorrow's reward should be worth less than today's. There is a more natural way to think about never-ending tasks.

## Discounted RL: a quick recap

In the discounted setting, you optimize

$$
\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right].
$$

The discount factor $\gamma < 1$ ensures this sum is finite. Rewards far in the future are worth exponentially less than immediate rewards. All the value functions, Bellman equations, and algorithms we have discussed work in this framework.

We discussed the philosophical implications of discounting in the [MDP lesson](./mdp-and-dp). The average-reward formulation resolves many of those concerns by removing $\gamma$ entirely. For many continuing tasks, what you really care about is: over the long run, what is my average reward per step?

## Average-reward RL: the steady-state view

In the average-reward formulation, you target the **long-run gain**:

$$
g^\pi(s)=\lim_{T\to\infty}\frac{1}{T}\mathbb{E}\left[\sum_{t=1}^{T}r_t \mid s_1=s\right].
$$

This is the average reward per time step, in the limit. For ergodic MDPs (where every state is reachable from every other state under any reasonable policy), the gain $g^\pi$ does not depend on the starting state — it is a single number that characterizes the steady-state performance of policy $\pi$.

Think of it like the average speed of a car on a long road trip. At the start, you accelerate and brake through city traffic. Eventually, you hit the highway and settle into a cruising speed. The average speed over the whole trip converges to something close to your cruising speed. The city traffic part is the transient; the highway cruising is the steady state.

## Gain and bias: steady state plus transient

The **gain-bias decomposition** separates value into two pieces. The **gain** $g^\pi$ captures the steady-state average reward — how well you do per step, once the system has settled. The **bias** $h^\pi(s)$ captures the transient advantage — how much better or worse it is to start in state $s$ compared to starting in steady state.

Together, they satisfy the Bellman-like equation: $g^\pi + h^\pi(s) = r^\pi(s) + \sum_{s'} P^\pi(s'|s) h^\pi(s')$. The gain is like the horizontal line of steady-state performance, and the bias tells you how far above or below that line you are at the beginning, depending on where you start.

Picture a graph of cumulative reward over time. After a while, the curve becomes approximately a straight line with slope $g^\pi$. The bias measures the vertical offset — some starting states get a head start (positive bias) and others start behind (negative bias), but eventually they all settle into the same slope.

## Algorithms for continuing tasks

**Relative value iteration** is the average-reward analog of value iteration. Instead of discounting, you subtract the gain at each step to keep the values from growing unboundedly.

**Differential TD** and **differential Q-learning** replace the discount factor with an estimate of the average reward. The TD error becomes $r - \bar{r} + V(s') - V(s)$, where $\bar{r}$ is a running estimate of the average reward per step.

**R-learning** maintains separate estimates of the average reward and the relative action-values, updating both as experience accumulates.

## Regret in online RL

When you bring the online learning perspective to RL, regret compares your cumulative reward to the optimal policy in the same environment class. You do not know the transition probabilities; you have to learn them by exploring. The regret measures how much reward you sacrifice during the learning phase.

Modern methods attack this with **optimism** or **posterior sampling**. **UCRL-style** algorithms build confidence sets for the transition probabilities and plan optimistically within those sets — just like UCB1 was optimistic about arm means, UCRL is optimistic about the MDP dynamics. **PSRL-style** (posterior sampling RL) algorithms maintain a Bayesian posterior over MDPs, sample one, solve it, and follow that policy for a while before re-sampling. Both achieve near-optimal regret scaling in tabular MDPs. These methods — particularly UCRL and PSRL — dominate modern regret analysis for tabular online RL and form the theoretical backbone of model-based RL, with their optimism and posterior-sampling principles extending to continuous and function-approximation settings. This closes the loop between the online learning theory from the beginning of the course and the full RL setting.

[[simulation average-reward-vs-discounted]]

## Big Ideas
* The discount factor smuggles in a philosophical assumption (see the [MDP lesson](./mdp-and-dp) for the full discussion). Average reward refuses that assumption and asks instead what happens in steady state.
* Gain and bias are the MDP analog of mean and variance: the gain captures steady-state performance, the bias captures the transient advantage of good starting positions.
* The exploration-exploitation trade-off (see the [bandits lesson](./bandits-ucb-exp3)) remains central here — UCRL extends the same optimism principle to MDP dynamics, closing the loop between bandit theory and full RL.
* The regret framework closes the loop: it takes the online learning question — how much do you lose by not knowing the best policy from the start? — and applies it to the full sequential decision problem. That is the deepest connection in this entire topic.

## What Comes Next

This lesson closes a long arc. We started with a simple question — how do you measure the cost of not knowing? — and followed it from regret through bandits, contextual bandits, MDPs, Monte Carlo, TD learning, deep function approximation, and finally continuing tasks with average reward.

The thread throughout is exploration versus exploitation. Every algorithm in this topic is a different answer to the same question: how do you act well under uncertainty, and how fast do you recover as uncertainty shrinks?

The ideas do not stop at the textbook. Average-reward RL underlies real-time control, UCRL-style optimism is the theoretical spine of model-based RL, and the deadly triad still keeps researchers up at night. The best way to consolidate is to build something: implement one algorithm end to end, break it, fix it, and understand in your hands why each design choice matters.

## Check Your Understanding
1. In a continuing task, why is the discounted objective $\mathbb{E}[\sum_{t=0}^\infty \gamma^t r_t]$ mathematically well-defined but philosophically awkward for a robot that runs indefinitely? What does the average-reward objective $g^\pi$ avoid?
2. Describe the gain-bias decomposition in your own words. Why does the bias function $h^\pi(s)$ make the Bellman-like equation well-posed even when there is no discount?
3. UCRL plans optimistically within a confidence set of MDPs. What happens to the size of that confidence set as the number of interactions grows? What does this imply about the policy UCRL follows as $T \to \infty$?

## Challenge (Advanced)

**Advanced challenge (optional).** This exercise is aimed at students who want to go beyond the core material.

Consider a 3-state ergodic MDP with two actions, where you do not know the transition probabilities. Design a UCRL-style algorithm from scratch: construct a confidence set for the transition probabilities using concentration inequalities, define the "optimistic MDP" as the one inside the confidence set that maximizes the gain, and specify how often you replan. Implement it and run it against a fixed (non-adaptive) policy on your MDP. Plot the regret — cumulative reward of the optimal policy minus your algorithm's cumulative reward — over 10,000 steps. Does it grow sublinearly? What is the empirical exponent, and how does it compare to the theoretical $O(\sqrt{T})$ bound?
