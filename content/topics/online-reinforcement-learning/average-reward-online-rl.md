# Online RL in Average-Reward and Discounted Settings

## What if the game never ends?

Every algorithm we have built so far assumed the game eventually stops. An episode starts, rewards accumulate, the episode ends, and you get a clean total. But what about a server that runs 24/7? A robot that never shuts down? A factory production line that just keeps going? There is no "end of episode." There is no final score to look back on.

When the game never ends, the discounted return $\sum_{t=0}^{\infty}\gamma^t r_t$ still makes mathematical sense — the geometric discount makes far-future rewards negligible. But the discount factor $\gamma$ becomes an awkward modeling choice. If $\gamma$ is close to 1, you care about the far future but the problem is nearly ill-conditioned. If $\gamma$ is small, you are myopic. There is a more natural way to think about never-ending tasks.

## Discounted RL: a quick recap

In the discounted setting, you optimize

$$
\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right].
$$

The discount factor $\gamma < 1$ ensures this sum is finite. Rewards far in the future are worth exponentially less than immediate rewards. All the value functions, Bellman equations, and algorithms we have discussed work in this framework.

But here is the philosophical issue: in a continuing task, there is no reason why tomorrow's reward should be worth less than today's. The discount factor is a mathematical convenience, not a feature of the world. For many continuing tasks, what you really care about is: over the long run, what is my average reward per step?

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

Modern methods attack this with **optimism** or **posterior sampling**. **UCRL-style** algorithms build confidence sets for the transition probabilities and plan optimistically within those sets — just like UCB1 was optimistic about arm means, UCRL is optimistic about the MDP dynamics. **PSRL-style** (posterior sampling RL) algorithms maintain a Bayesian posterior over MDPs, sample one, solve it, and follow that policy for a while before re-sampling. Both achieve near-optimal regret scaling in tabular MDPs, closing the loop between the online learning theory from the beginning of the course and the full RL setting.

[[simulation average-reward-vs-discounted]]

---

*We have now covered the full arc: from regret in online learning, through bandits and contextual bandits, to full RL with MDPs, TD methods, deep function approximation, and continuing tasks. The next and final section gives you assignments and project ideas to cement all of these ideas through hands-on practice.*
