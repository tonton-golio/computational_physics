# Online RL in Average-Reward and Discounted Settings

## What if the game never ends?

Every algorithm we've built assumes the game eventually stops. An episode starts, rewards pile up, the episode ends. But what about a server running 24/7? A robot that never shuts down? A factory line that keeps going? No "end of episode." No final score.

When the game never ends, the discounted return $\sum_{t=0}^{\infty}\gamma^t r_t$ still works mathematically -- geometric discount makes far-future rewards negligible. But as we saw in the [MDP lesson](./mdp-and-dp), the discount factor gets awkward for continuing tasks. There's a more natural way.

## Discounted RL: the quick version

In the discounted setting, you optimize

$$
\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right].
$$

The discount factor $\gamma < 1$ keeps the sum finite. Rewards far in the future are worth exponentially less. All the value functions, Bellman equations, and algorithms we've covered work in this framework.

But for many continuing tasks, what you really care about is simpler: over the long run, what's my average reward per step?

## Average-reward RL: the steady-state view

In the average-reward formulation, you target the **long-run gain**:

$$
g^\pi(s)=\lim_{T\to\infty}\frac{1}{T}\mathbb{E}\left[\sum_{t=1}^{T}r_t \mid s_1=s\right].
$$

That's the average reward per step, in the limit. For ergodic MDPs (every state reachable from every other), the gain $g^\pi$ doesn't depend on where you start -- it's a single number capturing steady-state performance.

Think of it like average speed on a long road trip. At first you're accelerating and braking through city traffic. Eventually you hit the highway and settle into a cruising speed. The average over the whole trip converges to something close to that cruising speed. City traffic is the transient; highway cruising is the steady state.

## Gain and bias: steady state plus transient

The **gain-bias decomposition** splits value into two pieces. The **gain** $g^\pi$ captures steady-state average reward. The **bias** $h^\pi(s)$ captures the transient advantage -- how much better or worse it is to start in state $s$ compared to steady state.

Together they satisfy a Bellman-like equation: $g^\pi + h^\pi(s) = r^\pi(s) + \sum_{s'} P^\pi(s'|s) h^\pi(s')$. The gain is the horizontal line of steady-state performance. The bias tells you how far above or below that line you start.

Picture cumulative reward over time. After a while, the curve becomes roughly a straight line with slope $g^\pi$. The bias is the vertical offset -- some starting states get a head start (positive bias), others start behind (negative bias), but they all settle into the same slope.

## Regret in online RL

Now here's where the circle closes. Bring the online learning perspective to RL, and regret compares your cumulative reward to the optimal policy. You don't know the transition probabilities. You have to learn them by exploring. Regret measures how much reward you sacrifice during the learning phase.

This connects all the way back to [Lesson 2](./regret). Same question, richer setting.

Modern methods attack this with the same optimism principle from [UCB1](./bandits-ucb-exp3), extended to MDPs. **UCRL-style** algorithms build confidence sets for the transition probabilities and plan optimistically -- just like UCB1 was optimistic about arm means, UCRL is optimistic about MDP dynamics. **PSRL-style** (posterior sampling RL) algorithms maintain a Bayesian posterior over MDPs, sample one, solve it, and follow that policy for a while.

Modern methods simply bring the same optimism trick from UCB1 into the full MDP world. That's the closing of the circle.

## Algorithms for continuing tasks

**Relative value iteration** is the average-reward analog of value iteration. Instead of discounting, you subtract the gain at each step to keep values from growing without bound.

**Differential TD** and **differential Q-learning** replace the discount factor with a running estimate of average reward. The TD error becomes $r - \bar{r} + V(s') - V(s)$, where $\bar{r}$ estimates the average reward per step.

**R-learning** maintains separate estimates of the average reward and relative action-values, updating both as experience accumulates.

[[simulation average-reward-vs-discounted]]

## What Comes Next

This lesson closes a long arc. We started with a simple question -- how do you measure the cost of not knowing? -- and followed it from regret through bandits, contextual bandits, MDPs, Monte Carlo, TD learning, deep function approximation, and finally continuing tasks with average reward.

The thread throughout is exploration versus exploitation. Every algorithm in this topic is a different answer to the same question: how do you act well under uncertainty, and how fast do you recover as uncertainty shrinks?

The ideas don't stop at the textbook. Average-reward RL underlies real-time control, UCRL-style optimism is the theoretical spine of model-based RL, and the deadly triad still keeps researchers up at night. The best way to consolidate is to build something: implement one algorithm end to end, break it, fix it, and understand in your hands why each design choice matters. Oh... that's why.

## Challenge (Optional)

Consider a 3-state ergodic MDP with two actions, where you don't know the transition probabilities. Design a UCRL-style algorithm from scratch: build a confidence set for transitions using concentration inequalities, define the "optimistic MDP" as the one inside the set that maximizes the gain, and specify how often to replan. Implement it and run against a fixed policy on your MDP. Plot regret over 10,000 steps. Does it grow sublinearly? What's the empirical exponent, and how does it compare to the theoretical $O(\sqrt{T})$ bound?
